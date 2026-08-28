// Package drift detects when live traffic stops looking like the data a model
// was trained on.
//
// A model does not know its inputs have changed. It keeps returning confident
// numbers while an upstream service starts sending amounts in cents instead of
// dollars, or a country expands into a new market, or a feature pipeline breaks
// and quietly sends zeros. Nothing in the serving path notices, because every
// request is individually well-formed — only the distribution moved. Comparing
// the live distribution against the training distribution is what turns that
// into an alert instead of a slow, invisible decline in prediction quality.
//
// The measure here is the Population Stability Index. For a feature binned into
// k buckets, with expected proportion e_i from the baseline and actual
// proportion a_i from live traffic:
//
//	PSI = Σ (a_i - e_i) · ln(a_i / e_i)
//
// It is symmetric, zero when the distributions match, and grows with the size
// of the shift. The conventional readings — below 0.1 stable, 0.1 to 0.25
// moderate, above 0.25 significant — are what Thresholds encodes.
package drift

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

// Conventional PSI thresholds.
const (
	// ThresholdModerate is where a shift stops being noise and is worth
	// looking at.
	ThresholdModerate = 0.1
	// ThresholdSignificant is where the live distribution has moved enough
	// that the model's calibration should not be trusted.
	ThresholdSignificant = 0.25
)

// Severity is a PSI reading turned into a verdict.
type Severity string

const (
	SeverityStable      Severity = "stable"
	SeverityModerate    Severity = "moderate"
	SeveritySignificant Severity = "significant"
)

// Classify turns a PSI value into a severity.
func Classify(psi float64) Severity {
	switch {
	case psi >= ThresholdSignificant:
		return SeveritySignificant
	case psi >= ThresholdModerate:
		return SeverityModerate
	default:
		return SeverityStable
	}
}

// Baseline is the reference distribution of one feature, captured from training
// data at registration time.
type Baseline struct {
	Feature string `json:"feature"`
	// Edges are the internal bin boundaries: len(Edges) == len(Expected)-1.
	// A value falls in bin i when Edges[i-1] <= v < Edges[i].
	Edges []float64 `json:"edges"`
	// Expected is the proportion of baseline samples in each bin. Sums to 1.
	Expected []float64 `json:"expected"`
}

// Bins returns the number of buckets in the baseline.
func (b Baseline) Bins() int { return len(b.Expected) }

// NewBaseline builds a baseline from training samples using quantile bins.
//
// Quantile bins rather than equal-width bins, because equal-width bins on a
// skewed feature — which most real features are — put nearly every sample in
// one bucket and leave the rest empty. PSI computed over bins that were already
// empty in the baseline measures almost nothing: the statistic's resolution
// lives where the data is, and quantiles put the boundaries there by
// construction.
//
// Ties are why the bin count can come back lower than requested. A feature that
// is zero for 60% of rows cannot have distinct deciles, and duplicating an edge
// would create an empty bin whose expected proportion is zero — which makes the
// PSI term for it undefined. Collapsing to the distinct edges is the honest
// outcome, so a caller checking Bins() can see what it actually got.
func NewBaseline(feature string, samples []float64, bins int) (Baseline, error) {
	if feature == "" {
		return Baseline{}, errors.New("drift: baseline needs a feature name")
	}
	if bins < 2 {
		return Baseline{}, fmt.Errorf("drift: need at least 2 bins, got %d", bins)
	}
	clean := make([]float64, 0, len(samples))
	for _, v := range samples {
		// A NaN cannot be placed in an ordered bin. Missing values are a
		// legitimate part of the input and are counted separately by the
		// monitor rather than being forced into a numeric bucket.
		if !math.IsNaN(v) && !math.IsInf(v, 0) {
			clean = append(clean, v)
		}
	}
	if len(clean) < bins {
		return Baseline{}, fmt.Errorf("drift: need at least %d finite samples for %d bins, got %d", bins, bins, len(clean))
	}

	sorted := append([]float64(nil), clean...)
	sort.Float64s(sorted)

	// Internal edges at the bin quantiles, deduplicated.
	//
	// Edges at or below the smallest sample are dropped. Bins are half-open
	// [lo, hi), so an edge equal to the minimum would make the first bin
	// [-inf, min) — a bin no value can ever fall into, whose expected
	// proportion is therefore zero and whose PSI term is undefined. That is
	// not a theoretical case: it is what a feature that is zero for most rows
	// produces, since its lower quantiles are all the minimum.
	//
	// With that filter every bin is guaranteed non-empty. Edges are distinct
	// values drawn from the data, so bin 0 contains the minimum, each middle
	// bin [e_j, e_j+1) contains e_j itself, and the last contains the largest
	// edge.
	edges := make([]float64, 0, bins-1)
	for i := 1; i < bins; i++ {
		idx := i * len(sorted) / bins
		if idx >= len(sorted) {
			idx = len(sorted) - 1
		}
		e := sorted[idx]
		if e <= sorted[0] {
			continue
		}
		if len(edges) == 0 || e > edges[len(edges)-1] {
			edges = append(edges, e)
		}
	}
	if len(edges) == 0 {
		// Every sample is the same value, or near enough that no quantile
		// exceeds the minimum. There is no distribution to stabilise against
		// and PSI would be meaningless.
		return Baseline{}, fmt.Errorf("drift: feature %q is constant across the baseline, so drift is not measurable", feature)
	}

	b := Baseline{Feature: feature, Edges: edges, Expected: make([]float64, len(edges)+1)}
	counts := make([]float64, len(b.Expected))
	for _, v := range clean {
		counts[bucket(edges, v)]++
	}
	total := float64(len(clean))
	for i, c := range counts {
		if c == 0 {
			// Unreachable given the edge filter above, but asserted rather
			// than assumed: a zero-expected bin silently makes that bin's PSI
			// term undefined for the life of the model, and the reading would
			// look plausible while ignoring part of the distribution.
			return Baseline{}, fmt.Errorf("drift: feature %q produced an empty bin %d; the baseline is too degenerate to bin", feature, i)
		}
		b.Expected[i] = c / total
	}
	return b, nil
}

// bucket returns the index of the bin a value falls in.
func bucket(edges []float64, v float64) int {
	// sort.SearchFloat64s finds the first edge >= v; a value equal to an edge
	// belongs to the bin starting at that edge, which is what makes the bins
	// half-open [lo, hi) and consistent with how the baseline was counted.
	i := sort.SearchFloat64s(edges, v)
	for i < len(edges) && edges[i] == v {
		i++
	}
	return i
}

// PSI compares an observed set of bin counts against the baseline.
//
// The subtlety is empty bins. If a bin has no observations its actual
// proportion is zero, ln(0) is negative infinity, and the PSI for the whole
// feature becomes infinite — which is not a useful alert, since an empty bin is
// entirely ordinary at small sample sizes. The fix here is a continuity
// correction: an empty bin is treated as holding half an observation, the same
// convention used for zero cells in a contingency table. That keeps the value
// finite while still making a genuinely vanished bucket contribute a large
// term, and it shrinks as the sample grows, so it does not distort a reading
// taken over a lot of traffic.
func (b Baseline) PSI(counts []int64) (float64, error) {
	if len(counts) != len(b.Expected) {
		return 0, fmt.Errorf("drift: got %d bin counts for a %d-bin baseline", len(counts), len(b.Expected))
	}
	var total int64
	for _, c := range counts {
		if c < 0 {
			return 0, fmt.Errorf("drift: negative bin count %d", c)
		}
		total += c
	}
	if total == 0 {
		return 0, errors.New("drift: no observations")
	}

	n := float64(total)
	var psi float64
	for i, c := range counts {
		expected := b.Expected[i]
		if expected <= 0 {
			// A bin the baseline never populated cannot be a reference point;
			// PSI's ratio is undefined against zero. NewBaseline does not
			// create these, but a hand-written baseline could.
			continue
		}
		actual := float64(c) / n
		if c == 0 {
			actual = 0.5 / n // continuity correction
		}
		psi += (actual - expected) * math.Log(actual/expected)
	}
	return psi, nil
}

// Bucket returns which bin a value belongs to, for callers accumulating counts.
func (b Baseline) Bucket(v float64) int { return bucket(b.Edges, v) }
