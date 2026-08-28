package drift

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"
	"time"
)

func normalSamples(n int, mean, sd float64, seed uint64) []float64 {
	r := rand.New(rand.NewPCG(seed, 0x9e3779b9))
	out := make([]float64, n)
	for i := range out {
		out[i] = r.NormFloat64()*sd + mean
	}
	return out
}

func countInto(b Baseline, values []float64) []int64 {
	counts := make([]int64, b.Bins())
	for _, v := range values {
		if math.IsNaN(v) {
			continue
		}
		counts[b.Bucket(v)]++
	}
	return counts
}

func TestNewBaselineUsesQuantileBins(t *testing.T) {
	// A deliberately skewed feature: equal-width bins would put nearly
	// everything in one bucket, which is the case quantile binning exists for.
	samples := make([]float64, 10000)
	r := rand.New(rand.NewPCG(1, 2))
	for i := range samples {
		samples[i] = math.Exp(r.NormFloat64() * 2)
	}

	b, err := NewBaseline("amount", samples, 10)
	if err != nil {
		t.Fatalf("NewBaseline: %v", err)
	}
	if b.Bins() != 10 {
		t.Fatalf("got %d bins, want 10", b.Bins())
	}

	// Quantile bins should each hold roughly a tenth of the data. That is the
	// property that keeps PSI's resolution where the data actually is.
	for i, e := range b.Expected {
		if e < 0.05 || e > 0.15 {
			t.Errorf("bin %d holds %.1f%% of the baseline, want about 10%%", i, e*100)
		}
	}

	var sum float64
	for _, e := range b.Expected {
		sum += e
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Errorf("expected proportions sum to %v, want 1", sum)
	}
}

// TestPSIIsNearZeroForTheSameDistribution is the calibration check: a fresh
// draw from the distribution the baseline came from must not look like drift.
func TestPSIIsNearZeroForTheSameDistribution(t *testing.T) {
	base := normalSamples(20000, 0, 1, 42)
	b, err := NewBaseline("x", base, 10)
	if err != nil {
		t.Fatal(err)
	}

	fresh := normalSamples(20000, 0, 1, 99) // same distribution, different draw
	psi, err := b.PSI(countInto(b, fresh))
	if err != nil {
		t.Fatal(err)
	}
	if psi >= ThresholdModerate {
		t.Errorf("PSI = %.4f for an identically distributed sample; it should read as stable", psi)
	}
	if Classify(psi) != SeverityStable {
		t.Errorf("classified as %s, want stable", Classify(psi))
	}
	t.Logf("same distribution: PSI %.4f", psi)
}

// TestPSIGrowsWithTheShift checks the statistic is monotone in the thing it is
// supposed to measure — a bigger move must produce a bigger number, or the
// thresholds mean nothing.
func TestPSIGrowsWithTheShift(t *testing.T) {
	b, err := NewBaseline("x", normalSamples(20000, 0, 1, 7), 10)
	if err != nil {
		t.Fatal(err)
	}

	var last float64
	for i, shift := range []float64{0, 0.25, 0.5, 1.0, 2.0} {
		psi, err := b.PSI(countInto(b, normalSamples(20000, shift, 1, 11)))
		if err != nil {
			t.Fatal(err)
		}
		if i > 0 && psi <= last {
			t.Errorf("PSI did not increase for a mean shift of %v: %.4f after %.4f", shift, psi, last)
		}
		t.Logf("mean shift %.2f -> PSI %.4f (%s)", shift, psi, Classify(psi))
		last = psi
	}
	if last < ThresholdSignificant {
		t.Errorf("a two-sigma mean shift gave PSI %.4f, which does not read as significant", last)
	}
}

// TestPSIIsFiniteWhenABinEmpties is the numerical case the continuity
// correction exists for. Without it ln(0) makes PSI infinite, and an infinite
// alert value is unusable — it cannot be compared, averaged or graphed.
func TestPSIIsFiniteWhenABinEmpties(t *testing.T) {
	b, err := NewBaseline("x", normalSamples(10000, 0, 1, 3), 10)
	if err != nil {
		t.Fatal(err)
	}

	// Everything collapses into the middle of the range, emptying the tails.
	collapsed := make([]float64, 5000)
	for i := range collapsed {
		collapsed[i] = b.Edges[len(b.Edges)/2]
	}

	psi, err := b.PSI(countInto(b, collapsed))
	if err != nil {
		t.Fatal(err)
	}
	if math.IsInf(psi, 0) || math.IsNaN(psi) {
		t.Fatalf("PSI = %v with empty bins; the continuity correction did not apply", psi)
	}
	if Classify(psi) != SeveritySignificant {
		t.Errorf("a fully collapsed distribution classified as %s, want significant (PSI %.3f)", Classify(psi), psi)
	}
	t.Logf("collapsed distribution: PSI %.3f", psi)
}

// TestContinuityCorrectionShrinksWithSampleSize checks the correction does not
// distort large-sample readings: half an observation is a smaller share of a
// bigger window, so its contribution has to fall away.
func TestContinuityCorrectionShrinksWithSampleSize(t *testing.T) {
	b, err := NewBaseline("x", normalSamples(10000, 0, 1, 5), 4)
	if err != nil {
		t.Fatal(err)
	}

	var last float64
	for i, scale := range []int64{1, 10, 100, 1000} {
		// One bin empty, the rest carrying identical counts.
		counts := make([]int64, b.Bins())
		for j := 1; j < len(counts); j++ {
			counts[j] = 100 * scale
		}
		psi, err := b.PSI(counts)
		if err != nil {
			t.Fatal(err)
		}
		if i > 0 && psi <= last {
			// The empty bin's term grows as its imputed proportion shrinks, so
			// PSI must rise with sample size for a genuinely absent bucket.
			t.Errorf("PSI for an empty bin did not grow with sample size: %.4f after %.4f", psi, last)
		}
		last = psi
	}
}

func TestBaselineValidation(t *testing.T) {
	for _, tc := range []struct {
		name    string
		feature string
		samples []float64
		bins    int
		wantErr string
	}{
		{"no name", "", []float64{1, 2, 3, 4}, 2, "feature name"},
		{"too few bins", "x", []float64{1, 2, 3, 4}, 1, "at least 2 bins"},
		{"too few samples", "x", []float64{1}, 4, "at least 4 finite samples"},
		{"constant feature", "x", []float64{5, 5, 5, 5, 5, 5}, 3, "constant"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewBaseline(tc.feature, tc.samples, tc.bins)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("NewBaseline = %v, want an error containing %q", err, tc.wantErr)
			}
		})
	}
}

// TestBaselineCollapsesTiedEdges covers the heavily-tied feature: a value that
// is zero for most rows cannot have distinct deciles, and duplicating an edge
// would create a zero-expected bin whose PSI term is undefined.
func TestBaselineCollapsesTiedEdges(t *testing.T) {
	samples := make([]float64, 1000)
	for i := range samples {
		if i >= 700 {
			samples[i] = float64(i)
		} // the first 700 are all 0
	}

	b, err := NewBaseline("mostly_zero", samples, 10)
	if err != nil {
		t.Fatalf("NewBaseline: %v", err)
	}
	if b.Bins() >= 10 {
		t.Errorf("got %d bins for a feature that is 70%% ties; edges should have collapsed", b.Bins())
	}
	for i, e := range b.Expected {
		if e <= 0 {
			t.Errorf("bin %d has zero expected proportion, which makes its PSI term undefined", i)
		}
	}
}

func TestBaselineIgnoresNonFiniteSamples(t *testing.T) {
	samples := append(normalSamples(1000, 0, 1, 4), math.NaN(), math.Inf(1), math.Inf(-1))
	b, err := NewBaseline("x", samples, 5)
	if err != nil {
		t.Fatalf("NewBaseline: %v", err)
	}
	var sum float64
	for _, e := range b.Expected {
		sum += e
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Errorf("proportions sum to %v; non-finite samples were counted", sum)
	}
}

func TestPSIInputValidation(t *testing.T) {
	b, err := NewBaseline("x", normalSamples(1000, 0, 1, 6), 4)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := b.PSI([]int64{1, 2}); err == nil {
		t.Error("PSI accepted the wrong number of bins")
	}
	if _, err := b.PSI(make([]int64, b.Bins())); err == nil {
		t.Error("PSI accepted an empty window")
	}
	bad := make([]int64, b.Bins())
	bad[0] = -1
	if _, err := b.PSI(bad); err == nil {
		t.Error("PSI accepted a negative count")
	}
}

func TestPSISkipsBinsTheBaselineNeverPopulated(t *testing.T) {
	// A hand-written baseline with a zero-expected bin. NewBaseline never
	// produces one, but PSI must not divide by it.
	b := Baseline{Feature: "x", Edges: []float64{0, 1}, Expected: []float64{0.5, 0, 0.5}}
	psi, err := b.PSI([]int64{50, 10, 40})
	if err != nil {
		t.Fatal(err)
	}
	if math.IsInf(psi, 0) || math.IsNaN(psi) {
		t.Fatalf("PSI = %v, want a finite value", psi)
	}
}

func TestClassify(t *testing.T) {
	for _, tc := range []struct {
		psi  float64
		want Severity
	}{
		{0, SeverityStable},
		{0.09, SeverityStable},
		{0.1, SeverityModerate},
		{0.24, SeverityModerate},
		{0.25, SeveritySignificant},
		{5, SeveritySignificant},
	} {
		if got := Classify(tc.psi); got != tc.want {
			t.Errorf("Classify(%v) = %s, want %s", tc.psi, got, tc.want)
		}
	}
}

func TestBucketBoundariesAreHalfOpen(t *testing.T) {
	b := Baseline{Edges: []float64{10, 20}, Expected: []float64{0.3, 0.4, 0.3}}
	for _, tc := range []struct {
		v    float64
		want int
	}{
		{5, 0}, {9.999, 0},
		{10, 1}, {15, 1}, {19.999, 1},
		{20, 2}, {100, 2},
	} {
		if got := b.Bucket(tc.v); got != tc.want {
			t.Errorf("Bucket(%v) = %d, want %d", tc.v, got, tc.want)
		}
	}
}

// --- Monitor ---

type fakeClock struct{ t time.Time }

func (c *fakeClock) now() time.Time          { return c.t }
func (c *fakeClock) advance(d time.Duration) { c.t = c.t.Add(d) }

func testMonitor(t *testing.T, clock *fakeClock, window time.Duration) *Monitor {
	t.Helper()
	b1, err := NewBaseline("amount", normalSamples(5000, 0, 1, 1), 10)
	if err != nil {
		t.Fatal(err)
	}
	b2, err := NewBaseline("age", normalSamples(5000, 0, 1, 2), 10)
	if err != nil {
		t.Fatal(err)
	}
	pred, err := NewBaseline("prediction", normalSamples(5000, 0.5, 0.1, 3), 10)
	if err != nil {
		t.Fatal(err)
	}

	m, err := NewMonitor(Config{
		Model: "fraud", Version: 3, Window: window, Buckets: 6, Now: clock.now,
	}, []Baseline{b1, b2}, &pred)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestMonitorNeedsBaselines(t *testing.T) {
	if _, err := NewMonitor(Config{Model: "m"}, nil, nil); err == nil {
		t.Fatal("NewMonitor accepted no baselines")
	}
}

// TestMonitorWithholdsSmallSamples covers the alerting-noise decision. PSI over
// a handful of rows is dominated by which bins happened to be hit.
func TestMonitorWithholdsSmallSamples(t *testing.T) {
	clock := &fakeClock{t: time.Date(2026, 8, 28, 12, 0, 0, 0, time.UTC)}
	m := testMonitor(t, clock, 15*time.Minute)

	for range MinSamples - 1 {
		m.Observe([]float64{0, 0}, []float64{0.5})
	}
	if _, ok := m.Report(); ok {
		t.Fatalf("reported a PSI on fewer than %d samples", MinSamples)
	}

	m.Observe([]float64{0, 0}, []float64{0.5})
	rep, ok := m.Report()
	if !ok {
		t.Fatal("no report after reaching the sample threshold")
	}
	if rep.Samples != MinSamples {
		t.Errorf("Samples = %d, want %d", rep.Samples, MinSamples)
	}
}

func TestMonitorSeesNoDriftOnBaselineTraffic(t *testing.T) {
	clock := &fakeClock{t: time.Date(2026, 8, 28, 12, 0, 0, 0, time.UTC)}
	m := testMonitor(t, clock, 15*time.Minute)

	a := normalSamples(4000, 0, 1, 1001)
	b := normalSamples(4000, 0, 1, 1002)
	for i := range a {
		m.Observe([]float64{a[i], b[i]}, []float64{0.5})
	}

	rep, ok := m.Report()
	if !ok {
		t.Fatal("no report")
	}
	for _, f := range rep.Features {
		if f.Severity != SeverityStable {
			t.Errorf("feature %q read %s (PSI %.4f) on baseline-distributed traffic", f.Feature, f.Severity, f.PSI)
		}
	}
}

func TestMonitorDetectsAShiftedFeature(t *testing.T) {
	clock := &fakeClock{t: time.Date(2026, 8, 28, 12, 0, 0, 0, time.UTC)}
	m := testMonitor(t, clock, 15*time.Minute)

	// "amount" moves by three sigma; "age" does not.
	shifted := normalSamples(4000, 3, 1, 2001)
	steady := normalSamples(4000, 0, 1, 2002)
	for i := range shifted {
		m.Observe([]float64{shifted[i], steady[i]}, []float64{0.5})
	}

	rep, ok := m.Report()
	if !ok {
		t.Fatal("no report")
	}
	byName := map[string]Reading{}
	for _, f := range rep.Features {
		byName[f.Feature] = f
	}
	if byName["amount"].Severity != SeveritySignificant {
		t.Errorf("shifted feature read %s (PSI %.3f), want significant",
			byName["amount"].Severity, byName["amount"].PSI)
	}
	if byName["age"].Severity != SeverityStable {
		t.Errorf("unshifted feature read %s (PSI %.3f), want stable",
			byName["age"].Severity, byName["age"].PSI)
	}

	// The alert should name the feature that actually moved.
	worst, ok := rep.Worst()
	if !ok || worst.Feature != "amount" {
		t.Errorf("Worst() = %+v, want the amount feature", worst)
	}
	t.Logf("amount PSI %.3f, age PSI %.3f", byName["amount"].PSI, byName["age"].PSI)
}

// TestMonitorTracksMissingRateSeparately covers the broken-pipeline case: a
// feature that becomes mostly NaN is a serious problem, and folding those rows
// into a numeric bin would hide it while also corrupting the value distribution.
func TestMonitorTracksMissingRateSeparately(t *testing.T) {
	clock := &fakeClock{t: time.Date(2026, 8, 28, 12, 0, 0, 0, time.UTC)}
	m := testMonitor(t, clock, 15*time.Minute)

	vals := normalSamples(1000, 0, 1, 3001)
	for i, v := range vals {
		amount := v
		if i%2 == 0 {
			amount = math.NaN()
		}
		m.Observe([]float64{amount, v}, []float64{0.5})
	}

	rep, ok := m.Report()
	if !ok {
		t.Fatal("no report")
	}
	var amount Reading
	for _, f := range rep.Features {
		if f.Feature == "amount" {
			amount = f
		}
	}
	if math.Abs(amount.MissingRate-0.5) > 0.02 {
		t.Errorf("MissingRate = %.3f, want about 0.5", amount.MissingRate)
	}
	// The half that did arrive is baseline-distributed, so the values
	// themselves must still read as stable.
	if amount.Severity != SeverityStable {
		t.Errorf("value drift read %s (PSI %.3f); missing rows should be excluded from the bins, not binned",
			amount.Severity, amount.PSI)
	}
}

func TestMonitorDetectsPredictionDrift(t *testing.T) {
	clock := &fakeClock{t: time.Date(2026, 8, 28, 12, 0, 0, 0, time.UTC)}
	m := testMonitor(t, clock, 15*time.Minute)

	inputs := normalSamples(3000, 0, 1, 4001)
	// The prediction baseline is centred at 0.5 with sd 0.1; the model has
	// started returning much higher scores.
	for _, v := range inputs {
		m.Observe([]float64{v, v}, []float64{0.95})
	}

	rep, ok := m.Report()
	if !ok {
		t.Fatal("no report")
	}
	if rep.Prediction == nil {
		t.Fatal("no prediction reading")
	}
	if rep.Prediction.Severity != SeveritySignificant {
		t.Errorf("prediction drift read %s (PSI %.3f), want significant",
			rep.Prediction.Severity, rep.Prediction.PSI)
	}
}

// TestWindowSlides is what makes the reading current rather than cumulative: a
// drift that has stopped must stop being reported.
func TestWindowSlides(t *testing.T) {
	clock := &fakeClock{t: time.Date(2026, 8, 28, 12, 0, 0, 0, time.UTC)}
	m := testMonitor(t, clock, 6*time.Minute) // 6 buckets of 1 minute

	// A burst of badly shifted traffic.
	shifted := normalSamples(1000, 4, 1, 5001)
	for _, v := range shifted {
		m.Observe([]float64{v, v}, []float64{0.5})
	}
	rep, _ := m.Report()
	worst, _ := rep.Worst()
	if worst.Severity != SeveritySignificant {
		t.Fatalf("shifted burst read %s, want significant", worst.Severity)
	}

	// Time passes and healthy traffic replaces it.
	clock.advance(7 * time.Minute)
	healthy := normalSamples(1000, 0, 1, 5002)
	for _, v := range healthy {
		m.Observe([]float64{v, v}, []float64{0.5})
	}

	rep, ok := m.Report()
	if !ok {
		t.Fatal("no report after the window slid")
	}
	if rep.Samples != int64(len(healthy)) {
		t.Errorf("Samples = %d, want only the %d in-window observations", rep.Samples, len(healthy))
	}
	for _, f := range rep.Features {
		if f.Severity != SeverityStable {
			t.Errorf("feature %q still reads %s after the shifted traffic aged out", f.Feature, f.Severity)
		}
	}
}

func TestObserveIgnoresWrongWidthRows(t *testing.T) {
	clock := &fakeClock{t: time.Date(2026, 8, 28, 12, 0, 0, 0, time.UTC)}
	m := testMonitor(t, clock, 15*time.Minute)

	// Drift monitoring must never be the thing that fails a request.
	for range 1000 {
		m.Observe([]float64{1}, []float64{0.5})
		m.Observe([]float64{1, 2, 3}, []float64{0.5})
	}
	if rep, ok := m.Report(); ok || rep.Samples != 0 {
		t.Errorf("wrong-width rows were recorded: %d samples", rep.Samples)
	}
}

func TestMonitorFeatures(t *testing.T) {
	clock := &fakeClock{t: time.Date(2026, 8, 28, 12, 0, 0, 0, time.UTC)}
	m := testMonitor(t, clock, 15*time.Minute)
	got := m.Features()
	if len(got) != 2 || got[0] != "amount" || got[1] != "age" {
		t.Errorf("Features() = %v", got)
	}
}

func TestMonitorWithoutPredictionBaseline(t *testing.T) {
	clock := &fakeClock{t: time.Date(2026, 8, 28, 12, 0, 0, 0, time.UTC)}
	b, err := NewBaseline("x", normalSamples(1000, 0, 1, 9), 5)
	if err != nil {
		t.Fatal(err)
	}
	m, err := NewMonitor(Config{Model: "m", Version: 1, Now: clock.now}, []Baseline{b}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for range MinSamples {
		m.Observe([]float64{0.1}, []float64{0.5})
	}
	rep, ok := m.Report()
	if !ok {
		t.Fatal("no report")
	}
	if rep.Prediction != nil {
		t.Error("reported prediction drift without a prediction baseline")
	}
}

func TestReportWorstOnEmptyReport(t *testing.T) {
	if _, ok := (Report{}).Worst(); ok {
		t.Error("Worst() found a reading in an empty report")
	}
}
