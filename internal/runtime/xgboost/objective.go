package xgboost

import (
	"fmt"
	"math"
)

// Objective describes how a model's raw margin becomes a prediction.
//
// Two functions matter and they are inverses of each other. Transform runs
// forward, turning the summed leaf values into whatever the model is supposed
// to output. MarginFromScore runs backward, and exists only because XGBoost
// stores base_score in prediction space for single-output models — see
// resolveIntercept.
//
// Keeping both on one value, rather than two switch statements somewhere in the
// scoring path, is what makes it impossible to add an objective that can be
// scored forward but whose intercept is then silently wrong.
type Objective struct {
	// Name is the objective string as XGBoost writes it, e.g. "binary:logistic".
	Name string

	// MultiClass reports whether the model emits one margin per class.
	MultiClass bool

	// OutputWidth returns the number of values a prediction holds for a model
	// with the given number of groups. It is the number of groups for every
	// objective except multi:softmax, which collapses its margins to a single
	// class index.
	OutputWidth func(numGroup int) int

	// transform maps margins to predictions in place-friendly fashion: it
	// receives the margin slice for one row and returns that row's prediction.
	transform func(margin []float64) []float64

	// marginFromScore inverts the per-element output transform. nil means the
	// objective has no meaningful single-output inverse, which is only the case
	// for multi-class objectives — and those never need it, because they store
	// their intercept in margin space already.
	marginFromScore func(score float64) (float64, error)
}

// Transform converts a row's raw margins into its prediction.
func (o Objective) Transform(margin []float64) []float64 { return o.transform(margin) }

// MarginFromScore converts a stored base_score from prediction space into
// margin space.
func (o Objective) MarginFromScore(score float64) (float64, error) {
	if o.marginFromScore == nil {
		return 0, fmt.Errorf("xgboost: objective %q has no scalar inverse link", o.Name)
	}
	return o.marginFromScore(score)
}

func identityWidth(numGroup int) int { return numGroup }

// objectives is the set this scorer will load. Every entry here is covered by a
// fixture generated from XGBoost in tools/fixtures/generate.py; an objective
// with no fixture does not belong in this map, because "supported" would then
// mean "compiles" rather than "produces the same numbers XGBoost does".
var objectives = map[string]Objective{
	"binary:logistic": {
		Name:            "binary:logistic",
		OutputWidth:     identityWidth,
		transform:       elementwise(sigmoid),
		marginFromScore: logit,
	},
	"reg:logistic": {
		Name:            "reg:logistic",
		OutputWidth:     identityWidth,
		transform:       elementwise(sigmoid),
		marginFromScore: logit,
	},
	"reg:squarederror": {
		Name:            "reg:squarederror",
		OutputWidth:     identityWidth,
		transform:       elementwise(func(v float64) float64 { return v }),
		marginFromScore: func(v float64) (float64, error) { return v, nil },
	},
	"count:poisson": {
		Name:        "count:poisson",
		OutputWidth: identityWidth,
		transform:   elementwise(math.Exp),
		marginFromScore: func(v float64) (float64, error) {
			if v <= 0 {
				return 0, fmt.Errorf("xgboost: count:poisson base_score must be positive, got %v", v)
			}
			return math.Log(v), nil
		},
	},
	"binary:logitraw": {
		Name:            "binary:logitraw",
		OutputWidth:     identityWidth,
		transform:       elementwise(func(v float64) float64 { return v }),
		marginFromScore: func(v float64) (float64, error) { return v, nil },
	},
	"multi:softprob": {
		Name:        "multi:softprob",
		MultiClass:  true,
		OutputWidth: identityWidth,
		transform:   softmax,
	},
	"multi:softmax": {
		Name:       "multi:softmax",
		MultiClass: true,
		// softmax is monotonic, so the arg max of the probabilities is the arg
		// max of the margins; there is no reason to exponentiate first.
		OutputWidth: func(int) int { return 1 },
		transform:   argmax,
	},
}

func lookupObjective(name string) (Objective, error) {
	obj, ok := objectives[name]
	if !ok {
		return Objective{}, fmt.Errorf("xgboost: unsupported objective %q", name)
	}
	return obj, nil
}

func elementwise(f func(float64) float64) func([]float64) []float64 {
	return func(margin []float64) []float64 {
		out := make([]float64, len(margin))
		for i, v := range margin {
			out[i] = f(v)
		}
		return out
	}
}

// sigmoid is written to avoid overflow for large-magnitude margins. The naive
// 1/(1+exp(-x)) form overflows exp for x below about -709 and returns +Inf
// rather than 0; boosted models reach margins like that on confidently
// classified rows, so this is a real input and not a theoretical one.
func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	e := math.Exp(x)
	return e / (1 + e)
}

func logit(p float64) (float64, error) {
	if p <= 0 || p >= 1 {
		return 0, fmt.Errorf("xgboost: logistic base_score must be in (0,1), got %v", p)
	}
	return math.Log(p / (1 - p)), nil
}

// softmax subtracts the maximum margin before exponentiating. Without that
// shift a margin above ~709 overflows to +Inf and the whole row becomes NaN
// after the division.
func softmax(margin []float64) []float64 {
	out := make([]float64, len(margin))
	if len(margin) == 0 {
		return out
	}
	max := margin[0]
	for _, v := range margin[1:] {
		if v > max {
			max = v
		}
	}
	var sum float64
	for i, v := range margin {
		e := math.Exp(v - max)
		out[i] = e
		sum += e
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// argmax returns the index of the largest margin as a single-element slice.
// Ties go to the lowest index, matching XGBoost.
func argmax(margin []float64) []float64 {
	best := 0
	for i, v := range margin {
		if v > margin[best] {
			best = i
		}
	}
	return []float64{float64(best)}
}
