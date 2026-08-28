package xgboost

import (
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"
)

// modelJSON builds a minimal but valid XGBoost model document so the loader's
// rejection paths can be tested one field at a time. Writing it by hand rather
// than mutating a fixture keeps each test's intent visible: the only thing
// different about a test's input is the thing it is testing.
func modelJSON(t *testing.T, opts ...func(map[string]any)) string {
	t.Helper()
	tree := map[string]any{
		"left_children":    []int{1, -1, -1},
		"right_children":   []int{2, -1, -1},
		"split_indices":    []int{0, 0, 0},
		"split_conditions": []float64{0.5, -1, 1},
		"base_weights":     []float64{0, -1, 1},
		"default_left":     []int{0, 0, 0},
		"split_type":       []int{0, 0, 0},
	}
	doc := map[string]any{
		"learner": map[string]any{
			"gradient_booster": map[string]any{
				"name":  "gbtree",
				"model": map[string]any{"trees": []any{tree}, "tree_info": []int{0}},
			},
			"learner_model_param": map[string]any{
				"base_score": "[5E-1]", "num_class": "0", "num_feature": "2", "num_target": "1",
			},
			"objective": map[string]any{"name": "reg:squarederror"},
		},
	}
	for _, o := range opts {
		o(doc)
	}
	return mustJSON(t, doc)
}

func learner(doc map[string]any) map[string]any {
	return doc["learner"].(map[string]any)
}

func firstTree(doc map[string]any) map[string]any {
	gb := learner(doc)["gradient_booster"].(map[string]any)
	return gb["model"].(map[string]any)["trees"].([]any)[0].(map[string]any)
}

func TestLoadRejects(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(map[string]any)
		wantErr string
	}{
		{
			name: "dart booster",
			mutate: func(d map[string]any) {
				learner(d)["gradient_booster"].(map[string]any)["name"] = "dart"
			},
			// dart applies per-tree weights this loader does not read, so
			// summing its trees unweighted would be quietly wrong.
			wantErr: "unsupported booster",
		},
		{
			name: "unknown objective",
			mutate: func(d map[string]any) {
				learner(d)["objective"].(map[string]any)["name"] = "rank:ndcg"
			},
			wantErr: "unsupported objective",
		},
		{
			name: "categorical split",
			mutate: func(d map[string]any) {
				firstTree(d)["split_type"] = []int{1, 0, 0}
			},
			wantErr: "categorical split",
		},
		{
			name: "split index past feature count",
			mutate: func(d map[string]any) {
				firstTree(d)["split_indices"] = []int{9, 0, 0}
			},
			wantErr: "outside [0,2)",
		},
		{
			name: "child index past node count",
			mutate: func(d map[string]any) {
				firstTree(d)["right_children"] = []int{7, -1, -1}
			},
			wantErr: "child index outside",
		},
		{
			name: "half-leaf node",
			mutate: func(d map[string]any) {
				firstTree(d)["right_children"] = []int{2, -1, 1}
			},
			wantErr: "one child but not the other",
		},
		{
			name: "ragged tree arrays",
			mutate: func(d map[string]any) {
				firstTree(d)["base_weights"] = []float64{0, -1}
			},
			wantErr: "base_weights has 2 entries",
		},
		{
			name: "no trees",
			mutate: func(d map[string]any) {
				gb := learner(d)["gradient_booster"].(map[string]any)
				gb["model"].(map[string]any)["trees"] = []any{}
			},
			wantErr: "no trees",
		},
		{
			name: "tree_info length mismatch",
			mutate: func(d map[string]any) {
				gb := learner(d)["gradient_booster"].(map[string]any)
				gb["model"].(map[string]any)["tree_info"] = []int{0, 0}
			},
			wantErr: "tree_info has 2 entries",
		},
		{
			name: "tree targets a group that does not exist",
			mutate: func(d map[string]any) {
				gb := learner(d)["gradient_booster"].(map[string]any)
				gb["model"].(map[string]any)["tree_info"] = []int{3}
			},
			wantErr: "outside [0,1)",
		},
		{
			name: "multi-class objective without num_class",
			mutate: func(d map[string]any) {
				learner(d)["objective"].(map[string]any)["name"] = "multi:softprob"
			},
			wantErr: "requires num_class >= 2",
		},
		{
			name: "multi-target",
			mutate: func(d map[string]any) {
				learner(d)["learner_model_param"].(map[string]any)["num_target"] = "3"
			},
			wantErr: "multi-target",
		},
		{
			name: "base_score outside the logistic range",
			mutate: func(d map[string]any) {
				learner(d)["objective"].(map[string]any)["name"] = "binary:logistic"
				learner(d)["learner_model_param"].(map[string]any)["base_score"] = "[0]"
			},
			wantErr: "must be in (0,1)",
		},
		{
			name: "non-positive poisson base_score",
			mutate: func(d map[string]any) {
				learner(d)["objective"].(map[string]any)["name"] = "count:poisson"
				learner(d)["learner_model_param"].(map[string]any)["base_score"] = "[0]"
			},
			wantErr: "must be positive",
		},
		{
			name: "base_score width disagrees with num_class",
			mutate: func(d map[string]any) {
				learner(d)["objective"].(map[string]any)["name"] = "multi:softprob"
				lmp := learner(d)["learner_model_param"].(map[string]any)
				lmp["num_class"] = "3"
				lmp["base_score"] = "[0.1,0.2]"
			},
			wantErr: "base_score has 2 values for 3 classes",
		},
		{
			name: "unparseable num_feature",
			mutate: func(d map[string]any) {
				learner(d)["learner_model_param"].(map[string]any)["num_feature"] = "many"
			},
			wantErr: "parse num_feature",
		},
		{
			name: "missing num_feature",
			mutate: func(d map[string]any) {
				learner(d)["learner_model_param"].(map[string]any)["num_feature"] = ""
			},
			wantErr: "num_feature is missing",
		},
		{
			name: "unparseable base_score",
			mutate: func(d map[string]any) {
				learner(d)["learner_model_param"].(map[string]any)["base_score"] = "[abc]"
			},
			wantErr: "parse base_score",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Load(strings.NewReader(modelJSON(t, tc.mutate)))
			if err == nil {
				t.Fatalf("Load succeeded, want error containing %q", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("Load error = %q, want it to contain %q", err, tc.wantErr)
			}
		})
	}
}

func TestLoadRejectsNonModelDocuments(t *testing.T) {
	for _, tc := range []struct{ name, body, wantErr string }{
		{"malformed json", "{not json", "parse model json"},
		{"empty object", "{}", "not an XGBoost model file"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Load(strings.NewReader(tc.body))
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("Load error = %v, want it to contain %q", err, tc.wantErr)
			}
		})
	}
}

func TestLoadFileMissing(t *testing.T) {
	if _, err := LoadFile("testdata/does-not-exist.json"); err == nil {
		t.Fatal("LoadFile on a missing path succeeded, want an error")
	}
}

// TestSplitComparisonUsesFloat32 is a regression test for the bug the fixtures
// caught: XGBoost compares a feature against a split threshold in float32, and
// writes that threshold to JSON at float32 precision.
//
// The threshold below is the float64 that 0.1 rounds to as a float32. The
// feature value is a different float64 that rounds to the same float32. Under a
// float64 comparison the value is strictly less than the threshold and the row
// goes left; under float32 they are equal, `<` is false, and it goes right —
// which is what XGBoost does. The left leaf and right leaf have different
// values, so the choice is visible in the output.
func TestSplitComparisonUsesFloat32(t *testing.T) {
	const threshold = 0.100000001490116119384765625 // float64(float32(0.1))
	value := math.Nextafter(threshold, 0)           // strictly below in float64

	if !(value < threshold) {
		t.Fatal("test setup is wrong: value should be below threshold in float64")
	}
	if float32(value) != float32(threshold) {
		t.Fatal("test setup is wrong: value and threshold should be equal as float32")
	}

	js := modelJSON(t, func(d map[string]any) {
		tr := firstTree(d)
		tr["split_conditions"] = []float64{threshold, -1, 1}
		tr["base_weights"] = []float64{0, -1, 1}
	})
	m, err := Load(strings.NewReader(js))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	got, err := m.Predict([]float64{value, 0})
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	// base_score 0.5 with an identity link, so the margin is 0.5 + leaf.
	if want := 0.5 + 1.0; math.Abs(got[0]-want) > 1e-12 {
		t.Fatalf("prediction = %v, want %v (the row must take the right branch, "+
			"as it does under XGBoost's float32 comparison)", got[0], want)
	}
}

func TestPredictRejectsWrongWidth(t *testing.T) {
	m, err := Load(strings.NewReader(modelJSON(t)))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	var werr ErrWidth
	_, err = m.Predict([]float64{1})
	if !errors.As(err, &werr) {
		t.Fatalf("Predict error = %v, want ErrWidth", err)
	}
	if werr.Got != 1 || werr.Want != 2 {
		t.Errorf("ErrWidth = %+v, want {Got:1 Want:2}", werr)
	}
	if !strings.Contains(werr.Error(), "model expects 2") {
		t.Errorf("ErrWidth.Error() = %q", werr.Error())
	}

	if _, err := m.PredictBatch([][]float64{{1, 2}, {1}}); err == nil ||
		!strings.Contains(err.Error(), "row 1") {
		t.Errorf("PredictBatch error = %v, want it to name row 1", err)
	}
	if _, err := m.MarginBatch([][]float64{{1}}); err == nil ||
		!strings.Contains(err.Error(), "row 0") {
		t.Errorf("MarginBatch error = %v, want it to name row 0", err)
	}
}

func TestModelAccessors(t *testing.T) {
	m, err := Load(strings.NewReader(modelJSON(t)))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := m.NumFeature(); got != 2 {
		t.Errorf("NumFeature = %d, want 2", got)
	}
	if got := m.NumGroup(); got != 1 {
		t.Errorf("NumGroup = %d, want 1", got)
	}
	if got := m.NumTree(); got != 1 {
		t.Errorf("NumTree = %d, want 1", got)
	}
	if got := m.OutputWidth(); got != 1 {
		t.Errorf("OutputWidth = %d, want 1", got)
	}
	if got := m.Objective().Name; got != "reg:squarederror" {
		t.Errorf("Objective = %q, want reg:squarederror", got)
	}
}

// TestAbsentBaseScoreMeansNoIntercept covers models saved before base_score was
// always written, where the field is empty rather than zero.
func TestAbsentBaseScoreMeansNoIntercept(t *testing.T) {
	js := modelJSON(t, func(d map[string]any) {
		learner(d)["learner_model_param"].(map[string]any)["base_score"] = ""
	})
	m, err := Load(strings.NewReader(js))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := m.Predict([]float64{0, 0}) // 0 < 0.5, so the left leaf: -1
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	if got[0] != -1 {
		t.Errorf("prediction = %v, want -1 (no intercept)", got[0])
	}
}

func TestMarginFromScoreOnMultiClassObjective(t *testing.T) {
	obj := objectives["multi:softprob"]
	if _, err := obj.MarginFromScore(0.5); err == nil {
		t.Fatal("MarginFromScore on a multi-class objective succeeded, want an error")
	}
}

// TestSigmoidAndSoftmaxAreOverflowSafe covers the saturated tails directly.
// Boosted models genuinely produce margins of this magnitude on confidently
// classified rows, and the naive formulations return +Inf or NaN there.
func TestSigmoidAndSoftmaxAreOverflowSafe(t *testing.T) {
	if got := sigmoid(-1000); got != 0 {
		t.Errorf("sigmoid(-1000) = %v, want 0", got)
	}
	if got := sigmoid(1000); got != 1 {
		t.Errorf("sigmoid(1000) = %v, want 1", got)
	}
	if got := sigmoid(0); got != 0.5 {
		t.Errorf("sigmoid(0) = %v, want 0.5", got)
	}

	out := softmax([]float64{1000, 999, -1000})
	var sum float64
	for _, v := range out {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("softmax produced a non-finite value: %v", out)
		}
		sum += v
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Errorf("softmax sums to %v, want 1", sum)
	}
	if out[0] <= out[1] {
		t.Errorf("softmax did not preserve ordering: %v", out)
	}
	if got := softmax(nil); len(got) != 0 {
		t.Errorf("softmax(nil) = %v, want empty", got)
	}
}

// TestArgmaxTiesGoToTheLowestIndex matches XGBoost's behaviour for
// multi:softmax when two classes have identical margins.
func TestArgmaxTiesGoToTheLowestIndex(t *testing.T) {
	if got := argmax([]float64{1, 1, 0}); got[0] != 0 {
		t.Errorf("argmax tie = %v, want class 0", got[0])
	}
	if got := argmax([]float64{0, 5, 2}); got[0] != 1 {
		t.Errorf("argmax = %v, want class 1", got[0])
	}
}

func TestParseBaseScoreRejectsNonFinite(t *testing.T) {
	if _, err := parseBaseScore("[NaN]"); err == nil {
		t.Fatal("parseBaseScore accepted NaN")
	}
	if _, err := parseBaseScore("[Inf]"); err == nil {
		t.Fatal("parseBaseScore accepted Inf")
	}
	// A bare, unbracketed value is how older XGBoost wrote the field.
	got, err := parseBaseScore("0.25")
	if err != nil || len(got) != 1 || got[0] != 0.25 {
		t.Fatalf("parseBaseScore(\"0.25\") = %v, %v", got, err)
	}
	if got, err := parseBaseScore("[]"); err != nil || got != nil {
		t.Fatalf("parseBaseScore(\"[]\") = %v, %v; want nil, nil", got, err)
	}
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal test model: %v", err)
	}
	return string(b)
}
