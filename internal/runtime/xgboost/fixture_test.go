package xgboost

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// fixture is the output of tools/fixtures/generate.py: an input matrix plus the
// margins and predictions XGBoost itself produced for it.
type fixture struct {
	Objective   string       `json:"objective"`
	NRows       int          `json:"n_rows"`
	NFeatures   int          `json:"n_features"`
	Inputs      [][]*float64 `json:"inputs"`
	Margins     [][]float64  `json:"margins"`
	Predictions [][]float64  `json:"predictions"`
}

// rows converts the fixture's inputs, turning JSON null back into NaN. The
// generator writes null because encoding/json refuses a bare NaN token.
func (f fixture) rows() [][]float64 {
	out := make([][]float64, len(f.Inputs))
	for i, row := range f.Inputs {
		out[i] = make([]float64, len(row))
		for j, v := range row {
			if v == nil {
				out[i][j] = math.NaN()
			} else {
				out[i][j] = *v
			}
		}
	}
	return out
}

func loadFixture(t *testing.T, name string) (*Model, fixture) {
	t.Helper()
	dir := filepath.Join("..", "..", "..", "testdata", "xgboost")

	m, err := LoadFile(filepath.Join(dir, name+".model.json"))
	if err != nil {
		t.Fatalf("load model %s: %v", name, err)
	}
	b, err := os.ReadFile(filepath.Join(dir, name+".expected.json"))
	if err != nil {
		t.Fatalf("read expectations %s: %v", name, err)
	}
	var f fixture
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatalf("parse expectations %s: %v", name, err)
	}
	return m, f
}

// TestAgainstXGBoost is the test this whole package exists to pass: for every
// fixture, the Go scorer must reproduce the numbers XGBoost produced for the
// same rows.
//
// Margins are checked as well as predictions, and that is deliberate. A
// sigmoid squashes a margin of 12 and a margin of 14 to 0.999994 and 0.999999,
// so a probability-only comparison would pass with a materially wrong margin on
// exactly the confident rows a threshold cares about most.
//
// The tolerance is 1e-5 rather than exact because XGBoost accumulates leaf
// values in float32 while this scorer uses float64. The residual is the
// float32 rounding, not a difference in what either one computes.
func TestAgainstXGBoost(t *testing.T) {
	const tol = 1e-5

	for _, name := range []string{
		"binary_stump",
		"binary_logistic",
		"binary_missing",
		"reg_squarederror",
		"count_poisson",
		"multi_softprob",
		"multi_softmax",
	} {
		t.Run(name, func(t *testing.T) {
			m, f := loadFixture(t, name)

			if got := m.Objective().Name; got != f.Objective {
				t.Errorf("objective = %q, want %q", got, f.Objective)
			}
			if got := m.NumFeature(); got != f.NFeatures {
				t.Errorf("NumFeature = %d, want %d", got, f.NFeatures)
			}

			rows := f.rows()
			if len(rows) != f.NRows {
				t.Fatalf("fixture has %d rows, declares %d", len(rows), f.NRows)
			}

			margins, err := m.MarginBatch(rows)
			if err != nil {
				t.Fatalf("MarginBatch: %v", err)
			}
			for i := range rows {
				assertClose(t, "margin", i, margins[i], f.Margins[i], tol)
			}

			preds, err := m.PredictBatch(rows)
			if err != nil {
				t.Fatalf("PredictBatch: %v", err)
			}
			for i := range rows {
				assertClose(t, "prediction", i, preds[i], f.Predictions[i], tol)
			}

			// Predict on a single row must agree with the batch path. They
			// share margin() but not their buffers, and a batch that reused a
			// buffer incorrectly would show up here as row N carrying row N-1's
			// answer.
			for i, row := range rows {
				one, err := m.Predict(row)
				if err != nil {
					t.Fatalf("Predict row %d: %v", i, err)
				}
				assertClose(t, "single-row prediction", i, one, preds[i], 0)
			}

			if got, want := m.OutputWidth(), len(f.Predictions[0]); got != want {
				t.Errorf("OutputWidth = %d, want %d", got, want)
			}
		})
	}
}

func assertClose(t *testing.T, what string, row int, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s row %d: got width %d, want %d", what, row, len(got), len(want))
	}
	for i := range got {
		if diff := math.Abs(got[i] - want[i]); diff > tol {
			t.Errorf("%s row %d col %d: got %.10g, want %.10g (diff %.3g)", what, row, i, got[i], want[i], diff)
		}
	}
}

// TestMissingFixtureExercisesDefaultDirections guards the fixture rather than
// the code. binary_missing is the only test that covers the NaN branch in
// leaf(), and it can only do that if the model actually learned default
// directions and the inputs actually contain NaNs. A regenerated fixture that
// quietly lost either property would leave that branch untested while every
// test still passed.
func TestMissingFixtureExercisesDefaultDirections(t *testing.T) {
	m, f := loadFixture(t, "binary_missing")

	var nans int
	for _, row := range f.rows() {
		for _, v := range row {
			if math.IsNaN(v) {
				nans++
			}
		}
	}
	if nans == 0 {
		t.Fatal("binary_missing fixture contains no NaN inputs, so the missing-value path is untested")
	}

	var defaultLeft int
	for _, tr := range m.trees {
		for i, dl := range tr.defaultLeft {
			if dl && tr.left[i] >= 0 {
				defaultLeft++
			}
		}
	}
	if defaultLeft == 0 {
		t.Fatal("binary_missing model has no internal node with default_left set, so the NaN branch is never taken")
	}
	t.Logf("fixture exercises %d NaN inputs against %d default-left nodes", nans, defaultLeft)
}

// TestMissingValuesChangeThePrediction proves the default direction is read
// rather than ignored. Sending a NaN and sending a large finite value can land
// in the same leaf by chance on any one row, so this asserts over the whole
// fixture that at least one row's prediction actually depends on it.
func TestMissingValuesChangeThePrediction(t *testing.T) {
	m, f := loadFixture(t, "binary_missing")
	rows := f.rows()

	var differed int
	for _, row := range rows {
		withNaN := append([]float64(nil), row...)
		withZero := append([]float64(nil), row...)
		var hasNaN bool
		for j, v := range row {
			if math.IsNaN(v) {
				hasNaN = true
				withZero[j] = 0
			}
		}
		if !hasNaN {
			continue
		}
		a, err := m.Predict(withNaN)
		if err != nil {
			t.Fatal(err)
		}
		b, err := m.Predict(withZero)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(a[0]-b[0]) > 1e-9 {
			differed++
		}
	}
	if differed == 0 {
		t.Fatal("substituting 0 for every missing value changed no prediction; the default direction is not being applied")
	}
	t.Logf("%d rows predict differently for NaN than for 0", differed)
}
