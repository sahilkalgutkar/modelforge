package xgboost

import (
	"fmt"
	"math"
)

// ErrWidth is returned when an input row does not have the width the model was
// trained on. XGBoost itself would treat a short row as one padded with missing
// values, which is a quietly dangerous default for a serving system: dropping a
// feature from a caller's payload would then produce a plausible prediction
// instead of an error. Requiring the exact width makes that a 400 at the edge.
type ErrWidth struct {
	Got, Want int
}

func (e ErrWidth) Error() string {
	return fmt.Sprintf("xgboost: input row has %d features, model expects %d", e.Got, e.Want)
}

// Margin returns the raw, untransformed scores for one row — one value per
// output group. This is the score to threshold, log, or compare across model
// versions: it is additive in the trees and unsquashed, so a difference of
// 0.01 means the same thing everywhere on the range, which is not true of a
// probability near 0 or 1.
func (m *Model) Margin(row []float64) ([]float64, error) {
	if len(row) != m.numFeature {
		return nil, ErrWidth{Got: len(row), Want: m.numFeature}
	}
	out := make([]float64, m.numGroup)
	m.margin(row, out)
	return out, nil
}

// margin accumulates into dst, which must have length numGroup. Splitting this
// out of Margin is what lets PredictBatch reuse one buffer per row instead of
// allocating two slices per row per request.
func (m *Model) margin(row []float64, dst []float64) {
	copy(dst, m.intercept)
	for i := range m.trees {
		dst[m.group[i]] += m.trees[i].leaf(row)
	}
}

// leaf walks one tree and returns the value of the leaf the row lands in.
//
// Two details here are load-bearing, and both were found by the fixture tests
// rather than by reading the format.
//
// The comparison happens in float32. XGBoost compares in float32 and writes
// split_conditions to JSON at float32 precision, so a threshold reads back as
// -0.3775961 while the feature value that produced it — a float32 widened to
// float64 — is -0.37759611010551453. Compared as float64 those differ and the
// row goes left; compared as float32 they are the same number and it goes
// right, which is what XGBoost does. It only matters for a row sitting exactly
// on a threshold, but thresholds are chosen from training values, so rows land
// exactly on them routinely: one row in 64 in the binary fixture and seven in
// the regression one. The symptom is the nastiest kind — predictions that are
// correct for ~99% of traffic and quietly wrong for the rest.
//
// A NaN feature takes the node's learned default direction rather than either
// branch of the comparison. NaN compares false against everything, so without
// the explicit check every missing value would silently go right and models
// trained on sparse data would score wrong while still returning a number.
func (t *tree) leaf(row []float64) float64 {
	i := int32(0)
	for t.left[i] >= 0 {
		v := row[t.splitIdx[i]]
		var goLeft bool
		if math.IsNaN(v) {
			goLeft = t.defaultLeft[i]
		} else {
			goLeft = float32(v) < t.splitCond[i]
		}
		if goLeft {
			i = t.left[i]
		} else {
			i = t.right[i]
		}
	}
	return t.leafValue[i]
}

// Predict scores one row and applies the objective's output transform.
func (m *Model) Predict(row []float64) ([]float64, error) {
	margin, err := m.Margin(row)
	if err != nil {
		return nil, err
	}
	return m.objective.Transform(margin), nil
}

// OutputWidth is the number of values Predict returns per row.
func (m *Model) OutputWidth() int { return m.objective.OutputWidth(m.numGroup) }

// PredictBatch scores every row in a batch.
//
// The rows are scored sequentially and deliberately so. A tree walk over a
// depth-6 model is on the order of a microsecond, and handing each row to a
// goroutine would spend more time on scheduling and on the false sharing
// between adjacent output slices than the walk itself costs. Throughput on this
// path comes from the batching layer above — amortising per-request overhead
// across many rows — not from splitting a batch that is already cheap.
func (m *Model) PredictBatch(rows [][]float64) ([][]float64, error) {
	out := make([][]float64, len(rows))
	margin := make([]float64, m.numGroup)
	for i, row := range rows {
		if len(row) != m.numFeature {
			return nil, fmt.Errorf("row %d: %w", i, ErrWidth{Got: len(row), Want: m.numFeature})
		}
		m.margin(row, margin)
		out[i] = m.objective.Transform(margin)
	}
	return out, nil
}

// MarginBatch returns raw margins for every row in a batch.
func (m *Model) MarginBatch(rows [][]float64) ([][]float64, error) {
	out := make([][]float64, len(rows))
	for i, row := range rows {
		if len(row) != m.numFeature {
			return nil, fmt.Errorf("row %d: %w", i, ErrWidth{Got: len(row), Want: m.numFeature})
		}
		dst := make([]float64, m.numGroup)
		m.margin(row, dst)
		out[i] = dst
	}
	return out, nil
}
