// Package xgboost loads models saved by XGBoost's own `save_model(*.json)` and
// scores them in pure Go.
//
// I wrote this rather than binding to libxgboost through cgo because the whole
// point of the serving layer above it is that a model version is an immutable
// artifact you can copy, hash, and roll back. A cgo dependency would put a
// C++ shared library on that path: the server would need a toolchain to build,
// the artifact's behaviour would depend on which libxgboost the host happened
// to have, and cross-compiling the static binary the deployment story assumes
// would stop working. A pure-Go scorer keeps `CGO_ENABLED=0` and makes the
// artifact the only thing that decides what a version predicts.
//
// The price is that this understands a subset of what XGBoost can save, and the
// loader is deliberately strict about saying so — see Load. A scorer that
// silently mis-scores a model it half-understands is far worse than one that
// refuses to load it, because the failure surfaces as slightly wrong
// predictions in production rather than an error at deploy time.
package xgboost

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
)

// Model is a loaded, immutable XGBoost model ready to score. It holds no
// mutable state, so a single Model is safe for concurrent use by any number of
// goroutines — which is what lets the serving layer share one loaded artifact
// across every request for that version instead of loading per request.
type Model struct {
	trees []tree
	// group[i] is the output index tree i contributes to. For binary and
	// regression objectives every tree is group 0; for multi-class, XGBoost
	// interleaves one tree per class per boosting round and tree_info is the
	// only thing that records which is which.
	group []int

	numFeature int
	numGroup   int

	// intercept is already in margin space, one entry per group. Getting here
	// from the stored base_score is the subtlest part of the format; see
	// resolveIntercept.
	intercept []float64

	objective Objective
}

// NumFeature is the width of the input row the model expects.
func (m *Model) NumFeature() int { return m.numFeature }

// NumGroup is the number of raw margin outputs per row: 1 for binary and
// regression objectives, one per class for multi-class.
func (m *Model) NumGroup() int { return m.numGroup }

// NumTree is the total number of trees, across all groups.
func (m *Model) NumTree() int { return len(m.trees) }

// Objective returns the objective the model was trained under, which determines
// how a raw margin becomes a prediction.
func (m *Model) Objective() Objective { return m.objective }

// tree is one decision tree in struct-of-arrays form.
//
// XGBoost's JSON already stores trees this way and I kept the layout rather
// than building a pointer-linked tree: traversal touches one element of four
// parallel slices per level, and the whole tree for a typical depth-6 model
// fits in a couple of cache lines per array. A pointer-chasing node struct
// would make every level a potential cache miss for no clarity gained.
type tree struct {
	left     []int32
	right    []int32
	splitIdx []int32
	// splitCond is float32, not float64, and that is a correctness requirement
	// rather than a size optimisation — see leaf() in predict.go.
	splitCond   []float32
	leafValue   []float64
	defaultLeft []bool
}

// rawModel mirrors the parts of XGBoost's JSON this loader reads. Fields it
// does not read are still checked for values it cannot honour.
type rawModel struct {
	Learner struct {
		GradientBooster struct {
			Name  string `json:"name"`
			Model struct {
				Trees    []rawTree `json:"trees"`
				TreeInfo []int     `json:"tree_info"`
			} `json:"model"`
		} `json:"gradient_booster"`
		LearnerModelParam struct {
			BaseScore  string `json:"base_score"`
			NumClass   string `json:"num_class"`
			NumFeature string `json:"num_feature"`
			NumTarget  string `json:"num_target"`
		} `json:"learner_model_param"`
		Objective struct {
			Name string `json:"name"`
		} `json:"objective"`
	} `json:"learner"`
}

type rawTree struct {
	LeftChildren    []int32   `json:"left_children"`
	RightChildren   []int32   `json:"right_children"`
	SplitIndices    []int32   `json:"split_indices"`
	SplitConditions []float64 `json:"split_conditions"`
	BaseWeights     []float64 `json:"base_weights"`
	DefaultLeft     []int     `json:"default_left"`
	SplitType       []int     `json:"split_type"`
}

// LoadFile reads and parses an XGBoost JSON model from disk.
func LoadFile(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("xgboost: open model: %w", err)
	}
	defer f.Close()
	return Load(f)
}

// Load parses an XGBoost JSON model.
//
// It fails rather than guesses on anything it cannot score exactly: a booster
// other than gbtree, a categorical split, or an objective whose link function
// is not implemented. Each of those would otherwise produce a number that looks
// like a prediction and is not one.
func Load(r io.Reader) (*Model, error) {
	var raw rawModel
	if err := json.NewDecoder(r).Decode(&raw); err != nil {
		return nil, fmt.Errorf("xgboost: parse model json: %w", err)
	}

	l := &raw.Learner
	if l.GradientBooster.Name == "" && len(l.GradientBooster.Model.Trees) == 0 {
		return nil, fmt.Errorf("xgboost: not an XGBoost model file (no learner.gradient_booster)")
	}
	// dart drops trees at training time and scores with per-tree weights that
	// this loader does not read, so summing its trees unweighted would be wrong.
	if name := l.GradientBooster.Name; name != "gbtree" {
		return nil, fmt.Errorf("xgboost: unsupported booster %q (only gbtree is supported)", name)
	}

	obj, err := lookupObjective(l.Objective.Name)
	if err != nil {
		return nil, err
	}

	numFeature, err := atoiParam("num_feature", l.LearnerModelParam.NumFeature)
	if err != nil {
		return nil, err
	}
	numClass, err := atoiParam("num_class", l.LearnerModelParam.NumClass)
	if err != nil {
		return nil, err
	}
	// num_class is 0 for anything that is not multi-class; both mean a single
	// output group.
	numGroup := numClass
	if numGroup < 1 {
		numGroup = 1
	}
	if obj.MultiClass && numClass < 2 {
		return nil, fmt.Errorf("xgboost: objective %q requires num_class >= 2, got %d", obj.Name, numClass)
	}
	if numTarget, err := atoiParam("num_target", l.LearnerModelParam.NumTarget); err == nil && numTarget > 1 {
		return nil, fmt.Errorf("xgboost: multi-target models are not supported (num_target=%d)", numTarget)
	}

	rawTrees := l.GradientBooster.Model.Trees
	if len(rawTrees) == 0 {
		return nil, fmt.Errorf("xgboost: model contains no trees")
	}
	treeInfo := l.GradientBooster.Model.TreeInfo
	if len(treeInfo) != 0 && len(treeInfo) != len(rawTrees) {
		return nil, fmt.Errorf("xgboost: tree_info has %d entries for %d trees", len(treeInfo), len(rawTrees))
	}

	m := &Model{
		trees:      make([]tree, 0, len(rawTrees)),
		group:      make([]int, len(rawTrees)),
		numFeature: numFeature,
		numGroup:   numGroup,
		objective:  obj,
	}

	for i := range rawTrees {
		t, err := convertTree(&rawTrees[i], numFeature)
		if err != nil {
			return nil, fmt.Errorf("xgboost: tree %d: %w", i, err)
		}
		m.trees = append(m.trees, t)

		g := 0
		if len(treeInfo) != 0 {
			g = treeInfo[i]
		}
		if g < 0 || g >= numGroup {
			return nil, fmt.Errorf("xgboost: tree %d targets group %d, outside [0,%d)", i, g, numGroup)
		}
		m.group[i] = g
	}

	m.intercept, err = resolveIntercept(l.LearnerModelParam.BaseScore, numGroup, obj)
	if err != nil {
		return nil, err
	}
	return m, nil
}

// convertTree validates one tree and copies it into the scoring layout.
//
// The validation here is not ceremony. Every one of these checks corresponds to
// a way a malformed or unsupported tree would otherwise score silently: an
// out-of-range split index reads the wrong feature, a child index past the end
// panics inside a request, and a categorical split compares a category code
// against a threshold as if it were a number.
func convertTree(rt *rawTree, numFeature int) (tree, error) {
	n := len(rt.LeftChildren)
	if n == 0 {
		return tree{}, fmt.Errorf("tree has no nodes")
	}
	for _, f := range []struct {
		name string
		got  int
	}{
		{"right_children", len(rt.RightChildren)},
		{"split_indices", len(rt.SplitIndices)},
		{"split_conditions", len(rt.SplitConditions)},
		{"base_weights", len(rt.BaseWeights)},
		{"default_left", len(rt.DefaultLeft)},
	} {
		if f.got != n {
			return tree{}, fmt.Errorf("%s has %d entries, expected %d", f.name, f.got, n)
		}
	}
	// split_type is absent on models saved by older XGBoost, which predate
	// categorical splits entirely — an empty slice means every split is
	// numerical, not that the field is unchecked.
	for i, st := range rt.SplitType {
		if st != splitTypeNumerical {
			return tree{}, fmt.Errorf("node %d uses a categorical split, which is not supported", i)
		}
	}

	t := tree{
		left:        make([]int32, n),
		right:       make([]int32, n),
		splitIdx:    make([]int32, n),
		splitCond:   make([]float32, n),
		leafValue:   make([]float64, n),
		defaultLeft: make([]bool, n),
	}
	for i := 0; i < n; i++ {
		l, r := rt.LeftChildren[i], rt.RightChildren[i]
		isLeaf := l == -1
		if isLeaf != (r == -1) {
			return tree{}, fmt.Errorf("node %d has one child but not the other (left=%d right=%d)", i, l, r)
		}
		if isLeaf {
			// XGBoost writes the leaf value into both base_weights and
			// split_conditions. base_weights is the field that means "the
			// value of this node" for every node type, so it is the one to
			// read; split_conditions holding a copy is an artefact.
			t.left[i], t.right[i] = -1, -1
			t.leafValue[i] = rt.BaseWeights[i]
			continue
		}
		if int(l) >= n || int(r) >= n || l < 0 || r < 0 {
			return tree{}, fmt.Errorf("node %d has child index outside [0,%d): left=%d right=%d", i, n, l, r)
		}
		if fi := rt.SplitIndices[i]; fi < 0 || int(fi) >= numFeature {
			return tree{}, fmt.Errorf("node %d splits on feature %d, outside [0,%d)", i, fi, numFeature)
		}
		t.left[i], t.right[i] = l, r
		t.splitIdx[i] = rt.SplitIndices[i]
		t.splitCond[i] = float32(rt.SplitConditions[i])
		t.defaultLeft[i] = rt.DefaultLeft[i] != 0
	}
	return t, nil
}

const splitTypeNumerical = 0

// resolveIntercept turns the stored base_score into per-group margin-space
// intercepts.
//
// This is the part of the format most worth writing down, because getting it
// wrong shifts every prediction by a constant — which is invisible in ranking
// metrics and very visible in a calibrated probability.
//
// XGBoost stores base_score in two different spaces depending on the model:
//
//   - Single-group models store it in *prediction* space, as the value the
//     model would output for a row that reached no trees. A binary:logistic
//     model with base_score 0.25 has a margin intercept of logit(0.25), and a
//     count:poisson model with base_score 1.16 has log(1.16). So the intercept
//     is the objective's output transform run backwards.
//   - Multi-class models store a vector of num_class values that is already in
//     margin space, and it is added straight through. Those values are not
//     probabilities — they are routinely negative, which is how you can tell
//     the two cases apart from the file alone.
//
// Both branches are pinned by fixtures generated from XGBoost itself, and the
// binary fixture deliberately uses a base_score other than 0.5 because logit(0.5)
// is 0 and would make an intercept applied in the wrong space look correct.
func resolveIntercept(stored string, numGroup int, obj Objective) ([]float64, error) {
	vals, err := parseBaseScore(stored)
	if err != nil {
		return nil, err
	}
	if len(vals) == 0 {
		// Absent base_score means no intercept, which is what older
		// regression models saved without the field expect.
		return make([]float64, numGroup), nil
	}

	if numGroup > 1 {
		if len(vals) != numGroup {
			return nil, fmt.Errorf("xgboost: base_score has %d values for %d classes", len(vals), numGroup)
		}
		out := make([]float64, numGroup)
		copy(out, vals)
		return out, nil
	}

	if len(vals) != 1 {
		return nil, fmt.Errorf("xgboost: base_score has %d values for a single-output model", len(vals))
	}
	margin, err := obj.MarginFromScore(vals[0])
	if err != nil {
		return nil, err
	}
	return []float64{margin}, nil
}

// parseBaseScore reads the base_score parameter, which XGBoost writes as a
// string holding either a bare number ("0.5", from older versions) or a
// bracketed vector ("[2.31E-1,-4.62E-1]").
func parseBaseScore(s string) ([]float64, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	s = strings.TrimSuffix(strings.TrimPrefix(s, "["), "]")
	if strings.TrimSpace(s) == "" {
		return nil, nil
	}
	parts := strings.Split(s, ",")
	out := make([]float64, 0, len(parts))
	for _, p := range parts {
		v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			return nil, fmt.Errorf("xgboost: parse base_score %q: %w", s, err)
		}
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("xgboost: base_score contains a non-finite value (%v)", v)
		}
		out = append(out, v)
	}
	return out, nil
}

func atoiParam(name, s string) (int, error) {
	if strings.TrimSpace(s) == "" {
		return 0, fmt.Errorf("xgboost: learner_model_param.%s is missing", name)
	}
	v, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil {
		return 0, fmt.Errorf("xgboost: parse %s %q: %w", name, s, err)
	}
	if v < 0 {
		return 0, fmt.Errorf("xgboost: %s is negative (%d)", name, v)
	}
	return v, nil
}
