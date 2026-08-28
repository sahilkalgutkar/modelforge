// Package serving holds the data plane: loaded model versions, the batchers in
// front of them, and the drift monitors behind them.
//
// The unit of everything here is a version, not a model. Each loaded version
// owns its own batcher and its own drift monitor, because they are the things a
// canary needs to be measured separately on — sharing a batcher across versions
// would put rows for two different models into one scoring call, and sharing a
// monitor would average a candidate's behaviour into the incumbent's and hide
// exactly the difference the rollout exists to observe.
package serving

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"sync"
	"time"

	"github.com/sahilkalgutkar/modelforge/internal/artifact"
	"github.com/sahilkalgutkar/modelforge/internal/batch"
	"github.com/sahilkalgutkar/modelforge/internal/drift"
	"github.com/sahilkalgutkar/modelforge/internal/registry"
	"github.com/sahilkalgutkar/modelforge/internal/runtime/xgboost"
)

// Errors callers are expected to handle.
var (
	ErrNotLoaded      = errors.New("serving: version is not loaded")
	ErrUnknownFeature = errors.New("serving: unknown feature")
)

// loaded is one model version, ready to serve.
type loaded struct {
	version registry.Version
	model   *xgboost.Model
	batcher *batch.Batcher
	monitor *drift.Monitor

	// index maps a feature name to its column in the dense row the model
	// expects. Built once at load time rather than per request.
	index map[string]int
}

// Manager loads model versions and scores against them.
type Manager struct {
	reg       *registry.Store
	artifacts artifact.Store

	batchCfg batch.Config
	driftCfg drift.Config

	mu     sync.RWMutex
	models map[string]*loaded
}

// Options configure a Manager.
type Options struct {
	Batch batch.Config
	Drift drift.Config
}

// NewManager creates a Manager.
func NewManager(reg *registry.Store, artifacts artifact.Store, opts Options) *Manager {
	return &Manager{
		reg:       reg,
		artifacts: artifacts,
		batchCfg:  opts.Batch,
		driftCfg:  opts.Drift,
		models:    make(map[string]*loaded),
	}
}

func key(model string, version int) string { return fmt.Sprintf("%s:%d", model, version) }

// Load reads a version's artifact and makes it servable. Loading a version that
// is already loaded is a no-op, so it is safe to call for every version a
// policy names on every policy change.
func (m *Manager) Load(ctx context.Context, model string, version int) error {
	k := key(model, version)

	m.mu.RLock()
	_, exists := m.models[k]
	m.mu.RUnlock()
	if exists {
		return nil
	}

	v, err := m.reg.GetVersion(ctx, model, version)
	if err != nil {
		return err
	}
	rc, err := m.artifacts.Open(v.Digest)
	if err != nil {
		return fmt.Errorf("serving: open artifact for %s: %w", v.Ref(), err)
	}
	defer rc.Close()

	parsed, err := xgboost.Load(rc)
	if err != nil {
		return fmt.Errorf("serving: load %s: %w", v.Ref(), err)
	}

	// The registry's declared feature list and the artifact's own width have
	// to agree. If they do not, every request would be scored with its values
	// in the wrong columns — the model would still return a number, and
	// nothing downstream could tell it was meaningless. Catching it at load
	// time makes it a deploy failure instead.
	if parsed.NumFeature() != len(v.Features) {
		return fmt.Errorf("serving: %s declares %d features but its artifact expects %d",
			v.Ref(), len(v.Features), parsed.NumFeature())
	}

	l := &loaded{version: v, model: parsed, index: make(map[string]int, len(v.Features))}
	for i, f := range v.Features {
		l.index[f] = i
	}

	l.batcher, err = batch.New(m.batchCfg, func(ctx context.Context, rows [][]float64) ([][]float64, error) {
		return parsed.PredictBatch(rows)
	})
	if err != nil {
		return fmt.Errorf("serving: batcher for %s: %w", v.Ref(), err)
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	// Another goroutine may have loaded it while the artifact was being read.
	if _, raced := m.models[k]; raced {
		l.batcher.Close()
		return nil
	}
	m.models[k] = l
	return nil
}

// AttachMonitor gives a loaded version its drift monitor.
//
// It is separate from Load because baselines come from training data that the
// artifact does not carry, so a version can be perfectly servable with no
// monitor at all. Refusing to serve without one would make drift monitoring a
// deployment blocker rather than an observability feature.
func (m *Manager) AttachMonitor(model string, version int, baselines []drift.Baseline, pred *drift.Baseline) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	l, ok := m.models[key(model, version)]
	if !ok {
		return fmt.Errorf("%w: %s", ErrNotLoaded, key(model, version))
	}
	if len(baselines) != len(l.version.Features) {
		return fmt.Errorf("serving: %d baselines for %d features", len(baselines), len(l.version.Features))
	}

	cfg := m.driftCfg
	cfg.Model, cfg.Version = model, version
	mon, err := drift.NewMonitor(cfg, baselines, pred)
	if err != nil {
		return err
	}
	l.monitor = mon
	return nil
}

// Unload removes a version and releases its batcher.
func (m *Manager) Unload(model string, version int) {
	m.mu.Lock()
	l, ok := m.models[key(model, version)]
	delete(m.models, key(model, version))
	m.mu.Unlock()

	if ok {
		l.batcher.Close()
	}
}

// IsLoaded reports whether a version is servable.
func (m *Manager) IsLoaded(model string, version int) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	_, ok := m.models[key(model, version)]
	return ok
}

// Loaded returns the versions currently in memory.
func (m *Manager) Loaded() []registry.Version {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make([]registry.Version, 0, len(m.models))
	for _, l := range m.models {
		out = append(out, l.version)
	}
	return out
}

func (m *Manager) get(model string, version int) (*loaded, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	l, ok := m.models[key(model, version)]
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrNotLoaded, key(model, version))
	}
	return l, nil
}

// BuildRow turns named features into the dense row a version's artifact
// expects.
//
// The two rules here are asymmetric on purpose. An unknown feature name is an
// error: it is almost always a typo or a caller built against a different
// version, and silently ignoring it would score the request with that feature
// missing while the caller believes it was supplied. An absent feature is not
// an error: XGBoost models learn an explicit direction for missing values, and
// sparse input is a legitimate, trained-for case rather than a mistake.
//
// So a caller who misspells "amount" gets a 400, and a caller who genuinely has
// no amount gets the model's learned missing-value behaviour. Treating both the
// same way would either reject legitimate sparse traffic or turn every typo
// into a silently degraded prediction.
func (m *Manager) BuildRow(model string, version int, features map[string]float64) ([]float64, error) {
	l, err := m.get(model, version)
	if err != nil {
		return nil, err
	}

	row := make([]float64, len(l.version.Features))
	for i := range row {
		row[i] = math.NaN()
	}
	for name, v := range features {
		i, ok := l.index[name]
		if !ok {
			return nil, fmt.Errorf("%w %q for %s", ErrUnknownFeature, name, l.version.Ref())
		}
		row[i] = v
	}
	return row, nil
}

// Features returns a version's declared feature names, in order.
func (m *Manager) Features(model string, version int) ([]string, error) {
	l, err := m.get(model, version)
	if err != nil {
		return nil, err
	}
	return l.version.Features, nil
}

// Score implements routing.Scorer: it scores one row against one version,
// through that version's batcher.
func (m *Manager) Score(ctx context.Context, model string, version int, row []float64) ([]float64, error) {
	l, err := m.get(model, version)
	if err != nil {
		return nil, err
	}

	pred, err := l.batcher.Submit(ctx, row)
	if err != nil {
		return nil, err
	}

	// Drift is observed after a successful score, so a monitor never sees a
	// row the model did not actually score, and never delays the response.
	if l.monitor != nil {
		l.monitor.Observe(row, pred)
	}
	return pred, nil
}

// DriftReport returns the current drift reading for a version.
func (m *Manager) DriftReport(model string, version int) (drift.Report, bool, error) {
	l, err := m.get(model, version)
	if err != nil {
		return drift.Report{}, false, err
	}
	if l.monitor == nil {
		return drift.Report{}, false, nil
	}
	rep, ok := l.monitor.Report()
	return rep, ok, nil
}

// BatchStats returns a version's batching counters.
func (m *Manager) BatchStats(model string, version int) (batch.Stats, error) {
	l, err := m.get(model, version)
	if err != nil {
		return batch.Stats{}, err
	}
	return l.batcher.Stats(), nil
}

// Close releases every loaded version.
func (m *Manager) Close() {
	m.mu.Lock()
	models := m.models
	m.models = make(map[string]*loaded)
	m.mu.Unlock()

	for _, l := range models {
		l.batcher.Close()
	}
}

// PutArtifact stores a model file and returns its digest and size, so callers
// do not need the artifact store directly.
func (m *Manager) PutArtifact(r io.Reader) (artifact.Digest, int64, error) {
	return m.artifacts.Put(r)
}

// ValidateArtifact parses a stored artifact and reports what it expects,
// without keeping it loaded.
//
// This runs at registration so a model that cannot be scored is rejected while
// someone is still looking at the upload, rather than at 3am when a policy
// first routes traffic to it.
func (m *Manager) ValidateArtifact(d artifact.Digest) (numFeature int, objective string, err error) {
	rc, err := m.artifacts.Open(d)
	if err != nil {
		return 0, "", err
	}
	defer rc.Close()

	parsed, err := xgboost.Load(rc)
	if err != nil {
		return 0, "", err
	}
	return parsed.NumFeature(), parsed.Objective().Name, nil
}

// DefaultOptions returns sensible serving defaults.
func DefaultOptions() Options {
	return Options{
		Batch: batch.Config{MaxSize: 32, MaxDelay: 5 * time.Millisecond, QueueSize: 2048, Workers: 4},
		Drift: drift.Config{Window: 15 * time.Minute, Buckets: 6},
	}
}
