package serving

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sahilkalgutkar/modelforge/internal/artifact"
	"github.com/sahilkalgutkar/modelforge/internal/drift"
	"github.com/sahilkalgutkar/modelforge/internal/registry"
)

func fixture(name string) string {
	return filepath.Join("..", "..", "testdata", "xgboost", name)
}

// setup builds a Manager over a real registry and artifact store, with the
// binary_logistic fixture available to register.
func setup(t *testing.T) (*Manager, *registry.Store) {
	t.Helper()

	dsn := os.Getenv("MODELFORGE_TEST_DATABASE_URL")
	if dsn == "" {
		t.Fatal("MODELFORGE_TEST_DATABASE_URL is not set; these tests require a real Postgres " +
			"(see internal/registry/postgres_test.go for why they fail rather than skip)")
	}
	ctx := context.Background()
	reg, err := registry.Open(ctx, dsn)
	if err != nil {
		t.Fatalf("connect to test Postgres: %v", err)
	}
	t.Cleanup(reg.Close)
	if err := reg.Reset(ctx); err != nil {
		t.Fatal(err)
	}

	blobs, err := artifact.NewFileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	m := NewManager(reg, blobs, DefaultOptions())
	t.Cleanup(m.Close)
	return m, reg
}

func register(t *testing.T, m *Manager, reg *registry.Store, model, fixtureName string, features []string) registry.Version {
	t.Helper()
	ctx := context.Background()

	if _, err := reg.GetModel(ctx, model); errors.Is(err, registry.ErrNotFound) {
		if _, err := reg.CreateModel(ctx, model, ""); err != nil {
			t.Fatal(err)
		}
	}

	f, err := os.Open(fixture(fixtureName))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	digest, size, err := m.PutArtifact(f)
	if err != nil {
		t.Fatal(err)
	}
	v, err := reg.CreateVersion(ctx, registry.NewVersion{
		Model: model, Runtime: registry.RuntimeXGBoost,
		Digest: digest, SizeBytes: size, Features: features,
	})
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func names(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = fmt.Sprintf("f%d", i)
	}
	return out
}

func TestLoadIsIdempotentAndUnloadReleases(t *testing.T) {
	ctx := context.Background()
	m, reg := setup(t)
	v := register(t, m, reg, "scorer", "binary_logistic.model.json", names(6))

	if m.IsLoaded("scorer", v.Version) {
		t.Fatal("version reported loaded before Load")
	}
	for range 3 {
		if err := m.Load(ctx, "scorer", v.Version); err != nil {
			t.Fatalf("Load: %v", err)
		}
	}
	if !m.IsLoaded("scorer", v.Version) {
		t.Fatal("version not loaded after Load")
	}
	if got := len(m.Loaded()); got != 1 {
		t.Errorf("Loaded() has %d entries after three Loads, want 1", got)
	}

	features, err := m.Features("scorer", v.Version)
	if err != nil || len(features) != 6 || features[0] != "f0" {
		t.Errorf("Features = %v, %v", features, err)
	}

	m.Unload("scorer", v.Version)
	if m.IsLoaded("scorer", v.Version) {
		t.Error("version still loaded after Unload")
	}
	m.Unload("scorer", v.Version) // unloading twice must not panic

	if _, err := m.Features("scorer", v.Version); !errors.Is(err, ErrNotLoaded) {
		t.Errorf("Features after Unload = %v, want ErrNotLoaded", err)
	}
	if _, err := m.Score(ctx, "scorer", v.Version, make([]float64, 6)); !errors.Is(err, ErrNotLoaded) {
		t.Errorf("Score after Unload = %v, want ErrNotLoaded", err)
	}
}

func TestLoadRejectsUnknownVersion(t *testing.T) {
	m, reg := setup(t)
	register(t, m, reg, "scorer", "binary_logistic.model.json", names(6))

	if err := m.Load(context.Background(), "scorer", 99); !errors.Is(err, registry.ErrNotFound) {
		t.Errorf("Load of a nonexistent version = %v, want ErrNotFound", err)
	}
}

// TestLoadRejectsFeatureCountMismatch is the guard that stops a caller's values
// being scored in the wrong columns. Registering 3 names for a 6-feature
// artifact would otherwise load fine and return confident nonsense.
func TestLoadRejectsFeatureCountMismatch(t *testing.T) {
	m, reg := setup(t)
	v := register(t, m, reg, "scorer", "binary_logistic.model.json", names(3))

	err := m.Load(context.Background(), "scorer", v.Version)
	if err == nil || !strings.Contains(err.Error(), "artifact expects 6") {
		t.Fatalf("Load = %v, want a feature-count mismatch error", err)
	}
}

func TestLoadRejectsMissingArtifact(t *testing.T) {
	ctx := context.Background()
	m, reg := setup(t)

	if _, err := reg.CreateModel(ctx, "ghost", ""); err != nil {
		t.Fatal(err)
	}
	// A digest that is well-formed but was never stored.
	v, err := reg.CreateVersion(ctx, registry.NewVersion{
		Model: "ghost", Runtime: registry.RuntimeXGBoost,
		Digest: artifact.Digest(strings.Repeat("ab", 32)), Features: names(2),
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Load(ctx, "ghost", v.Version); !errors.Is(err, artifact.ErrNotFound) {
		t.Errorf("Load with a missing artifact = %v, want ErrNotFound", err)
	}
}

// TestBuildRowPlacesFeaturesByName is the contract that lets callers send an
// object instead of a positional array.
func TestBuildRowPlacesFeaturesByName(t *testing.T) {
	ctx := context.Background()
	m, reg := setup(t)
	v := register(t, m, reg, "scorer", "binary_logistic.model.json", names(6))
	if err := m.Load(ctx, "scorer", v.Version); err != nil {
		t.Fatal(err)
	}

	// Deliberately out of order: the map has no order, so the index is the
	// only thing putting values in the right columns.
	row, err := m.BuildRow("scorer", v.Version, map[string]float64{
		"f3": 3, "f0": 0, "f5": 5, "f1": 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range map[int]float64{0: 0, 1: 1, 3: 3, 5: 5} {
		if row[i] != want {
			t.Errorf("column %d = %v, want %v", i, row[i], want)
		}
	}
	// Unsupplied features become missing, which the model has a direction for.
	for _, i := range []int{2, 4} {
		if !math.IsNaN(row[i]) {
			t.Errorf("column %d = %v, want NaN for an absent feature", i, row[i])
		}
	}

	if _, err := m.BuildRow("scorer", v.Version, map[string]float64{"nope": 1}); !errors.Is(err, ErrUnknownFeature) {
		t.Errorf("BuildRow with an unknown feature = %v, want ErrUnknownFeature", err)
	}
	if _, err := m.BuildRow("scorer", 99, nil); !errors.Is(err, ErrNotLoaded) {
		t.Errorf("BuildRow for an unloaded version = %v, want ErrNotLoaded", err)
	}
}

func TestValidateArtifact(t *testing.T) {
	m, _ := setup(t)

	f, err := os.Open(fixture("multi_softprob.model.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	digest, _, err := m.PutArtifact(f)
	if err != nil {
		t.Fatal(err)
	}

	numFeature, objective, err := m.ValidateArtifact(digest)
	if err != nil {
		t.Fatalf("ValidateArtifact: %v", err)
	}
	if numFeature != 5 || objective != "multi:softprob" {
		t.Errorf("ValidateArtifact = %d, %q; want 5, multi:softprob", numFeature, objective)
	}

	junk, _, err := m.PutArtifact(strings.NewReader("not a model"))
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := m.ValidateArtifact(junk); err == nil {
		t.Error("ValidateArtifact accepted a non-model file")
	}
	if _, _, err := m.ValidateArtifact(artifact.Digest(strings.Repeat("cd", 32))); err == nil {
		t.Error("ValidateArtifact accepted a digest that was never stored")
	}
}

// TestDriftMonitorSeesServedTraffic is the integration the drift package cannot
// test on its own: rows actually scored through the manager must reach the
// monitor, and a shift in live traffic must show up in the report.
func TestDriftMonitorSeesServedTraffic(t *testing.T) {
	ctx := context.Background()
	m, reg := setup(t)
	v := register(t, m, reg, "scorer", "binary_logistic.model.json", names(6))
	if err := m.Load(ctx, "scorer", v.Version); err != nil {
		t.Fatal(err)
	}

	// Baselines from standard-normal training data, matching how the fixture
	// was generated.
	r := rand.New(rand.NewPCG(11, 22))
	baselines := make([]drift.Baseline, 6)
	for i := range baselines {
		samples := make([]float64, 5000)
		for j := range samples {
			samples[j] = r.NormFloat64()
		}
		b, err := drift.NewBaseline(fmt.Sprintf("f%d", i), samples, 10)
		if err != nil {
			t.Fatal(err)
		}
		baselines[i] = b
	}

	if err := m.AttachMonitor("scorer", v.Version, baselines, nil); err != nil {
		t.Fatalf("AttachMonitor: %v", err)
	}

	// Before any traffic, there is nothing to report.
	if _, ready, err := m.DriftReport("scorer", v.Version); err != nil || ready {
		t.Errorf("DriftReport before traffic = ready %v, err %v", ready, err)
	}

	// Serve traffic whose first feature is shifted well away from the
	// baseline, and leave the rest where they were.
	for range drift.MinSamples * 3 {
		row := make([]float64, 6)
		for i := range row {
			row[i] = r.NormFloat64()
		}
		row[0] += 4
		if _, err := m.Score(ctx, "scorer", v.Version, row); err != nil {
			t.Fatal(err)
		}
	}

	rep, ready, err := m.DriftReport("scorer", v.Version)
	if err != nil {
		t.Fatal(err)
	}
	if !ready {
		t.Fatalf("no drift report after %d scored requests", drift.MinSamples*3)
	}
	worst, ok := rep.Worst()
	if !ok {
		t.Fatal("report has no feature readings")
	}
	if worst.Feature != "f0" {
		t.Errorf("worst drift is %q, want f0 — the feature that was actually shifted", worst.Feature)
	}
	if worst.Severity != drift.SeveritySignificant {
		t.Errorf("shifted feature reads %s (PSI %.3f), want significant", worst.Severity, worst.PSI)
	}
	t.Logf("f0 PSI %.3f over %d samples", worst.PSI, rep.Samples)
}

func TestAttachMonitorValidation(t *testing.T) {
	ctx := context.Background()
	m, reg := setup(t)
	v := register(t, m, reg, "scorer", "binary_logistic.model.json", names(6))
	if err := m.Load(ctx, "scorer", v.Version); err != nil {
		t.Fatal(err)
	}

	if err := m.AttachMonitor("scorer", 99, nil, nil); !errors.Is(err, ErrNotLoaded) {
		t.Errorf("AttachMonitor on an unloaded version = %v, want ErrNotLoaded", err)
	}

	// One baseline for a six-feature model would silently monitor the wrong
	// columns, so it is refused.
	one, err := drift.NewBaseline("f0", []float64{1, 2, 3, 4, 5, 6, 7, 8}, 4)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.AttachMonitor("scorer", v.Version, []drift.Baseline{one}, nil); err == nil {
		t.Error("AttachMonitor accepted 1 baseline for a 6-feature model")
	}
}

func TestDriftReportWithoutAMonitor(t *testing.T) {
	ctx := context.Background()
	m, reg := setup(t)
	v := register(t, m, reg, "scorer", "binary_logistic.model.json", names(6))
	if err := m.Load(ctx, "scorer", v.Version); err != nil {
		t.Fatal(err)
	}

	// A version with no baselines is servable; drift is an observability
	// feature, not a deployment gate.
	_, ready, err := m.DriftReport("scorer", v.Version)
	if err != nil || ready {
		t.Errorf("DriftReport with no monitor = ready %v, err %v; want false, nil", ready, err)
	}
	if _, _, err := m.DriftReport("scorer", 99); !errors.Is(err, ErrNotLoaded) {
		t.Errorf("DriftReport for an unloaded version = %v, want ErrNotLoaded", err)
	}
}

func TestBatchStats(t *testing.T) {
	ctx := context.Background()
	m, reg := setup(t)
	v := register(t, m, reg, "scorer", "binary_logistic.model.json", names(6))
	if err := m.Load(ctx, "scorer", v.Version); err != nil {
		t.Fatal(err)
	}

	for range 5 {
		if _, err := m.Score(ctx, "scorer", v.Version, make([]float64, 6)); err != nil {
			t.Fatal(err)
		}
	}
	stats, err := m.BatchStats("scorer", v.Version)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Rows != 5 {
		t.Errorf("Rows = %d, want 5", stats.Rows)
	}
	if _, err := m.BatchStats("scorer", 99); !errors.Is(err, ErrNotLoaded) {
		t.Errorf("BatchStats for an unloaded version = %v, want ErrNotLoaded", err)
	}
}

func TestDefaultOptionsAreUsable(t *testing.T) {
	opts := DefaultOptions()
	if opts.Batch.MaxSize <= 0 || opts.Batch.MaxDelay <= 0 {
		t.Errorf("batch defaults are unusable: %+v", opts.Batch)
	}
	if opts.Drift.Window <= 0 || opts.Drift.Buckets < 2 {
		t.Errorf("drift defaults are unusable: %+v", opts.Drift)
	}
	if opts.Batch.MaxDelay > 50*time.Millisecond {
		t.Errorf("default batch window of %v is too much added latency for a serving default", opts.Batch.MaxDelay)
	}
}
