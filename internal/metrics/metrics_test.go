package metrics

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/prometheus/common/expfmt"

	"github.com/sahilkalgutkar/modelforge/internal/drift"
	"github.com/sahilkalgutkar/modelforge/internal/routing"
)

func TestCollectorsRecord(t *testing.T) {
	reg := prometheus.NewPedanticRegistry()
	m := New(reg)

	m.ObservePrediction("fraud", 2, 3*time.Millisecond)
	m.ObservePrediction("fraud", 2, 4*time.Millisecond)
	m.ObserveError("fraud", "overloaded")
	m.ObserveBatch("fraud", 2, 16)

	if got := testutil.ToFloat64(m.predictions.WithLabelValues("fraud", "2")); got != 2 {
		t.Errorf("predictions = %v, want 2", got)
	}
	if got := testutil.ToFloat64(m.errors.WithLabelValues("fraud", "overloaded")); got != 1 {
		t.Errorf("errors = %v, want 1", got)
	}
}

// TestLatencyBucketsResolveSubMillisecond guards the reason for the custom
// buckets: the default Prometheus buckets start at 5ms, which would put every
// prediction in the first bucket and make the histogram useless for the
// batching window it exists to show.
func TestLatencyBucketsResolveSubMillisecond(t *testing.T) {
	reg := prometheus.NewPedanticRegistry()
	m := New(reg)

	m.ObservePrediction("m", 1, 300*time.Microsecond)
	m.ObservePrediction("m", 1, 30*time.Millisecond)

	out, err := testutil.CollectAndFormat(reg, expfmt.TypeTextPlain, "modelforge_prediction_duration_seconds")
	if err != nil {
		t.Fatal(err)
	}
	text := string(out)
	if !strings.Contains(text, `le="0.0005"`) {
		t.Errorf("no sub-millisecond bucket in the histogram:\n%s", text)
	}
}

func TestShadowOutcomes(t *testing.T) {
	reg := prometheus.NewPedanticRegistry()
	m := New(reg)

	m.ObserveShadow(routing.ShadowOutcome{Model: "m", MaxDelta: 0.001})
	m.ObserveShadow(routing.ShadowOutcome{Model: "m", Diverged: true, MaxDelta: 0.4})
	m.ObserveShadow(routing.ShadowOutcome{Model: "m", Err: errors.New("boom")})

	for outcome, want := range map[string]float64{"agreed": 1, "diverged": 1, "failed": 1} {
		if got := testutil.ToFloat64(m.shadowComparisons.WithLabelValues("m", outcome)); got != want {
			t.Errorf("%s = %v, want %v", outcome, got, want)
		}
	}
}

func TestDriftGauges(t *testing.T) {
	reg := prometheus.NewPedanticRegistry()
	m := New(reg)

	pred := drift.Reading{Feature: "prediction", PSI: 0.42}
	m.ObserveDrift(drift.Report{
		Model: "fraud", Version: 3,
		Features:   []drift.Reading{{Feature: "amount", PSI: 0.31}},
		Prediction: &pred,
	})

	if got := testutil.ToFloat64(m.driftPSI.WithLabelValues("fraud", "3", "amount")); got != 0.31 {
		t.Errorf("amount PSI gauge = %v, want 0.31", got)
	}
	if got := testutil.ToFloat64(m.driftPSI.WithLabelValues("fraud", "3", "prediction")); got != 0.42 {
		t.Errorf("prediction PSI gauge = %v, want 0.42", got)
	}
}

func TestAuthThrottledCounter(t *testing.T) {
	reg := prometheus.NewPedanticRegistry()
	m := New(reg)

	m.ObserveAuthThrottled()
	m.ObserveAuthThrottled()

	if got := testutil.ToFloat64(m.authThrottled); got != 2 {
		t.Errorf("auth throttled counter = %v, want 2", got)
	}

	// The client address must not be a label. It is attacker-controlled and
	// unbounded, so using it would let anybody mint arbitrary time series and
	// take the monitoring down — a worse outcome than the failed logins it
	// describes.
	out, err := testutil.CollectAndFormat(reg, expfmt.TypeTextPlain, "modelforge_auth_throttled_total")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(out), "{") {
		t.Errorf("the throttle counter carries labels, risking cardinality explosion:\n%s", out)
	}
}

func TestAuthReloadMetrics(t *testing.T) {
	reg := prometheus.NewPedanticRegistry()
	m := New(reg)

	m.ObserveAuthReload("applied", 3)
	m.ObserveAuthReload("unchanged", 3)
	m.ObserveAuthReload("failed", 0)

	for outcome, want := range map[string]float64{"applied": 1, "unchanged": 1, "failed": 1} {
		if got := testutil.ToFloat64(m.authReloads.WithLabelValues(outcome)); got != want {
			t.Errorf("reload outcome %q = %v, want %v", outcome, got, want)
		}
	}
	// A failed reload leaves the gauge showing what is actually in force,
	// which is the previous set — reporting zero would say the server has no
	// tokens when it is still happily serving with the old ones.
	if got := testutil.ToFloat64(m.authTokens); got != 3 {
		t.Errorf("token gauge = %v after a failed reload, want the count still in force (3)", got)
	}
}
