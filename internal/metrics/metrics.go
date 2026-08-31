// Package metrics exposes what an operator needs to answer three questions
// during a rollout: is the candidate serving, is it as fast, and is it
// predicting the same things.
package metrics

import (
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/sahilkalgutkar/modelforge/internal/drift"
	"github.com/sahilkalgutkar/modelforge/internal/routing"
)

// Metrics holds the collectors.
type Metrics struct {
	predictions *prometheus.CounterVec
	latency     *prometheus.HistogramVec
	errors      *prometheus.CounterVec

	batchSize *prometheus.HistogramVec

	shadowComparisons *prometheus.CounterVec
	shadowDelta       *prometheus.HistogramVec

	driftPSI *prometheus.GaugeVec

	authThrottled prometheus.Counter
	authReloads   *prometheus.CounterVec
	authTokens    prometheus.Gauge
	authByKind    *prometheus.CounterVec
}

// New registers the collectors with r.
func New(r prometheus.Registerer) *Metrics {
	m := &Metrics{
		predictions: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "modelforge_predictions_total",
			Help: "Predictions served, by model and version.",
		}, []string{"model", "version"}),

		latency: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name: "modelforge_prediction_duration_seconds",
			Help: "End-to-end prediction latency, by model and version.",
			// Buckets start at 250 microseconds because the batching window
			// is measured in milliseconds; the default Prometheus buckets
			// begin at 5ms and would put every request in the first bucket,
			// making the histogram unable to show the thing it exists to show.
			Buckets: []float64{0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1},
		}, []string{"model", "version"}),

		errors: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "modelforge_errors_total",
			Help: "Failed requests, by model and reason.",
		}, []string{"model", "reason"}),

		batchSize: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "modelforge_batch_size",
			Help:    "Rows per scoring call, by model and version.",
			Buckets: []float64{1, 2, 4, 8, 16, 32, 64, 128},
		}, []string{"model", "version"}),

		shadowComparisons: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "modelforge_shadow_comparisons_total",
			Help: "Shadow comparisons, by model and outcome (agreed, diverged, failed).",
		}, []string{"model", "outcome"}),

		shadowDelta: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "modelforge_shadow_delta",
			Help:    "Absolute difference between the served and shadow predictions.",
			Buckets: []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1},
		}, []string{"model"}),

		authThrottled: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "modelforge_auth_throttled_total",
			Help: "Clients newly throttled for repeated authentication failures.",
			// Counted per client newly throttled rather than per rejected
			// request: the request count is attacker-controlled and would make
			// the series say more about how hard somebody is trying than about
			// how many distinct sources are failing.
		}),

		authReloads: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "modelforge_auth_reloads_total",
			Help: "Token-set reload attempts, by outcome (applied, unchanged, failed).",
		}, []string{"outcome"}),

		authByKind: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "modelforge_authentications_total",
			Help: "Successful authentications, by credential kind (service, user).",
			// Kind, not identity. A per-user label would put an unbounded,
			// externally-controlled set of values into the metric names — the
			// same cardinality problem as labelling by client address. Who did
			// what belongs in the audit log.
		}, []string{"kind"}),

		authTokens: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "modelforge_auth_tokens",
			Help: "Tokens currently configured.",
		}),

		driftPSI: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "modelforge_feature_psi",
			Help: "Population Stability Index per feature over the current window.",
		}, []string{"model", "version", "feature"}),
	}

	r.MustRegister(m.predictions, m.latency, m.errors, m.batchSize,
		m.shadowComparisons, m.shadowDelta, m.driftPSI, m.authThrottled,
		m.authReloads, m.authTokens, m.authByKind)
	return m
}

// ObservePrediction records a served prediction.
func (m *Metrics) ObservePrediction(model string, version int, d time.Duration) {
	v := strconv.Itoa(version)
	m.predictions.WithLabelValues(model, v).Inc()
	m.latency.WithLabelValues(model, v).Observe(d.Seconds())
}

// ObserveError records a failed request.
func (m *Metrics) ObserveError(model, reason string) {
	m.errors.WithLabelValues(model, reason).Inc()
}

// ObserveBatch records the size of a scoring call.
func (m *Metrics) ObserveBatch(model string, version int, rows int) {
	m.batchSize.WithLabelValues(model, strconv.Itoa(version)).Observe(float64(rows))
}

// ObserveShadow records one shadow comparison.
func (m *Metrics) ObserveShadow(o routing.ShadowOutcome) {
	switch {
	case o.Err != nil:
		m.shadowComparisons.WithLabelValues(o.Model, "failed").Inc()
	case o.Diverged:
		m.shadowComparisons.WithLabelValues(o.Model, "diverged").Inc()
		m.shadowDelta.WithLabelValues(o.Model).Observe(o.MaxDelta)
	default:
		m.shadowComparisons.WithLabelValues(o.Model, "agreed").Inc()
		m.shadowDelta.WithLabelValues(o.Model).Observe(o.MaxDelta)
	}
}

// ObserveAuthThrottled records that a client was throttled for repeated
// authentication failures.
//
// The client's address is deliberately not a label. It is attacker-controlled
// and unbounded, so using it would let anybody create as many time series as
// they liked — a cardinality explosion that takes the monitoring down, which is
// a worse outcome than the failed logins it was meant to describe. The address
// is in the log line, where a high-cardinality value belongs.
func (m *Metrics) ObserveAuthThrottled() { m.authThrottled.Inc() }

// ObserveAuthReload records the outcome of a token-set reload.
//
// A failed reload is a counter rather than a gauge because it is an event worth
// alerting on even once: the running set is still serving, so nothing looks
// broken, and the only sign that a rotation did not take is this number moving.
func (m *Metrics) ObserveAuthReload(outcome string, tokens int) {
	m.authReloads.WithLabelValues(outcome).Inc()
	if tokens > 0 {
		m.authTokens.Set(float64(tokens))
	}
}

// ObserveAuthentication records a successful authentication by credential kind.
func (m *Metrics) ObserveAuthentication(kind string) {
	m.authByKind.WithLabelValues(kind).Inc()
}

// ObserveDrift publishes a drift report as gauges.
func (m *Metrics) ObserveDrift(rep drift.Report) {
	v := strconv.Itoa(rep.Version)
	for _, f := range rep.Features {
		m.driftPSI.WithLabelValues(rep.Model, v, f.Feature).Set(f.PSI)
	}
	if rep.Prediction != nil {
		m.driftPSI.WithLabelValues(rep.Model, v, "prediction").Set(rep.Prediction.PSI)
	}
}
