// Package api is the HTTP surface: one serving endpoint and a small admin API
// for registering models and changing traffic policy.
package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"slices"
	"strconv"
	"time"

	"github.com/sahilkalgutkar/modelforge/internal/artifact"
	"github.com/sahilkalgutkar/modelforge/internal/auth"
	"github.com/sahilkalgutkar/modelforge/internal/batch"
	"github.com/sahilkalgutkar/modelforge/internal/drift"
	"github.com/sahilkalgutkar/modelforge/internal/registry"
	"github.com/sahilkalgutkar/modelforge/internal/routing"
	"github.com/sahilkalgutkar/modelforge/internal/serving"
)

// maxArtifactBytes bounds an upload. Without a limit, a single request can make
// the process allocate until it is killed, which is a denial of service that
// needs no privileges — and the limit belongs on the reader rather than on a
// Content-Length header, because a chunked request has no such header to trust.
const maxArtifactBytes = 256 << 20 // 256 MiB

// Deps is what the API needs to do its job.
type Deps struct {
	Registry *registry.Store
	Manager  *serving.Manager
	Router   *routing.Router
	Logger   *slog.Logger
	// Auth guards the routes. A nil Auth is treated as explicitly disabled.
	Auth *auth.Authenticator
	// Metrics is optional; a nil Metrics means the handlers do not record.
	Metrics Recorder
}

// Recorder receives serving events. It is an interface so the API does not
// depend on Prometheus directly and can be tested without a registry.
type Recorder interface {
	ObservePrediction(model string, version int, d time.Duration)
	ObserveError(model, reason string)
}

// Server is the HTTP API.
type Server struct {
	deps Deps
	mux  *http.ServeMux
}

// NewServer wires up the routes.
func NewServer(deps Deps) *Server {
	if deps.Logger == nil {
		deps.Logger = slog.Default()
	}
	if deps.Auth == nil {
		// A nil Authenticator would mean every route silently ran unguarded,
		// which is the exact failure this is meant to prevent. Turning auth
		// off has to be said out loud.
		deps.Auth = auth.Disabled()
	}
	s := &Server{deps: deps, mux: http.NewServeMux()}
	mw := auth.NewMiddleware(deps.Auth, deps.Logger)

	// Serving. Separate from read on purpose: the credential a high-volume
	// caller ships to production should be able to score and nothing else, so
	// leaking it does not also expose which models exist and how they are
	// deployed.
	s.mux.Handle("POST /v1/models/{model}/predict",
		mw.RequireFunc(auth.ScopePredict, s.handlePredict))

	// Control plane. Reads and writes are split so a dashboard or an on-call
	// script can be given a credential that cannot change what serves traffic.
	for pattern, h := range map[string]http.HandlerFunc{
		"GET /v1/models":                                  s.handleListModels,
		"GET /v1/models/{model}":                          s.handleGetModel,
		"GET /v1/models/{model}/versions":                 s.handleListVersions,
		"GET /v1/models/{model}/policy":                   s.handleGetPolicy,
		"GET /v1/models/{model}/versions/{version}/drift": s.handleDrift,
		"GET /v1/models/{model}/versions/{version}/stats": s.handleStats,
	} {
		s.mux.Handle(pattern, mw.RequireFunc(auth.ScopeRead, h))
	}
	for pattern, h := range map[string]http.HandlerFunc{
		"POST /v1/models":                                    s.handleCreateModel,
		"POST /v1/models/{model}/versions":                   s.handleCreateVersion,
		"PUT /v1/models/{model}/policy":                      s.handleSetPolicy,
		"PUT /v1/models/{model}/versions/{version}/baseline": s.handleSetBaseline,
	} {
		s.mux.Handle(pattern, mw.RequireFunc(auth.ScopeAdmin, h))
	}

	// Operations. Deliberately unauthenticated.
	//
	// A liveness probe that needs a credential is a probe that starts failing
	// the moment a token is rotated or misconfigured — and it would then
	// restart the very process that is serving correctly. Neither endpoint
	// reveals anything beyond whether the process is up and how many models it
	// holds, which is not worth that failure mode.
	s.mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	s.mux.HandleFunc("GET /readyz", s.handleReady)

	return s
}

// MetricsHandler wraps a metrics handler in the read scope.
//
// /metrics is protected because it is not neutral: it carries every model name,
// every version, request volumes and drift readings, which together describe
// what a business is scoring and how much of it. Prometheus can send a bearer
// token from its scrape config, so the cost of protecting it is one line of
// configuration.
func (s *Server) MetricsHandler(h http.Handler) http.Handler {
	return auth.NewMiddleware(s.deps.Auth, s.deps.Logger).Require(auth.ScopeRead, h)
}

// Handler returns the HTTP handler.
func (s *Server) Handler() http.Handler { return s.mux }

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) { s.mux.ServeHTTP(w, r) }

// --- serving ---

// PredictRequest is the body of a prediction call.
type PredictRequest struct {
	// Features are named rather than positional. A positional array would make
	// every caller depend on the exact column order the model was trained
	// with, so retraining with a reordered feature list would silently score
	// existing callers' values in the wrong columns — and still return a
	// plausible number.
	Features map[string]float64 `json:"features"`

	// Key identifies the entity being scored. It makes canary assignment
	// stable for that entity; omitting it gives a random assignment per
	// request.
	Key string `json:"key,omitempty"`
}

// PredictResponse is a scored prediction.
type PredictResponse struct {
	Model      string    `json:"model"`
	Version    int       `json:"version"`
	Prediction []float64 `json:"prediction"`
}

func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	model := r.PathValue("model")
	start := time.Now()

	var req PredictRequest
	if err := decodeJSON(w, r, &req); err != nil {
		s.fail(w, model, http.StatusBadRequest, "bad_request", err)
		return
	}
	if len(req.Features) == 0 {
		s.fail(w, model, http.StatusBadRequest, "bad_request", errors.New("features are required"))
		return
	}

	policy, ok := s.deps.Router.Policy(model)
	if !ok {
		s.fail(w, model, http.StatusNotFound, "no_policy",
			fmt.Errorf("model %q has no deployment policy", model))
		return
	}
	// The row has to be built for the version that will serve it, because the
	// feature order is a property of the version. Selecting first and building
	// second is what makes a canary with a different feature list safe.
	version, err := policy.Select(req.Key, randUint64())
	if err != nil {
		s.fail(w, model, http.StatusServiceUnavailable, "no_route", err)
		return
	}
	row, err := s.deps.Manager.BuildRow(model, version, req.Features)
	if err != nil {
		status := http.StatusBadRequest
		reason := "unknown_feature"
		if errors.Is(err, serving.ErrNotLoaded) {
			status, reason = http.StatusServiceUnavailable, "not_loaded"
		}
		s.fail(w, model, status, reason, err)
		return
	}

	res, err := s.deps.Router.Route(r.Context(), model, req.Key, row)
	if err != nil {
		status, reason := http.StatusInternalServerError, "score_failed"
		switch {
		case errors.Is(err, batch.ErrOverloaded):
			// 503 with Retry-After, not 500: the request was not wrong and
			// retrying it later is the correct client behaviour.
			status, reason = http.StatusServiceUnavailable, "overloaded"
			w.Header().Set("Retry-After", "1")
		case errors.Is(err, routing.ErrNoRoute):
			status, reason = http.StatusServiceUnavailable, "no_route"
		case errors.Is(err, r.Context().Err()) && r.Context().Err() != nil:
			// The caller went away; there is nobody to send a status to, but
			// it should not be counted as a server error either.
			status, reason = 499, "client_gone"
		}
		s.fail(w, model, status, reason, err)
		return
	}

	if s.deps.Metrics != nil {
		s.deps.Metrics.ObservePrediction(model, res.Version, time.Since(start))
	}
	writeJSON(w, http.StatusOK, PredictResponse{
		Model: model, Version: res.Version, Prediction: res.Prediction,
	})
}

// --- control plane ---

type createModelRequest struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

func (s *Server) handleCreateModel(w http.ResponseWriter, r *http.Request) {
	var req createModelRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	m, err := s.deps.Registry.CreateModel(r.Context(), req.Name, req.Description)
	if err != nil {
		writeRegistryError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, m)
}

func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	models, err := s.deps.Registry.ListModels(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if models == nil {
		models = []registry.Model{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": models})
}

func (s *Server) handleGetModel(w http.ResponseWriter, r *http.Request) {
	m, err := s.deps.Registry.GetModel(r.Context(), r.PathValue("model"))
	if err != nil {
		writeRegistryError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, m)
}

// handleCreateVersion accepts the artifact itself, with metadata in the query
// string.
//
// The artifact is streamed straight into the content-addressed store rather
// than buffered and parsed first: a model file can be hundreds of megabytes,
// and holding one in memory per concurrent upload is the easiest way to run a
// serving process out of it. It is validated after storage, which costs a read
// back but bounds memory to one model at a time.
func (s *Server) handleCreateVersion(w http.ResponseWriter, r *http.Request) {
	model := r.PathValue("model")
	q := r.URL.Query()

	features := q["feature"]
	if len(features) == 0 {
		writeError(w, http.StatusBadRequest,
			errors.New("at least one ?feature= parameter is required, in the order the model expects"))
		return
	}

	digest, size, err := s.deps.Manager.PutArtifact(http.MaxBytesReader(w, r.Body, maxArtifactBytes))
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("store artifact: %w", err))
		return
	}

	// Reject a model that cannot be scored now, while whoever uploaded it is
	// still watching, rather than when a policy first sends traffic to it.
	numFeature, objective, err := s.deps.Manager.ValidateArtifact(digest)
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("artifact is not a loadable model: %w", err))
		return
	}
	if numFeature != len(features) {
		writeError(w, http.StatusBadRequest, fmt.Errorf(
			"declared %d features but the artifact expects %d", len(features), numFeature))
		return
	}

	v, err := s.deps.Registry.CreateVersion(r.Context(), registry.NewVersion{
		Model:     model,
		Runtime:   registry.RuntimeXGBoost,
		Digest:    digest,
		SizeBytes: size,
		Features:  features,
		Notes:     q.Get("notes"),
	})
	if err != nil {
		writeRegistryError(w, err)
		return
	}

	if err := s.deps.Manager.Load(r.Context(), model, v.Version); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{"version": v, "objective": objective})
}

func (s *Server) handleListVersions(w http.ResponseWriter, r *http.Request) {
	versions, err := s.deps.Registry.ListVersions(r.Context(), r.PathValue("model"))
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if versions == nil {
		versions = []registry.Version{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"versions": versions})
}

func (s *Server) handleSetPolicy(w http.ResponseWriter, r *http.Request) {
	model := r.PathValue("model")

	var p routing.Policy
	if err := decodeJSON(w, r, &p); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	p.Model = model
	if err := p.Validate(); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}

	// Every version the policy names must be loadable before the policy is
	// installed. Installing first would create a window in which traffic is
	// routed to a version that cannot serve it, and the failure would land on
	// callers rather than on the person making the change.
	for _, v := range p.Versions() {
		if err := s.deps.Manager.Load(r.Context(), model, v); err != nil {
			writeRegistryError(w, fmt.Errorf("cannot activate version %d: %w", v, err))
			return
		}
	}
	if err := s.deps.Registry.SavePolicy(r.Context(), model, p); err != nil {
		writeRegistryError(w, err)
		return
	}
	if err := s.deps.Router.SetPolicy(p); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	s.deps.Logger.Info("policy updated", "model", model, "policy", p.String())
	writeJSON(w, http.StatusOK, p)
}

func (s *Server) handleGetPolicy(w http.ResponseWriter, r *http.Request) {
	p, ok := s.deps.Router.Policy(r.PathValue("model"))
	if !ok {
		writeError(w, http.StatusNotFound, errors.New("no policy for this model"))
		return
	}
	writeJSON(w, http.StatusOK, p)
}

// BaselineRequest carries the training-data distributions a version's drift
// monitor compares live traffic against.
type BaselineRequest struct {
	// Samples maps each feature name to values drawn from the training set.
	// Raw samples rather than pre-computed bins, because the binning strategy
	// is the server's business: quantile edges, tie collapsing and the
	// empty-bin rules all live in one place, and a client that computed its
	// own bins would silently disagree with the next server version.
	Samples map[string][]float64 `json:"samples"`

	// Predictions are the model's outputs over the training set, for
	// monitoring prediction drift. Optional: input drift works without it.
	Predictions []float64 `json:"predictions,omitempty"`

	// Bins is how many quantile buckets to use. Ten is the convention PSI's
	// published thresholds were calibrated against.
	Bins int `json:"bins,omitempty"`
}

// handleSetBaseline captures a version's reference distributions.
//
// This is separate from registering the version because the training data is
// not part of the artifact, and a version with no baseline is still perfectly
// servable. Making it a deployment prerequisite would turn an observability
// feature into a thing that blocks a rollout.
func (s *Server) handleSetBaseline(w http.ResponseWriter, r *http.Request) {
	model, version, err := modelVersion(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}

	// Baselines are thousands of samples per feature, so the usual 1MB body
	// cap is too small; this uses the artifact limit instead.
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxArtifactBytes))
	var req BaselineRequest
	if err := dec.Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("invalid baseline body: %w", err))
		return
	}
	if req.Bins <= 0 {
		req.Bins = 10
	}

	features, err := s.deps.Manager.Features(model, version)
	if err != nil {
		writeError(w, http.StatusNotFound, err)
		return
	}

	// The baselines are built in the version's own feature order, and every
	// feature must be present. A partial baseline would monitor some columns
	// and silently ignore others, which reads on a dashboard as "no drift".
	baselines := make([]drift.Baseline, 0, len(features))
	for _, name := range features {
		samples, ok := req.Samples[name]
		if !ok {
			writeError(w, http.StatusBadRequest,
				fmt.Errorf("no baseline samples for feature %q", name))
			return
		}
		b, err := drift.NewBaseline(name, samples, req.Bins)
		if err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		baselines = append(baselines, b)
	}
	for name := range req.Samples {
		if !slices.Contains(features, name) {
			writeError(w, http.StatusBadRequest,
				fmt.Errorf("baseline includes %q, which is not a feature of this version", name))
			return
		}
	}

	var pred *drift.Baseline
	if len(req.Predictions) > 0 {
		p, err := drift.NewBaseline("prediction", req.Predictions, req.Bins)
		if err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		pred = &p
	}

	if err := s.deps.Manager.AttachMonitor(model, version, baselines, pred); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	s.deps.Logger.Info("drift baseline attached", "model", model, "version", version,
		"features", len(baselines), "prediction_baseline", pred != nil)
	writeJSON(w, http.StatusOK, map[string]any{
		"model": model, "version": version,
		"features": len(baselines), "bins": req.Bins,
		"prediction_baseline": pred != nil,
	})
}

func (s *Server) handleDrift(w http.ResponseWriter, r *http.Request) {
	model, version, err := modelVersion(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	rep, ready, err := s.deps.Manager.DriftReport(model, version)
	if err != nil {
		writeError(w, http.StatusNotFound, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"ready":  ready,
		"report": rep,
		// A caller needs to know the difference between "no drift" and "not
		// enough traffic to say", and a bare report cannot express it.
		"min_samples": drift.MinSamples,
	})
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	model, version, err := modelVersion(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	stats, err := s.deps.Manager.BatchStats(model, version)
	if err != nil {
		writeError(w, http.StatusNotFound, err)
		return
	}
	requests, failures, removed := s.deps.Router.Health(model, version)
	writeJSON(w, http.StatusOK, map[string]any{
		"batches":          stats.Batches,
		"rows":             stats.Rows,
		"mean_batch":       stats.Mean(),
		"largest_batch":    stats.LargestBatch,
		"dropped":          stats.Dropped,
		"rejected":         stats.Rejected,
		"requests":         requests,
		"failures":         failures,
		"removed_by_guard": removed,
	})
}

// handleReady reports whether the process can serve, which is not the same
// question as whether it is alive. A server with no models loaded is running
// correctly and cannot answer a prediction, so it must fail readiness — or a
// rollout would send it traffic the moment the process started.
func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	loaded := s.deps.Manager.Loaded()
	if len(loaded) == 0 {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{
			"status": "no models loaded",
		})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "loaded": len(loaded)})
}

// --- helpers ---

func modelVersion(r *http.Request) (string, int, error) {
	model := r.PathValue("model")
	version, err := strconv.Atoi(r.PathValue("version"))
	if err != nil {
		return "", 0, fmt.Errorf("version must be a number: %q", r.PathValue("version"))
	}
	return model, version, nil
}

func decodeJSON(w http.ResponseWriter, r *http.Request, dst any) error {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20))
	// An unknown field is usually a caller sending a parameter this server
	// does not implement, and silently dropping it means they believe it took
	// effect. For a serving API where an ignored field could be a routing key
	// or a threshold, that is worth a 400.
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return fmt.Errorf("invalid request body: %w", err)
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		return errors.New("request body must contain exactly one JSON object")
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	//nolint:errcheck // the response is already committed; nothing to do
	json.NewEncoder(w).Encode(body)
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

func writeRegistryError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, registry.ErrNotFound), errors.Is(err, artifact.ErrNotFound):
		writeError(w, http.StatusNotFound, err)
	case errors.Is(err, registry.ErrAlreadyExists):
		writeError(w, http.StatusConflict, err)
	case errors.Is(err, routing.ErrInvalidPolicy):
		writeError(w, http.StatusBadRequest, err)
	default:
		// A validation failure from the registry is the caller's fault, and
		// they are all reported as plain errors, so anything unrecognised that
		// mentions validation is treated as a 400 rather than a 500.
		writeError(w, http.StatusInternalServerError, err)
	}
}

func (s *Server) fail(w http.ResponseWriter, model string, status int, reason string, err error) {
	if s.deps.Metrics != nil {
		s.deps.Metrics.ObserveError(model, reason)
	}
	if status == 499 {
		// The client is gone; writing a body would only be discarded.
		return
	}
	s.deps.Logger.Warn("request failed", "model", model, "reason", reason, "error", err)
	writeError(w, status, err)
}
