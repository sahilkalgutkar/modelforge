// Package app wires the pieces together and owns the process lifecycle.
//
// It exists so that main() is only flag parsing. Everything with a decision in
// it — restoring deployment state on startup, the shutdown ordering, publishing
// drift as metrics — lives here where it can be tested, rather than in a
// function that can only be exercised by starting a process.
package app

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/sahilkalgutkar/modelforge/internal/api"
	"github.com/sahilkalgutkar/modelforge/internal/artifact"
	"github.com/sahilkalgutkar/modelforge/internal/auth"
	"github.com/sahilkalgutkar/modelforge/internal/metrics"
	"github.com/sahilkalgutkar/modelforge/internal/registry"
	"github.com/sahilkalgutkar/modelforge/internal/routing"
	"github.com/sahilkalgutkar/modelforge/internal/serving"
)

// Config is everything the server needs to start.
type Config struct {
	Addr          string
	DatabaseURL   string
	ArtifactDir   string
	ShadowTimeout time.Duration
	// DriftInterval is how often drift readings are refreshed into metrics.
	DriftInterval time.Duration
	Logger        *slog.Logger

	// Tokens are `name:scopes:sha256hex` entries; see internal/auth.
	Tokens []string

	// AuthDisabled runs the server with no authentication. It has to be set
	// explicitly: an unset Tokens list is a configuration mistake, not a
	// request to serve the control plane to anyone who can reach the port.
	AuthDisabled bool

	// AuthMaxFailures is how many authentication failures one client may make
	// before it is throttled. Zero uses the default; negative disables
	// throttling.
	AuthMaxFailures int

	// AuthFailureWindow is how long an exhausted failure budget takes to
	// refill.
	AuthFailureWindow time.Duration

	// TrustForwardedFor makes the rate limiter key on X-Forwarded-For rather
	// than the socket address. Set it only when a proxy in front of this
	// server overwrites that header, because it is otherwise trivially
	// spoofable and the limit becomes evadable.
	TrustForwardedFor bool
}

// App is a running server's dependencies.
type App struct {
	cfg      Config
	log      *slog.Logger
	registry *registry.Store
	manager  *serving.Manager
	router   *routing.Router
	metrics  *metrics.Metrics
	handler  http.Handler
	auth     *auth.Authenticator
}

// New connects to the database, opens the artifact store, and restores the
// deployment state that was in effect when the process last stopped.
func New(ctx context.Context, cfg Config) (*App, error) {
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	if cfg.DriftInterval <= 0 {
		cfg.DriftInterval = 30 * time.Second
	}

	reg, err := registry.Open(ctx, cfg.DatabaseURL)
	if err != nil {
		return nil, err
	}
	blobs, err := artifact.NewFileStore(cfg.ArtifactDir)
	if err != nil {
		reg.Close()
		return nil, err
	}

	authn, err := buildAuth(cfg)
	if err != nil {
		return nil, err
	}

	promReg := prometheus.NewRegistry()
	met := metrics.New(promReg)

	manager := serving.NewManager(reg, blobs, serving.DefaultOptions())
	router := routing.NewRouter(manager, routing.Options{
		ShadowTimeout: cfg.ShadowTimeout,
		OnShadow:      met.ObserveShadow,
	})

	a := &App{
		cfg: cfg, log: cfg.Logger, registry: reg,
		manager: manager, router: router, metrics: met,
	}

	if err := a.RestorePolicies(ctx); err != nil {
		reg.Close()
		manager.Close()
		return nil, err
	}

	apiSrv := api.NewServer(api.Deps{
		Registry: reg, Manager: manager, Router: router,
		Logger: cfg.Logger, Metrics: met, Auth: authn,
		Limiter:           buildLimiter(cfg, met),
		TrustForwardedFor: cfg.TrustForwardedFor,
	})

	mux := http.NewServeMux()
	mux.Handle("/metrics", apiSrv.MetricsHandler(
		promhttp.HandlerFor(promReg, promhttp.HandlerOpts{})))
	mux.Handle("/", apiSrv)
	a.handler = mux
	a.auth = authn
	return a, nil
}

// buildLimiter builds the failed-authentication rate limiter, or nil.
//
// Throttling is off when authentication is off: with no credential to get
// wrong there are no failures to count, and a limiter would only be machinery
// that never fires.
func buildLimiter(cfg Config, met *metrics.Metrics) *auth.Limiter {
	if cfg.AuthDisabled || cfg.AuthMaxFailures < 0 {
		return nil
	}
	return auth.NewLimiter(auth.LimiterConfig{
		MaxFailures: cfg.AuthMaxFailures,
		Window:      cfg.AuthFailureWindow,
		OnLimit:     func(string) { met.ObserveAuthThrottled() },
	})
}

// buildAuth turns the configuration into an Authenticator, failing closed.
//
// Starting with no tokens and no explicit opt-out is refused rather than
// defaulted to open. Getting this wrong is silent in the worst way: the server
// comes up, serves, and looks entirely healthy while its control plane is
// available to anybody who can reach the port. There is no log line anybody
// reads that reliably beats simply not starting.
//
// The opt-out exists because local development and the test suite genuinely do
// not want tokens, and an escape hatch that has to be named is very different
// from a default that nobody chose.
func buildAuth(cfg Config) (*auth.Authenticator, error) {
	if cfg.AuthDisabled {
		if len(cfg.Tokens) > 0 {
			// Both set is ambiguous, and guessing which one the operator meant
			// is exactly the wrong thing to do about a security control.
			return nil, errors.New("app: tokens are configured but authentication is disabled; " +
				"remove one or the other rather than leaving it ambiguous")
		}
		cfg.Logger.Warn("authentication is DISABLED; the control plane is open to anyone who can reach this port")
		return auth.Disabled(), nil
	}

	if len(cfg.Tokens) == 0 {
		return nil, errors.New("app: no API tokens configured. Mint one with " +
			"`modelforgectl token <name> <scope>` and set MODELFORGE_TOKENS, or pass " +
			"-auth-disabled to run without authentication deliberately")
	}

	authn, err := auth.New(cfg.Tokens)
	if err != nil {
		return nil, err
	}
	for _, t := range authn.Tokens() {
		cfg.Logger.Info("api token configured", "token", t.Name, "scopes", t.Scopes)
	}
	return authn, nil
}

// Handler returns the HTTP handler, for tests and for main.
func (a *App) Handler() http.Handler { return a.handler }

// RestorePolicies reloads every stored deployment policy and the model versions
// it names.
//
// Without this a restart would come up with an empty router: every model would
// answer 404 until somebody re-applied its policy by hand, which is the worst
// possible thing to discover during an incident. The policies are the
// authoritative record of what should be serving, so startup reads them back
// rather than waiting to be told again.
//
// A policy whose versions cannot be loaded is logged and skipped rather than
// failing startup. Refusing to start because one model's artifact is missing
// would take down every other model with it, which turns a single broken
// deployment into a total outage.
func (a *App) RestorePolicies(ctx context.Context) error {
	raw, err := a.registry.ListPolicies(ctx)
	if err != nil {
		return err
	}

	var restored, skipped int
	for model, body := range raw {
		var p routing.Policy
		if err := json.Unmarshal(body, &p); err != nil {
			a.log.Error("stored policy is unreadable", "model", model, "error", err)
			skipped++
			continue
		}
		p.Model = model

		if err := a.loadVersions(ctx, p); err != nil {
			a.log.Error("skipping model whose versions could not be loaded",
				"model", model, "error", err)
			skipped++
			continue
		}
		if err := a.router.SetPolicy(p); err != nil {
			a.log.Error("stored policy is no longer valid", "model", model, "error", err)
			skipped++
			continue
		}
		restored++
		a.log.Info("restored deployment", "model", model, "policy", p.String())
	}

	a.log.Info("policy restore complete", "restored", restored, "skipped", skipped)
	return nil
}

func (a *App) loadVersions(ctx context.Context, p routing.Policy) error {
	for _, v := range p.Versions() {
		if err := a.manager.Load(ctx, p.Model, v); err != nil {
			return fmt.Errorf("version %d: %w", v, err)
		}
	}
	return nil
}

// PublishDrift copies the current drift readings for every loaded version into
// the metrics gauges.
//
// It runs on a timer rather than per request because PSI is a property of a
// window, not of one prediction: recomputing it on every request would do the
// same aggregation thousands of times a second to produce a number that changes
// slowly by construction.
func (a *App) PublishDrift() {
	for _, v := range a.manager.Loaded() {
		rep, ready, err := a.manager.DriftReport(v.Model, v.Version)
		if err != nil || !ready {
			continue
		}
		a.metrics.ObserveDrift(rep)

		if worst, ok := rep.Worst(); ok && worst.Severity != "stable" {
			a.log.Warn("input drift detected",
				"model", v.Model, "version", v.Version,
				"feature", worst.Feature, "psi", worst.PSI, "severity", worst.Severity)
		}
	}
}

// Run serves until ctx is cancelled, then shuts down.
func (a *App) Run(ctx context.Context) error {
	srv := &http.Server{
		Addr:    a.cfg.Addr,
		Handler: a.handler,
		// A serving request should be fast; a model upload is a large body on
		// the same server, so the read timeout has to accommodate it while the
		// write timeout stays tight.
		ReadHeaderTimeout: 5 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	drift := time.NewTicker(a.cfg.DriftInterval)
	defer drift.Stop()
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case <-drift.C:
				a.PublishDrift()
			}
		}
	}()

	errc := make(chan error, 1)
	go func() {
		a.log.Info("listening", "addr", a.cfg.Addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errc <- err
			return
		}
		errc <- nil
	}()

	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
	}

	// Shutdown order matters. The listener stops first so no new request is
	// accepted, then in-flight requests are given time to finish, and only
	// then are the batchers closed — closing them first would fail requests
	// that were already accepted and are waiting on a batch.
	a.log.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	err := srv.Shutdown(shutdownCtx)

	a.router.Wait()
	a.manager.Close()
	a.registry.Close()
	return err
}

// Close releases resources without having run.
func (a *App) Close() {
	a.manager.Close()
	a.registry.Close()
}
