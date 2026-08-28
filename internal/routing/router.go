package routing

import (
	"context"
	"math"
	"math/rand/v2"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Scorer scores one row against one version of a model.
type Scorer interface {
	Score(ctx context.Context, model string, version int, row []float64) ([]float64, error)
}

// Result is what a routed request produced.
type Result struct {
	Version    int
	Prediction []float64
}

// ShadowOutcome is one comparison between the served version and the shadow.
type ShadowOutcome struct {
	Model    string
	Served   int
	Shadow   int
	Diverged bool
	// MaxDelta is the largest absolute difference across the output vector.
	MaxDelta float64
	Err      error
}

// Router serves requests according to a model's policy.
type Router struct {
	scorer Scorer

	// shadowTimeout bounds how long a mirrored request may run. A shadow that
	// hangs must not keep a goroutine and a worker slot indefinitely, because
	// the whole premise is that it costs the served path nothing.
	shadowTimeout time.Duration

	// divergenceTol is how far two versions' outputs may differ before the
	// comparison counts as a divergence. Exact equality is the wrong test:
	// two models that agree to fifteen decimal places are the same model for
	// any decision anyone makes with them, and floating-point noise would
	// otherwise report constant divergence.
	divergenceTol float64

	mu       sync.RWMutex
	policies map[string]Policy
	health   map[string]*versionHealth

	onShadow func(ShadowOutcome)

	// shadows tracks in-flight mirrored requests so shutdown can wait for
	// them rather than leaving goroutines writing to a closed world.
	shadows sync.WaitGroup
}

// Options configure a Router.
type Options struct {
	ShadowTimeout time.Duration
	DivergenceTol float64
	// OnShadow receives every shadow comparison. It is called from the shadow
	// goroutine, off the served request's path.
	OnShadow func(ShadowOutcome)
}

// NewRouter creates a Router.
func NewRouter(scorer Scorer, opts Options) *Router {
	if opts.ShadowTimeout <= 0 {
		opts.ShadowTimeout = 2 * time.Second
	}
	if opts.DivergenceTol <= 0 {
		opts.DivergenceTol = 1e-6
	}
	return &Router{
		scorer:        scorer,
		shadowTimeout: opts.ShadowTimeout,
		divergenceTol: opts.DivergenceTol,
		policies:      make(map[string]Policy),
		health:        make(map[string]*versionHealth),
		onShadow:      opts.OnShadow,
	}
}

// SetPolicy installs or replaces a model's policy.
func (r *Router) SetPolicy(p Policy) error {
	if err := p.Validate(); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.policies[p.Model] = p
	return nil
}

// Policy returns a model's current policy.
func (r *Router) Policy(model string) (Policy, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	p, ok := r.policies[model]
	return p, ok
}

// RemovePolicy forgets a model.
func (r *Router) RemovePolicy(model string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.policies, model)
	for k := range r.health {
		if strings.HasPrefix(k, model+":") {
			delete(r.health, k)
		}
	}
}

// Route scores a request against whichever version the policy selects, and
// mirrors it to the shadow version if one is configured.
//
// key identifies the entity being scored — a user, an account, a device. It is
// what makes version assignment stable across that entity's requests. Passing
// an empty key is allowed and gives a random assignment per request.
func (r *Router) Route(ctx context.Context, model, key string, row []float64) (Result, error) {
	r.mu.RLock()
	p, ok := r.policies[model]
	r.mu.RUnlock()
	if !ok {
		return Result{}, ErrNoRoute
	}

	version, err := p.Select(key, rand.Uint64())
	if err != nil {
		return Result{}, err
	}

	pred, err := r.scorer.Score(ctx, model, version, row)
	r.observe(model, version, err, p.Guard)
	if err != nil {
		return Result{}, err
	}

	if p.Shadow != nil {
		r.mirror(model, version, *p.Shadow, row, pred)
	}
	return Result{Version: version, Prediction: pred}, nil
}

// mirror scores the same row against the shadow version and compares.
//
// It deliberately runs on its own goroutine with its own context. Inheriting
// the request's context would tie the shadow's lifetime to a response that has
// already been sent — the caller's context is cancelled the moment the handler
// returns, so the shadow would be killed before it produced anything useful for
// almost every request. Doing the work inline would be worse still: the whole
// point of a shadow is that it costs the served request nothing, and inline
// mirroring would put the candidate model's latency directly into production's.
func (r *Router) mirror(model string, served, shadow int, row []float64, servedPred []float64) {
	rowCopy := append([]float64(nil), row...)
	predCopy := append([]float64(nil), servedPred...)

	r.shadows.Add(1)
	go func() {
		defer r.shadows.Done()

		ctx, cancel := context.WithTimeout(context.Background(), r.shadowTimeout)
		defer cancel()

		out := ShadowOutcome{Model: model, Served: served, Shadow: shadow}
		shadowPred, err := r.scorer.Score(ctx, model, shadow, rowCopy)
		switch {
		case err != nil:
			// A failing shadow is information, not an incident. It is recorded
			// and never propagated: the served request already succeeded, and
			// the candidate's problems are the candidate's.
			out.Err = err
		case len(shadowPred) != len(predCopy):
			out.Diverged = true
			out.MaxDelta = math.Inf(1)
		default:
			for i := range predCopy {
				if d := math.Abs(predCopy[i] - shadowPred[i]); d > out.MaxDelta {
					out.MaxDelta = d
				}
			}
			out.Diverged = out.MaxDelta > r.divergenceTol
		}

		if r.onShadow != nil {
			r.onShadow(out)
		}
	}()
}

// versionHealth counts outcomes for one version of one model.
type versionHealth struct {
	requests int64
	failures int64
	removed  bool
}

func healthKey(model string, version int) string {
	return model + ":" + strconv.Itoa(version)
}

// observe records the outcome of a request and, if the guard says so, takes the
// version out of the split.
//
// Removing a version means setting its weight to zero rather than deleting the
// route. The route staying visible is what lets an operator see that a version
// was pulled and why; deleting it would make an automatic rollback look
// indistinguishable from a policy nobody ever set.
func (r *Router) observe(model string, version int, err error, guard *GuardConfig) {
	r.mu.Lock()
	defer r.mu.Unlock()

	key := healthKey(model, version)
	h := r.health[key]
	if h == nil {
		h = &versionHealth{}
		r.health[key] = h
	}
	h.requests++
	if err != nil {
		h.failures++
	}

	if guard == nil || h.removed {
		return
	}
	if h.requests < int64(guard.MinRequests) {
		return
	}
	if float64(h.failures)/float64(h.requests) <= guard.MaxErrorRate {
		return
	}

	p, ok := r.policies[model]
	if !ok {
		return
	}
	// Never remove the last version receiving traffic. A guard that can empty
	// the split converts "this version is failing" into "this model serves
	// nothing", which is a strictly worse outcome than continuing to serve a
	// degraded version — there is nothing left to fall back to.
	var living int
	for _, rt := range p.Routes {
		if rt.Weight > 0 && rt.Version != version {
			living++
		}
	}
	if living == 0 {
		return
	}

	routes := make([]Route, len(p.Routes))
	copy(routes, p.Routes)
	for i := range routes {
		if routes[i].Version == version {
			routes[i].Weight = 0
		}
	}
	p.Routes = routes
	r.policies[model] = p
	h.removed = true
}

// Health reports what the guard has seen for a version.
func (r *Router) Health(model string, version int) (requests, failures int64, removed bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if h := r.health[healthKey(model, version)]; h != nil {
		return h.requests, h.failures, h.removed
	}
	return 0, 0, false
}

// Wait blocks until every in-flight shadow request has finished. It exists so
// shutdown is orderly and so tests can assert on shadow outcomes without
// sleeping.
func (r *Router) Wait() { r.shadows.Wait() }
