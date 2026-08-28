// Package routing decides which version of a model answers a request.
//
// Three things live here, and they are separate on purpose. A weighted split
// sends a share of traffic to a candidate version. A shadow mirrors traffic to
// a second version whose answer is recorded and thrown away. A guard watches
// the error rate of each version and removes one that is failing. Together they
// are what makes shipping a new model a gradual, reversible act rather than a
// switch someone flips at 2am.
package routing

import (
	"errors"
	"fmt"
	"hash/fnv"
	"sort"
	"strings"
)

// Route sends a share of a model's traffic to one version.
type Route struct {
	Version int `json:"version"`
	// Weight is relative, not a percentage. Weights of 1 and 9 mean the same
	// thing as 10 and 90, which keeps a two-way split expressible without
	// forcing every change to re-derive numbers that add to 100.
	Weight int `json:"weight"`
}

// Policy is how a model's traffic is divided.
type Policy struct {
	Model  string  `json:"model"`
	Routes []Route `json:"routes"`

	// Shadow, when set, names a version that receives a copy of every request.
	// Its prediction is compared against the served one and discarded. This is
	// how a candidate is evaluated on real traffic before it answers anybody:
	// full production input distribution, zero blast radius.
	Shadow *int `json:"shadow,omitempty"`

	// Guard, when set, removes a version from the split once it fails often
	// enough to be doing damage.
	Guard *GuardConfig `json:"guard,omitempty"`
}

// GuardConfig configures automatic removal of a failing version.
type GuardConfig struct {
	// MaxErrorRate is the fraction of failed requests above which a version is
	// removed from the split.
	MaxErrorRate float64 `json:"max_error_rate"`

	// MinRequests is how many requests a version must serve before the guard
	// will act on its error rate. Without it the first request to fail puts a
	// version at a 100% error rate and the guard removes a version that may be
	// perfectly healthy — the smaller the sample, the noisier the rate, and
	// the very start of a rollout is the smallest sample there is.
	MinRequests int `json:"min_requests"`
}

// Errors callers are expected to handle.
var (
	ErrNoRoute       = errors.New("routing: no version is receiving traffic")
	ErrInvalidPolicy = errors.New("routing: invalid policy")
)

// Validate checks a policy is servable.
func (p Policy) Validate() error {
	if len(p.Routes) == 0 {
		return fmt.Errorf("%w: at least one route is required", ErrInvalidPolicy)
	}

	seen := make(map[int]struct{}, len(p.Routes))
	total := 0
	for _, r := range p.Routes {
		if r.Version <= 0 {
			return fmt.Errorf("%w: version must be positive, got %d", ErrInvalidPolicy, r.Version)
		}
		if r.Weight < 0 {
			return fmt.Errorf("%w: weight must not be negative, got %d for version %d", ErrInvalidPolicy, r.Weight, r.Version)
		}
		if _, dup := seen[r.Version]; dup {
			// Two routes to one version make the split ambiguous to read and
			// change: an operator lowering "the canary's weight" would fix one
			// entry and leave the other.
			return fmt.Errorf("%w: version %d appears in two routes", ErrInvalidPolicy, r.Version)
		}
		seen[r.Version] = struct{}{}
		total += r.Weight
	}
	if total == 0 {
		// Every weight being zero is not an empty split, it is an outage: the
		// model would be registered and deployed and answer nothing.
		return fmt.Errorf("%w: all weights are zero, so no version would receive traffic", ErrInvalidPolicy)
	}

	if p.Shadow != nil {
		if *p.Shadow <= 0 {
			return fmt.Errorf("%w: shadow version must be positive, got %d", ErrInvalidPolicy, *p.Shadow)
		}
		if _, serving := seen[*p.Shadow]; serving {
			// Shadowing a version that is already serving compares it against
			// itself for some fraction of traffic, which produces a divergence
			// rate that is a mix of two different comparisons and means
			// nothing.
			return fmt.Errorf("%w: version %d cannot be both shadow and serving", ErrInvalidPolicy, *p.Shadow)
		}
	}

	if g := p.Guard; g != nil {
		if g.MaxErrorRate <= 0 || g.MaxErrorRate > 1 {
			return fmt.Errorf("%w: guard max_error_rate must be in (0,1], got %v", ErrInvalidPolicy, g.MaxErrorRate)
		}
		if g.MinRequests < 1 {
			return fmt.Errorf("%w: guard min_requests must be at least 1", ErrInvalidPolicy)
		}
	}
	return nil
}

// Versions returns every version the policy references, serving and shadow,
// so the server knows which artifacts to load.
func (p Policy) Versions() []int {
	seen := make(map[int]struct{})
	var out []int
	for _, r := range p.Routes {
		if _, ok := seen[r.Version]; !ok {
			seen[r.Version] = struct{}{}
			out = append(out, r.Version)
		}
	}
	if p.Shadow != nil {
		if _, ok := seen[*p.Shadow]; !ok {
			out = append(out, *p.Shadow)
		}
	}
	sort.Ints(out)
	return out
}

// normalised returns the routes with a positive weight, in a stable order.
//
// The ordering is what makes selection reproducible. Weighted choice walks the
// routes accumulating weight, so two servers that walked them in different
// orders would map the same key to different versions — and the whole value of
// hashing the key is that every server agrees.
func (p Policy) normalised() ([]Route, int) {
	out := make([]Route, 0, len(p.Routes))
	total := 0
	for _, r := range p.Routes {
		if r.Weight > 0 {
			out = append(out, r)
			total += r.Weight
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Version < out[j].Version })
	return out, total
}

// Select picks the version that serves a request.
//
// When key is non-empty the choice is a deterministic function of the key, so
// the same entity always reaches the same version for as long as the policy is
// unchanged. That is not a nicety: if a user flips between the control and the
// canary from request to request, every per-user metric mixes both models and
// the experiment measures nothing. It also means a user does not see a
// prediction change under them mid-session.
//
// The model name is mixed into the hash as well. Without it an entity that
// landed in the canary bucket for one model would land in the canary bucket for
// every model, and the experiments would be correlated rather than independent —
// so a bad canary anywhere would look like it affected the same unlucky users
// everywhere.
//
// An empty key falls back to the caller-supplied fallback value, which lets the
// server pass a random number for requests that carry no entity to key on.
func (p Policy) Select(key string, fallback uint64) (int, error) {
	routes, total := p.normalised()
	if total == 0 {
		return 0, fmt.Errorf("%w for model %q", ErrNoRoute, p.Model)
	}

	// One route is the overwhelmingly common case — no canary in flight — and
	// it does not need hashing at all.
	if len(routes) == 1 {
		return routes[0].Version, nil
	}

	var point uint64
	if key == "" {
		point = fallback
	} else {
		h := fnv.New64a()
		h.Write([]byte(p.Model))
		h.Write([]byte{0}) // separator, so "ab"+"c" and "a"+"bc" differ
		h.Write([]byte(key))
		point = h.Sum64()
	}

	target := point % uint64(total)
	var acc uint64
	for _, r := range routes {
		acc += uint64(r.Weight)
		if target < acc {
			return r.Version, nil
		}
	}
	// Unreachable: acc reaches total and target is strictly below it.
	return routes[len(routes)-1].Version, nil
}

// String renders a policy the way the CLI shows it.
func (p Policy) String() string {
	routes, total := p.normalised()
	parts := make([]string, 0, len(routes))
	for _, r := range routes {
		parts = append(parts, fmt.Sprintf("v%d=%d%%", r.Version, r.Weight*100/total))
	}
	s := strings.Join(parts, " ")
	if p.Shadow != nil {
		s += fmt.Sprintf(" shadow=v%d", *p.Shadow)
	}
	return s
}
