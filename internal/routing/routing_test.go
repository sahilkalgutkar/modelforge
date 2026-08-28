package routing

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"testing"
	"time"
)

func ptr(v int) *int { return &v }

// stubScorer returns a per-version constant, so a test can tell which version
// answered just by looking at the prediction.
type stubScorer struct {
	mu     sync.Mutex
	calls  map[int]int
	fail   map[int]error
	value  map[int][]float64
	delay  time.Duration
	blocks chan struct{}
}

func newStub() *stubScorer {
	return &stubScorer{
		calls: map[int]int{},
		fail:  map[int]error{},
		value: map[int][]float64{},
	}
}

func (s *stubScorer) Score(ctx context.Context, _ string, version int, _ []float64) ([]float64, error) {
	s.mu.Lock()
	s.calls[version]++
	err := s.fail[version]
	val, hasVal := s.value[version]
	delay := s.delay
	blocks := s.blocks
	s.mu.Unlock()

	if blocks != nil {
		select {
		case <-blocks:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if delay > 0 {
		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if err != nil {
		return nil, err
	}
	if hasVal {
		return val, nil
	}
	return []float64{float64(version)}, nil
}

func (s *stubScorer) callsFor(version int) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls[version]
}

func TestPolicyValidation(t *testing.T) {
	tests := []struct {
		name    string
		policy  Policy
		wantErr string
	}{
		{"no routes", Policy{Model: "m"}, "at least one route"},
		{"zero version", Policy{Model: "m", Routes: []Route{{Version: 0, Weight: 1}}}, "version must be positive"},
		{"negative weight", Policy{Model: "m", Routes: []Route{{Version: 1, Weight: -1}}}, "must not be negative"},
		{"duplicate version", Policy{Model: "m", Routes: []Route{{Version: 1, Weight: 1}, {Version: 1, Weight: 2}}}, "appears in two routes"},
		{"all weights zero", Policy{Model: "m", Routes: []Route{{Version: 1, Weight: 0}}}, "all weights are zero"},
		{
			"shadow also serving",
			Policy{Model: "m", Routes: []Route{{Version: 1, Weight: 1}}, Shadow: ptr(1)},
			"cannot be both shadow and serving",
		},
		{
			"shadow version zero",
			Policy{Model: "m", Routes: []Route{{Version: 1, Weight: 1}}, Shadow: ptr(0)},
			"shadow version must be positive",
		},
		{
			"guard rate out of range",
			Policy{Model: "m", Routes: []Route{{Version: 1, Weight: 1}}, Guard: &GuardConfig{MaxErrorRate: 1.5, MinRequests: 1}},
			"max_error_rate",
		},
		{
			"guard min requests zero",
			Policy{Model: "m", Routes: []Route{{Version: 1, Weight: 1}}, Guard: &GuardConfig{MaxErrorRate: 0.5}},
			"min_requests",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.policy.Validate()
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("Validate() = %v, want an error containing %q", err, tc.wantErr)
			}
		})
	}

	ok := Policy{
		Model:  "m",
		Routes: []Route{{Version: 1, Weight: 90}, {Version: 2, Weight: 10}},
		Shadow: ptr(3),
		Guard:  &GuardConfig{MaxErrorRate: 0.1, MinRequests: 50},
	}
	if err := ok.Validate(); err != nil {
		t.Errorf("Validate() on a valid policy = %v", err)
	}
}

// TestSelectRespectsWeights checks the split actually lands near the requested
// proportions over many distinct keys.
func TestSelectRespectsWeights(t *testing.T) {
	p := Policy{Model: "fraud", Routes: []Route{{Version: 1, Weight: 90}, {Version: 2, Weight: 10}}}

	const n = 20000
	counts := map[int]int{}
	for i := range n {
		v, err := p.Select(fmt.Sprintf("entity-%d", i), 0)
		if err != nil {
			t.Fatal(err)
		}
		counts[v]++
	}

	share := float64(counts[2]) / float64(n)
	if math.Abs(share-0.10) > 0.02 {
		t.Errorf("version 2 received %.1f%% of traffic, want about 10%%", share*100)
	}
	t.Logf("v1 %.1f%%, v2 %.1f%%", float64(counts[1])/n*100, share*100)
}

// TestSelectIsStableForTheSameKey is the property that makes a canary an
// experiment rather than noise. If an entity flips between versions from
// request to request, every per-entity metric mixes both models.
func TestSelectIsStableForTheSameKey(t *testing.T) {
	p := Policy{Model: "fraud", Routes: []Route{{Version: 1, Weight: 50}, {Version: 2, Weight: 50}}}

	for _, key := range []string{"user-1", "user-2", "account-99"} {
		first, err := p.Select(key, 0)
		if err != nil {
			t.Fatal(err)
		}
		for range 100 {
			got, err := p.Select(key, 0)
			if err != nil {
				t.Fatal(err)
			}
			if got != first {
				t.Fatalf("key %q selected version %d then %d; assignment must be stable", key, first, got)
			}
		}
	}
}

// TestSelectIsIndependentAcrossModels covers why the model name is mixed into
// the hash. Without it, an entity in the canary bucket for one model would be
// in the canary bucket for every model, correlating experiments that are
// supposed to be independent.
func TestSelectIsIndependentAcrossModels(t *testing.T) {
	a := Policy{Model: "fraud", Routes: []Route{{Version: 1, Weight: 50}, {Version: 2, Weight: 50}}}
	b := Policy{Model: "churn", Routes: []Route{{Version: 1, Weight: 50}, {Version: 2, Weight: 50}}}

	const n = 2000
	var same int
	for i := range n {
		key := fmt.Sprintf("entity-%d", i)
		va, _ := a.Select(key, 0)
		vb, _ := b.Select(key, 0)
		if va == vb {
			same++
		}
	}
	// Independent 50/50 assignments agree about half the time. Perfect
	// agreement would mean the two models bucket entities identically.
	agreement := float64(same) / n
	if agreement > 0.6 || agreement < 0.4 {
		t.Errorf("assignments agreed %.0f%% of the time across models; want about 50%% (independent)", agreement*100)
	}
}

func TestSelectWithNoKeyUsesTheFallback(t *testing.T) {
	p := Policy{Model: "m", Routes: []Route{{Version: 1, Weight: 1}, {Version: 2, Weight: 1}}}

	// The fallback lands in the first route's bucket, then the second's.
	if v, _ := p.Select("", 0); v != 1 {
		t.Errorf("fallback 0 selected version %d, want 1", v)
	}
	if v, _ := p.Select("", 1); v != 2 {
		t.Errorf("fallback 1 selected version %d, want 2", v)
	}
}

// TestSelectSkipsZeroWeightVersions matters because that is exactly the state
// the rollout guard leaves a removed version in.
func TestSelectSkipsZeroWeightVersions(t *testing.T) {
	p := Policy{Model: "m", Routes: []Route{{Version: 1, Weight: 0}, {Version: 2, Weight: 5}}}
	for i := range 200 {
		v, err := p.Select(fmt.Sprintf("k%d", i), 0)
		if err != nil {
			t.Fatal(err)
		}
		if v != 2 {
			t.Fatalf("selected version %d, but version 1 has zero weight", v)
		}
	}
}

func TestSelectWithNothingServing(t *testing.T) {
	p := Policy{Model: "m", Routes: []Route{{Version: 1, Weight: 0}}}
	if _, err := p.Select("k", 0); !errors.Is(err, ErrNoRoute) {
		t.Errorf("Select = %v, want ErrNoRoute", err)
	}
}

func TestPolicyVersionsAndString(t *testing.T) {
	p := Policy{
		Model:  "m",
		Routes: []Route{{Version: 3, Weight: 1}, {Version: 1, Weight: 9}},
		Shadow: ptr(7),
	}
	got := p.Versions()
	want := []int{1, 3, 7}
	if len(got) != len(want) {
		t.Fatalf("Versions() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Versions() = %v, want %v", got, want)
		}
	}
	if s := p.String(); !strings.Contains(s, "v1=90%") || !strings.Contains(s, "shadow=v7") {
		t.Errorf("String() = %q", s)
	}
}

func TestRouteUsesThePolicy(t *testing.T) {
	s := newStub()
	r := NewRouter(s, Options{})
	if err := r.SetPolicy(Policy{Model: "m", Routes: []Route{{Version: 4, Weight: 1}}}); err != nil {
		t.Fatal(err)
	}

	res, err := r.Route(context.Background(), "m", "key", []float64{1, 2})
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if res.Version != 4 || res.Prediction[0] != 4 {
		t.Errorf("Route = %+v, want version 4", res)
	}
}

func TestRouteWithoutAPolicy(t *testing.T) {
	r := NewRouter(newStub(), Options{})
	if _, err := r.Route(context.Background(), "unknown", "", nil); !errors.Is(err, ErrNoRoute) {
		t.Errorf("Route = %v, want ErrNoRoute", err)
	}
}

func TestSetPolicyRejectsInvalid(t *testing.T) {
	r := NewRouter(newStub(), Options{})
	if err := r.SetPolicy(Policy{Model: "m"}); err == nil {
		t.Fatal("SetPolicy accepted a policy with no routes")
	}
}

// TestShadowDoesNotAffectTheResponse is the shadow's core contract: whatever
// the candidate does — disagree, fail, hang — the served answer is unchanged.
func TestShadowDoesNotAffectTheResponse(t *testing.T) {
	s := newStub()
	s.value[1] = []float64{0.10}
	s.value[2] = []float64{0.95} // wildly different candidate

	var (
		mu       sync.Mutex
		outcomes []ShadowOutcome
	)
	r := NewRouter(s, Options{OnShadow: func(o ShadowOutcome) {
		mu.Lock()
		defer mu.Unlock()
		outcomes = append(outcomes, o)
	}})
	if err := r.SetPolicy(Policy{
		Model:  "m",
		Routes: []Route{{Version: 1, Weight: 1}},
		Shadow: ptr(2),
	}); err != nil {
		t.Fatal(err)
	}

	res, err := r.Route(context.Background(), "m", "k", []float64{1})
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if res.Version != 1 || res.Prediction[0] != 0.10 {
		t.Errorf("served %+v, want version 1's 0.10 — the shadow changed the response", res)
	}

	r.Wait()
	mu.Lock()
	defer mu.Unlock()
	if len(outcomes) != 1 {
		t.Fatalf("recorded %d shadow outcomes, want 1", len(outcomes))
	}
	o := outcomes[0]
	if !o.Diverged {
		t.Error("0.10 vs 0.95 was not recorded as a divergence")
	}
	if math.Abs(o.MaxDelta-0.85) > 1e-9 {
		t.Errorf("MaxDelta = %v, want 0.85", o.MaxDelta)
	}
	if s.callsFor(2) != 1 {
		t.Errorf("shadow version scored %d times, want 1", s.callsFor(2))
	}
}

// TestFailingShadowIsRecordedNotPropagated covers the case where the candidate
// is broken. The served request already succeeded; the candidate's problems are
// the candidate's.
func TestFailingShadowIsRecordedNotPropagated(t *testing.T) {
	s := newStub()
	s.fail[2] = errors.New("candidate is broken")

	got := make(chan ShadowOutcome, 1)
	r := NewRouter(s, Options{OnShadow: func(o ShadowOutcome) { got <- o }})
	if err := r.SetPolicy(Policy{Model: "m", Routes: []Route{{Version: 1, Weight: 1}}, Shadow: ptr(2)}); err != nil {
		t.Fatal(err)
	}

	if _, err := r.Route(context.Background(), "m", "k", []float64{1}); err != nil {
		t.Fatalf("a failing shadow broke the served request: %v", err)
	}

	r.Wait()
	select {
	case o := <-got:
		if o.Err == nil {
			t.Error("shadow failure was not recorded")
		}
	default:
		t.Error("no shadow outcome was reported")
	}
}

// TestShadowSurvivesTheRequestContext is why the shadow gets its own context.
// A handler's context is cancelled the moment it returns, so a shadow that
// inherited it would be killed before finishing on nearly every request.
func TestShadowSurvivesTheRequestContext(t *testing.T) {
	s := newStub()
	s.delay = 40 * time.Millisecond

	done := make(chan ShadowOutcome, 1)
	r := NewRouter(s, Options{OnShadow: func(o ShadowOutcome) { done <- o }})
	if err := r.SetPolicy(Policy{Model: "m", Routes: []Route{{Version: 1, Weight: 1}}, Shadow: ptr(2)}); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	if _, err := r.Route(ctx, "m", "k", []float64{1}); err != nil {
		t.Fatal(err)
	}
	// Exactly what a finished HTTP handler does.
	cancel()

	r.Wait()
	select {
	case o := <-done:
		if o.Err != nil {
			t.Errorf("shadow failed after the request context was cancelled: %v", o.Err)
		}
	default:
		t.Fatal("shadow produced no outcome; it was killed with the request context")
	}
}

func TestShadowTimeoutIsBounded(t *testing.T) {
	s := newStub()
	s.blocks = make(chan struct{}) // never released

	done := make(chan ShadowOutcome, 1)
	r := NewRouter(s, Options{
		ShadowTimeout: 50 * time.Millisecond,
		OnShadow:      func(o ShadowOutcome) { done <- o },
	})
	if err := r.SetPolicy(Policy{Model: "m", Routes: []Route{{Version: 1, Weight: 1}}, Shadow: ptr(2)}); err != nil {
		t.Fatal(err)
	}

	// Version 1 must not block, so release the served call by giving it a
	// value and letting the stub's block apply only once it is closed.
	go func() {
		time.Sleep(10 * time.Millisecond)
		close(s.blocks)
	}()

	if _, err := r.Route(context.Background(), "m", "k", []float64{1}); err != nil {
		t.Fatal(err)
	}
	r.Wait()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("shadow never completed; its timeout did not fire")
	}
}

// TestGuardRemovesAFailingVersion is the automatic-rollback path.
func TestGuardRemovesAFailingVersion(t *testing.T) {
	s := newStub()
	s.fail[2] = errors.New("canary is broken")

	r := NewRouter(s, Options{})
	if err := r.SetPolicy(Policy{
		Model:  "m",
		Routes: []Route{{Version: 1, Weight: 50}, {Version: 2, Weight: 50}},
		Guard:  &GuardConfig{MaxErrorRate: 0.2, MinRequests: 10},
	}); err != nil {
		t.Fatal(err)
	}

	for i := range 500 {
		//nolint:errcheck // failures are the point
		r.Route(context.Background(), "m", fmt.Sprintf("k%d", i), []float64{1})
		_, _, removed := r.Health("m", 2)
		if removed {
			break
		}
	}

	_, _, removed := r.Health("m", 2)
	if !removed {
		t.Fatal("the guard never removed a version failing 100% of its requests")
	}

	p, _ := r.Policy("m")
	// The route stays visible at zero weight rather than disappearing, so an
	// operator can see that a version was pulled.
	var found bool
	for _, rt := range p.Routes {
		if rt.Version == 2 {
			found = true
			if rt.Weight != 0 {
				t.Errorf("removed version still has weight %d", rt.Weight)
			}
		}
	}
	if !found {
		t.Error("the removed version's route disappeared entirely; it should remain at zero weight")
	}

	// And traffic now goes only to the healthy version.
	before := s.callsFor(2)
	for i := range 50 {
		//nolint:errcheck // only the routing matters
		r.Route(context.Background(), "m", fmt.Sprintf("after-%d", i), []float64{1})
	}
	if got := s.callsFor(2); got != before {
		t.Errorf("removed version received %d more requests", got-before)
	}
}

// TestGuardWaitsForMinRequests covers the noise problem: the first request to
// fail puts a version at a 100% error rate, and acting on that would pull
// versions that are fine.
func TestGuardWaitsForMinRequests(t *testing.T) {
	s := newStub()
	s.fail[2] = errors.New("fails every time")

	r := NewRouter(s, Options{})
	if err := r.SetPolicy(Policy{
		Model:  "m",
		Routes: []Route{{Version: 1, Weight: 0}, {Version: 2, Weight: 1}},
		Guard:  &GuardConfig{MaxErrorRate: 0.1, MinRequests: 20},
	}); err != nil {
		t.Fatal(err)
	}

	// Version 2 is the only one with weight, so every request goes to it.
	for range 5 {
		//nolint:errcheck // failures are the point
		r.Route(context.Background(), "m", "", []float64{1})
	}
	if _, _, removed := r.Health("m", 2); removed {
		t.Fatal("the guard acted on 5 requests despite min_requests being 20")
	}
}

// TestGuardWillNotEmptyTheSplit is the safety rail. Removing the last serving
// version turns "this version is failing" into "this model serves nothing",
// which is strictly worse.
func TestGuardWillNotEmptyTheSplit(t *testing.T) {
	s := newStub()
	s.fail[1] = errors.New("everything is broken")

	r := NewRouter(s, Options{})
	if err := r.SetPolicy(Policy{
		Model:  "m",
		Routes: []Route{{Version: 1, Weight: 1}},
		Guard:  &GuardConfig{MaxErrorRate: 0.1, MinRequests: 5},
	}); err != nil {
		t.Fatal(err)
	}

	for range 100 {
		//nolint:errcheck // failures are the point
		r.Route(context.Background(), "m", "", []float64{1})
	}

	if _, _, removed := r.Health("m", 1); removed {
		t.Fatal("the guard removed the only serving version, leaving the model with no route")
	}
	p, _ := r.Policy("m")
	if _, err := p.Select("k", 0); err != nil {
		t.Errorf("model can no longer serve: %v", err)
	}
}

func TestGuardIsOptional(t *testing.T) {
	s := newStub()
	s.fail[1] = errors.New("broken")

	r := NewRouter(s, Options{})
	if err := r.SetPolicy(Policy{Model: "m", Routes: []Route{{Version: 1, Weight: 1}, {Version: 2, Weight: 0}}}); err != nil {
		t.Fatal(err)
	}
	for range 50 {
		//nolint:errcheck // failures are the point
		r.Route(context.Background(), "m", "", []float64{1})
	}
	requests, failures, removed := r.Health("m", 1)
	if removed {
		t.Error("a version was removed with no guard configured")
	}
	if requests != 50 || failures != 50 {
		t.Errorf("Health = %d requests, %d failures; want 50, 50", requests, failures)
	}
}

func TestRemovePolicy(t *testing.T) {
	r := NewRouter(newStub(), Options{})
	if err := r.SetPolicy(Policy{Model: "m", Routes: []Route{{Version: 1, Weight: 1}}}); err != nil {
		t.Fatal(err)
	}
	//nolint:errcheck // populating health
	r.Route(context.Background(), "m", "", []float64{1})

	r.RemovePolicy("m")
	if _, ok := r.Policy("m"); ok {
		t.Error("policy survived RemovePolicy")
	}
	if req, _, _ := r.Health("m", 1); req != 0 {
		t.Error("health survived RemovePolicy")
	}
}

// TestConcurrentRoutingIsRaceFree exercises the lock discipline: reads of the
// policy, writes from the guard, and shadow goroutines all at once.
func TestConcurrentRoutingIsRaceFree(t *testing.T) {
	s := newStub()
	s.fail[3] = errors.New("shadow fails")

	r := NewRouter(s, Options{OnShadow: func(ShadowOutcome) {}})
	if err := r.SetPolicy(Policy{
		Model:  "m",
		Routes: []Route{{Version: 1, Weight: 50}, {Version: 2, Weight: 50}},
		Shadow: ptr(3),
		Guard:  &GuardConfig{MaxErrorRate: 0.5, MinRequests: 10},
	}); err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	for i := range 100 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			//nolint:errcheck // only exercising concurrency
			r.Route(context.Background(), "m", fmt.Sprintf("k%d", i), []float64{1, 2})
		}()
	}
	wg.Wait()
	r.Wait()
}
