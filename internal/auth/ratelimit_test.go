package auth

import (
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

type clock struct {
	mu sync.Mutex
	t  time.Time
}

func newClock() *clock {
	return &clock{t: time.Date(2026, 8, 31, 12, 0, 0, 0, time.UTC)}
}

func (c *clock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *clock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(d)
}

func testLimiter(t *testing.T, failures int, window time.Duration, clk *clock) *Limiter {
	t.Helper()
	return NewLimiter(LimiterConfig{MaxFailures: failures, Window: window, Now: clk.now})
}

// TestBudgetIsSpentOnlyByFailures is the property that keeps a busy, correctly
// configured caller from ever being throttled: success costs nothing.
func TestBudgetIsSpentOnlyByFailures(t *testing.T) {
	clk := newClock()
	l := testLimiter(t, 3, time.Minute, clk)

	// Any number of allowed checks without a recorded failure changes nothing.
	for range 100 {
		if ok, _ := l.Allow("1.2.3.4"); !ok {
			t.Fatal("a client with no failures was throttled")
		}
	}
	if l.Tracked() != 0 {
		t.Errorf("checking a key created a bucket for it (%d tracked); only failures should", l.Tracked())
	}

	for range 3 {
		l.RecordFailure("1.2.3.4")
	}
	if ok, _ := l.Allow("1.2.3.4"); ok {
		t.Fatal("a client past its failure budget was still allowed")
	}
}

func TestBurstThenThrottle(t *testing.T) {
	clk := newClock()
	l := testLimiter(t, 5, time.Minute, clk)

	for i := range 5 {
		if ok, _ := l.Allow("client"); !ok {
			t.Fatalf("throttled after only %d failures, want a burst of 5", i)
		}
		l.RecordFailure("client")
	}

	ok, wait := l.Allow("client")
	if ok {
		t.Fatal("not throttled after exhausting the burst")
	}
	if wait <= 0 {
		t.Error("throttled without a retry hint")
	}
	// The window is a minute for 5 failures, so one token accrues in about 12
	// seconds — the hint must be in that neighbourhood, not a token amount.
	if wait > time.Minute {
		t.Errorf("retry hint of %v is longer than the whole window", wait)
	}
}

// TestBudgetRefillsOverTime covers the recovery path: a client that backs off
// gets its budget back rather than being locked out permanently.
func TestBudgetRefillsOverTime(t *testing.T) {
	clk := newClock()
	l := testLimiter(t, 10, time.Minute, clk)

	for range 10 {
		l.RecordFailure("client")
	}
	if ok, _ := l.Allow("client"); ok {
		t.Fatal("not throttled after exhausting the budget")
	}

	// Ten failures per minute is one per six seconds.
	clk.advance(6 * time.Second)
	if ok, _ := l.Allow("client"); !ok {
		t.Error("no budget after waiting for one token to accrue")
	}

	// And a full window restores the whole burst.
	clk.advance(time.Minute)
	for i := range 10 {
		if ok, _ := l.Allow("client"); !ok {
			t.Fatalf("only %d of 10 tokens back after a full window", i)
		}
		l.RecordFailure("client")
	}
}

// TestSuccessClearsTheBudget is the usability case: somebody pasting a stale
// token, then the right one, should not stay throttled for the rest of the
// window over attempts they have already corrected.
func TestSuccessClearsTheBudget(t *testing.T) {
	clk := newClock()
	l := testLimiter(t, 3, time.Minute, clk)

	for range 3 {
		l.RecordFailure("client")
	}
	if ok, _ := l.Allow("client"); ok {
		t.Fatal("not throttled after exhausting the budget")
	}

	l.RecordSuccess("client")
	if ok, _ := l.Allow("client"); !ok {
		t.Fatal("still throttled after authenticating successfully")
	}
	if l.Tracked() != 0 {
		t.Errorf("a successful client is still tracked (%d)", l.Tracked())
	}
}

// TestClientsAreIndependent stops one offender throttling everybody else.
func TestClientsAreIndependent(t *testing.T) {
	clk := newClock()
	l := testLimiter(t, 2, time.Minute, clk)

	for range 5 {
		l.RecordFailure("attacker")
	}
	if ok, _ := l.Allow("attacker"); ok {
		t.Fatal("the attacker was not throttled")
	}
	if ok, _ := l.Allow("innocent"); !ok {
		t.Fatal("an unrelated client was throttled by somebody else's failures")
	}
}

// TestTableIsBoundedAndSweepsRecoveredClients is the memory-safety property. A
// naive map keyed on client address grows with every address ever seen, which
// is its own denial of service.
func TestTableIsBoundedAndSweepsRecoveredClients(t *testing.T) {
	clk := newClock()
	l := NewLimiter(LimiterConfig{MaxFailures: 2, Window: time.Minute, MaxKeys: 50, Now: clk.now})

	for i := range 500 {
		l.RecordFailure(fmt.Sprintf("client-%d", i))
	}
	if got := l.Tracked(); got > 50 {
		t.Fatalf("tracking %d clients with a cap of 50", got)
	}

	// After a full window every tracked bucket has refilled, and a refilled
	// bucket is indistinguishable from an absent one — so the sweep should
	// reclaim them all rather than holding entries for clients that recovered.
	clk.advance(2 * time.Minute)
	for i := range 60 {
		l.RecordFailure(fmt.Sprintf("later-%d", i))
	}
	if got := l.Tracked(); got > 50 {
		t.Fatalf("tracking %d after the sweep, cap is 50", got)
	}
}

// TestSaturatedTableFailsOpen is the decision that stops the limiter becoming
// the outage it exists to prevent: if an attacker fills the table, new clients
// go untracked rather than denied.
func TestSaturatedTableFailsOpen(t *testing.T) {
	clk := newClock()
	l := NewLimiter(LimiterConfig{MaxFailures: 5, Window: time.Hour, MaxKeys: 10, Now: clk.now})

	// Fill the table with clients that are all still failing, so nothing can
	// be swept.
	for i := range 10 {
		key := fmt.Sprintf("attacker-%d", i)
		for range 5 {
			l.RecordFailure(key)
		}
	}
	if l.Tracked() != 10 {
		t.Fatalf("expected a full table, got %d", l.Tracked())
	}

	// A new client cannot be tracked, and must therefore be allowed.
	l.RecordFailure("newcomer")
	if ok, _ := l.Allow("newcomer"); !ok {
		t.Fatal("a client the limiter could not track was denied; a full table must not lock people out")
	}
	// The clients that are tracked stay throttled.
	if ok, _ := l.Allow("attacker-0"); ok {
		t.Error("a tracked attacker stopped being throttled when the table filled")
	}
}

// TestLogSuppression covers the thing this limiter mainly exists to protect. If
// every throttled request logged, the defence against log flooding would itself
// flood the log.
func TestLogSuppression(t *testing.T) {
	clk := newClock()
	l := testLimiter(t, 2, time.Minute, clk)

	for range 2 {
		l.RecordFailure("noisy")
	}

	logged, suppressed := l.ShouldLog("noisy")
	if !logged {
		t.Fatal("the first throttled request was not logged")
	}
	if suppressed != 0 {
		t.Errorf("first log reported %d suppressed, want 0", suppressed)
	}

	for range 1000 {
		if again, _ := l.ShouldLog("noisy"); again {
			t.Fatal("a subsequent throttled request logged again")
		}
	}

	// Recovering resets the suppression, so the next incident is visible.
	clk.advance(2 * time.Minute)
	l.RecordFailure("noisy")
	if again, _ := l.ShouldLog("noisy"); !again {
		t.Error("a client that recovered and reoffended did not log again")
	}
}

func TestOnLimitFires(t *testing.T) {
	clk := newClock()
	var mu sync.Mutex
	var limited []string

	l := NewLimiter(LimiterConfig{
		MaxFailures: 1, Window: time.Minute, Now: clk.now,
		OnLimit: func(key string) {
			mu.Lock()
			defer mu.Unlock()
			limited = append(limited, key)
		},
	})

	l.RecordFailure("client")
	l.ShouldLog("client")
	l.ShouldLog("client")
	l.ShouldLog("client")

	mu.Lock()
	defer mu.Unlock()
	if len(limited) != 1 || limited[0] != "client" {
		t.Errorf("OnLimit fired %v, want exactly one call for client", limited)
	}
}

func TestLimiterDefaults(t *testing.T) {
	var cfg LimiterConfig
	cfg.setDefaults()
	if cfg.MaxFailures <= 0 || cfg.Window <= 0 || cfg.MaxKeys <= 0 || cfg.Now == nil {
		t.Fatalf("setDefaults left something unusable: %+v", cfg)
	}
	// A burst of one would throttle a human who mistyped a token once.
	if cfg.MaxFailures < 3 {
		t.Errorf("default burst of %d is too tight for an honest mistake", cfg.MaxFailures)
	}
}

func TestLimiterIsRaceFree(t *testing.T) {
	l := NewLimiter(LimiterConfig{MaxFailures: 5, Window: time.Second, MaxKeys: 100})

	var wg sync.WaitGroup
	for i := range 50 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			key := "client-" + strconv.Itoa(i%10)
			for range 50 {
				l.Allow(key)
				l.RecordFailure(key)
				l.ShouldLog(key)
				if i%7 == 0 {
					l.RecordSuccess(key)
				}
			}
		}()
	}
	wg.Wait()
}

// --- client key selection ---

func TestClientKeyUsesRemoteAddrByDefault(t *testing.T) {
	r := httptest.NewRequest("GET", "/", nil)
	r.RemoteAddr = "203.0.113.9:54321"
	r.Header.Set("X-Forwarded-For", "1.2.3.4")

	// The header is not trusted, so a client cannot pick its own bucket by
	// setting one — which would let it evade the limit entirely.
	if got := clientKey(r, false); got != "203.0.113.9" {
		t.Errorf("clientKey = %q, want the socket address", got)
	}
	if got := clientKey(r, true); got != "1.2.3.4" {
		t.Errorf("clientKey with a trusted proxy = %q, want the forwarded address", got)
	}
}

func TestClientKeyHandlesProxyChainsAndOddAddresses(t *testing.T) {
	r := httptest.NewRequest("GET", "/", nil)
	r.RemoteAddr = "10.0.0.1:1234"

	// Left-most is the original client; the rest is the proxy chain.
	r.Header.Set("X-Forwarded-For", " 198.51.100.7 , 10.0.0.5, 10.0.0.6")
	if got := clientKey(r, true); got != "198.51.100.7" {
		t.Errorf("clientKey = %q, want the left-most entry", got)
	}

	// An empty or absent header falls back rather than producing a blank key,
	// which would collapse every such client into one bucket.
	r.Header.Set("X-Forwarded-For", "   ")
	if got := clientKey(r, true); got != "10.0.0.1" {
		t.Errorf("clientKey with a blank header = %q, want the socket address", got)
	}
	r.Header.Del("X-Forwarded-For")
	if got := clientKey(r, true); got != "10.0.0.1" {
		t.Errorf("clientKey with no header = %q", got)
	}

	// RemoteAddr is not always host:port.
	r.RemoteAddr = "@"
	if got := clientKey(r, false); got != "@" {
		t.Errorf("clientKey with an unsplittable address = %q", got)
	}
}

func TestRetryAfterIsWholeSeconds(t *testing.T) {
	for _, tc := range []struct {
		in   time.Duration
		want string
	}{
		{0, "1"},
		{time.Millisecond, "1"},
		{time.Second, "1"},
		{1500 * time.Millisecond, "2"},
		{90 * time.Second, "90"},
	} {
		if got := retryAfter(tc.in); got != tc.want {
			t.Errorf("retryAfter(%v) = %s, want %s", tc.in, got, tc.want)
		}
	}
}

// --- middleware integration ---

func limitedMiddleware(t *testing.T, clk *clock, failures int) (*Middleware, *Limiter) {
	t.Helper()
	a := testAuth(t, entry("ci", "admin", knownToken))
	l := NewLimiter(LimiterConfig{MaxFailures: failures, Window: time.Minute, Now: clk.now})
	return WithLimiter(a, slog.New(slog.NewTextHandler(io.Discard, nil)), l, false), l
}

func callFrom(t *testing.T, m *Middleware, scope Scope, token, addr string) *httptest.ResponseRecorder {
	t.Helper()
	r := httptest.NewRequest("GET", "/v1/models", nil)
	r.RemoteAddr = addr
	if token != "" {
		r.Header.Set("Authorization", "Bearer "+token)
	}
	rec := httptest.NewRecorder()
	m.Require(scope, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})).ServeHTTP(rec, r)
	return rec
}

// TestRepeatedBadTokensGet429 is the end-to-end behaviour: 401 while there is
// budget, then 429 with a Retry-After once there is not.
func TestRepeatedBadTokensGet429(t *testing.T) {
	clk := newClock()
	m, _ := limitedMiddleware(t, clk, 3)

	for i := range 3 {
		if got := callFrom(t, m, ScopeAdmin, "wrong", "9.9.9.9:1").Code; got != http.StatusUnauthorized {
			t.Fatalf("attempt %d = %d, want 401 while budget remains", i, got)
		}
	}

	rec := callFrom(t, m, ScopeAdmin, "wrong", "9.9.9.9:1")
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("after exhausting the budget = %d, want 429", rec.Code)
	}
	if ra := rec.Header().Get("Retry-After"); ra == "" {
		t.Error("429 without a Retry-After header")
	} else if n, err := strconv.Atoi(ra); err != nil || n < 1 {
		t.Errorf("Retry-After = %q, want whole seconds", ra)
	}
	// The 429 must not leak whether the token was close to anything.
	if strings.Contains(rec.Body.String(), "wrong") {
		t.Errorf("the 429 body echoed the attempted token: %s", rec.Body.String())
	}
}

// TestAValidCredentialIsNeverThrottled is the reason authentication runs before
// the rate-limit check. An IP is a shared resource: behind a NAT or a shared
// egress, refusing a correct credential because a neighbour is failing is an
// outage for somebody who did nothing wrong. The saving from checking the limit
// first is one SHA-256 over a short string, which is not worth that.
func TestAValidCredentialIsNeverThrottled(t *testing.T) {
	clk := newClock()
	m, _ := limitedMiddleware(t, clk, 2)

	// Exhaust the budget from this address with bad tokens.
	for range 10 {
		callFrom(t, m, ScopeAdmin, "wrong", "5.5.5.5:1")
	}
	if got := callFrom(t, m, ScopeAdmin, "wrong", "5.5.5.5:1").Code; got != http.StatusTooManyRequests {
		t.Fatal("the failing client was not throttled")
	}

	// Same address, correct token: it must still work.
	if got := callFrom(t, m, ScopeAdmin, knownToken, "5.5.5.5:1").Code; got != http.StatusOK {
		t.Fatalf("a valid credential from a throttled address = %d, want 200", got)
	}
}

// TestThrottledFailuresDoNotDeepenTheHole checks a throttled client's budget
// does not go further negative while it is being refused, which would make the
// wait grow the longer an attacker keeps trying and turn a transient mistake
// into an unbounded lockout.
func TestThrottledFailuresDoNotDeepenTheHole(t *testing.T) {
	clk := newClock()
	m, l := limitedMiddleware(t, clk, 2)

	for range 2 {
		callFrom(t, m, ScopeAdmin, "wrong", "3.3.3.3:1")
	}
	first := callFrom(t, m, ScopeAdmin, "wrong", "3.3.3.3:1")
	firstWait := first.Header().Get("Retry-After")

	for range 200 {
		callFrom(t, m, ScopeAdmin, "wrong", "3.3.3.3:1")
	}
	later := callFrom(t, m, ScopeAdmin, "wrong", "3.3.3.3:1")
	if got := later.Header().Get("Retry-After"); got != firstWait {
		t.Errorf("Retry-After grew from %s to %s while the client kept trying", firstWait, got)
	}
	if l.Tracked() != 1 {
		t.Errorf("tracking %d clients, want 1", l.Tracked())
	}
}

// TestAValidTokenAtVolumeIsUnaffected keeps this off the serving path: a
// correct caller sending any number of requests is never throttled, because
// success spends no budget.
func TestAValidTokenAtVolumeIsUnaffected(t *testing.T) {
	clk := newClock()
	m, _ := limitedMiddleware(t, clk, 2)

	for i := range 500 {
		if got := callFrom(t, m, ScopeAdmin, knownToken, "7.7.7.7:1").Code; got != http.StatusOK {
			t.Fatalf("request %d from a valid client = %d, want 200", i, got)
		}
	}
}

// TestScopeFailuresDoNotConsumeBudget covers the deliberate exemption: a valid
// credential refused for lacking a scope is a misconfigured client, and
// throttling it would take a deploy script offline for holding the wrong role.
func TestScopeFailuresDoNotConsumeBudget(t *testing.T) {
	clk := newClock()
	a := testAuth(t, entry("dash", "read", knownToken))
	l := NewLimiter(LimiterConfig{MaxFailures: 2, Window: time.Minute, Now: clk.now})
	m := WithLimiter(a, slog.New(slog.NewTextHandler(io.Discard, nil)), l, false)

	for i := range 50 {
		if got := callFrom(t, m, ScopeAdmin, knownToken, "8.8.8.8:1").Code; got != http.StatusForbidden {
			t.Fatalf("attempt %d = %d, want 403 every time", i, got)
		}
	}
	if l.Tracked() != 0 {
		t.Errorf("scope failures were counted against the rate limit (%d tracked)", l.Tracked())
	}
}

// TestRecoveryAfterBackoff is what a well-behaved client experiences: obey the
// Retry-After, present the right token, and carry on.
func TestRecoveryAfterBackoff(t *testing.T) {
	clk := newClock()
	m, _ := limitedMiddleware(t, clk, 2)

	for range 2 {
		callFrom(t, m, ScopeAdmin, "wrong", "4.4.4.4:1")
	}
	rec := callFrom(t, m, ScopeAdmin, "wrong", "4.4.4.4:1")
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("= %d, want 429", rec.Code)
	}

	wait, _ := strconv.Atoi(rec.Header().Get("Retry-After"))
	clk.advance(time.Duration(wait) * time.Second)

	// A correct token now works, and clears the client's history with it.
	if got := callFrom(t, m, ScopeAdmin, knownToken, "4.4.4.4:1").Code; got != http.StatusOK {
		t.Fatalf("after backing off, a valid token = %d, want 200", got)
	}
	for range 2 {
		if got := callFrom(t, m, ScopeAdmin, "wrong", "4.4.4.4:1").Code; got != http.StatusUnauthorized {
			t.Fatalf("budget was not restored after a success: %d", got)
		}
	}
}

func TestOneAttackerDoesNotThrottleOthersOverHTTP(t *testing.T) {
	clk := newClock()
	m, _ := limitedMiddleware(t, clk, 2)

	for range 20 {
		callFrom(t, m, ScopeAdmin, "wrong", "6.6.6.6:1")
	}
	if got := callFrom(t, m, ScopeAdmin, "wrong", "6.6.6.6:1").Code; got != http.StatusTooManyRequests {
		t.Fatal("the attacker was not throttled")
	}
	// A different address, still wrong token: 401, not 429.
	if got := callFrom(t, m, ScopeAdmin, "wrong", "1.1.1.1:1").Code; got != http.StatusUnauthorized {
		t.Errorf("an unrelated client got %d; one attacker must not throttle everyone", got)
	}
	if got := callFrom(t, m, ScopeAdmin, knownToken, "1.1.1.1:1").Code; got != http.StatusOK {
		t.Errorf("a legitimate client got %d while another address was throttled", got)
	}
}

// TestNoLimiterMeansNoThrottling keeps the feature optional, which is what the
// pre-existing middleware tests rely on.
func TestNoLimiterMeansNoThrottling(t *testing.T) {
	a := testAuth(t, entry("ci", "admin", knownToken))
	m := NewMiddleware(a, slog.New(slog.NewTextHandler(io.Discard, nil)))

	for range 200 {
		if got := callFrom(t, m, ScopeAdmin, "wrong", "2.2.2.2:1").Code; got != http.StatusUnauthorized {
			t.Fatalf("= %d, want 401 with no limiter configured", got)
		}
	}
}
