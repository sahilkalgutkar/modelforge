package auth

import (
	"fmt"
	"math"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Limiter throttles clients that keep failing authentication.
//
// It is worth being exact about what this buys, because the obvious claim is
// the wrong one. It does *not* meaningfully protect against credential
// guessing: the tokens are 256 bits of randomness, so an attacker at a million
// attempts a second is still not going to find one before the heat death of
// the universe, and a rate limit does not change that arithmetic in any way
// that matters. Saying it "prevents brute force" would be the security theatre
// worth avoiding.
//
// What it does bound is the cost of somebody hammering the endpoint. Every
// rejection runs the HTTP stack, a SHA-256, and — the expensive part — writes a
// log line containing attacker-controlled request data. Unbounded, that is a
// free way to fill a disk, run up a log-ingest bill, and bury the handful of
// entries an operator actually needs during an incident. Bounding it costs a
// map lookup.
//
// So this is a cost control and a log-flood defence, not an authentication
// control. A gateway is still the right place for volumetric defence, and this
// does not replace one.
type Limiter struct {
	burst        float64
	refillPerSec float64
	maxKeys      int
	now          func() time.Time

	mu      sync.Mutex
	buckets map[string]*bucket

	// onLimit fires the first time a key is throttled, so the server can count
	// it. It is a callback rather than a metric so this package does not have
	// to know about Prometheus.
	onLimit func(key string)
}

// bucket is one client's remaining failure budget.
type bucket struct {
	tokens float64
	last   time.Time

	// limited records whether this key is currently being throttled, so the
	// first rejection logs and the rest do not.
	limited bool
	// suppressed counts rejections that were not logged while limited, so the
	// eventual summary can say how many there were.
	suppressed int
}

// LimiterConfig configures a Limiter.
type LimiterConfig struct {
	// MaxFailures is the burst: how many authentication failures a client may
	// make before it is throttled at all. It is not 1, because a
	// misconfigured deploy script or a human retrying a stale token should get
	// a few honest attempts before being slowed down.
	MaxFailures int

	// Window is how long a fully exhausted budget takes to refill. The
	// sustained rate is MaxFailures/Window.
	Window time.Duration

	// MaxKeys bounds how many clients are tracked at once, which bounds the
	// memory this can consume. See Allow for what happens at the cap.
	MaxKeys int

	// Now is injectable so tests can drive the clock rather than sleep.
	Now func() time.Time

	// OnLimit is called the first time a key is throttled.
	OnLimit func(key string)
}

func (c *LimiterConfig) setDefaults() {
	if c.MaxFailures <= 0 {
		c.MaxFailures = 10
	}
	if c.Window <= 0 {
		c.Window = time.Minute
	}
	if c.MaxKeys <= 0 {
		c.MaxKeys = 10000
	}
	if c.Now == nil {
		c.Now = time.Now
	}
}

// NewLimiter creates a Limiter.
func NewLimiter(cfg LimiterConfig) *Limiter {
	cfg.setDefaults()
	return &Limiter{
		burst:        float64(cfg.MaxFailures),
		refillPerSec: float64(cfg.MaxFailures) / cfg.Window.Seconds(),
		maxKeys:      cfg.MaxKeys,
		now:          cfg.Now,
		buckets:      make(map[string]*bucket),
		onLimit:      cfg.OnLimit,
	}
}

// Allow reports whether a request from key should be processed at all, and how
// long to tell the client to wait if not.
//
// It does not consume budget: a request that succeeds must not count against
// anybody. Only RecordFailure spends a token, so a legitimate caller sending
// valid credentials at any volume is never throttled by this.
func (l *Limiter) Allow(key string) (bool, time.Duration) {
	l.mu.Lock()
	defer l.mu.Unlock()

	b, ok := l.buckets[key]
	if !ok {
		// An untracked key has its full budget by definition, and creating the
		// bucket here would let read-only traffic populate the table. It is
		// created on the first failure instead.
		return true, 0
	}
	l.refill(b)

	if b.tokens >= 1 {
		return true, 0
	}
	// Time until one token is available, rounded up so a client that obeys the
	// hint is not immediately rejected again.
	wait := time.Duration(math.Ceil((1-b.tokens)/l.refillPerSec)) * time.Second
	if wait < time.Second {
		wait = time.Second
	}
	return false, wait
}

// RecordFailure spends one unit of a client's budget.
//
// Only authentication failures are recorded — a token that is missing or
// unrecognised. A *valid* token refused for lacking a scope is deliberately not
// counted: that is a misconfigured client rather than somebody guessing, and
// throttling it would take a deploy script offline for holding the wrong role,
// which is a confusing outage to debug and does nothing for security.
func (l *Limiter) RecordFailure(key string) {
	l.mu.Lock()
	defer l.mu.Unlock()

	b, ok := l.buckets[key]
	if !ok {
		if !l.makeRoom() {
			// The table is full of active offenders. Failing open is
			// deliberate: refusing to track a new key and denying it instead
			// would let an attacker fill the table and lock everybody else
			// out, turning a rate limiter into the outage it exists to
			// prevent. Untracked traffic is no worse off than it would be with
			// no limiter at all.
			return
		}
		b = &bucket{tokens: l.burst, last: l.now()}
		l.buckets[key] = b
	}
	l.refill(b)

	if b.tokens >= 1 {
		b.tokens--
	}
}

// RecordSuccess restores a client's budget.
//
// A caller that fixed its credentials should not stay throttled for the rest of
// the window because of attempts it has already corrected.
func (l *Limiter) RecordSuccess(key string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	delete(l.buckets, key)
}

// refill adds the tokens that have accrued since this bucket was last touched.
// The caller must hold the lock.
func (l *Limiter) refill(b *bucket) {
	now := l.now()
	if elapsed := now.Sub(b.last).Seconds(); elapsed > 0 {
		b.tokens = math.Min(l.burst, b.tokens+elapsed*l.refillPerSec)
		b.last = now
	}
	if b.tokens >= l.burst {
		// Fully refilled means the client has no history worth remembering.
		b.limited = false
		b.suppressed = 0
	}
}

// makeRoom ensures there is space for one more key, and reports whether there
// is. The caller must hold the lock.
//
// Sweeping fully-refilled buckets is lossless: a bucket at full budget is
// indistinguishable from one that does not exist, so dropping it forgets
// nothing. That is what keeps the table proportional to the number of clients
// currently failing rather than to every client ever seen.
func (l *Limiter) makeRoom() bool {
	if len(l.buckets) < l.maxKeys {
		return true
	}
	for k, b := range l.buckets {
		l.refill(b)
		if b.tokens >= l.burst {
			delete(l.buckets, k)
		}
	}
	return len(l.buckets) < l.maxKeys
}

// ShouldLog reports whether this rejection should be written to the log, and
// how many were suppressed since the last one that was.
//
// The first rejection from a key is logged; the rest are counted silently until
// the key recovers. Log volume is the main thing this limiter is defending, so
// logging every throttled request would be self-defeating.
func (l *Limiter) ShouldLog(key string) (bool, int) {
	l.mu.Lock()
	defer l.mu.Unlock()

	b, ok := l.buckets[key]
	if !ok {
		return true, 0
	}
	if !b.limited {
		b.limited = true
		suppressed := b.suppressed
		b.suppressed = 0
		if l.onLimit != nil {
			l.onLimit(key)
		}
		return true, suppressed
	}
	b.suppressed++
	return false, 0
}

// Tracked returns how many clients are currently being tracked. A value at
// MaxKeys means the table is saturated and new offenders are going untracked,
// which is the signal that volumetric defence is needed upstream.
func (l *Limiter) Tracked() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return len(l.buckets)
}

// clientKey identifies the client a request came from, for rate-limiting.
//
// This is the part of the design with a genuine tension and no answer that is
// right everywhere.
//
// RemoteAddr is the only value the server actually observes, and it cannot be
// forged over a completed TCP handshake. But behind a load balancer every
// request carries the proxy's address, so a single bucket would cover the whole
// internet and the first attacker would throttle every real user — a limiter
// that causes the outage it exists to prevent.
//
// X-Forwarded-For fixes that and is trivially spoofable, so trusting it by
// default would let an attacker rotate a header and evade the limit entirely,
// while also letting them write arbitrary strings into the log and the bucket
// table.
//
// So the header is used only when the operator says their deployment is behind
// a proxy that overwrites it. Only they know the topology, and neither default
// is safe to guess.
func clientKey(r *http.Request, trustForwardedFor bool) string {
	if trustForwardedFor {
		if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
			// The left-most entry is the original client; everything after is
			// the proxy chain. A trusted proxy overwrites this header, so the
			// value is only as good as that promise.
			if i := strings.Index(xff, ","); i >= 0 {
				xff = xff[:i]
			}
			if ip := strings.TrimSpace(xff); ip != "" {
				return ip
			}
		}
	}

	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		// RemoteAddr is not always host:port (a unix socket, or a test
		// harness), and the raw value is still a stable key.
		return r.RemoteAddr
	}
	return host
}

// retryAfter renders a duration for the Retry-After header, which is in whole
// seconds.
func retryAfter(d time.Duration) string {
	secs := int(math.Ceil(d.Seconds()))
	if secs < 1 {
		secs = 1
	}
	return fmt.Sprintf("%d", secs)
}
