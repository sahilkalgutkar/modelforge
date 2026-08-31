package auth

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// Cookie names. The host-only, path-wide defaults are deliberate; see SetCookie.
const (
	// SessionCookie holds an opaque session id and nothing else.
	SessionCookie = "modelforge_session"
	// CSRFCookie holds the double-submit token. It is deliberately *not*
	// HttpOnly, because the whole mechanism depends on same-origin script
	// being able to read it and echo it back in a header.
	CSRFCookie = "modelforge_csrf"
	// LoginCookie holds the id of an in-progress login.
	LoginCookie = "modelforge_login"
)

// CSRFHeader is where a client echoes the CSRF cookie's value.
const CSRFHeader = "X-CSRF-Token"

// Errors callers are expected to handle.
var (
	ErrNoSession  = errors.New("auth: no session")
	ErrCSRFFailed = errors.New("auth: CSRF check failed")
)

// Session is a signed-in browser.
//
// Everything about the identity lives here, on the server, and the cookie
// carries nothing but an opaque random id. The alternative — putting the
// identity in the cookie and signing it — means a session cannot be revoked
// before it expires, because the server has nothing to delete. Keeping the
// state here makes logout mean something.
type Session struct {
	ID      string
	Token   Token
	CSRF    string
	Created time.Time
	Expires time.Time
}

// Expired reports whether the session is past its deadline.
func (s *Session) Expired(now time.Time) bool { return now.After(s.Expires) }

// SessionStore holds browser sessions in memory.
//
// In memory rather than in Postgres, which means a restart signs everybody out.
// That is the right trade for this: sessions are minutes-to-hours of
// convenience for a human looking at a dashboard, restarts are rare, and the
// alternative writes credential-equivalent material into a database that is
// backed up, replicated, and read by people debugging models. Losing a session
// costs one click; leaking one costs rather more.
type SessionStore struct {
	ttl     time.Duration
	maxSize int
	now     func() time.Time

	mu       sync.Mutex
	sessions map[string]*Session
}

// SessionConfig configures a SessionStore.
type SessionConfig struct {
	// TTL caps how long a session lasts regardless of what the identity
	// provider said. A session also ends when the underlying token expires,
	// whichever comes first.
	TTL time.Duration
	// MaxSessions bounds memory. See Create for what happens at the cap.
	MaxSessions int
	Now         func() time.Time
}

// NewSessionStore creates a SessionStore.
func NewSessionStore(cfg SessionConfig) *SessionStore {
	if cfg.TTL <= 0 {
		cfg.TTL = 12 * time.Hour
	}
	if cfg.MaxSessions <= 0 {
		cfg.MaxSessions = 10000
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	return &SessionStore{
		ttl:      cfg.TTL,
		maxSize:  cfg.MaxSessions,
		now:      cfg.Now,
		sessions: make(map[string]*Session),
	}
}

// Create starts a session for an authenticated identity.
//
// A fresh id is minted here and nowhere else, which is what makes session
// fixation impossible: there is no way to hand somebody an id before they
// authenticate and have it become their session afterwards.
func (s *SessionStore) Create(tok Token) (*Session, error) {
	id, err := randomID()
	if err != nil {
		return nil, err
	}
	csrf, err := randomID()
	if err != nil {
		return nil, err
	}

	now := s.now()
	expires := now.Add(s.ttl)
	// The session must not outlive the credential behind it. A user whose
	// access was revoked at the provider should stop working here when their
	// token would have expired, not hours later because a cookie said so.
	if !tok.NotAfter.IsZero() && tok.NotAfter.Before(expires) {
		expires = tok.NotAfter
	}

	sess := &Session{ID: id, Token: tok, CSRF: csrf, Created: now, Expires: expires}

	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.sessions) >= s.maxSize {
		s.sweepLocked()
	}
	if len(s.sessions) >= s.maxSize {
		// Refusing is right here, unlike the rate limiter's table. There the
		// cost of failing closed was locking out innocent clients; here it is
		// one person being told to try again, and the alternative is an
		// unbounded map an attacker can grow by logging in repeatedly.
		return nil, errors.New("auth: too many active sessions")
	}
	s.sessions[id] = sess
	return sess, nil
}

// Get returns a live session, or ErrNoSession.
func (s *SessionStore) Get(id string) (*Session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	sess, ok := s.sessions[id]
	if !ok {
		return nil, ErrNoSession
	}
	if sess.Expired(s.now()) {
		delete(s.sessions, id)
		return nil, ErrNoSession
	}
	return sess, nil
}

// Delete ends a session.
func (s *SessionStore) Delete(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.sessions[id]; !ok {
		return false
	}
	delete(s.sessions, id)
	return true
}

// Count is how many sessions are live, for tests and metrics.
func (s *SessionStore) Count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sweepLocked()
	return len(s.sessions)
}

func (s *SessionStore) sweepLocked() {
	now := s.now()
	for id, sess := range s.sessions {
		if sess.Expired(now) {
			delete(s.sessions, id)
		}
	}
}

// --- pending logins ---

// pendingLogin is an in-progress browser login.
//
// The PKCE verifier, the state and the nonce are held here rather than in a
// cookie. They could be put in a signed cookie, but there is no reason to hand
// a browser the very secret that proves the token exchange belongs to this
// login — keeping it server-side means an attacker who can read cookies still
// cannot complete somebody else's exchange.
type pendingLogin struct {
	State    string
	Nonce    string
	Verifier string
	Next     string
	Created  time.Time
}

// LoginStore holds in-progress logins.
type LoginStore struct {
	ttl     time.Duration
	maxSize int
	now     func() time.Time

	mu      sync.Mutex
	pending map[string]*pendingLogin
}

// NewLoginStore creates a LoginStore. The TTL is short because it only has to
// cover the time somebody spends typing a password.
func NewLoginStore(now func() time.Time) *LoginStore {
	if now == nil {
		now = time.Now
	}
	return &LoginStore{
		ttl:     10 * time.Minute,
		maxSize: 10000,
		now:     now,
		pending: make(map[string]*pendingLogin),
	}
}

// Start records a new in-progress login and returns its id.
func (l *LoginStore) Start(state, nonce, verifier, next string) (string, error) {
	id, err := randomID()
	if err != nil {
		return "", err
	}

	l.mu.Lock()
	defer l.mu.Unlock()
	if len(l.pending) >= l.maxSize {
		l.sweepLocked()
	}
	if len(l.pending) >= l.maxSize {
		return "", errors.New("auth: too many logins in progress")
	}
	l.pending[id] = &pendingLogin{
		State: state, Nonce: nonce, Verifier: verifier, Next: next, Created: l.now(),
	}
	return id, nil
}

// Take consumes an in-progress login.
//
// It is removed whether or not it turns out to match, so a callback can be
// completed exactly once. Leaving it in place would let an authorization code
// be replayed against the same pending login.
func (l *LoginStore) Take(id string) (*pendingLogin, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()

	p, ok := l.pending[id]
	if !ok {
		return nil, false
	}
	delete(l.pending, id)
	if l.now().Sub(p.Created) > l.ttl {
		return nil, false
	}
	return p, true
}

func (l *LoginStore) sweepLocked() {
	now := l.now()
	for id, p := range l.pending {
		if now.Sub(p.Created) > l.ttl {
			delete(l.pending, id)
		}
	}
}

// Pending is how many logins are in progress, for tests.
func (l *LoginStore) Pending() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.sweepLocked()
	return len(l.pending)
}

// --- cookies ---

// CookieOptions controls how session cookies are written.
type CookieOptions struct {
	// Insecure drops the Secure attribute so cookies work over plain HTTP.
	//
	// It exists for local development and has to be asked for. Without Secure
	// a session cookie is sent over any http:// request to the same host,
	// where anybody on the path can read it — which is the entire session.
	Insecure bool
}

// SetCookie writes a cookie with the attributes a session cookie needs.
//
// HttpOnly keeps script from reading the session id, so an XSS bug somewhere
// else cannot walk off with the session. Secure keeps it off plaintext HTTP.
// SameSite=Lax is the primary CSRF defence: the cookie is not sent on
// cross-site form posts or fetches, only on top-level navigations, which are
// GETs and safe. No Domain attribute is set, which makes the cookie host-only —
// setting one would share it with every subdomain, and a single compromised
// subdomain would then be able to read it.
func SetCookie(w http.ResponseWriter, name, value string, expires time.Time, opts CookieOptions) {
	http.SetCookie(w, &http.Cookie{
		Name:     name,
		Value:    value,
		Path:     "/",
		Expires:  expires,
		HttpOnly: name != CSRFCookie,
		Secure:   !opts.Insecure,
		SameSite: http.SameSiteLaxMode,
	})
}

// ClearCookie expires a cookie.
func ClearCookie(w http.ResponseWriter, name string, opts CookieOptions) {
	http.SetCookie(w, &http.Cookie{
		Name:     name,
		Value:    "",
		Path:     "/",
		MaxAge:   -1,
		HttpOnly: name != CSRFCookie,
		Secure:   !opts.Insecure,
		SameSite: http.SameSiteLaxMode,
	})
}

// CheckCSRF verifies the double-submit token on a state-changing request.
//
// SameSite=Lax already stops a cross-site page from sending the session cookie
// on a POST, and this is the second lock on the same door. It is worth having
// because SameSite is a browser behaviour rather than a server one: an old
// browser, a misconfigured proxy that strips the attribute, or a future
// same-site-but-untrusted page all defeat it, and none of them defeat a token
// the attacker cannot read.
//
// The comparison is constant-time. The token is a secret the attacker is trying
// to guess, compared byte by byte — which is exactly the shape that leaks, and
// unlike the token digests elsewhere in this package there is no hash in front
// of it to scramble a near miss.
func CheckCSRF(r *http.Request, sess *Session) error {
	if isSafeMethod(r.Method) {
		return nil
	}

	presented := r.Header.Get(CSRFHeader)
	if presented == "" {
		return fmt.Errorf("%w: no %s header", ErrCSRFFailed, CSRFHeader)
	}
	if subtle.ConstantTimeCompare([]byte(presented), []byte(sess.CSRF)) != 1 {
		return fmt.Errorf("%w: the %s header does not match this session", ErrCSRFFailed, CSRFHeader)
	}
	return nil
}

func isSafeMethod(method string) bool {
	switch method {
	case http.MethodGet, http.MethodHead, http.MethodOptions:
		return true
	}
	return false
}

// RandomToken returns 256 bits of randomness, base64url encoded. It is exported
// for the state and nonce values a browser login needs.
func RandomToken() (string, error) { return randomID() }

// randomID returns 256 bits of randomness, base64url encoded.
func randomID() (string, error) {
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		return "", fmt.Errorf("auth: generate random id: %w", err)
	}
	return base64.RawURLEncoding.EncodeToString(buf), nil
}
