package auth

import (
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func testSessions(t *testing.T, clk *clock, mutate ...func(*SessionConfig)) *SessionStore {
	t.Helper()
	cfg := SessionConfig{TTL: time.Hour, Now: clk.now}
	for _, m := range mutate {
		m(&cfg)
	}
	return NewSessionStore(cfg)
}

func userToken(expires time.Time) Token {
	return Token{
		Name: "sahil@example.com", Scopes: []Scope{ScopeAdmin},
		Subject: "sub-1", Issuer: "https://idp.example.com", NotAfter: expires,
	}
}

func TestSessionRoundTrip(t *testing.T) {
	clk := newClock()
	store := testSessions(t, clk)

	sess, err := store.Create(userToken(time.Time{}))
	if err != nil {
		t.Fatal(err)
	}
	if sess.ID == "" || sess.CSRF == "" {
		t.Fatal("session has no id or CSRF token")
	}
	if sess.ID == sess.CSRF {
		t.Error("the session id and the CSRF token are the same value")
	}

	got, err := store.Get(sess.ID)
	if err != nil || got.Token.Name != "sahil@example.com" {
		t.Fatalf("Get = %+v, %v", got, err)
	}

	if !store.Delete(sess.ID) {
		t.Error("Delete reported nothing to remove")
	}
	if _, err := store.Get(sess.ID); !errors.Is(err, ErrNoSession) {
		t.Errorf("Get after Delete = %v, want ErrNoSession", err)
	}
	if store.Delete(sess.ID) {
		t.Error("Delete removed the same session twice")
	}
}

// TestSessionIDsAreUnpredictable is the whole security of the scheme: the
// cookie is a bearer credential, and a guessable one is no credential at all.
func TestSessionIDsAreUnpredictable(t *testing.T) {
	clk := newClock()
	store := testSessions(t, clk, func(c *SessionConfig) { c.MaxSessions = 2000 })

	seen := make(map[string]bool, 500)
	for range 500 {
		sess, err := store.Create(userToken(time.Time{}))
		if err != nil {
			t.Fatal(err)
		}
		if seen[sess.ID] {
			t.Fatal("a session id was issued twice")
		}
		seen[sess.ID] = true
		if len(sess.ID) < 40 {
			t.Fatalf("session id is only %d characters", len(sess.ID))
		}
	}
}

// TestSessionNeverOutlivesItsToken: a session that survived the credential
// behind it would be an access grant the identity provider can no longer
// withdraw.
func TestSessionNeverOutlivesItsToken(t *testing.T) {
	clk := newClock()
	store := testSessions(t, clk, func(c *SessionConfig) { c.TTL = 12 * time.Hour })

	// Token expires long before the session TTL would.
	tokenExpiry := clk.now().Add(30 * time.Minute)
	sess, err := store.Create(userToken(tokenExpiry))
	if err != nil {
		t.Fatal(err)
	}
	if !sess.Expires.Equal(tokenExpiry) {
		t.Errorf("session expires at %v, want the token's %v", sess.Expires, tokenExpiry)
	}

	clk.advance(31 * time.Minute)
	if _, err := store.Get(sess.ID); !errors.Is(err, ErrNoSession) {
		t.Error("a session outlived the token it was built from")
	}

	// And the TTL caps a token that outlives it.
	clk2 := newClock()
	short := NewSessionStore(SessionConfig{TTL: time.Minute, Now: clk2.now})
	long, err := short.Create(userToken(clk2.now().Add(100 * time.Hour)))
	if err != nil {
		t.Fatal(err)
	}
	if long.Expires.After(clk2.now().Add(time.Minute)) {
		t.Errorf("session expires at %v, past the one-minute TTL", long.Expires)
	}
}

// TestSessionTableIsBounded stops an attacker growing memory by logging in
// repeatedly.
func TestSessionTableIsBounded(t *testing.T) {
	clk := newClock()
	store := testSessions(t, clk, func(c *SessionConfig) { c.MaxSessions = 10 })

	for range 10 {
		if _, err := store.Create(userToken(time.Time{})); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := store.Create(userToken(time.Time{})); err == nil {
		t.Fatal("the store accepted an eleventh session past its cap of 10")
	}

	// Expired sessions are reclaimed, so the cap bounds *live* sessions rather
	// than sessions ever created.
	clk.advance(2 * time.Hour)
	if _, err := store.Create(userToken(time.Time{})); err != nil {
		t.Fatalf("the store did not reclaim expired sessions: %v", err)
	}
	if n := store.Count(); n != 1 {
		t.Errorf("Count = %d after the sweep, want 1", n)
	}
}

// --- CSRF ---

func TestCSRFAllowsSafeMethods(t *testing.T) {
	sess := &Session{CSRF: "the-token"}
	for _, method := range []string{"GET", "HEAD", "OPTIONS"} {
		r := httptest.NewRequest(method, "/v1/models", nil)
		if err := CheckCSRF(r, sess); err != nil {
			t.Errorf("%s was refused: %v", method, err)
		}
	}
}

// TestCSRFRequiredForStateChangingMethods is the second lock on the door
// SameSite already closes. SameSite is a browser behaviour; a token the
// attacker cannot read does not depend on one.
func TestCSRFRequiredForStateChangingMethods(t *testing.T) {
	sess := &Session{CSRF: "the-token"}

	for _, method := range []string{"POST", "PUT", "PATCH", "DELETE"} {
		r := httptest.NewRequest(method, "/v1/models", nil)
		if err := CheckCSRF(r, sess); !errors.Is(err, ErrCSRFFailed) {
			t.Errorf("%s with no token = %v, want ErrCSRFFailed", method, err)
		}

		r = httptest.NewRequest(method, "/v1/models", nil)
		r.Header.Set(CSRFHeader, "not-the-token")
		if err := CheckCSRF(r, sess); !errors.Is(err, ErrCSRFFailed) {
			t.Errorf("%s with the wrong token = %v, want ErrCSRFFailed", method, err)
		}

		r = httptest.NewRequest(method, "/v1/models", nil)
		r.Header.Set(CSRFHeader, "the-token")
		if err := CheckCSRF(r, sess); err != nil {
			t.Errorf("%s with the right token was refused: %v", method, err)
		}
	}
}

// --- cookies ---

// TestSessionCookieAttributes are the difference between a session cookie and a
// liability. Each one closes a specific hole.
func TestSessionCookieAttributes(t *testing.T) {
	rec := httptest.NewRecorder()
	SetCookie(rec, SessionCookie, "abc", time.Now().Add(time.Hour), CookieOptions{})

	cookies := rec.Result().Cookies()
	if len(cookies) != 1 {
		t.Fatalf("got %d cookies", len(cookies))
	}
	c := cookies[0]

	// Script cannot read it, so an XSS bug elsewhere cannot steal the session.
	if !c.HttpOnly {
		t.Error("the session cookie is not HttpOnly")
	}
	// It never travels over plaintext.
	if !c.Secure {
		t.Error("the session cookie is not Secure")
	}
	// Not sent on cross-site form posts, which is the primary CSRF defence.
	if c.SameSite != http.SameSiteLaxMode {
		t.Errorf("SameSite = %v, want Lax", c.SameSite)
	}
	// Host-only: a Domain attribute would share the session with every
	// subdomain, so one compromised subdomain could read it.
	if c.Domain != "" {
		t.Errorf("the cookie sets Domain=%q, making it readable by subdomains", c.Domain)
	}
	if c.Path != "/" {
		t.Errorf("Path = %q, want /", c.Path)
	}
}

// TestCSRFCookieIsReadableByScript is the one cookie that must not be HttpOnly,
// because the double-submit pattern depends on same-origin script reading it.
func TestCSRFCookieIsReadableByScript(t *testing.T) {
	rec := httptest.NewRecorder()
	SetCookie(rec, CSRFCookie, "abc", time.Now().Add(time.Hour), CookieOptions{})

	c := rec.Result().Cookies()[0]
	if c.HttpOnly {
		t.Error("the CSRF cookie is HttpOnly, so no client can echo it back")
	}
	if !c.Secure || c.SameSite != http.SameSiteLaxMode {
		t.Error("the CSRF cookie lost its other protections")
	}
}

func TestInsecureCookiesAreOptIn(t *testing.T) {
	rec := httptest.NewRecorder()
	SetCookie(rec, SessionCookie, "abc", time.Now().Add(time.Hour), CookieOptions{Insecure: true})
	if rec.Result().Cookies()[0].Secure {
		t.Error("Insecure did not drop the Secure attribute")
	}

	rec = httptest.NewRecorder()
	ClearCookie(rec, SessionCookie, CookieOptions{})
	c := rec.Result().Cookies()[0]
	if c.MaxAge >= 0 || c.Value != "" {
		t.Errorf("ClearCookie did not expire the cookie: %+v", c)
	}
}

// --- middleware integration ---

func sessionMiddleware(t *testing.T, store *SessionStore) *Middleware {
	t.Helper()
	a := testAuth(t, entry("ci", "admin", knownToken))
	return NewMiddleware(a, slog.New(slog.NewTextHandler(io.Discard, nil))).WithSessions(store)
}

func sessionRequest(method string, sess *Session, csrf string) *http.Request {
	r := httptest.NewRequest(method, "/v1/models", nil)
	if sess != nil {
		r.AddCookie(&http.Cookie{Name: SessionCookie, Value: sess.ID})
	}
	if csrf != "" {
		r.Header.Set(CSRFHeader, csrf)
	}
	return r
}

func serveSession(t *testing.T, m *Middleware, scope Scope, r *http.Request) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	m.Require(scope, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})).ServeHTTP(rec, r)
	return rec
}

func TestCookieAuthenticatesABrowser(t *testing.T) {
	clk := newClock()
	store := testSessions(t, clk)
	m := sessionMiddleware(t, store)

	sess, err := store.Create(userToken(time.Time{}))
	if err != nil {
		t.Fatal(err)
	}

	if got := serveSession(t, m, ScopeRead, sessionRequest("GET", sess, "")).Code; got != http.StatusOK {
		t.Errorf("a session cookie on a GET = %d, want 200", got)
	}
	// An unknown cookie is not a credential.
	r := httptest.NewRequest("GET", "/v1/models", nil)
	r.AddCookie(&http.Cookie{Name: SessionCookie, Value: "made-up"})
	if got := serveSession(t, m, ScopeRead, r).Code; got != http.StatusUnauthorized {
		t.Errorf("an unknown session cookie = %d, want 401", got)
	}
}

// TestCookieWriteNeedsCSRF is the attack this defends against: a page on
// another origin making the browser POST here with its ambient cookie.
func TestCookieWriteNeedsCSRF(t *testing.T) {
	clk := newClock()
	store := testSessions(t, clk)
	m := sessionMiddleware(t, store)

	sess, err := store.Create(userToken(time.Time{}))
	if err != nil {
		t.Fatal(err)
	}

	if got := serveSession(t, m, ScopeAdmin, sessionRequest("POST", sess, "")).Code; got != http.StatusForbidden {
		t.Errorf("a cookie POST with no CSRF token = %d, want 403", got)
	}
	if got := serveSession(t, m, ScopeAdmin, sessionRequest("POST", sess, "wrong")).Code; got != http.StatusForbidden {
		t.Errorf("a cookie POST with the wrong CSRF token = %d, want 403", got)
	}
	if got := serveSession(t, m, ScopeAdmin, sessionRequest("POST", sess, sess.CSRF)).Code; got != http.StatusOK {
		t.Errorf("a cookie POST with the right CSRF token = %d, want 200", got)
	}
}

// TestOneSessionsCSRFTokenDoesNotWorkForAnother: the token is bound to the
// session, not global, so obtaining one from your own login does not let you
// act on somebody else's.
func TestOneSessionsCSRFTokenDoesNotWorkForAnother(t *testing.T) {
	clk := newClock()
	store := testSessions(t, clk)
	m := sessionMiddleware(t, store)

	victim, err := store.Create(userToken(time.Time{}))
	if err != nil {
		t.Fatal(err)
	}
	attacker, err := store.Create(userToken(time.Time{}))
	if err != nil {
		t.Fatal(err)
	}

	if got := serveSession(t, m, ScopeAdmin, sessionRequest("POST", victim, attacker.CSRF)).Code; got != http.StatusForbidden {
		t.Errorf("another session's CSRF token was accepted: %d", got)
	}
}

// TestBearerTokenBeatsCookie: a client that presented a credential explicitly
// meant it, and an ambient cookie must not silently override it.
func TestBearerTokenBeatsCookie(t *testing.T) {
	clk := newClock()
	store := testSessions(t, clk)
	m := sessionMiddleware(t, store)

	sess, err := store.Create(Token{Name: "browser-user", Scopes: []Scope{ScopeRead}, Subject: "s"})
	if err != nil {
		t.Fatal(err)
	}

	// The cookie carries only read; the bearer token carries admin. The
	// request must be admin, and must not need a CSRF token, because it did
	// not authenticate with a cookie.
	r := httptest.NewRequest("POST", "/v1/models", nil)
	r.AddCookie(&http.Cookie{Name: SessionCookie, Value: sess.ID})
	r.Header.Set("Authorization", "Bearer "+knownToken)

	if got := serveSession(t, m, ScopeAdmin, r).Code; got != http.StatusOK {
		t.Errorf("a bearer token alongside a cookie = %d, want 200", got)
	}
}

// TestExpiredSessionIsRefused covers the deadline through the middleware.
func TestExpiredSessionIsRefused(t *testing.T) {
	clk := newClock()
	store := testSessions(t, clk)
	m := sessionMiddleware(t, store)

	sess, err := store.Create(userToken(time.Time{}))
	if err != nil {
		t.Fatal(err)
	}
	clk.advance(2 * time.Hour)

	if got := serveSession(t, m, ScopeRead, sessionRequest("GET", sess, "")).Code; got != http.StatusUnauthorized {
		t.Errorf("an expired session = %d, want 401", got)
	}
}

// TestSessionScopesAreEnforced: a cookie is not a bypass of authorisation.
func TestSessionScopesAreEnforced(t *testing.T) {
	clk := newClock()
	store := testSessions(t, clk)
	m := sessionMiddleware(t, store)

	sess, err := store.Create(Token{Name: "reader", Scopes: []Scope{ScopeRead}, Subject: "s"})
	if err != nil {
		t.Fatal(err)
	}

	if got := serveSession(t, m, ScopeRead, sessionRequest("GET", sess, "")).Code; got != http.StatusOK {
		t.Errorf("a read session on a read route = %d, want 200", got)
	}
	if got := serveSession(t, m, ScopeAdmin, sessionRequest("POST", sess, sess.CSRF)).Code; got != http.StatusForbidden {
		t.Errorf("a read session on an admin route = %d, want 403", got)
	}
}

func TestNoSessionStoreMeansCookiesAreIgnored(t *testing.T) {
	a := testAuth(t, entry("ci", "admin", knownToken))
	m := NewMiddleware(a, slog.New(slog.NewTextHandler(io.Discard, nil)))

	r := httptest.NewRequest("GET", "/v1/models", nil)
	r.AddCookie(&http.Cookie{Name: SessionCookie, Value: "anything"})
	if got := serveSession(t, m, ScopeRead, r).Code; got != http.StatusUnauthorized {
		t.Errorf("a cookie without a session store = %d, want 401", got)
	}
}

// TestLoginStoreIsSingleUse stops an authorization code being replayed against
// the same pending login.
func TestLoginStoreIsSingleUse(t *testing.T) {
	clk := newClock()
	store := NewLoginStore(clk.now)

	id, err := store.Start("state", "nonce", "verifier", "/next")
	if err != nil {
		t.Fatal(err)
	}
	if store.Pending() != 1 {
		t.Errorf("Pending = %d, want 1", store.Pending())
	}

	p, ok := store.Take(id)
	if !ok || p.State != "state" || p.Verifier != "verifier" || p.Next != "/next" {
		t.Fatalf("Take = %+v, %v", p, ok)
	}
	if _, ok := store.Take(id); ok {
		t.Error("the same pending login was consumed twice")
	}
}

func TestLoginStoreExpires(t *testing.T) {
	clk := newClock()
	store := NewLoginStore(clk.now)

	id, err := store.Start("s", "n", "v", "/")
	if err != nil {
		t.Fatal(err)
	}
	clk.advance(11 * time.Minute)
	if _, ok := store.Take(id); ok {
		t.Error("an expired pending login was accepted")
	}
	if store.Pending() != 0 {
		t.Errorf("Pending = %d after expiry, want 0", store.Pending())
	}
}

func TestRandomTokenIsUsable(t *testing.T) {
	seen := map[string]bool{}
	for range 200 {
		tok, err := RandomToken()
		if err != nil {
			t.Fatal(err)
		}
		if seen[tok] {
			t.Fatal("RandomToken repeated a value")
		}
		seen[tok] = true
		if strings.ContainsAny(tok, " \t\n;,") {
			t.Fatalf("token contains a character awkward in a cookie: %q", tok)
		}
	}
}
