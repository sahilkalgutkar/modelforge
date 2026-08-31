package auth

import (
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

const knownToken = "a-known-token"

func testAuth(t *testing.T, entries ...string) *Authenticator {
	t.Helper()
	a, err := New(entries)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return a
}

func entry(name, scopes, token string) string {
	return name + ":" + scopes + ":" + Digest(token)
}

func request(token string) *http.Request {
	r := httptest.NewRequest("GET", "/v1/models", nil)
	if token != "" {
		r.Header.Set("Authorization", "Bearer "+token)
	}
	return r
}

func TestAuthenticateAcceptsAConfiguredToken(t *testing.T) {
	a := testAuth(t, entry("ci", "admin", knownToken))

	tok, err := a.Authenticate(request(knownToken))
	if err != nil {
		t.Fatalf("Authenticate: %v", err)
	}
	if tok.Name != "ci" {
		t.Errorf("token name = %q, want ci", tok.Name)
	}
}

func TestAuthenticateRejects(t *testing.T) {
	a := testAuth(t, entry("ci", "admin", knownToken))

	for _, tc := range []struct {
		name    string
		req     *http.Request
		wantErr error
	}{
		{"no header", request(""), ErrNoCredential},
		{"wrong token", request("not-the-token"), ErrBadCredential},
		{"empty bearer", func() *http.Request {
			r := httptest.NewRequest("GET", "/", nil)
			r.Header.Set("Authorization", "Bearer ")
			return r
		}(), ErrNoCredential},
		{"wrong scheme", func() *http.Request {
			r := httptest.NewRequest("GET", "/", nil)
			r.Header.Set("Authorization", "Basic "+knownToken)
			return r
		}(), ErrNoCredential},
		{"token with no scheme", func() *http.Request {
			r := httptest.NewRequest("GET", "/", nil)
			r.Header.Set("Authorization", knownToken)
			return r
		}(), ErrNoCredential},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := a.Authenticate(tc.req); !errors.Is(err, tc.wantErr) {
				t.Fatalf("Authenticate = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

// TestBearerSchemeIsCaseInsensitive matches RFC 7235; clients do send "bearer".
func TestBearerSchemeIsCaseInsensitive(t *testing.T) {
	a := testAuth(t, entry("ci", "admin", knownToken))

	for _, scheme := range []string{"Bearer", "bearer", "BEARER", "BeArEr"} {
		r := httptest.NewRequest("GET", "/", nil)
		r.Header.Set("Authorization", scheme+" "+knownToken)
		if _, err := a.Authenticate(r); err != nil {
			t.Errorf("scheme %q rejected: %v", scheme, err)
		}
	}
}

// TestAdminImpliesEverything is the hierarchy decision: an operator who can
// replace the model answering every request is not restrained by being unable
// to read which model that is.
func TestAdminImpliesEverything(t *testing.T) {
	admin := Token{Name: "a", Scopes: []Scope{ScopeAdmin}}
	for _, s := range AllScopes {
		if !admin.Allows(s) {
			t.Errorf("admin does not allow %q", s)
		}
	}

	read := Token{Name: "r", Scopes: []Scope{ScopeRead}}
	if !read.Allows(ScopeRead) {
		t.Error("read does not allow read")
	}
	// read must not imply predict or admin: the point of the split is that a
	// dashboard credential cannot change what serves traffic, and a serving
	// credential cannot enumerate the deployment.
	if read.Allows(ScopeAdmin) {
		t.Error("read implies admin, which defeats the split")
	}
	if read.Allows(ScopePredict) {
		t.Error("read implies predict, which defeats the split")
	}

	predict := Token{Name: "p", Scopes: []Scope{ScopePredict}}
	if predict.Allows(ScopeRead) || predict.Allows(ScopeAdmin) {
		t.Error("predict implies more than predict")
	}

	multi := Token{Name: "m", Scopes: []Scope{ScopeRead, ScopePredict}}
	if !multi.Allows(ScopeRead) || !multi.Allows(ScopePredict) {
		t.Error("a multi-scope token lost a scope")
	}
	if multi.Allows(ScopeAdmin) {
		t.Error("read+predict implies admin")
	}
}

func TestNewRejectsBadConfiguration(t *testing.T) {
	for _, tc := range []struct {
		name    string
		entries []string
		wantErr string
	}{
		{"nothing", nil, "no tokens configured"},
		{"only blanks", []string{"", "  "}, "no tokens configured"},
		{"too few fields", []string{"ci:admin"}, "must be name:scopes:sha256hex"},
		{"too many fields", []string{"ci:admin:aa:bb"}, "must be name:scopes:sha256hex"},
		{"empty name", []string{":admin:" + Digest("x")}, "empty name"},
		{"unknown scope", []string{"ci:superuser:" + Digest("x")}, "unknown scope"},
		{"short digest", []string{"ci:admin:abc"}, "sha256 hex digest"},
		{"non-hex digest", []string{"ci:admin:" + strings.Repeat("z", 64)}, "non-hex digest"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := New(tc.entries)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("New = %v, want an error containing %q", err, tc.wantErr)
			}
		})
	}
}

// TestDuplicateDigestIsRejected covers the revocation trap: two names sharing a
// secret means removing one leaves the other working, and the operator has no
// way to see that from the config.
func TestDuplicateDigestIsRejected(t *testing.T) {
	_, err := New([]string{
		entry("ci", "admin", knownToken),
		entry("deploy", "read", knownToken),
	})
	if err == nil || !strings.Contains(err.Error(), "same digest") {
		t.Fatalf("New = %v, want a duplicate-digest error", err)
	}
}

// TestMalformedEntryIsRedacted is a leak check. A malformed entry is exactly
// the case where somebody pasted the raw token where the digest belongs, and
// echoing it into an error would write the secret to wherever logs go.
func TestMalformedEntryIsRedacted(t *testing.T) {
	secret := "super-secret-token-value"

	// The classic mistake: name:token, with no scope, so the raw token is in
	// the entry.
	_, err := New([]string{"ci:" + secret})
	if err == nil {
		t.Fatal("New accepted a malformed entry")
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatalf("the error leaked the raw token: %s", err)
	}
	if !strings.Contains(err.Error(), "redacted") {
		t.Errorf("error should say it redacted something: %s", err)
	}

	// And an entry with no colon at all.
	_, err = New([]string{secret})
	if err == nil || strings.Contains(err.Error(), secret) {
		t.Fatalf("error leaked the raw token: %v", err)
	}
}

// TestScopeParsingAndCaseHandling covers the accepted spellings of the config.
func TestScopeParsingAndCaseHandling(t *testing.T) {
	a := testAuth(t, "dash: read + predict :"+strings.ToUpper(Digest(knownToken)))

	tok, err := a.Authenticate(request(knownToken))
	if err != nil {
		t.Fatalf("Authenticate: %v", err)
	}
	if !tok.Allows(ScopeRead) || !tok.Allows(ScopePredict) {
		t.Errorf("scopes = %v, want read and predict", tok.Scopes)
	}
	if tok.Allows(ScopeAdmin) {
		t.Error("read+predict granted admin")
	}
}

func TestGenerateTokenIsRandomAndUsable(t *testing.T) {
	seen := make(map[string]bool, 200)
	for range 200 {
		tok, err := GenerateToken()
		if err != nil {
			t.Fatalf("GenerateToken: %v", err)
		}
		if seen[tok] {
			t.Fatal("GenerateToken returned a duplicate")
		}
		seen[tok] = true

		// It has to survive a header, a shell and a YAML file unquoted.
		if strings.ContainsAny(tok, " \t\n\"'`$:+,") {
			t.Fatalf("token contains an awkward character: %q", tok)
		}
		if len(tok) < 40 {
			t.Fatalf("token is only %d characters; 32 random bytes should be longer", len(tok))
		}
	}
}

// TestDigestIsStableAndOneWay is the property the config format depends on.
func TestDigestIsStableAndOneWay(t *testing.T) {
	d1, d2 := Digest("abc"), Digest("abc")
	if d1 != d2 {
		t.Fatal("Digest is not deterministic")
	}
	if len(d1) != 64 {
		t.Fatalf("digest is %d characters, want 64", len(d1))
	}
	if strings.Contains(d1, "abc") {
		t.Fatal("the digest contains the input")
	}
	// A one-character change must scramble the whole digest. That is what
	// makes hash-then-lookup timing-safe: a near miss and a wild miss produce
	// unrelated keys, so there is no "how close was I" signal.
	if shared := commonPrefix(d1, Digest("abd")); shared > 4 {
		t.Errorf("digests of similar inputs share %d leading characters", shared)
	}
}

func commonPrefix(a, b string) int {
	n := 0
	for n < len(a) && n < len(b) && a[n] == b[n] {
		n++
	}
	return n
}

func TestDisabledAllowsEverything(t *testing.T) {
	a := Disabled()
	if !a.IsDisabled() {
		t.Error("IsDisabled = false on a disabled Authenticator")
	}

	tok, err := a.Authenticate(request(""))
	if err != nil {
		t.Fatalf("disabled Authenticate rejected a request: %v", err)
	}
	for _, s := range AllScopes {
		if !tok.Allows(s) {
			t.Errorf("disabled auth withheld scope %q", s)
		}
	}
}

func TestTokensListsNamesAndScopesOnly(t *testing.T) {
	a := testAuth(t,
		entry("zeta", "read", "token-z"),
		entry("alpha", "admin", "token-a"),
	)

	got := a.Tokens()
	if len(got) != 2 {
		t.Fatalf("Tokens() returned %d, want 2", len(got))
	}
	// Sorted, so startup logging is stable rather than map-ordered.
	if got[0].Name != "alpha" || got[1].Name != "zeta" {
		t.Errorf("Tokens() = %v, want sorted by name", got)
	}
	// The struct must carry no secret to leak in the first place.
	for _, tok := range got {
		if strings.Contains(tok.String(), "token-") {
			t.Errorf("Token.String() leaked a secret: %s", tok)
		}
	}
	if s := (Token{Name: "n", Scopes: []Scope{ScopeRead, ScopePredict}}).String(); s != "n[read+predict]" {
		t.Errorf("Token.String() = %q", s)
	}
}

// --- middleware ---

func testMiddleware(a *Authenticator) *Middleware {
	return NewMiddleware(a, slog.New(slog.NewTextHandler(io.Discard, nil)))
}

func serve(t *testing.T, m *Middleware, scope Scope, req *http.Request) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	m.Require(scope, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})).ServeHTTP(rec, req)
	return rec
}

// TestUnauthenticatedIs401AndScopeFailureIs403 is the distinction that matters
// operationally: 401 means "you have not proved who you are", and retrying with
// a credential fixes it; 403 means "you have, and the answer is still no", so
// retrying is pointless and somebody has to grant the scope. Collapsing them
// sends operators hunting a permissions problem when the real one is an unset
// environment variable.
func TestUnauthenticatedIs401AndScopeFailureIs403(t *testing.T) {
	a := testAuth(t, entry("dash", "read", knownToken))
	m := testMiddleware(a)

	rec := serve(t, m, ScopeRead, request(""))
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("no credential = %d, want 401", rec.Code)
	}
	// RFC 7235 requires the challenge on a 401, and it is what tells a client
	// this is authentication rather than a generic refusal.
	if got := rec.Header().Get("WWW-Authenticate"); !strings.Contains(got, "Bearer") {
		t.Errorf("WWW-Authenticate = %q, want a Bearer challenge", got)
	}

	rec = serve(t, m, ScopeRead, request("wrong"))
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("bad credential = %d, want 401", rec.Code)
	}

	rec = serve(t, m, ScopeAdmin, request(knownToken))
	if rec.Code != http.StatusForbidden {
		t.Errorf("valid token, missing scope = %d, want 403", rec.Code)
	}
	if got := rec.Header().Get("WWW-Authenticate"); got != "" {
		t.Errorf("403 sent an authentication challenge (%q); retrying with a credential is not the fix", got)
	}
	if !strings.Contains(rec.Body.String(), "admin") {
		t.Errorf("403 body should name the scope required: %s", rec.Body.String())
	}

	rec = serve(t, m, ScopeRead, request(knownToken))
	if rec.Code != http.StatusOK {
		t.Errorf("valid token with the right scope = %d, want 200", rec.Code)
	}
}

// TestErrorBodiesNeverEchoTheCredential is a leak check on the response path.
func TestErrorBodiesNeverEchoTheCredential(t *testing.T) {
	a := testAuth(t, entry("dash", "read", knownToken))
	m := testMiddleware(a)

	const attempted = "an-attacker-supplied-token"
	rec := serve(t, m, ScopeRead, request(attempted))
	if strings.Contains(rec.Body.String(), attempted) {
		t.Fatalf("the 401 body echoed the presented token: %s", rec.Body.String())
	}

	rec = serve(t, m, ScopeAdmin, request(knownToken))
	if strings.Contains(rec.Body.String(), knownToken) {
		t.Fatalf("the 403 body echoed the valid token: %s", rec.Body.String())
	}
}

// TestTokenReachesTheHandlerContext is what makes audit logging possible in the
// handlers underneath.
func TestTokenReachesTheHandlerContext(t *testing.T) {
	a := testAuth(t, entry("ci", "admin", knownToken))
	m := testMiddleware(a)

	var seen Token
	var ok bool
	rec := httptest.NewRecorder()
	m.Require(ScopeAdmin, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen, ok = FromContext(r.Context())
	})).ServeHTTP(rec, request(knownToken))

	if !ok {
		t.Fatal("no token on the handler's context")
	}
	if seen.Name != "ci" {
		t.Errorf("context token = %q, want ci", seen.Name)
	}

	// And nothing is on the context when the request never got that far.
	if _, present := FromContext(httptest.NewRequest("GET", "/", nil).Context()); present {
		t.Error("FromContext found a token on a bare request")
	}
}

func TestDisabledMiddlewareLetsEverythingThrough(t *testing.T) {
	m := testMiddleware(Disabled())

	for _, scope := range AllScopes {
		if rec := serve(t, m, scope, request("")); rec.Code != http.StatusOK {
			t.Errorf("disabled auth blocked scope %q with %d", scope, rec.Code)
		}
	}
}

func TestNewMiddlewareToleratesANilLogger(t *testing.T) {
	m := NewMiddleware(Disabled(), nil)
	if rec := serve(t, m, ScopeRead, request("")); rec.Code != http.StatusOK {
		t.Errorf("nil logger broke the middleware: %d", rec.Code)
	}
}

// TestOnlyControlPlaneChangesAreAudited is a regression test for a bug the
// smoke run caught: auditing on the HTTP method alone logged a line for every
// prediction, because scoring is a POST. At serving volume that is millions of
// entries a day burying the handful that record a real change.
func TestOnlyControlPlaneChangesAreAudited(t *testing.T) {
	a := testAuth(t,
		entry("ci", "admin", knownToken),
		entry("edge", "predict", "predict-token"),
	)

	var buf strings.Builder
	m := NewMiddleware(a, slog.New(slog.NewTextHandler(&buf, nil)))

	post := func(path, token string, scope Scope) {
		r := httptest.NewRequest("POST", path, nil)
		r.Header.Set("Authorization", "Bearer "+token)
		m.Require(scope, http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})).
			ServeHTTP(httptest.NewRecorder(), r)
	}

	// Scoring is a POST and must not be audited.
	for range 5 {
		post("/v1/models/m/predict", "predict-token", ScopePredict)
	}
	if strings.Contains(buf.String(), "authorised change") {
		t.Fatalf("a prediction was written to the audit log:\n%s", buf.String())
	}

	// A control-plane write must be.
	post("/v1/models", knownToken, ScopeAdmin)
	if !strings.Contains(buf.String(), "authorised change") {
		t.Fatalf("a control-plane change was not audited:\n%s", buf.String())
	}
	// And it must name the credential, which is the whole point.
	if !strings.Contains(buf.String(), "ci") {
		t.Errorf("the audit line does not name the token:\n%s", buf.String())
	}
	// It must never contain the secret itself.
	if strings.Contains(buf.String(), knownToken) {
		t.Fatalf("the audit log leaked the token:\n%s", buf.String())
	}
}

// TestReadsAreNotAudited keeps the audit log to things that changed something.
func TestReadsAreNotAudited(t *testing.T) {
	a := testAuth(t, entry("ci", "admin", knownToken))
	var buf strings.Builder
	m := NewMiddleware(a, slog.New(slog.NewTextHandler(&buf, nil)))

	serve(t, m, ScopeRead, request(knownToken))
	if strings.Contains(buf.String(), "authorised change") {
		t.Errorf("a GET was audited as a change:\n%s", buf.String())
	}
}
