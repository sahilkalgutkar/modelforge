package auth

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	jose "github.com/go-jose/go-jose/v4"
	"github.com/go-jose/go-jose/v4/jwt"
)

// idp is a minimal identity provider: a real RSA key, a real JWKS endpoint and
// real signed tokens.
//
// The tokens are genuinely signed and genuinely verified against keys fetched
// over HTTP, because the failures worth catching here are cryptographic. A stub
// that returned "valid: true" would pass every test in this file while leaving
// the server accepting forged tokens.
type idp struct {
	server *httptest.Server

	mu    sync.Mutex
	key   *rsa.PrivateKey
	keyID string
}

// rotate replaces the provider's signing key, as a real provider does
// periodically.
func (p *idp) rotate(t *testing.T) {
	t.Helper()
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.key, p.keyID = key, "test-key-2"
}

func newIDP(t *testing.T) *idp {
	t.Helper()

	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	p := &idp{key: key, keyID: "test-key-1"}

	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		//nolint:errcheck // test server
		json.NewEncoder(w).Encode(map[string]any{
			"issuer":                                p.server.URL,
			"jwks_uri":                              p.server.URL + "/keys",
			"id_token_signing_alg_values_supported": []string{"RS256"},
		})
	})
	mux.HandleFunc("/keys", func(w http.ResponseWriter, r *http.Request) {
		// Reads p.key rather than closing over the key created above, so that
		// rotating the provider's key is actually visible to a client
		// refetching the key set.
		p.mu.Lock()
		current, kid := p.key, p.keyID
		p.mu.Unlock()

		set := jose.JSONWebKeySet{Keys: []jose.JSONWebKey{{
			Key: current.Public(), KeyID: kid, Algorithm: string(jose.RS256), Use: "sig",
		}}}
		w.Header().Set("Content-Type", "application/json")
		//nolint:errcheck // test server
		json.NewEncoder(w).Encode(set)
	})

	p.server = httptest.NewServer(mux)
	t.Cleanup(p.server.Close)
	return p
}

// mint signs a token with the given claims.
func (p *idp) mint(t *testing.T, extra map[string]any) string {
	t.Helper()
	p.mu.Lock()
	key, kid := p.key, p.keyID
	p.mu.Unlock()
	return p.mintWithKey(t, key, kid, extra)
}

func (p *idp) mintWithKey(t *testing.T, key *rsa.PrivateKey, kid string, extra map[string]any) string {
	t.Helper()

	signer, err := jose.NewSigner(
		jose.SigningKey{Algorithm: jose.RS256, Key: key},
		(&jose.SignerOptions{}).WithType("JWT").WithHeader("kid", kid),
	)
	if err != nil {
		t.Fatal(err)
	}

	now := time.Now()
	claims := map[string]any{
		"iss": p.server.URL,
		"aud": "modelforge",
		"sub": "user-123",
		"iat": now.Unix(),
		"exp": now.Add(time.Hour).Unix(),
	}
	for k, v := range extra {
		if v == nil {
			delete(claims, k)
			continue
		}
		claims[k] = v
	}

	raw, err := jwt.Signed(signer).Claims(claims).Serialize()
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func testVerifier(t *testing.T, p *idp, mutate ...func(*OIDCConfig)) *OIDCVerifier {
	t.Helper()

	cfg := OIDCConfig{
		Issuer:      p.server.URL,
		Audience:    "modelforge",
		GroupsClaim: "groups",
		ScopeMap: map[string][]Scope{
			"platform-oncall": {ScopeAdmin},
			"ml-engineers":    {ScopeRead},
			"batch-scorers":   {ScopePredict},
		},
	}
	for _, m := range mutate {
		m(&cfg)
	}

	v, err := NewOIDCVerifier(context.Background(), cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := v.WaitReady(ctx); err != nil {
		t.Fatalf("provider never became ready: %v", err)
	}
	return v
}

func jwtRequest(token string) *http.Request {
	r := httptest.NewRequest("GET", "/v1/models", nil)
	r.Header.Set("Authorization", "Bearer "+token)
	return r
}

// TestOIDCTokenCarriesPerUserIdentity is the point of the whole feature: an
// audit line can name a person rather than a shared credential.
func TestOIDCTokenCarriesPerUserIdentity(t *testing.T) {
	p := newIDP(t)
	v := testVerifier(t, p)

	raw := p.mint(t, map[string]any{
		"sub":    "8f14e45f",
		"email":  "sahil@example.com",
		"groups": []string{"platform-oncall"},
	})

	tok, err := v.Verify(context.Background(), raw)
	if err != nil {
		t.Fatalf("Verify: %v", err)
	}
	if !tok.Human() {
		t.Error("a token from an identity provider does not report as human")
	}
	if tok.Subject != "8f14e45f" {
		t.Errorf("Subject = %q", tok.Subject)
	}
	// The readable name is preferred for logs, with the stable subject kept
	// alongside it.
	if tok.Name != "sahil@example.com" {
		t.Errorf("Name = %q, want the email", tok.Name)
	}
	if tok.Email != "sahil@example.com" {
		t.Errorf("Email = %q", tok.Email)
	}
	if tok.Issuer != p.server.URL {
		t.Errorf("Issuer = %q, want %q", tok.Issuer, p.server.URL)
	}
	if !tok.Allows(ScopeAdmin) {
		t.Errorf("scopes = %v, want admin from the platform-oncall group", tok.Scopes)
	}
	// The deadline comes from the token, so a short-lived login expires on its
	// own without anybody revoking anything.
	if tok.NotAfter.IsZero() {
		t.Error("NotAfter is unset; an OIDC token's exp should bound the session")
	}
}

// TestForgedSignatureIsRejected is the test that matters most. Everything else
// here is policy; this is whether the server can be trivially impersonated.
func TestForgedSignatureIsRejected(t *testing.T) {
	p := newIDP(t)
	v := testVerifier(t, p)

	// An attacker's own key, presented with the real provider's key id.
	attacker, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	forged := p.mintWithKey(t, attacker, "test-key-1", map[string]any{
		"sub": "attacker", "groups": []string{"platform-oncall"},
	})

	if _, err := v.Verify(context.Background(), forged); !errors.Is(err, ErrBadCredential) {
		t.Fatalf("a token signed with the wrong key was accepted: %v", err)
	}
}

// TestTamperedClaimsAreRejected covers modifying a real token: the signature no
// longer matches, which is the property that makes claims trustworthy at all.
func TestTamperedClaimsAreRejected(t *testing.T) {
	p := newIDP(t)
	v := testVerifier(t, p)

	raw := p.mint(t, map[string]any{"groups": []string{"ml-engineers"}})
	parts := strings.Split(raw, ".")
	if len(parts) != 3 {
		t.Fatalf("expected a three-part JWS, got %d parts", len(parts))
	}
	// Swap in a different payload, keeping the original signature.
	other := p.mint(t, map[string]any{"groups": []string{"platform-oncall"}})
	tampered := parts[0] + "." + strings.Split(other, ".")[1] + "." + parts[2]

	if _, err := v.Verify(context.Background(), tampered); !errors.Is(err, ErrBadCredential) {
		t.Fatalf("a token with swapped claims was accepted: %v", err)
	}
}

// TestUnsignedTokenIsRejected covers the alg:none family of attacks, which is
// the reason not to hand-roll JWT parsing.
func TestUnsignedTokenIsRejected(t *testing.T) {
	p := newIDP(t)
	v := testVerifier(t, p)

	// An unsecured JWS: {"alg":"none"}, real-looking claims, empty signature.
	header := `{"alg":"none","typ":"JWT"}`
	payload := fmt.Sprintf(`{"iss":%q,"aud":"modelforge","sub":"attacker","exp":%d,"groups":["platform-oncall"]}`,
		p.server.URL, time.Now().Add(time.Hour).Unix())
	unsigned := b64(header) + "." + b64(payload) + "."

	if _, err := v.Verify(context.Background(), unsigned); err == nil {
		t.Fatal("an unsigned alg:none token was accepted")
	}
}

func b64(s string) string {
	return base64.RawURLEncoding.EncodeToString([]byte(s))
}

// TestWrongAudienceIsRejected is the confused-deputy case: a real user with a
// real token their provider issued for a different service.
func TestWrongAudienceIsRejected(t *testing.T) {
	p := newIDP(t)
	v := testVerifier(t, p)

	raw := p.mint(t, map[string]any{
		"aud":    "the-expense-tool",
		"groups": []string{"platform-oncall"},
	})

	if _, err := v.Verify(context.Background(), raw); !errors.Is(err, ErrBadCredential) {
		t.Fatalf("a token minted for another service was accepted: %v", err)
	}
}

func TestExpiredOIDCTokenIsRejected(t *testing.T) {
	p := newIDP(t)
	v := testVerifier(t, p)

	raw := p.mint(t, map[string]any{
		"exp":    time.Now().Add(-time.Minute).Unix(),
		"groups": []string{"platform-oncall"},
	})

	if _, err := v.Verify(context.Background(), raw); !errors.Is(err, ErrBadCredential) {
		t.Fatalf("an expired token was accepted: %v", err)
	}
}

// TestAuthenticatedButNotAuthorised is the default-deny property. A valid
// employee login must not become access just because it verified.
func TestAuthenticatedButNotAuthorised(t *testing.T) {
	p := newIDP(t)
	v := testVerifier(t, p)

	for _, groups := range []any{
		[]string{"finance", "everyone"}, // real groups, none mapped
		[]string{},                      // in nothing
		nil,                             // no groups claim at all
	} {
		raw := p.mint(t, map[string]any{"email": "someone@example.com", "groups": groups})
		_, err := v.Verify(context.Background(), raw)
		if !errors.Is(err, ErrForbidden) {
			t.Errorf("groups %v gave %v, want ErrForbidden", groups, err)
		}
		// The message should name who was refused, so the fix is obvious.
		if err != nil && !strings.Contains(err.Error(), "someone@example.com") {
			t.Errorf("error does not name the user: %v", err)
		}
	}
}

func TestScopesAreTheUnionOfGroups(t *testing.T) {
	p := newIDP(t)
	v := testVerifier(t, p)

	raw := p.mint(t, map[string]any{"groups": []string{"ml-engineers", "batch-scorers", "unmapped"}})
	tok, err := v.Verify(context.Background(), raw)
	if err != nil {
		t.Fatal(err)
	}
	if !tok.Allows(ScopeRead) || !tok.Allows(ScopePredict) {
		t.Errorf("scopes = %v, want read and predict", tok.Scopes)
	}
	if tok.Allows(ScopeAdmin) {
		t.Errorf("scopes = %v; an unmapped group must not grant admin", tok.Scopes)
	}
}

// TestGroupsClaimShapes covers what providers actually emit.
func TestGroupsClaimShapes(t *testing.T) {
	p := newIDP(t)
	v := testVerifier(t, p)

	for _, tc := range []struct {
		name   string
		groups any
	}{
		{"list of strings", []string{"platform-oncall"}},
		{"single bare string", "platform-oncall"},
		{"space separated string", "finance platform-oncall"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := p.mint(t, map[string]any{"groups": tc.groups})
			tok, err := v.Verify(context.Background(), raw)
			if err != nil {
				t.Fatalf("Verify: %v", err)
			}
			if !tok.Allows(ScopeAdmin) {
				t.Errorf("scopes = %v, want admin", tok.Scopes)
			}
		})
	}

	// A claim that is neither is refused rather than silently treated as empty,
	// which would turn a provider misconfiguration into a quiet lockout.
	raw := p.mint(t, map[string]any{"groups": 42})
	if _, err := v.Verify(context.Background(), raw); err == nil {
		t.Error("a numeric groups claim was accepted")
	}
}

func TestConfigurableGroupsClaim(t *testing.T) {
	p := newIDP(t)
	v := testVerifier(t, p, func(c *OIDCConfig) { c.GroupsClaim = "roles" })

	raw := p.mint(t, map[string]any{"roles": []string{"platform-oncall"}})
	if _, err := v.Verify(context.Background(), raw); err != nil {
		t.Fatalf("a custom groups claim was not read: %v", err)
	}

	// And the default claim is now ignored, so a provider emitting both does
	// not accidentally grant through the wrong one.
	raw = p.mint(t, map[string]any{"groups": []string{"platform-oncall"}})
	if _, err := v.Verify(context.Background(), raw); !errors.Is(err, ErrForbidden) {
		t.Errorf("the unconfigured claim granted access: %v", err)
	}
}

func TestTokenWithoutASubjectIsRejected(t *testing.T) {
	p := newIDP(t)
	v := testVerifier(t, p)

	raw := p.mint(t, map[string]any{"sub": nil, "groups": []string{"platform-oncall"}})
	if _, err := v.Verify(context.Background(), raw); err == nil {
		t.Fatal("a token with no subject was accepted; there is no identity to record")
	}
}

func TestSubjectIsUsedWhenNoReadableNameExists(t *testing.T) {
	p := newIDP(t)
	v := testVerifier(t, p)

	raw := p.mint(t, map[string]any{"sub": "opaque-uuid", "groups": []string{"ml-engineers"}})
	tok, err := v.Verify(context.Background(), raw)
	if err != nil {
		t.Fatal(err)
	}
	if tok.Name != "opaque-uuid" {
		t.Errorf("Name = %q, want the subject as a fallback", tok.Name)
	}

	raw = p.mint(t, map[string]any{
		"sub": "opaque-uuid", "preferred_username": "sahil", "groups": []string{"ml-engineers"},
	})
	tok, err = v.Verify(context.Background(), raw)
	if err != nil {
		t.Fatal(err)
	}
	if tok.Name != "sahil" {
		t.Errorf("Name = %q, want preferred_username", tok.Name)
	}
}

// TestVerifyBeforeDiscoveryCompletes covers the window where the provider is
// not reachable yet. It must be a clean refusal, not a panic or a hang.
func TestVerifyBeforeDiscoveryCompletes(t *testing.T) {
	v, err := NewOIDCVerifier(context.Background(), OIDCConfig{
		// A port nothing is listening on, so discovery keeps failing.
		Issuer:   "http://127.0.0.1:1",
		Audience: "modelforge",
		ScopeMap: map[string][]Scope{"g": {ScopeRead}},
	}, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatal(err)
	}

	if v.Ready() {
		t.Error("Ready reported true against an unreachable provider")
	}
	if _, err := v.Verify(context.Background(), "a.b.c"); !errors.Is(err, ErrNoIdentityProvider) {
		t.Errorf("Verify before discovery = %v, want ErrNoIdentityProvider", err)
	}
}

func TestOIDCConfigValidation(t *testing.T) {
	valid := OIDCConfig{
		Issuer:   "https://idp.example.com",
		Audience: "modelforge",
		ScopeMap: map[string][]Scope{"g": {ScopeAdmin}},
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("a valid config was rejected: %v", err)
	}

	for _, tc := range []struct {
		name    string
		mutate  func(*OIDCConfig)
		wantErr string
	}{
		{"no issuer", func(c *OIDCConfig) { c.Issuer = "" }, "issuer URL is required"},
		{"issuer is not a url", func(c *OIDCConfig) { c.Issuer = "idp.example.com" }, "must be a URL"},
		// Plain HTTP to a remote host means the signing keys — the entire trust
		// anchor — are fetched over a channel anybody on the path can rewrite.
		{"remote http issuer", func(c *OIDCConfig) { c.Issuer = "http://idp.example.com" }, "unauthenticated channel"},
		{"no audience", func(c *OIDCConfig) { c.Audience = "" }, "audience is required"},
		{"no scope map", func(c *OIDCConfig) { c.ScopeMap = nil }, "scope map is required"},
		{"empty group", func(c *OIDCConfig) { c.ScopeMap = map[string][]Scope{"": {ScopeAdmin}} }, "empty group"},
		{"unknown scope", func(c *OIDCConfig) { c.ScopeMap = map[string][]Scope{"g": {"wizard"}} }, "unknown scope"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := valid
			tc.mutate(&cfg)
			err := cfg.Validate()
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("Validate = %v, want an error containing %q", err, tc.wantErr)
			}
		})
	}

	// Loopback over http is allowed, for a provider running locally.
	local := valid
	local.Issuer = "http://127.0.0.1:5556/dex"
	if err := local.Validate(); err != nil {
		t.Errorf("a loopback http issuer was rejected: %v", err)
	}
}

func TestParseScopeMap(t *testing.T) {
	got, err := ParseScopeMap([]string{
		"platform-oncall=admin",
		" ml-eng = read + predict ",
		"",
	})
	if err != nil {
		t.Fatalf("ParseScopeMap: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("parsed %d groups, want 2", len(got))
	}
	if len(got["ml-eng"]) != 2 {
		t.Errorf("ml-eng = %v, want two scopes", got["ml-eng"])
	}

	for _, bad := range []string{"no-equals-sign", "=admin", "g=wizard"} {
		if _, err := ParseScopeMap([]string{bad}); err == nil {
			t.Errorf("ParseScopeMap accepted %q", bad)
		}
	}
}

// TestStaticTokensAndOIDCCoexist is the deployment this is actually for: a
// service holds a token, a person logs in, and both work at once.
func TestStaticTokensAndOIDCCoexist(t *testing.T) {
	p := newIDP(t)
	v := testVerifier(t, p)

	a, err := New([]string{entry("ci", "predict", knownToken)}, WithOIDC(v))
	if err != nil {
		t.Fatal(err)
	}

	// The machine credential.
	tok, err := a.Authenticate(request(knownToken))
	if err != nil {
		t.Fatalf("the static token stopped working: %v", err)
	}
	if tok.Human() {
		t.Error("a static token reports as a human identity")
	}
	if !tok.Allows(ScopePredict) || tok.Allows(ScopeAdmin) {
		t.Errorf("static token scopes = %v", tok.Scopes)
	}

	// The person.
	raw := p.mint(t, map[string]any{"email": "sahil@example.com", "groups": []string{"platform-oncall"}})
	user, err := a.Authenticate(jwtRequest(raw))
	if err != nil {
		t.Fatalf("the OIDC login failed: %v", err)
	}
	if !user.Human() || user.Name != "sahil@example.com" {
		t.Errorf("user token = %+v", user)
	}
	if !user.Allows(ScopeAdmin) {
		t.Errorf("user scopes = %v, want admin", user.Scopes)
	}

	// And nonsense is still nonsense.
	if _, err := a.Authenticate(request("neither-of-those")); !errors.Is(err, ErrBadCredential) {
		t.Errorf("an unknown credential gave %v", err)
	}
}

// TestJWTWithNoProviderConfiguredIsRefusedClearly avoids the worst debugging
// experience: a valid JWT silently failing a digest lookup.
func TestJWTWithNoProviderConfiguredIsRefusedClearly(t *testing.T) {
	a := testAuth(t, entry("ci", "admin", knownToken))

	_, err := a.Authenticate(jwtRequest("header.payload.signature"))
	if !errors.Is(err, ErrBadCredential) {
		t.Fatalf("= %v, want ErrBadCredential", err)
	}
	if !strings.Contains(err.Error(), "no identity provider is configured") {
		t.Errorf("the error should say why a JWT cannot work here: %v", err)
	}
}

func TestLooksLikeJWT(t *testing.T) {
	// Static tokens are base64url with no dots, so the two shapes never
	// collide.
	for _, tok := range []string{"abc", "", "abcDEF-_123", "one.dot"} {
		if looksLikeJWT(tok) {
			t.Errorf("looksLikeJWT(%q) = true", tok)
		}
	}
	for _, tok := range []string{"a.b.c", "header.payload.sig"} {
		if !looksLikeJWT(tok) {
			t.Errorf("looksLikeJWT(%q) = false", tok)
		}
	}
	generated, err := GenerateToken()
	if err != nil {
		t.Fatal(err)
	}
	if looksLikeJWT(generated) {
		t.Errorf("a minted static token looks like a JWT: %q", generated)
	}
}

// TestKeyRotationAtTheProvider: go-oidc refetches the key set when it sees an
// unknown key id, so a provider rotating its signing key does not require
// anything here.
func TestKeyRotationAtTheProvider(t *testing.T) {
	p := newIDP(t)
	v := testVerifier(t, p)

	raw := p.mint(t, map[string]any{"groups": []string{"platform-oncall"}})
	if _, err := v.Verify(context.Background(), raw); err != nil {
		t.Fatalf("the original key does not work: %v", err)
	}

	// The provider rotates.
	p.rotate(t)

	rotated := p.mint(t, map[string]any{"groups": []string{"platform-oncall"}})
	if _, err := v.Verify(context.Background(), rotated); err != nil {
		t.Fatalf("a token signed with the provider's rotated key was rejected: %v", err)
	}
}

// TestVerifierIsRaceFree exercises concurrent verification, which is how it
// will actually be used.
func TestVerifierIsRaceFree(t *testing.T) {
	p := newIDP(t)
	v := testVerifier(t, p)

	raw := p.mint(t, map[string]any{"groups": []string{"platform-oncall"}})

	errs := make(chan error, 32)
	for range 32 {
		go func() {
			_, err := v.Verify(context.Background(), raw)
			errs <- err
		}()
	}
	for range 32 {
		if err := <-errs; err != nil {
			t.Errorf("concurrent Verify: %v", err)
		}
	}
}

// TestUnauthorisedIdentityDoesNotSpendRateLimitBudget: a real employee in the
// wrong group would otherwise be throttled for a permissions problem, which is
// a confusing outage and does nothing for security.
func TestUnauthorisedIdentityDoesNotSpendRateLimitBudget(t *testing.T) {
	p := newIDP(t)
	v := testVerifier(t, p)

	a, err := New([]string{entry("ci", "admin", knownToken)}, WithOIDC(v))
	if err != nil {
		t.Fatal(err)
	}
	l := NewLimiter(LimiterConfig{MaxFailures: 2, Window: time.Minute})
	m := WithLimiter(a, slog.New(slog.NewTextHandler(io.Discard, nil)), l, false)

	raw := p.mint(t, map[string]any{"email": "finance@example.com", "groups": []string{"finance"}})

	for i := range 20 {
		r := jwtRequest(raw)
		r.RemoteAddr = "9.9.9.9:1"
		rec := httptest.NewRecorder()
		m.Require(ScopeRead, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusOK)
		})).ServeHTTP(rec, r)

		if rec.Code != http.StatusForbidden {
			t.Fatalf("attempt %d = %d, want 403 every time", i, rec.Code)
		}
	}
	if l.Tracked() != 0 {
		t.Errorf("an authenticated-but-unauthorised user was rate limited (%d tracked)", l.Tracked())
	}
}

// TestForgedTokensDoSpendRateLimitBudget is the other half: a bad signature is
// somebody guessing, and should be throttled like any other bad credential.
func TestForgedTokensDoSpendRateLimitBudget(t *testing.T) {
	p := newIDP(t)
	v := testVerifier(t, p)

	attacker, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	forged := p.mintWithKey(t, attacker, "test-key-1", map[string]any{"groups": []string{"platform-oncall"}})

	a, err := New([]string{entry("ci", "admin", knownToken)}, WithOIDC(v))
	if err != nil {
		t.Fatal(err)
	}
	l := NewLimiter(LimiterConfig{MaxFailures: 2, Window: time.Minute})
	m := WithLimiter(a, slog.New(slog.NewTextHandler(io.Discard, nil)), l, false)

	call := func() int {
		r := jwtRequest(forged)
		r.RemoteAddr = "7.7.7.7:1"
		rec := httptest.NewRecorder()
		m.Require(ScopeRead, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {})).
			ServeHTTP(rec, r)
		return rec.Code
	}

	for range 2 {
		if got := call(); got != http.StatusUnauthorized {
			t.Fatalf("a forged token = %d, want 401", got)
		}
	}
	if got := call(); got != http.StatusTooManyRequests {
		t.Errorf("repeated forged tokens = %d, want 429", got)
	}
}
