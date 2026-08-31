package app

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/rsa"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	jose "github.com/go-jose/go-jose/v4"
	"github.com/go-jose/go-jose/v4/jwt"
)

// fakeIDP is a real identity provider for the app tests: a real RSA key, a real
// JWKS endpoint, real signed tokens. The server discovers it over HTTP exactly
// as it would discover Okta.
type fakeIDP struct {
	server *httptest.Server
	key    *rsa.PrivateKey

	mu     sync.Mutex
	codes  map[string]string // code -> nonce
	groups []string
	email  string
}

func newFakeIDP(t *testing.T) *fakeIDP {
	t.Helper()

	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	p := &fakeIDP{
		key:    key,
		codes:  map[string]string{},
		groups: []string{"platform-oncall"},
		email:  "sahil@example.com",
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		//nolint:errcheck // test server
		json.NewEncoder(w).Encode(map[string]any{
			"issuer":                                p.server.URL,
			"authorization_endpoint":                p.server.URL + "/authorize",
			"token_endpoint":                        p.server.URL + "/token",
			"jwks_uri":                              p.server.URL + "/keys",
			"id_token_signing_alg_values_supported": []string{"RS256"},
		})
	})

	// The authorization endpoint. A real provider renders a login page here;
	// this records the nonce and redirects straight back, which is what makes
	// the browser flow drivable from a test client.
	mux.HandleFunc("/authorize", func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()
		code := "code-" + q.Get("state")[:8]

		p.mu.Lock()
		p.codes[code] = q.Get("nonce")
		p.mu.Unlock()

		redirect, _ := url.Parse(q.Get("redirect_uri"))
		rq := redirect.Query()
		rq.Set("code", code)
		rq.Set("state", q.Get("state"))
		redirect.RawQuery = rq.Encode()
		http.Redirect(w, r, redirect.String(), http.StatusFound)
	})

	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		//nolint:errcheck // test server
		r.ParseForm()

		p.mu.Lock()
		nonce, known := p.codes[r.Form.Get("code")]
		delete(p.codes, r.Form.Get("code")) // single use, as a real provider does
		groups, email := p.groups, p.email
		p.mu.Unlock()

		if !known {
			http.Error(w, `{"error":"invalid_grant"}`, http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		//nolint:errcheck // test server
		json.NewEncoder(w).Encode(map[string]any{
			"access_token": "opaque", "token_type": "Bearer", "expires_in": 3600,
			"id_token": p.mintWithNonce(t, email, groups, nonce),
		})
	})
	mux.HandleFunc("/keys", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		//nolint:errcheck // test server
		json.NewEncoder(w).Encode(jose.JSONWebKeySet{Keys: []jose.JSONWebKey{{
			Key: p.key.Public(), KeyID: "k1", Algorithm: string(jose.RS256), Use: "sig",
		}}})
	})

	p.server = httptest.NewServer(mux)
	t.Cleanup(p.server.Close)
	return p
}

func (p *fakeIDP) mint(t *testing.T, email string, groups []string) string {
	t.Helper()
	return p.mintWithNonce(t, email, groups, "")
}

func (p *fakeIDP) mintWithNonce(t *testing.T, email string, groups []string, nonce string) string {
	t.Helper()

	signer, err := jose.NewSigner(
		jose.SigningKey{Algorithm: jose.RS256, Key: p.key},
		(&jose.SignerOptions{}).WithType("JWT").WithHeader("kid", "k1"),
	)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	claims := map[string]any{
		"iss": p.server.URL, "aud": "modelforge", "sub": "sub-" + email,
		"email": email, "groups": groups,
		"iat": now.Unix(), "exp": now.Add(time.Hour).Unix(),
	}
	if nonce != "" {
		claims["nonce"] = nonce
	}
	raw, err := jwt.Signed(signer).Claims(claims).Serialize()
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// oidcConfig builds an app config with both a static token and per-user login.
func oidcConfig(t *testing.T, p *fakeIDP, logTo io.Writer) Config {
	t.Helper()
	cfg := testConfig(t, t.TempDir())
	cfg.OIDCIssuer = p.server.URL
	cfg.OIDCAudience = "modelforge"
	cfg.OIDCScopeMap = []string{"platform-oncall=admin", "ml-eng=read", "scorers=predict"}
	if logTo != nil {
		cfg.Logger = slog.New(slog.NewJSONHandler(logTo, nil))
	}
	return cfg
}

func startWithOIDC(t *testing.T, cfg Config) (*App, string) {
	t.Helper()
	a, err := New(context.Background(), cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(a.Close)

	// Discovery runs in the background; wait for it so the test is not racing
	// the provider.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := a.WaitForIdentityProvider(ctx); err != nil {
		t.Fatalf("identity provider never became ready: %v", err)
	}

	ts := httptest.NewServer(a.Handler())
	t.Cleanup(ts.Close)
	return a, ts.URL
}

// TestPerUserLoginEndToEnd is the feature through the whole stack: a person's
// JWT authenticates, and their group decides what they may do.
func TestPerUserLoginEndToEnd(t *testing.T) {
	p := newFakeIDP(t)
	cfg := oidcConfig(t, p, nil)
	resetRegistry(t, cfg.DatabaseURL)
	_, url := startWithOIDC(t, cfg)

	admin := p.mint(t, "sahil@example.com", []string{"platform-oncall"})
	reader := p.mint(t, "analyst@example.com", []string{"ml-eng"})
	outsider := p.mint(t, "finance@example.com", []string{"finance"})

	// An admin user can change what serves traffic.
	if got := probe(t, url, admin); got != http.StatusOK {
		t.Errorf("an admin user reading models = %d, want 200", got)
	}
	body, _ := json.Marshal(map[string]string{"name": "owned-by-a-person"})
	req, _ := http.NewRequest("POST", url+"/v1/models", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+admin)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		t.Errorf("an admin user creating a model = %d, want 201", resp.StatusCode)
	}

	// A reader can read but not write.
	if got := probe(t, url, reader); got != http.StatusOK {
		t.Errorf("a read user = %d, want 200", got)
	}
	req, _ = http.NewRequest("POST", url+"/v1/models", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+reader)
	req.Header.Set("Content-Type", "application/json")
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusForbidden {
		t.Errorf("a read user creating a model = %d, want 403", resp.StatusCode)
	}

	// Somebody in no mapped group is authenticated and still refused.
	if got := probe(t, url, outsider); got != http.StatusForbidden {
		t.Errorf("a user in no mapped group = %d, want 403", got)
	}
}

// TestStaticTokensStillWorkAlongsideOIDC is the coexistence property: a service
// holding a token is unaffected by adding per-user login.
func TestStaticTokensStillWorkAlongsideOIDC(t *testing.T) {
	p := newFakeIDP(t)
	cfg := oidcConfig(t, p, nil)
	resetRegistry(t, cfg.DatabaseURL)
	_, url := startWithOIDC(t, cfg)

	if got := probe(t, url, testToken); got != http.StatusOK {
		t.Errorf("the static token = %d after enabling OIDC, want 200", got)
	}
	user := p.mint(t, "sahil@example.com", []string{"platform-oncall"})
	if got := probe(t, url, user); got != http.StatusOK {
		t.Errorf("the user token = %d, want 200", got)
	}
}

// TestAuditLogNamesThePerson is what per-user identity is for. The point is not
// that a request authenticated; it is that six months later somebody can answer
// who changed the model.
func TestAuditLogNamesThePerson(t *testing.T) {
	var logs bytes.Buffer
	p := newFakeIDP(t)
	cfg := oidcConfig(t, p, &logs)
	resetRegistry(t, cfg.DatabaseURL)
	_, url := startWithOIDC(t, cfg)

	admin := p.mint(t, "sahil@example.com", []string{"platform-oncall"})
	body, _ := json.Marshal(map[string]string{"name": "audited-model"})
	req, _ := http.NewRequest("POST", url+"/v1/models", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+admin)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	out := logs.String()
	if !strings.Contains(out, `"msg":"authorised change"`) {
		t.Fatalf("no audit line was written:\n%s", out)
	}
	if !strings.Contains(out, "sahil@example.com") {
		t.Errorf("the audit line does not name the person:\n%s", out)
	}
	// The stable subject is recorded alongside the readable name, because the
	// readable one changes and the subject does not.
	if !strings.Contains(out, `"subject":"sub-sahil@example.com"`) {
		t.Errorf("the audit line has no stable subject:\n%s", out)
	}
	if !strings.Contains(out, `"kind":"user"`) {
		t.Errorf("the audit line does not distinguish a person from a service:\n%s", out)
	}
	// And no part of the credential is ever written down.
	if strings.Contains(out, admin) {
		t.Error("the audit log contains the raw JWT")
	}
}

// TestForgedTokenIsRefusedByTheServer runs the signature check through the real
// HTTP stack, not just the verifier.
func TestForgedTokenIsRefusedByTheServer(t *testing.T) {
	p := newFakeIDP(t)
	cfg := oidcConfig(t, p, nil)
	resetRegistry(t, cfg.DatabaseURL)
	_, url := startWithOIDC(t, cfg)

	attacker := newFakeIDP(t)
	// Signed by the attacker's key, but claiming the real issuer and a group
	// that grants admin.
	forged := attacker.mint(t, "attacker@evil.example", []string{"platform-oncall"})

	if got := probe(t, url, forged); got != http.StatusUnauthorized {
		t.Fatalf("a forged token = %d, want 401", got)
	}
}

// TestOIDCOnlyDeploymentNeedsNoStaticTokens covers the configuration where
// every caller is a person.
func TestOIDCOnlyDeploymentNeedsNoStaticTokens(t *testing.T) {
	p := newFakeIDP(t)
	cfg := oidcConfig(t, p, nil)
	resetRegistry(t, cfg.DatabaseURL)
	cfg.Tokens = nil

	_, url := startWithOIDC(t, cfg)

	user := p.mint(t, "sahil@example.com", []string{"platform-oncall"})
	if got := probe(t, url, user); got != http.StatusOK {
		t.Errorf("a user login in an OIDC-only deployment = %d, want 200", got)
	}
	if got := probe(t, url, "some-static-token"); got != http.StatusUnauthorized {
		t.Errorf("an invented static token = %d, want 401", got)
	}
}

// TestStartupSurvivesAnUnreachableProvider is the availability decision: an
// identity provider being down degrades logins, it does not stop the server.
func TestStartupSurvivesAnUnreachableProvider(t *testing.T) {
	cfg := testConfig(t, t.TempDir())
	resetRegistry(t, cfg.DatabaseURL)
	cfg.OIDCIssuer = "http://127.0.0.1:1" // nothing is listening
	cfg.OIDCAudience = "modelforge"
	cfg.OIDCScopeMap = []string{"platform-oncall=admin"}

	a, err := New(context.Background(), cfg)
	if err != nil {
		t.Fatalf("the server refused to start because the identity provider was down: %v", err)
	}
	defer a.Close()

	ts := httptest.NewServer(a.Handler())
	defer ts.Close()

	// Static credentials keep working, which is the whole reason not to fail
	// startup.
	if got := probe(t, ts.URL, testToken); got != http.StatusOK {
		t.Errorf("the static token = %d while the provider was unreachable, want 200", got)
	}
	// And a JWT is cleanly refused rather than hanging.
	if got := probe(t, ts.URL, "a.b.c"); got != http.StatusUnauthorized {
		t.Errorf("a JWT with no provider available = %d, want 401", got)
	}
}

func TestOIDCConfigurationIsValidatedAtStartup(t *testing.T) {
	p := newFakeIDP(t)

	for _, tc := range []struct {
		name    string
		mutate  func(*Config)
		wantErr string
	}{
		{"no audience", func(c *Config) { c.OIDCAudience = "" }, "audience is required"},
		{"no scope map", func(c *Config) { c.OIDCScopeMap = nil }, "scope map is required"},
		{"bad scope map entry", func(c *Config) { c.OIDCScopeMap = []string{"nonsense"} }, "group=scope"},
		{"unknown scope", func(c *Config) { c.OIDCScopeMap = []string{"g=wizard"} }, "unknown scope"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := oidcConfig(t, p, nil)
			resetRegistry(t, cfg.DatabaseURL)
			tc.mutate(&cfg)

			a, err := New(context.Background(), cfg)
			if err == nil {
				a.Close()
				t.Fatal("New accepted an invalid OIDC configuration")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("error = %v, want it to contain %q", err, tc.wantErr)
			}
		})
	}
}

// TestUserAndServiceAreCountedSeparately checks the metric distinguishes them
// without labelling by identity.
func TestUserAndServiceAreCountedSeparately(t *testing.T) {
	p := newFakeIDP(t)
	cfg := oidcConfig(t, p, nil)
	resetRegistry(t, cfg.DatabaseURL)
	_, url := startWithOIDC(t, cfg)

	probe(t, url, testToken)
	probe(t, url, p.mint(t, "sahil@example.com", []string{"platform-oncall"}))

	resp, err := getAuthed(t, url+"/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	for _, want := range []string{`modelforge_authentications_total{kind="service"}`,
		`modelforge_authentications_total{kind="user"}`} {
		if !strings.Contains(string(body), want) {
			t.Errorf("metrics missing %s", want)
		}
	}
	// Identity must not become a label; that is unbounded cardinality.
	if strings.Contains(string(body), "sahil@example.com") {
		t.Error("a user identity leaked into the metrics as a label")
	}
}
