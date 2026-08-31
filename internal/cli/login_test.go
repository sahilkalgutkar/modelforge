package cli

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	jose "github.com/go-jose/go-jose/v4"
	"github.com/go-jose/go-jose/v4/jwt"

	"github.com/sahilkalgutkar/modelforge/internal/app"
	"github.com/sahilkalgutkar/modelforge/internal/registry"
)

// loginIDP is an identity provider with a working authorization endpoint and
// token endpoint, so the login flow can be driven end to end.
//
// It enforces PKCE rather than ignoring it: the token endpoint recomputes the
// challenge from the verifier the client sends and refuses a mismatch. A stub
// that accepted any verifier would let the CLI ship without PKCE and every test
// here would still pass, which is exactly the bug worth catching.
type loginIDP struct {
	server *httptest.Server
	key    *rsa.PrivateKey

	mu        sync.Mutex
	codes     map[string]codeGrant
	groups    []string
	email     string
	issueNoID bool

	// refresh tokens the provider considers live. Rotation issues a new one
	// and retires the old, which is what OAuth 2.1 tells providers to do.
	refresh      map[string]bool
	rotate       bool
	refreshCount int
	revoked      []string
	noRevocation bool
	idTTL        time.Duration
}

type codeGrant struct {
	challenge string
	nonce     string
}

func newLoginIDP(t *testing.T) *loginIDP {
	t.Helper()

	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	p := &loginIDP{
		key:     key,
		codes:   map[string]codeGrant{},
		refresh: map[string]bool{},
		groups:  []string{"platform-oncall"},
		email:   "sahil@example.com",
		idTTL:   time.Hour,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, r *http.Request) {
		//nolint:errcheck // test server
		json.NewEncoder(w).Encode(map[string]any{
			"issuer":                                p.server.URL,
			"authorization_endpoint":                p.server.URL + "/authorize",
			"token_endpoint":                        p.server.URL + "/token",
			"jwks_uri":                              p.server.URL + "/keys",
			"revocation_endpoint":                   p.revocationEndpoint(),
			"id_token_signing_alg_values_supported": []string{"RS256"},
		})
	})
	mux.HandleFunc("/revoke", func(w http.ResponseWriter, r *http.Request) {
		//nolint:errcheck // test server
		r.ParseForm()
		tok := r.Form.Get("token")
		p.mu.Lock()
		p.revoked = append(p.revoked, tok)
		delete(p.refresh, tok)
		p.mu.Unlock()
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/keys", func(w http.ResponseWriter, r *http.Request) {
		//nolint:errcheck // test server
		json.NewEncoder(w).Encode(jose.JSONWebKeySet{Keys: []jose.JSONWebKey{{
			Key: key.Public(), KeyID: "k1", Algorithm: string(jose.RS256), Use: "sig",
		}}})
	})

	// The authorization endpoint. A browser would render a login page here;
	// this records the PKCE challenge and redirects straight back.
	mux.HandleFunc("/authorize", func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()
		code := "code-" + q.Get("state")[:8]

		p.mu.Lock()
		p.codes[code] = codeGrant{challenge: q.Get("code_challenge"), nonce: q.Get("nonce")}
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

		if r.Form.Get("grant_type") == "refresh_token" {
			p.serveRefresh(t, w, r)
			return
		}

		code := r.Form.Get("code")

		p.mu.Lock()
		grant, known := p.codes[code]
		delete(p.codes, code) // one use only, as a real provider does
		groups, email, noID := p.groups, p.email, p.issueNoID
		p.mu.Unlock()

		if !known {
			http.Error(w, `{"error":"invalid_grant"}`, http.StatusBadRequest)
			return
		}
		// PKCE, actually verified.
		sum := sha256.Sum256([]byte(r.Form.Get("code_verifier")))
		if base64.RawURLEncoding.EncodeToString(sum[:]) != grant.challenge {
			http.Error(w, `{"error":"invalid_grant","error_description":"pkce mismatch"}`, http.StatusBadRequest)
			return
		}

		body := map[string]any{"access_token": "opaque", "token_type": "Bearer", "expires_in": 3600}
		if !noID {
			body["id_token"] = p.mintID(t, email, groups, grant.nonce)
		}
		rt := "refresh-" + code
		p.mu.Lock()
		p.refresh[rt] = true
		p.mu.Unlock()
		body["refresh_token"] = rt

		w.Header().Set("Content-Type", "application/json")
		//nolint:errcheck // test server
		json.NewEncoder(w).Encode(body)
	})

	p.server = httptest.NewServer(mux)
	t.Cleanup(p.server.Close)
	return p
}

func (p *loginIDP) revocationEndpoint() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.noRevocation {
		return ""
	}
	return p.server.URL + "/revoke"
}

// serveRefresh handles a refresh_token grant, honouring rotation.
func (p *loginIDP) serveRefresh(t *testing.T, w http.ResponseWriter, r *http.Request) {
	t.Helper()
	presented := r.Form.Get("refresh_token")

	p.mu.Lock()
	live := p.refresh[presented]
	if live {
		p.refreshCount++
	}
	rotate, groups, email, ttl := p.rotate, p.groups, p.email, p.idTTL
	next := presented
	if live && rotate {
		delete(p.refresh, presented)
		next = presented + "-rotated"
		p.refresh[next] = true
	}
	p.mu.Unlock()

	if !live {
		w.Header().Set("Content-Type", "application/json")
		http.Error(w, `{"error":"invalid_grant","error_description":"refresh token is not valid"}`,
			http.StatusBadRequest)
		return
	}

	body := map[string]any{
		"access_token": "opaque", "token_type": "Bearer",
		"expires_in": int(ttl.Seconds()),
		"id_token":   p.mintIDWithTTL(t, email, groups, "", ttl),
	}
	if rotate {
		body["refresh_token"] = next
	}
	w.Header().Set("Content-Type", "application/json")
	//nolint:errcheck // test server
	json.NewEncoder(w).Encode(body)
}

func (p *loginIDP) mintID(t *testing.T, email string, groups []string, nonce string) string {
	t.Helper()
	return p.mintIDWithTTL(t, email, groups, nonce, time.Hour)
}

func (p *loginIDP) mintIDWithTTL(t *testing.T, email string, groups []string, nonce string, ttl time.Duration) string {
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
		"iat": now.Unix(), "exp": now.Add(ttl).Unix(),
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

// loginServer starts a modelforge configured for interactive login.
func loginServer(t *testing.T, p *loginIDP) string {
	t.Helper()

	dsn := os.Getenv("MODELFORGE_TEST_DATABASE_URL")
	if dsn == "" {
		t.Fatal("MODELFORGE_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	reg, err := registry.Open(ctx, dsn)
	if err != nil {
		t.Fatal(err)
	}
	if err := reg.Reset(ctx); err != nil {
		t.Fatal(err)
	}
	reg.Close()

	a, err := app.New(ctx, app.Config{
		DatabaseURL:  dsn,
		ArtifactDir:  t.TempDir(),
		Logger:       slog.New(slog.NewTextHandler(io.Discard, nil)),
		OIDCIssuer:   p.server.URL,
		OIDCAudience: "modelforge",
		OIDCClientID: "modelforge-cli",
		OIDCScopeMap: []string{"platform-oncall=admin", "ml-eng=read"},
	})
	if err != nil {
		t.Fatalf("start server: %v", err)
	}
	t.Cleanup(a.Close)

	waitCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	if err := a.WaitForIdentityProvider(waitCtx); err != nil {
		t.Fatalf("identity provider never became ready: %v", err)
	}

	ts := httptest.NewServer(a.Handler())
	t.Cleanup(ts.Close)

	// Each test gets its own credential file, so a login never touches the
	// developer's real one.
	t.Setenv("MODELFORGE_CREDENTIAL_FILE", filepath.Join(t.TempDir(), "credential"))
	t.Setenv("MODELFORGE_TOKEN", "")
	return ts.URL
}

// browser stands in for a person completing the login: it simply follows the
// URL the CLI would have opened.
func browser(t *testing.T) func(string) error {
	t.Helper()
	return func(authURL string) error {
		go func() {
			resp, err := http.Get(authURL)
			if err == nil {
				resp.Body.Close()
			}
		}()
		return nil
	}
}

// TestLoginEndToEnd is the flow as a person experiences it: run the command,
// authenticate in a browser, and end up holding a working credential.
func TestLoginEndToEnd(t *testing.T) {
	p := newLoginIDP(t)
	addr := loginServer(t, p)

	var out strings.Builder
	c := NewClient(addr, &out)
	if err := c.Login(context.Background(), browser(t)); err != nil {
		t.Fatalf("Login: %v", err)
	}

	if !strings.Contains(out.String(), "signed in as sahil@example.com") {
		t.Errorf("login did not report who signed in:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "admin") {
		t.Errorf("login did not report the granted scopes:\n%s", out.String())
	}

	// The credential is stored and a fresh client picks it up without any
	// environment variable.
	stored := LoadCredential()
	if !stored.Valid() {
		t.Fatal("no credential was stored")
	}
	if out2, code := run(t, addr, "whoami"); code != 0 ||
		!strings.Contains(out2, "sahil@example.com") {
		t.Fatalf("whoami after login: %d %s", code, out2)
	}

	// And it actually authenticates against the API.
	if out3, code := run(t, addr, "models"); code != 0 {
		t.Fatalf("a stored login could not call the API: %d %s", code, out3)
	}
}

// TestStoredCredentialIsPrivate covers the file permissions. A credential left
// world-readable in a home directory is a credential shared with every other
// account on the machine.
func TestStoredCredentialIsPrivate(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nested", "credential")
	t.Setenv("MODELFORGE_CREDENTIAL_FILE", path)

	if err := SaveCredential(Credential{IDToken: "a-token"}); err != nil {
		t.Fatal(err)
	}

	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if perm := fi.Mode().Perm(); perm != 0o600 {
		t.Errorf("credential file mode = %o, want 600", perm)
	}
	di, err := os.Stat(filepath.Dir(path))
	if err != nil {
		t.Fatal(err)
	}
	if perm := di.Mode().Perm(); perm != 0o700 {
		t.Errorf("credential directory mode = %o, want 700", perm)
	}

	// A pre-existing loose directory is tightened rather than trusted, since
	// MkdirAll leaves an existing mode alone.
	loose := filepath.Join(dir, "loose")
	if err := os.MkdirAll(loose, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("MODELFORGE_CREDENTIAL_FILE", filepath.Join(loose, "credential"))
	if err := SaveCredential(Credential{IDToken: "another"}); err != nil {
		t.Fatal(err)
	}
	di, _ = os.Stat(loose)
	if perm := di.Mode().Perm(); perm != 0o700 {
		t.Errorf("a pre-existing directory was left at %o, want it tightened to 700", perm)
	}
}

// TestPKCEIsActuallySent is the security property of this flow. Without a
// verifier the provider refuses, which proves the CLI is sending one rather
// than relying on the provider not to check.
func TestPKCEIsActuallySent(t *testing.T) {
	p := newLoginIDP(t)
	addr := loginServer(t, p)

	var seenChallenge string
	var out strings.Builder
	c := NewClient(addr, &out)

	err := c.Login(context.Background(), func(authURL string) error {
		u, perr := url.Parse(authURL)
		if perr != nil {
			return perr
		}
		seenChallenge = u.Query().Get("code_challenge")
		if got := u.Query().Get("code_challenge_method"); got != "S256" {
			t.Errorf("code_challenge_method = %q, want S256", got)
		}
		// The plain method is not acceptable: it puts the verifier in the
		// authorization request, where anything that can read the redirect can
		// also replay the code.
		go func() {
			resp, gerr := http.Get(authURL)
			if gerr == nil {
				resp.Body.Close()
			}
		}()
		return nil
	})
	if err != nil {
		t.Fatalf("Login: %v", err)
	}
	if seenChallenge == "" {
		t.Fatal("no code_challenge was sent; the flow is not using PKCE")
	}
}

// TestCallbackRejectsAForgedState is the CSRF defence. Without it, anybody who
// can make this browser load a URL can complete a login of their choosing, and
// the victim ends up holding the attacker's session.
func TestCallbackRejectsAForgedState(t *testing.T) {
	p := newLoginIDP(t)
	addr := loginServer(t, p)

	var out strings.Builder
	c := NewClient(addr, &out)
	c.HTTP = &http.Client{Timeout: 30 * time.Second}

	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()

	err := c.Login(ctx, func(authURL string) error {
		u, perr := url.Parse(authURL)
		if perr != nil {
			return perr
		}
		// Complete the callback with a state the CLI never issued.
		redirect := u.Query().Get("redirect_uri") + "?code=stolen&state=not-the-right-state"
		go func() {
			resp, gerr := http.Get(redirect)
			if gerr == nil {
				resp.Body.Close()
			}
		}()
		return nil
	})
	if err == nil {
		t.Fatal("a callback with a forged state completed the login")
	}
	if !strings.Contains(err.Error(), "state") {
		t.Errorf("error should name the state mismatch: %v", err)
	}
	if LoadCredential().Valid() {
		t.Error("a failed login stored a credential anyway")
	}
}

// TestProviderErrorIsReported covers the user declining consent, or the client
// being misconfigured at the provider.
func TestProviderErrorIsReported(t *testing.T) {
	p := newLoginIDP(t)
	addr := loginServer(t, p)

	var out strings.Builder
	c := NewClient(addr, &out)
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()

	err := c.Login(ctx, func(authURL string) error {
		u, _ := url.Parse(authURL)
		redirect := u.Query().Get("redirect_uri") +
			"?error=access_denied&error_description=user+declined&state=" + u.Query().Get("state")
		go func() {
			resp, gerr := http.Get(redirect)
			if gerr == nil {
				resp.Body.Close()
			}
		}()
		return nil
	})
	if err == nil {
		t.Fatal("a provider error completed the login")
	}
	if !strings.Contains(err.Error(), "access_denied") {
		t.Errorf("error should carry the provider's reason: %v", err)
	}
}

// TestLoginWithoutAnIDTokenFails: an access token is for calling the
// provider's APIs, and is not what this server authenticates.
func TestLoginWithoutAnIDTokenFails(t *testing.T) {
	p := newLoginIDP(t)
	p.issueNoID = true
	addr := loginServer(t, p)

	var out strings.Builder
	c := NewClient(addr, &out)
	err := c.Login(context.Background(), browser(t))
	if err == nil {
		t.Fatal("a token response with no id_token was accepted")
	}
	if !strings.Contains(err.Error(), "openid scope") {
		t.Errorf("error should suggest the likely cause: %v", err)
	}
}

// TestLoginAgainstAServerWithoutOIDC explains itself rather than failing
// obscurely.
func TestLoginAgainstAServerWithoutOIDC(t *testing.T) {
	addr := serverURL(t) // static tokens only
	var out strings.Builder
	c := NewClient(addr, &out)

	err := c.Login(context.Background(), browser(t))
	if err == nil {
		t.Fatal("login succeeded against a server with no identity provider")
	}
	if !strings.Contains(err.Error(), "does not support interactive login") {
		t.Errorf("error should say the server cannot do this: %v", err)
	}
}

// TestUserInNoMappedGroupCannotLogIn: the refusal happens at the server, and
// the CLI should surface it rather than storing a useless credential.
func TestUserInNoMappedGroupCannotLogIn(t *testing.T) {
	p := newLoginIDP(t)
	p.groups = []string{"finance"}
	p.email = "finance@example.com"
	addr := loginServer(t, p)

	var out strings.Builder
	c := NewClient(addr, &out)
	if err := c.Login(context.Background(), browser(t)); err != nil {
		t.Fatalf("Login: %v", err)
	}
	// The token itself is valid — the provider issued it — so it is stored,
	// and the server is the thing that refuses. Saying so is more useful than
	// pretending the login failed.
	if !strings.Contains(out.String(), "could not confirm with the server") {
		t.Errorf("output should report that the server refused this identity:\n%s", out.String())
	}
}

func TestLogoutRemovesTheCredential(t *testing.T) {
	t.Setenv("MODELFORGE_CREDENTIAL_FILE", filepath.Join(t.TempDir(), "credential"))

	var out strings.Builder
	c := NewClient("http://localhost:1", &out)

	// Nothing stored yet.
	if err := c.Logout(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "no stored login") {
		t.Errorf("logout with nothing stored: %s", out.String())
	}

	if err := SaveCredential(Credential{IDToken: "a-token"}); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	if err := c.Logout(context.Background()); err != nil {
		t.Fatal(err)
	}
	if LoadCredential().Valid() {
		t.Error("the credential survived logout")
	}
}

func TestEnvironmentTokenOverridesAStoredLogin(t *testing.T) {
	t.Setenv("MODELFORGE_CREDENTIAL_FILE", filepath.Join(t.TempDir(), "credential"))
	if err := SaveCredential(Credential{IDToken: "stored-token"}); err != nil {
		t.Fatal(err)
	}

	t.Setenv("MODELFORGE_TOKEN", "")
	if got := credential(); got != "stored-token" {
		t.Errorf("credential() = %q, want the stored login", got)
	}

	// A script should be able to override whoever is signed in on the machine.
	t.Setenv("MODELFORGE_TOKEN", "env-token")
	if got := credential(); got != "env-token" {
		t.Errorf("credential() = %q, want the environment to win", got)
	}
}

func TestWhoAmIWithoutACredential(t *testing.T) {
	t.Setenv("MODELFORGE_CREDENTIAL_FILE", filepath.Join(t.TempDir(), "credential"))
	t.Setenv("MODELFORGE_TOKEN", "")

	var out strings.Builder
	c := NewClient("http://localhost:1", &out)
	if err := c.WhoAmI(); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "not signed in") {
		t.Errorf("whoami with no credential: %s", out.String())
	}
}

// TestCallbackPageDoesNotReflectProviderInput is an XSS check on a page that
// has just handled an authorization code.
func TestCallbackPageDoesNotReflectProviderInput(t *testing.T) {
	result := make(chan loginResult, 1)
	h := callbackHandler("the-state", result)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET",
		"/callback?error=nope&error_description=%3Cscript%3Ealert(1)%3C/script%3E&state=the-state", nil)
	h.ServeHTTP(rec, req)

	if strings.Contains(rec.Body.String(), "<script>") {
		t.Errorf("the callback page reflected provider input unescaped:\n%s", rec.Body.String())
	}
}

func TestBrowserCommand(t *testing.T) {
	const target = "https://idp.example.com/authorize?state=abc"

	for _, tc := range []struct {
		goos string
		want string
	}{
		{"darwin", "open"},
		{"linux", "xdg-open"},
		{"freebsd", "xdg-open"},
		{"windows", "rundll32"},
	} {
		name, args := browserCommand(tc.goos, target)
		if name != tc.want {
			t.Errorf("%s uses %q, want %q", tc.goos, name, tc.want)
		}
		// The URL must be its own argument. Interpolating it into a shell
		// string would make a URL containing metacharacters a command, and
		// this URL comes from a server response.
		if args[len(args)-1] != target {
			t.Errorf("%s: the URL is not passed as a distinct argument: %v", tc.goos, args)
		}
	}
}

func TestSaveCredentialRejectsAnUnusablePath(t *testing.T) {
	// A path whose parent is a regular file cannot be created.
	file := filepath.Join(t.TempDir(), "not-a-dir")
	if err := os.WriteFile(file, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("MODELFORGE_CREDENTIAL_FILE", filepath.Join(file, "sub", "credential"))

	if err := SaveCredential(Credential{IDToken: "token"}); err == nil {
		t.Fatal("SaveCredential succeeded under a regular file")
	}
}

func TestLoadCredentialWithNoFile(t *testing.T) {
	t.Setenv("MODELFORGE_CREDENTIAL_FILE", filepath.Join(t.TempDir(), "absent"))
	if got := LoadCredential(); got.Valid() {
		t.Errorf("LoadCredential with no file = %+v, want empty", got)
	}

	// A bare token is what older versions wrote. Reading it keeps somebody who
	// upgrades signed in rather than logging them out for a format change they
	// did not ask for.
	path := filepath.Join(t.TempDir(), "credential")
	t.Setenv("MODELFORGE_CREDENTIAL_FILE", path)
	if err := os.WriteFile(path, []byte("  a-token\n\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	got := LoadCredential()
	if got.IDToken != "a-token" {
		t.Errorf("LoadCredential = %+v, want the trimmed bare token", got)
	}
	if got.CanRefresh() {
		t.Error("a bare token was reported as refreshable")
	}
}

func TestScopeNamesForAnIdentityWithNone(t *testing.T) {
	if got := scopeNames(nil); len(got) != 1 || got[0] != "no scopes" {
		t.Errorf("scopeNames(nil) = %v", got)
	}
	if got := scopeNames([]string{"read"}); len(got) != 1 || got[0] != "read" {
		t.Errorf("scopeNames = %v", got)
	}
}

// TestAuthConfigIsUnauthenticated: a client cannot present a credential before
// it knows where to obtain one.
func TestAuthConfigIsUnauthenticated(t *testing.T) {
	p := newLoginIDP(t)
	addr := loginServer(t, p)

	resp, err := http.Get(addr + "/v1/auth/config")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("/v1/auth/config without a credential = %d, want 200", resp.StatusCode)
	}

	var meta authMeta
	if err := json.NewDecoder(resp.Body).Decode(&meta); err != nil {
		t.Fatal(err)
	}
	if !meta.Login || meta.ClientID != "modelforge-cli" || meta.Issuer != p.server.URL {
		t.Errorf("auth config = %+v", meta)
	}
}

// TestAuthConfigOnAServerWithoutLogin says so plainly rather than returning
// an empty document a client has to guess at.
func TestAuthConfigOnAServerWithoutLogin(t *testing.T) {
	addr := serverURL(t) // static tokens only

	resp, err := http.Get(addr + "/v1/auth/config")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var meta authMeta
	if err := json.NewDecoder(resp.Body).Decode(&meta); err != nil {
		t.Fatal(err)
	}
	if meta.Login {
		t.Error("a server with no identity provider advertised login")
	}
	if meta.Reason == "" {
		t.Error("no reason was given for why login is unavailable")
	}
}

// TestWhoAmIWorksForAServiceToken: the endpoint is about the caller, so any
// valid credential can ask.
func TestWhoAmIWorksForAServiceToken(t *testing.T) {
	addr := serverURL(t)

	out, code := run(t, addr, "whoami")
	if code != 0 {
		t.Fatalf("whoami with a static token: %d %s", code, out)
	}
	if !strings.Contains(out, "service") {
		t.Errorf("whoami should identify a static token as a service:\n%s", out)
	}
	if !strings.Contains(out, "admin") {
		t.Errorf("whoami should report scopes:\n%s", out)
	}
}

// TestNoBrowserStillCompletes covers the headless path: the URL is printed and
// the flow waits, rather than reporting a browser failure that did not happen.
func TestNoBrowserStillCompletes(t *testing.T) {
	p := newLoginIDP(t)
	addr := loginServer(t, p)

	var out strings.Builder
	c := NewClient(addr, &out)

	err := c.Login(context.Background(), func(authURL string) error {
		go func() {
			resp, gerr := http.Get(authURL)
			if gerr == nil {
				resp.Body.Close()
			}
		}()
		return errNoBrowserRequested
	})
	if err != nil {
		t.Fatalf("Login: %v", err)
	}
	if strings.Contains(out.String(), "could not open a browser") {
		t.Errorf("--no-browser was reported as a failure:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "waiting for you to complete") {
		t.Errorf("output should say it is waiting:\n%s", out.String())
	}
	// The URL is always printed, browser or not, because the browser may open
	// somewhere the person cannot see.
	if !strings.Contains(out.String(), "to sign in, visit:") {
		t.Errorf("the sign-in URL was not printed:\n%s", out.String())
	}
}

// TestBrowserFailureIsNotFatal: a machine with no browser is an ordinary place
// to run this, and the printed URL still works.
func TestBrowserFailureIsNotFatal(t *testing.T) {
	p := newLoginIDP(t)
	addr := loginServer(t, p)

	var out strings.Builder
	c := NewClient(addr, &out)

	err := c.Login(context.Background(), func(authURL string) error {
		go func() {
			resp, gerr := http.Get(authURL)
			if gerr == nil {
				resp.Body.Close()
			}
		}()
		return errors.New("exec: \"xdg-open\": executable file not found in $PATH")
	})
	if err != nil {
		t.Fatalf("a browser that could not launch failed the whole login: %v", err)
	}
	if !strings.Contains(out.String(), "could not open a browser") {
		t.Errorf("the browser failure was not mentioned:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "signed in as") {
		t.Errorf("the login did not complete:\n%s", out.String())
	}
}
