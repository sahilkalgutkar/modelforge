package app

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sahilkalgutkar/modelforge/internal/artifact"
	"github.com/sahilkalgutkar/modelforge/internal/auth"
	"github.com/sahilkalgutkar/modelforge/internal/registry"
	"github.com/sahilkalgutkar/modelforge/internal/routing"
)

func testConfig(t *testing.T, artifactDir string) Config {
	t.Helper()
	dsn := os.Getenv("MODELFORGE_TEST_DATABASE_URL")
	if dsn == "" {
		t.Fatal("MODELFORGE_TEST_DATABASE_URL is not set; these tests require a real Postgres " +
			"(see internal/registry/postgres_test.go for why they fail rather than skip)")
	}
	return Config{
		Addr:        "127.0.0.1:0",
		DatabaseURL: dsn,
		ArtifactDir: artifactDir,
		Logger:      slog.New(slog.NewTextHandler(io.Discard, nil)),
		// A real token rather than AuthDisabled, so the harness exercises the
		// path production uses. Tests that want the unauthenticated path ask
		// for it explicitly.
		Tokens: []string{"test:admin:" + auth.Digest(testToken)},
	}
}

// testToken is the credential the app tests present. It is a fixed literal
// rather than a generated one so a failing test is reproducible.
const testToken = "test-token-for-the-app-suite"

// authed adds the bearer credential to a request.
func authed(t *testing.T, req *http.Request) *http.Request {
	t.Helper()
	req.Header.Set("Authorization", "Bearer "+testToken)
	return req
}

// getAuthed is http.Get with the test credential.
func getAuthed(t *testing.T, url string) (*http.Response, error) {
	t.Helper()
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	return http.DefaultClient.Do(authed(t, req))
}

// postAuthed is http.Post with the test credential.
func postAuthed(t *testing.T, url, contentType string, body io.Reader) (*http.Response, error) {
	t.Helper()
	req, err := http.NewRequest("POST", url, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", contentType)
	return http.DefaultClient.Do(authed(t, req))
}

func resetRegistry(t *testing.T, dsn string) {
	t.Helper()
	reg, err := registry.Open(context.Background(), dsn)
	if err != nil {
		t.Fatalf("connect to test Postgres: %v", err)
	}
	defer reg.Close()
	if err := reg.Reset(context.Background()); err != nil {
		t.Fatal(err)
	}
}

// deploy brings up an App, registers a model from a fixture and installs a
// policy, all through the HTTP surface, then returns the app.
func deploy(t *testing.T, cfg Config, model string) *App {
	t.Helper()

	a, err := New(context.Background(), cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ts := httptest.NewServer(a.Handler())
	t.Cleanup(ts.Close)

	post := func(path string, body any) {
		t.Helper()
		b, _ := json.Marshal(body)
		resp, err := postAuthed(t, ts.URL+path, "application/json", bytes.NewReader(b))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode >= 300 {
			out, _ := io.ReadAll(resp.Body)
			t.Fatalf("POST %s: %d %s", path, resp.StatusCode, out)
		}
	}

	post("/v1/models", map[string]string{"name": model})

	f, err := os.Open(filepath.Join("..", "..", "testdata", "xgboost", "binary_logistic.model.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	q := "?feature=f0&feature=f1&feature=f2&feature=f3&feature=f4&feature=f5"
	resp, err := postAuthed(t, ts.URL+"/v1/models/"+model+"/versions"+q, "application/octet-stream", f)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		out, _ := io.ReadAll(resp.Body)
		t.Fatalf("upload: %d %s", resp.StatusCode, out)
	}

	b, _ := json.Marshal(routing.Policy{Routes: []routing.Route{{Version: 1, Weight: 1}}})
	req, _ := http.NewRequest("PUT", ts.URL+"/v1/models/"+model+"/policy", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	presp, err := http.DefaultClient.Do(authed(t, req))
	if err != nil {
		t.Fatal(err)
	}
	defer presp.Body.Close()
	if presp.StatusCode != http.StatusOK {
		out, _ := io.ReadAll(presp.Body)
		t.Fatalf("set policy: %d %s", presp.StatusCode, out)
	}
	return a
}

// TestPoliciesSurviveARestart is the reason RestorePolicies exists. A process
// that comes back with an empty router answers 404 for every model until
// someone re-applies its policy by hand, which is the worst thing to discover
// during an incident.
func TestPoliciesSurviveARestart(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(t, dir)
	resetRegistry(t, cfg.DatabaseURL)

	first := deploy(t, cfg, "restarted")
	first.Close()

	// A brand-new process against the same database and artifact directory.
	second, err := New(context.Background(), cfg)
	if err != nil {
		t.Fatalf("restart: %v", err)
	}
	defer second.Close()

	if _, ok := second.router.Policy("restarted"); !ok {
		t.Fatal("policy was not restored after restart")
	}
	if !second.manager.IsLoaded("restarted", 1) {
		t.Fatal("version was not loaded after restart")
	}

	// And it can actually serve without anyone touching it.
	ts := httptest.NewServer(second.Handler())
	defer ts.Close()

	resp, err := postAuthed(t, ts.URL+"/v1/models/restarted/predict", "application/json",
		strings.NewReader(`{"features":{"f0":1,"f1":2}}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("predict after restart: %d %s", resp.StatusCode, body)
	}

	// Readiness must also pass, or a rollout would never send it traffic.
	rresp, err := http.Get(ts.URL + "/readyz") // deliberately unauthenticated
	if err != nil {
		t.Fatal(err)
	}
	defer rresp.Body.Close()
	if rresp.StatusCode != http.StatusOK {
		t.Errorf("readyz after restart = %d, want 200", rresp.StatusCode)
	}
}

// TestOneBrokenDeploymentDoesNotBlockStartup covers the isolation decision:
// refusing to start because one model's artifact is missing would take every
// other model down with it.
func TestOneBrokenDeploymentDoesNotBlockStartup(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(t, dir)
	resetRegistry(t, cfg.DatabaseURL)

	healthy := deploy(t, cfg, "healthy")
	healthy.Close()

	// A model with a policy pointing at a version whose artifact was never
	// stored — the shape of a half-finished deploy or a lost blob.
	ctx := context.Background()
	reg, err := registry.Open(ctx, cfg.DatabaseURL)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := reg.CreateModel(ctx, "broken", ""); err != nil {
		t.Fatal(err)
	}
	if _, err := reg.CreateVersion(ctx, registry.NewVersion{
		Model: "broken", Runtime: registry.RuntimeXGBoost,
		Digest: artifact.Digest(strings.Repeat("ab", 32)), Features: []string{"f0"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := reg.SavePolicy(ctx, "broken", routing.Policy{
		Model: "broken", Routes: []routing.Route{{Version: 1, Weight: 1}},
	}); err != nil {
		t.Fatal(err)
	}
	reg.Close()

	a, err := New(ctx, cfg)
	if err != nil {
		t.Fatalf("startup failed because one deployment was broken: %v", err)
	}
	defer a.Close()

	if _, ok := a.router.Policy("healthy"); !ok {
		t.Error("the healthy model was not restored")
	}
	if _, ok := a.router.Policy("broken"); ok {
		t.Error("the broken model was installed despite its artifact being missing")
	}
}

func TestUnreadableStoredPolicyIsSkipped(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(t, dir)
	resetRegistry(t, cfg.DatabaseURL)

	healthy := deploy(t, cfg, "fine")
	healthy.Close()

	ctx := context.Background()
	reg, err := registry.Open(ctx, cfg.DatabaseURL)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := reg.CreateModel(ctx, "corrupt", ""); err != nil {
		t.Fatal(err)
	}
	// A policy document that is valid JSON but not a policy: no routes at all.
	if err := reg.SavePolicy(ctx, "corrupt", map[string]string{"nonsense": "yes"}); err != nil {
		t.Fatal(err)
	}
	reg.Close()

	a, err := New(ctx, cfg)
	if err != nil {
		t.Fatalf("startup failed on an unusable stored policy: %v", err)
	}
	defer a.Close()

	if _, ok := a.router.Policy("corrupt"); ok {
		t.Error("an invalid policy was installed")
	}
	if _, ok := a.router.Policy("fine"); !ok {
		t.Error("the valid policy was not restored alongside it")
	}
}

func TestNewFailsOnAnUnreachableDatabase(t *testing.T) {
	cfg := testConfig(t, t.TempDir())
	cfg.DatabaseURL = "postgres://nobody@127.0.0.1:1/none?sslmode=disable&connect_timeout=1"
	if _, err := New(context.Background(), cfg); err == nil {
		t.Fatal("New succeeded against an unreachable database")
	}
}

func TestNewFailsOnAnUnusableArtifactDir(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "not-a-dir")
	if err := os.WriteFile(file, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg := testConfig(t, filepath.Join(file, "sub"))
	if _, err := New(context.Background(), cfg); err == nil {
		t.Fatal("New succeeded with an artifact directory that cannot exist")
	}
}

// TestPublishDriftIsSafeWithoutMonitors checks the metrics timer does not
// depend on baselines being configured.
func TestPublishDriftIsSafeWithoutMonitors(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(t, dir)
	resetRegistry(t, cfg.DatabaseURL)

	a := deploy(t, cfg, "nodrift")
	defer a.Close()

	a.PublishDrift() // must not panic and must not report anything
}

func TestMetricsEndpointIsServed(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(t, dir)
	resetRegistry(t, cfg.DatabaseURL)

	a := deploy(t, cfg, "metered")
	defer a.Close()

	ts := httptest.NewServer(a.Handler())
	defer ts.Close()

	//nolint:errcheck // generating a metric to scrape
	postAuthed(t, ts.URL+"/v1/models/metered/predict", "application/json",
		strings.NewReader(`{"features":{"f0":1}}`))

	resp, err := getAuthed(t, ts.URL+"/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("/metrics returned %d", resp.StatusCode)
	}
	if !strings.Contains(string(body), "modelforge_predictions_total") {
		t.Errorf("/metrics does not expose prediction counters:\n%s", truncate(string(body)))
	}
}

func truncate(s string) string {
	if len(s) > 400 {
		return s[:400] + "..."
	}
	return s
}

// freePort asks the kernel for an unused port and gives it straight back.
// There is a race between releasing it and Run binding it, but it is the only
// way to test the real listener path — Run owns its own listener, so the port
// cannot be handed in already bound.
func freePort(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := l.Addr().String()
	l.Close()
	return addr
}

// TestRunServesThenShutsDownCleanly exercises the real listener and the
// shutdown ordering: the listener closes first, in-flight requests drain, and
// only then are the batchers released. Closing them first would fail requests
// that had already been accepted and were waiting on a batch.
func TestRunServesThenShutsDownCleanly(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(t, dir)
	resetRegistry(t, cfg.DatabaseURL)

	// Deploy a model with one process, then serve it with another.
	deploy(t, cfg, "served").Close()

	cfg.Addr = freePort(t)
	cfg.DriftInterval = 20 * time.Millisecond // exercise the drift ticker too

	a, err := New(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- a.Run(ctx) }()

	// Wait for the listener to come up.
	base := "http://" + cfg.Addr
	var up bool
	for range 100 {
		resp, err := http.Get(base + "/healthz")
		if err == nil {
			resp.Body.Close()
			up = true
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !up {
		cancel()
		<-done
		t.Fatal("server never started listening")
	}

	resp, err := postAuthed(t, base+"/v1/models/served/predict", "application/json",
		strings.NewReader(`{"features":{"f0":1,"f1":2}}`))
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("predict against the real listener: %d %s", resp.StatusCode, body)
	}

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("Run returned %v, want a clean shutdown", err)
		}
	case <-time.After(20 * time.Second):
		t.Fatal("Run did not return after its context was cancelled")
	}

	// The listener must actually be closed, not merely unreferenced.
	if _, err := http.Get(base + "/healthz"); err == nil {
		t.Error("the server is still accepting connections after shutdown")
	}
}

// TestRunFailsOnAnAddressAlreadyInUse checks the listen error is surfaced
// rather than swallowed, since a process that silently fails to bind looks
// healthy to everything except its callers.
func TestRunFailsOnAnAddressAlreadyInUse(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(t, dir)
	resetRegistry(t, cfg.DatabaseURL)

	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	cfg.Addr = l.Addr().String()

	a, err := New(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := a.Run(ctx); err == nil {
		t.Fatal("Run succeeded on an address already in use")
	}
}

// --- authentication configuration ---

// TestStartupFailsClosedWithoutTokens is the decision that matters most in this
// whole feature. Defaulting to open would mean a server that comes up, serves,
// and looks entirely healthy while its control plane is available to anybody
// who can reach the port — and no log line anybody actually reads beats simply
// not starting.
func TestStartupFailsClosedWithoutTokens(t *testing.T) {
	cfg := testConfig(t, t.TempDir())
	resetRegistry(t, cfg.DatabaseURL)
	cfg.Tokens = nil

	a, err := New(context.Background(), cfg)
	if err == nil {
		a.Close()
		t.Fatal("the server started with no tokens and no explicit opt-out")
	}
	// The error has to say what to do, or it just moves the confusion.
	for _, want := range []string{"modelforgectl token", "MODELFORGE_TOKENS", "-auth-disabled"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("startup error does not mention %q: %v", want, err)
		}
	}
}

// TestAuthCanBeDisabledDeliberately covers the escape hatch. It exists because
// local development and this suite genuinely do not want tokens, and a flag
// somebody had to type is very different from a default nobody chose.
func TestAuthCanBeDisabledDeliberately(t *testing.T) {
	cfg := testConfig(t, t.TempDir())
	resetRegistry(t, cfg.DatabaseURL)
	cfg.Tokens = nil
	cfg.AuthDisabled = true

	a, err := New(context.Background(), cfg)
	if err != nil {
		t.Fatalf("New with -auth-disabled: %v", err)
	}
	defer a.Close()

	if !a.auth.IsDisabled() {
		t.Fatal("auth is not disabled despite AuthDisabled")
	}

	ts := httptest.NewServer(a.Handler())
	defer ts.Close()
	resp, err := http.Get(ts.URL + "/v1/models") // no credential
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("disabled auth still refused a request: %d", resp.StatusCode)
	}
}

// TestTokensAndDisabledTogetherIsRejected refuses to guess. Deciding for the
// operator which half of a contradictory configuration they meant is exactly
// the wrong instinct to have about a security control.
func TestTokensAndDisabledTogetherIsRejected(t *testing.T) {
	cfg := testConfig(t, t.TempDir())
	resetRegistry(t, cfg.DatabaseURL)
	cfg.AuthDisabled = true // Tokens is already set by testConfig

	a, err := New(context.Background(), cfg)
	if err == nil {
		a.Close()
		t.Fatal("New accepted both tokens and -auth-disabled")
	}
	if !strings.Contains(err.Error(), "ambiguous") {
		t.Errorf("error should call the configuration ambiguous: %v", err)
	}
}

func TestStartupRejectsMalformedTokens(t *testing.T) {
	cfg := testConfig(t, t.TempDir())
	resetRegistry(t, cfg.DatabaseURL)
	cfg.Tokens = []string{"broken-entry-with-no-digest"}

	a, err := New(context.Background(), cfg)
	if err == nil {
		a.Close()
		t.Fatal("New accepted a malformed token entry")
	}
}

// TestServingRequiresACredential is the end-to-end version of the same claim,
// through a fully wired app rather than a hand-built server.
func TestServingRequiresACredential(t *testing.T) {
	cfg := testConfig(t, t.TempDir())
	resetRegistry(t, cfg.DatabaseURL)

	a := deploy(t, cfg, "guarded")
	defer a.Close()

	ts := httptest.NewServer(a.Handler())
	defer ts.Close()

	resp, err := http.Post(ts.URL+"/v1/models/guarded/predict", "application/json",
		strings.NewReader(`{"features":{"f0":1}}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("predict without a credential = %d, want 401", resp.StatusCode)
	}

	// And the same request with the credential succeeds, so the 401 above is
	// authentication rather than something else being broken.
	ok, err := postAuthed(t, ts.URL+"/v1/models/guarded/predict", "application/json",
		strings.NewReader(`{"features":{"f0":1}}`))
	if err != nil {
		t.Fatal(err)
	}
	defer ok.Body.Close()
	if ok.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(ok.Body)
		t.Fatalf("predict with a credential = %d: %s", ok.StatusCode, body)
	}
}

// TestMetricsRequiresACredentialEndToEnd checks the wiring in app.New actually
// wrapped the metrics handler, not just that the API can wrap one.
func TestMetricsRequiresACredentialEndToEnd(t *testing.T) {
	cfg := testConfig(t, t.TempDir())
	resetRegistry(t, cfg.DatabaseURL)

	a := deploy(t, cfg, "metered")
	defer a.Close()

	ts := httptest.NewServer(a.Handler())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("/metrics without a credential = %d, want 401", resp.StatusCode)
	}
}
