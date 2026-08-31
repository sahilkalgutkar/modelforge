package api

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/sahilkalgutkar/modelforge/internal/auth"
	"github.com/sahilkalgutkar/modelforge/internal/drift"
	"github.com/sahilkalgutkar/modelforge/internal/routing"
)

// getPage fetches an HTML page with a credential.
func (h *harness) getPage(t *testing.T, path, token string) (*http.Response, string) {
	t.Helper()
	req, err := http.NewRequest("GET", h.http.URL+path, nil)
	if err != nil {
		t.Fatal(err)
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	// Do not follow the sign-in redirect; the redirect itself is the answer.
	client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	return resp, string(body)
}

func TestDashboardRendersModels(t *testing.T) {
	h := newHarness(t)

	h.do("POST", "/v1/models", createModelRequest{Name: "fraud-score", Description: "scores a transaction"}) //nolint:errcheck
	version := h.upload("fraud-score", "binary_logistic.model.json", featureNames(6))
	h.do("PUT", "/v1/models/fraud-score/policy", routing.Policy{ //nolint:errcheck
		Routes: []routing.Route{{Version: version, Weight: 90}, {Version: version, Weight: 0}}[:1],
	})

	resp, body := h.getPage(t, "/", "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("dashboard = %d: %s", resp.StatusCode, body)
	}
	for _, want := range []string{"fraud-score", "v1=100%", "modelforge"} {
		if !strings.Contains(body, want) {
			t.Errorf("dashboard does not mention %q:\n%s", want, truncate(body))
		}
	}
	// It links to the detail page rather than being a dead end.
	if !strings.Contains(body, `href="/models/fraud-score"`) {
		t.Errorf("no link to the model page:\n%s", truncate(body))
	}
}

func TestDashboardEmptyState(t *testing.T) {
	h := newHarness(t)

	_, body := h.getPage(t, "/", "")
	if !strings.Contains(body, "No models registered") {
		t.Errorf("no empty state:\n%s", truncate(body))
	}
	// And it says what to do next rather than just being blank.
	if !strings.Contains(body, "modelforgectl push") {
		t.Errorf("the empty state does not say how to add a model:\n%s", truncate(body))
	}
}

func TestModelPageShowsVersionsAndTraffic(t *testing.T) {
	h := newHarness(t)

	h.do("POST", "/v1/models", createModelRequest{Name: "canary-demo"}) //nolint:errcheck
	v1 := h.upload("canary-demo", "binary_logistic.model.json", featureNames(6))
	v2 := h.upload("canary-demo", "binary_logistic.model.json", featureNames(6))
	h.do("PUT", "/v1/models/canary-demo/policy", routing.Policy{ //nolint:errcheck
		Routes: []routing.Route{{Version: v1, Weight: 90}, {Version: v2, Weight: 10}},
	})
	for i := range 20 {
		h.do("POST", "/v1/models/canary-demo/predict", PredictRequest{ //nolint:errcheck
			Features: map[string]float64{"f0": 1}, Key: string(rune('a' + i)),
		})
	}

	resp, body := h.getPage(t, "/models/canary-demo", "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("model page = %d: %s", resp.StatusCode, body)
	}
	for _, want := range []string{"v1", "v2", "90%", "10%", "Versions"} {
		if !strings.Contains(body, want) {
			t.Errorf("model page does not show %q:\n%s", want, truncate(body))
		}
	}
}

func TestUnknownModelPage(t *testing.T) {
	h := newHarness(t)
	resp, body := h.getPage(t, "/models/not-a-model", "")
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("unknown model page = %d, want 404: %s", resp.StatusCode, truncate(body))
	}
}

// TestDashboardEscapesHostileContent is the XSS check. These pages are rendered
// for a browser holding a session cookie, so a script injected here would run
// with somebody's session.
func TestDashboardEscapesHostileContent(t *testing.T) {
	h := newAuthedHarness(t)

	const payload = `<script>alert(1)</script>`
	// A model description is free text and reaches the page unmodified.
	h.call(t, "POST", "/v1/models", adminToken, createModelRequest{
		Name: "hostile", Description: payload,
	})

	req, _ := http.NewRequest("GET", h.http.URL+"/models/hostile", nil)
	req.Header.Set("Authorization", "Bearer "+adminToken)
	resp, err := h.http.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if strings.Contains(string(body), payload) {
		t.Fatalf("the description was rendered unescaped:\n%s", truncate(string(body)))
	}
	if !strings.Contains(string(body), "&lt;script&gt;") {
		t.Errorf("expected the payload escaped, got:\n%s", truncate(string(body)))
	}
}

// TestDashboardSendsAStrictCSP: the escaping above is the defence, and this is
// the browser enforcing it independently.
func TestDashboardSendsAStrictCSP(t *testing.T) {
	h := newHarness(t)
	resp, _ := h.getPage(t, "/", "")

	csp := resp.Header.Get("Content-Security-Policy")
	if csp == "" {
		t.Fatal("no Content-Security-Policy on a dashboard page")
	}
	// default-src 'none' with no script-src means no script can run at all.
	if !strings.Contains(csp, "default-src 'none'") {
		t.Errorf("CSP does not deny by default: %q", csp)
	}
	if strings.Contains(csp, "script-src") && !strings.Contains(csp, "script-src 'none'") {
		t.Errorf("CSP permits scripts: %q", csp)
	}
	if !strings.Contains(csp, "frame-ancestors 'none'") {
		t.Errorf("CSP allows framing: %q", csp)
	}
	if got := resp.Header.Get("X-Frame-Options"); got != "DENY" {
		t.Errorf("X-Frame-Options = %q, want DENY", got)
	}
	// Model names, traffic volumes and drift readings should not sit in a
	// shared cache.
	if got := resp.Header.Get("Cache-Control"); !strings.Contains(got, "no-store") {
		t.Errorf("Cache-Control = %q, want no-store", got)
	}
}

// TestDashboardShipsNoJavaScript keeps the CSP honest: a script tag added later
// would be silently blocked by the policy, which is a confusing way to find out.
func TestDashboardShipsNoJavaScript(t *testing.T) {
	h := newHarness(t)
	h.do("POST", "/v1/models", createModelRequest{Name: "scripted"}) //nolint:errcheck

	for _, path := range []string{"/", "/models/scripted"} {
		_, body := h.getPage(t, path, "")
		lower := strings.ToLower(body)
		if strings.Contains(lower, "<script") {
			t.Errorf("%s contains a script tag", path)
		}
		if strings.Contains(lower, "onclick=") || strings.Contains(lower, "onload=") {
			t.Errorf("%s contains an inline event handler", path)
		}
	}
}

// TestDashboardRequiresTheReadScope: the UI is not a way around authorisation.
func TestDashboardRequiresTheReadScope(t *testing.T) {
	h := newAuthedHarness(t)

	for _, tc := range []struct {
		token string
		want  int
	}{
		{adminToken, http.StatusOK},
		{readToken, http.StatusOK},
		{predictToken, http.StatusForbidden},
		{"made-up", http.StatusUnauthorized},
	} {
		resp, body := h.getPage(t, "/", tc.token)
		if resp.StatusCode != tc.want {
			t.Errorf("dashboard with %s = %d, want %d: %s",
				nameOf(tc.token), resp.StatusCode, tc.want, truncate(body))
		}
	}
}

// TestDashboardRedirectsABrowserToSignIn is the HTML/API split: a browser gets
// somewhere to go, an API client gets a status it can act on.
func TestDashboardRedirectsABrowserToSignIn(t *testing.T) {
	h := newAuthedHarness(t)
	// Sessions have to be configured for a redirect to make sense; without a
	// sign-in flow there is nowhere to send anybody.
	h.server = NewServer(Deps{
		Registry: h.reg, Manager: h.manager, Router: h.router,
		Auth:     h.server.deps.Auth,
		Sessions: auth.NewSessionStore(auth.SessionConfig{}),
		Logins:   auth.NewLoginStore(nil),
	})
	h.http.Close()
	h.http = httptest.NewServer(h.server.Handler())
	t.Cleanup(h.http.Close)

	resp, _ := h.getPage(t, "/models/anything", "")
	if resp.StatusCode != http.StatusFound {
		t.Fatalf("an unauthenticated browser = %d, want a redirect", resp.StatusCode)
	}
	loc := resp.Header.Get("Location")
	if !strings.HasPrefix(loc, "/login?next=") {
		t.Fatalf("Location = %q, want a sign-in redirect", loc)
	}
	// It comes back to where you were trying to go.
	if !strings.Contains(loc, "models") {
		t.Errorf("the redirect does not preserve the destination: %q", loc)
	}

	// The JSON API is unaffected: it still gets a 401 it can act on.
	resp, _ = h.getPage(t, "/v1/models", "")
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("the API returned %d instead of 401", resp.StatusCode)
	}
}

// TestDashboardShowsDriftWhenMonitored, and says "not enough traffic" rather
// than implying health when there is none.
func TestDashboardShowsDriftState(t *testing.T) {
	h := newHarness(t)
	h.do("POST", "/v1/models", createModelRequest{Name: "monitored"}) //nolint:errcheck
	v := h.upload("monitored", "binary_logistic.model.json", featureNames(6))

	samples := map[string][]float64{}
	for _, f := range featureNames(6) {
		vals := make([]float64, 500)
		for i := range vals {
			vals[i] = float64(i%100) / 10
		}
		samples[f] = vals
	}
	h.do("PUT", fmt.Sprintf("/v1/models/monitored/versions/%d/baseline", v), BaselineRequest{ //nolint:errcheck
		Samples: samples, Bins: 10,
	})

	_, body := h.getPage(t, "/models/monitored", "")
	if !strings.Contains(body, "Drift") {
		t.Errorf("no drift panel for a monitored version:\n%s", truncate(body))
	}
	// The distinction that matters: silence is not health.
	if !strings.Contains(body, "Not enough traffic") {
		t.Errorf("the drift panel does not distinguish no-data from no-drift:\n%s", truncate(body))
	}
}

// TestIndexIsAListNotAControlSurface. The overview shows every model, so a
// stray action there would be a click away from changing the wrong one; the
// forms live on a page that is already about one model.
func TestIndexIsAListNotAControlSurface(t *testing.T) {
	h := newHarness(t)
	h.do("POST", "/v1/models", createModelRequest{Name: "listed-only"}) //nolint:errcheck
	h.upload("listed-only", "binary_logistic.model.json", featureNames(6))

	_, body := h.getPage(t, "/", "")
	if strings.Contains(strings.ToLower(body), "<form") {
		t.Errorf("the index contains a form:\n%s", truncate(body))
	}
}

// TestEveryActionFormCarriesACSRFToken. A form rendered without one would be
// refused on submit, which reads as a broken page rather than as the defence
// working.
func TestEveryActionFormCarriesACSRFToken(t *testing.T) {
	h := newHarness(t)
	h.do("POST", "/v1/models", createModelRequest{Name: "formful"}) //nolint:errcheck
	h.upload("formful", "binary_logistic.model.json", featureNames(6))

	_, body := h.getPage(t, "/models/formful", "")
	forms := strings.Count(strings.ToLower(body), "<form")
	if forms == 0 {
		t.Fatalf("no action forms rendered:\n%s", truncate(body))
	}
	tokens := strings.Count(body, `name="csrf_token"`)
	if tokens != forms {
		t.Errorf("%d forms but %d CSRF fields", forms, tokens)
	}
	// And every one posts, so none can be triggered by a link or a prefetch.
	if strings.Count(strings.ToLower(body), `method="post"`) != forms {
		t.Errorf("not every form uses POST:\n%s", truncate(body))
	}
}

// TestGetRoutesDoNotMutate: the action endpoints exist only under POST, so no
// amount of link-following changes what serves traffic.
func TestGetRoutesDoNotMutate(t *testing.T) {
	h := newHarness(t)
	h.deployable(t, "immutable-by-get")

	for _, path := range []string{
		"/models/immutable-by-get/plan?action=deploy&version=2",
		"/models/immutable-by-get/apply?action=deploy&version=2",
	} {
		resp, _ := h.getPage(t, path, "")
		if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusSeeOther {
			t.Errorf("GET %s was accepted with %d", path, resp.StatusCode)
		}
	}
	if p, _ := h.router.Policy("immutable-by-get"); policyLabel(p) != "v1=100%" {
		t.Errorf("a GET changed the policy to %q", policyLabel(p))
	}
}

// TestDashboardIdentityIsShown so somebody can tell whose view they are looking
// at, which matters on a shared machine.
func TestDashboardIdentityIsShown(t *testing.T) {
	h := newAuthedHarness(t)
	_, body := h.getPage(t, "/", readToken)

	if !strings.Contains(body, "dash") {
		t.Errorf("the page does not name the viewer:\n%s", truncate(body))
	}
	if !strings.Contains(body, "read") {
		t.Errorf("the page does not show the viewer's scopes:\n%s", truncate(body))
	}
	if !strings.Contains(body, "/logout") {
		t.Errorf("no way to sign out:\n%s", truncate(body))
	}
}

func TestTemplatesParseAtStartup(t *testing.T) {
	// The templates ship inside the binary, so a broken one is a build-time
	// mistake rather than something a request can cause. This asserts they are
	// actually embedded rather than silently missing.
	entries, err := templateFS.ReadDir("templates")
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) < 3 {
		t.Fatalf("only %d templates embedded", len(entries))
	}
	if indexTemplate == nil || modelTemplate == nil {
		t.Fatal("templates did not parse")
	}
}

func truncate(s string) string {
	if len(s) > 600 {
		return s[:600] + "..."
	}
	return s
}

// TestDashboardRendersRealDriftReadings covers the path where a version has
// enough traffic to report, which is the one that actually shows numbers.
func TestDashboardRendersRealDriftReadings(t *testing.T) {
	h := newHarness(t)
	h.do("POST", "/v1/models", createModelRequest{Name: "drifting"}) //nolint:errcheck
	v := h.upload("drifting", "binary_logistic.model.json", featureNames(6))
	h.do("PUT", "/v1/models/drifting/policy", routing.Policy{ //nolint:errcheck
		Routes: []routing.Route{{Version: v, Weight: 1}},
	})

	// A baseline centred near zero.
	samples := map[string][]float64{}
	for _, f := range featureNames(6) {
		vals := make([]float64, 1000)
		for i := range vals {
			vals[i] = float64(i%200)/100 - 1 // spread over [-1, 1)
		}
		samples[f] = vals
	}
	h.do("PUT", fmt.Sprintf("/v1/models/drifting/versions/%d/baseline", v), BaselineRequest{ //nolint:errcheck
		Samples: samples, Bins: 10,
	})

	// Traffic well outside the baseline, enough to clear the sample threshold.
	for i := range drift.MinSamples + 50 {
		h.do("POST", "/v1/models/drifting/predict", PredictRequest{ //nolint:errcheck
			Features: map[string]float64{
				"f0": 9, "f1": 9, "f2": 9, "f3": 9, "f4": 9, "f5": 9,
			},
			Key: fmt.Sprintf("k%d", i),
		})
	}

	_, body := h.getPage(t, "/models/drifting", "")
	if strings.Contains(body, "Not enough traffic") {
		t.Fatalf("drift still reports insufficient traffic:\n%s", truncate(body))
	}
	if !strings.Contains(body, "significant") {
		t.Errorf("a large shift did not render as significant:\n%s", truncate(body))
	}
	// The severity class drives the colour, so it has to reach the markup.
	if !strings.Contains(body, `class="bad"`) {
		t.Errorf("no severity class rendered:\n%s", truncate(body))
	}

	// And the index rolls that up to the model.
	_, index := h.getPage(t, "/", "")
	if !strings.Contains(index, "significant") {
		t.Errorf("the index does not surface the model's worst drift:\n%s", truncate(index))
	}
}

func TestSeverityMapping(t *testing.T) {
	for _, tc := range []struct {
		sev   drift.Severity
		class string
		rank  int
	}{
		{drift.SeveritySignificant, "bad", 3},
		{drift.SeverityModerate, "warn", 2},
		{drift.SeverityStable, "ok", 1},
	} {
		if got := severityClass(tc.sev); got != tc.class {
			t.Errorf("severityClass(%s) = %q, want %q", tc.sev, got, tc.class)
		}
		if got := severityRank(string(tc.sev)); got != tc.rank {
			t.Errorf("severityRank(%s) = %d, want %d", tc.sev, got, tc.rank)
		}
	}
	// Ranking has to order worst-first, since the index shows the worst.
	if severityRank("significant") <= severityRank("moderate") {
		t.Error("significant does not outrank moderate")
	}
	if severityRank("nonsense") != 0 {
		t.Error("an unknown severity should rank below everything")
	}
	if got := severityClass("nonsense"); got != "ok" {
		t.Errorf("severityClass of an unknown value = %q", got)
	}
}

func TestScopeSummary(t *testing.T) {
	if got := scopeSummary(auth.Token{Scopes: []auth.Scope{auth.ScopeRead, auth.ScopePredict}}); got != "read, predict" {
		t.Errorf("scopeSummary = %q", got)
	}
	// An identity with nothing granted should say so rather than render blank.
	if got := scopeSummary(auth.Token{}); got != "no scopes" {
		t.Errorf("scopeSummary of an empty token = %q", got)
	}
}

func TestVersionWeights(t *testing.T) {
	weights, total := versionWeights(routing.Policy{
		Routes: []routing.Route{{Version: 1, Weight: 90}, {Version: 2, Weight: 10}},
	})
	if total != 100 || weights[1] != 90 || weights[2] != 10 {
		t.Errorf("versionWeights = %v, %d", weights, total)
	}

	// An undeployed model has no routes, and the caller divides by the total.
	if _, total := versionWeights(routing.Policy{}); total != 0 {
		t.Errorf("an empty policy totalled %d, want 0", total)
	}
}

// TestSafeRedirect is the open-redirect guard the dashboard shares with the
// login flow.
func TestSafeRedirect(t *testing.T) {
	for _, bad := range []string{
		"", "https://evil.example", "//evil.example", "http://evil.example/x",
		"javascript:alert(1)", "evil.example", "/",
	} {
		if got := safeRedirect(bad); got != "" {
			t.Errorf("safeRedirect(%q) = %q, want empty", bad, got)
		}
	}
	for _, good := range []string{"/models/x", "/v1/models?a=b", "/models/x#frag"} {
		if got := safeRedirect(good); got == "" {
			t.Errorf("safeRedirect(%q) rejected a same-site path", good)
		}
	}
}
