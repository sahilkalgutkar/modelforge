package app

import (
	"context"
	"io"
	"net/http"
	"net/http/cookiejar"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sahilkalgutkar/modelforge/internal/auth"
)

// browserServer starts a server with browser sessions enabled, plus a client
// with a cookie jar that follows redirects — a browser, near enough.
func browserServer(t *testing.T, p *fakeIDP) (*httptest.Server, *http.Client) {
	t.Helper()

	cfg := oidcConfig(t, p, nil)
	resetRegistry(t, cfg.DatabaseURL)
	cfg.OIDCClientID = "modelforge-web"
	cfg.InsecureCookies = true // the test server is plain HTTP

	// The external URL has to be the server's own address, and the server has
	// to exist before it is known. Built in two steps.
	var srv *httptest.Server
	holder := &deferredHandler{}
	srv = httptest.NewServer(holder)
	t.Cleanup(srv.Close)
	cfg.ExternalURL = srv.URL

	a, err := New(context.Background(), cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(a.Close)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := a.WaitForIdentityProvider(ctx); err != nil {
		t.Fatalf("identity provider never became ready: %v", err)
	}
	holder.h = a.Handler()

	jar, err := cookiejar.New(nil)
	if err != nil {
		t.Fatal(err)
	}
	return srv, &http.Client{Jar: jar, Timeout: 15 * time.Second}
}

type deferredHandler struct{ h http.Handler }

func (d *deferredHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if d.h == nil {
		http.Error(w, "not ready", http.StatusServiceUnavailable)
		return
	}
	d.h.ServeHTTP(w, r)
}

func cookieNamed(jar http.CookieJar, base, name string) *http.Cookie {
	u, _ := url.Parse(base)
	for _, c := range jar.Cookies(u) {
		if c.Name == name {
			return c
		}
	}
	return nil
}

// TestBrowserSessionEndToEnd is the feature: visit /login, authenticate, and
// come back holding a session that works.
func TestBrowserSessionEndToEnd(t *testing.T) {
	p := newFakeIDP(t)
	srv, client := browserServer(t, p)

	resp, err := client.Get(srv.URL + "/login")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("the login flow ended at %d", resp.StatusCode)
	}

	if cookieNamed(client.Jar, srv.URL, auth.SessionCookie) == nil {
		t.Fatal("no session cookie was set")
	}
	csrf := cookieNamed(client.Jar, srv.URL, auth.CSRFCookie)
	if csrf == nil {
		t.Fatal("no CSRF cookie was set")
	}

	// The session authenticates a read with no Authorization header at all.
	read, err := client.Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	read.Body.Close()
	if read.StatusCode != http.StatusOK {
		t.Fatalf("a session cookie on /v1/models = %d, want 200", read.StatusCode)
	}

	// A write needs the CSRF token echoed back.
	body := strings.NewReader(`{"name":"made-in-a-browser"}`)
	req, _ := http.NewRequest("POST", srv.URL+"/v1/models", body)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(auth.CSRFHeader, csrf.Value)
	write, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	write.Body.Close()
	if write.StatusCode != http.StatusCreated {
		t.Fatalf("a session POST with the CSRF token = %d, want 201", write.StatusCode)
	}

	// whoami reports the person, not a shared credential.
	who, err := client.Get(srv.URL + "/v1/auth/whoami")
	if err != nil {
		t.Fatal(err)
	}
	defer who.Body.Close()
	out, _ := io.ReadAll(who.Body)
	if !strings.Contains(string(out), "sahil@example.com") {
		t.Errorf("whoami over a session = %s", out)
	}
}

// TestCrossSiteWriteIsRefused is the attack browser sessions invite: a page on
// another origin making this browser POST with its ambient cookie. It cannot
// read the CSRF cookie, so it cannot send the header.
func TestCrossSiteWriteIsRefused(t *testing.T) {
	p := newFakeIDP(t)
	srv, client := browserServer(t, p)

	resp, err := client.Get(srv.URL + "/login")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	body := strings.NewReader(`{"name":"forged"}`)
	req, _ := http.NewRequest("POST", srv.URL+"/v1/models", body)
	req.Header.Set("Content-Type", "application/json")
	// No CSRF header, exactly as a cross-site attacker would be stuck.
	forged, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	forged.Body.Close()
	if forged.StatusCode != http.StatusForbidden {
		t.Fatalf("a cookie POST with no CSRF token = %d, want 403", forged.StatusCode)
	}
}

// TestLogoutEndsTheSessionServerSide: the cookie is cleared and the session is
// destroyed, so replaying a captured cookie does not work.
func TestLogoutEndsTheSessionServerSide(t *testing.T) {
	p := newFakeIDP(t)
	srv, client := browserServer(t, p)

	resp, err := client.Get(srv.URL + "/login")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	captured := cookieNamed(client.Jar, srv.URL, auth.SessionCookie)
	if captured == nil {
		t.Fatal("no session cookie")
	}

	out, err := client.Get(srv.URL + "/logout")
	if err != nil {
		t.Fatal(err)
	}
	out.Body.Close()

	// Replay the cookie with a fresh client, as somebody who copied it would.
	replay := &http.Client{}
	req, _ := http.NewRequest("GET", srv.URL+"/v1/models", nil)
	req.AddCookie(&http.Cookie{Name: auth.SessionCookie, Value: captured.Value})
	got, err := replay.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	got.Body.Close()
	if got.StatusCode != http.StatusUnauthorized {
		t.Fatalf("a logged-out session cookie still worked: %d", got.StatusCode)
	}
}

// TestLoginRejectsAnOpenRedirect. A login endpoint is the most valuable place
// to have one: the victim authenticates for real at their real provider and is
// then bounced somewhere the attacker controls.
func TestLoginRejectsAnOpenRedirect(t *testing.T) {
	p := newFakeIDP(t)
	// Each case uses its own jar, so a session from one does not carry over.
	srv, _ := browserServer(t, p)

	for _, next := range []string{
		"https://evil.example/steal",
		"//evil.example/steal", // protocol-relative: a browser reads it as another origin
		"http://evil.example",
		"javascript:alert(1)",
	} {
		jar, _ := cookiejar.New(nil)
		c := &http.Client{Jar: jar, Timeout: 15 * time.Second,
			CheckRedirect: func(r *http.Request, via []*http.Request) error {
				if strings.Contains(r.URL.Host, "evil.example") {
					t.Errorf("next=%q produced a redirect to %s", next, r.URL)
					return http.ErrUseLastResponse
				}
				return nil
			}}
		resp, err := c.Get(srv.URL + "/login?next=" + url.QueryEscape(next))
		if err != nil {
			t.Fatalf("next=%q: %v", next, err)
		}
		resp.Body.Close()

		if final := resp.Request.URL; strings.Contains(final.Host, "evil.example") {
			t.Errorf("next=%q landed on %s", next, final)
		}
	}
}

// TestLoginHonoursASafeNext keeps the feature useful: a relative path is fine.
func TestLoginHonoursASafeNext(t *testing.T) {
	p := newFakeIDP(t)
	srv, client := browserServer(t, p)

	resp, err := client.Get(srv.URL + "/login?next=" + url.QueryEscape("/v1/models"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if got := resp.Request.URL.Path; got != "/v1/models" {
		t.Errorf("landed on %q, want /v1/models", got)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("the redirect target returned %d", resp.StatusCode)
	}
}

// TestCallbackWithoutAStartedLoginIsRefused: a callback that did not begin here
// has no pending login to match, which is what stops a forged one.
func TestCallbackWithoutAStartedLoginIsRefused(t *testing.T) {
	p := newFakeIDP(t)
	srv, _ := browserServer(t, p)

	bare := &http.Client{}
	resp, err := bare.Get(srv.URL + "/auth/callback?code=whatever&state=whatever")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("a bare callback = %d, want 400", resp.StatusCode)
	}
	if !strings.Contains(string(body), "did not start here") {
		t.Errorf("unexpected page: %s", body)
	}
	// And no session was created.
	for _, c := range resp.Cookies() {
		if c.Name == auth.SessionCookie && c.Value != "" {
			t.Error("a forged callback created a session")
		}
	}
}

// TestBrowserPagesDoNotReflectProviderInput. These pages render at the moment a
// browser is holding an authorization code and a session cookie.
func TestBrowserPagesDoNotReflectProviderInput(t *testing.T) {
	p := newFakeIDP(t)
	srv, client := browserServer(t, p)

	resp, err := client.Get(srv.URL +
		"/auth/callback?error=" + url.QueryEscape("<script>alert(1)</script>"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if strings.Contains(string(body), "<script>") {
		t.Errorf("the callback page reflected provider input unescaped:\n%s", body)
	}
	// And it says it is not framable, since it handles credentials.
	if got := resp.Header.Get("X-Frame-Options"); got != "DENY" {
		t.Errorf("X-Frame-Options = %q, want DENY", got)
	}
}

// TestBrowserEndpointsAbsentWithoutConfiguration: half-enabled browser sessions
// would be worse than none.
func TestBrowserEndpointsAbsentWithoutConfiguration(t *testing.T) {
	cfg := testConfig(t, t.TempDir())
	resetRegistry(t, cfg.DatabaseURL)
	// No ExternalURL, so browser sessions stay off.

	a, err := New(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()

	ts := httptest.NewServer(a.Handler())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/login")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("/login without browser sessions = %d, want 404", resp.StatusCode)
	}

	// And a stray cookie is not a credential.
	req, _ := http.NewRequest("GET", ts.URL+"/v1/models", nil)
	req.AddCookie(&http.Cookie{Name: auth.SessionCookie, Value: "anything"})
	got, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	got.Body.Close()
	if got.StatusCode != http.StatusUnauthorized {
		t.Errorf("a cookie with sessions disabled = %d, want 401", got.StatusCode)
	}
}

// TestSessionCarriesTheSameAuthorisationAsABearerToken: the cookie must not be
// a second, weaker front door.
func TestSessionCarriesTheSameAuthorisationAsABearerToken(t *testing.T) {
	p := newFakeIDP(t)
	p.groups = []string{"ml-eng"} // read only
	srv, client := browserServer(t, p)

	resp, err := client.Get(srv.URL + "/login")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	csrf := cookieNamed(client.Jar, srv.URL, auth.CSRFCookie)
	if csrf == nil {
		t.Fatal("no CSRF cookie")
	}

	read, err := client.Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	read.Body.Close()
	if read.StatusCode != http.StatusOK {
		t.Errorf("a read session reading = %d, want 200", read.StatusCode)
	}

	req, _ := http.NewRequest("POST", srv.URL+"/v1/models", strings.NewReader(`{"name":"nope"}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(auth.CSRFHeader, csrf.Value)
	write, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	write.Body.Close()
	if write.StatusCode != http.StatusForbidden {
		t.Errorf("a read session writing = %d, want 403", write.StatusCode)
	}
}

// TestUserInNoGroupCannotStartASession: authenticated is not authorised, and
// that has to hold for the browser path too.
func TestUserInNoGroupCannotStartASession(t *testing.T) {
	p := newFakeIDP(t)
	p.groups = []string{"finance"}
	srv, client := browserServer(t, p)

	resp, err := client.Get(srv.URL + "/login")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("an unauthorised user's login = %d, want 403", resp.StatusCode)
	}
	if cookieNamed(client.Jar, srv.URL, auth.SessionCookie) != nil {
		t.Error("a session was created for a user with no granted scopes")
	}
}

func TestAuthConfigAdvertisesBrowserLogin(t *testing.T) {
	p := newFakeIDP(t)
	srv, _ := browserServer(t, p)

	resp, err := http.Get(srv.URL + "/v1/auth/config")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if !strings.Contains(string(body), `"browser_login":true`) {
		t.Errorf("auth config does not advertise browser login: %s", body)
	}
}

// TestDeployActionThroughABrowserSession is the flow as an operator performs
// it: sign in, review the change, apply it — with the CSRF token coming from a
// form field, because a page with no JavaScript cannot set a header.
func TestDeployActionThroughABrowserSession(t *testing.T) {
	p := newFakeIDP(t)
	srv, client := browserServer(t, p)

	resp, err := client.Get(srv.URL + "/login")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	csrf := cookieNamed(client.Jar, srv.URL, auth.CSRFCookie)
	if csrf == nil {
		t.Fatal("no CSRF cookie")
	}

	// Register a model with two versions and deploy the first, over the API,
	// using the same session.
	post := func(path string, body string) *http.Response {
		t.Helper()
		req, _ := http.NewRequest("POST", srv.URL+path, strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set(auth.CSRFHeader, csrf.Value)
		r, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		return r
	}
	post("/v1/models", `{"name":"ui-deploy"}`).Body.Close()

	for range 2 {
		f, err := os.Open(filepath.Join("..", "..", "testdata", "xgboost", "binary_logistic.model.json"))
		if err != nil {
			t.Fatal(err)
		}
		q := "?feature=f0&feature=f1&feature=f2&feature=f3&feature=f4&feature=f5"
		req, _ := http.NewRequest("POST", srv.URL+"/v1/models/ui-deploy/versions"+q, f)
		req.Header.Set("Content-Type", "application/octet-stream")
		req.Header.Set(auth.CSRFHeader, csrf.Value)
		r, err := client.Do(req)
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
		r.Body.Close()
	}

	req, _ := http.NewRequest("PUT", srv.URL+"/v1/models/ui-deploy/policy",
		strings.NewReader(`{"routes":[{"version":1,"weight":100}]}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(auth.CSRFHeader, csrf.Value)
	r, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()

	// The model page offers the actions, with a CSRF token embedded.
	pageResp, err := client.Get(srv.URL + "/models/ui-deploy")
	if err != nil {
		t.Fatal(err)
	}
	pageBody, _ := io.ReadAll(pageResp.Body)
	pageResp.Body.Close()
	if !strings.Contains(string(pageBody), "Change what serves traffic") {
		t.Fatalf("no actions on the model page:\n%s", string(pageBody)[:min(800, len(pageBody))])
	}
	if !strings.Contains(string(pageBody), csrf.Value) {
		t.Error("the forms do not carry a CSRF token, so submitting one would be refused")
	}

	// Plan, using a form field for the token exactly as the rendered form does.
	form := url.Values{"action": {"deploy"}, "version": {"2"}, auth.CSRFField: {csrf.Value}}
	planResp, err := client.PostForm(srv.URL+"/models/ui-deploy/plan", form)
	if err != nil {
		t.Fatal(err)
	}
	planBody, _ := io.ReadAll(planResp.Body)
	planResp.Body.Close()
	if planResp.StatusCode != http.StatusOK {
		t.Fatalf("plan = %d: %s", planResp.StatusCode, planBody)
	}
	if !strings.Contains(string(planBody), "Confirm this change") {
		t.Errorf("no confirmation page:\n%s", planBody)
	}

	// Apply.
	form.Set("expected", "v1=100%")
	applyResp, err := client.PostForm(srv.URL+"/models/ui-deploy/apply", form)
	if err != nil {
		t.Fatal(err)
	}
	applyResp.Body.Close()
	if applyResp.StatusCode != http.StatusOK {
		t.Fatalf("apply landed on %d", applyResp.StatusCode)
	}

	final, err := client.Get(srv.URL + "/v1/models/ui-deploy/policy")
	if err != nil {
		t.Fatal(err)
	}
	defer final.Body.Close()
	body, _ := io.ReadAll(final.Body)
	if !strings.Contains(string(body), `"version":2`) {
		t.Errorf("the deploy did not take: %s", body)
	}
}

// TestDeployActionWithoutCSRFIsRefused: the forms are the reason the CSRF check
// accepts a field, and that must not become a way around it.
func TestDeployActionWithoutCSRFIsRefused(t *testing.T) {
	p := newFakeIDP(t)
	srv, client := browserServer(t, p)

	resp, err := client.Get(srv.URL + "/login")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	// A cross-site page can make the browser post, but cannot read the cookie
	// to fill in the field.
	bad, err := client.PostForm(srv.URL+"/models/anything/plan",
		url.Values{"action": {"deploy"}, "version": {"1"}})
	if err != nil {
		t.Fatal(err)
	}
	bad.Body.Close()
	if bad.StatusCode != http.StatusForbidden {
		t.Fatalf("a form post with no CSRF token = %d, want 403", bad.StatusCode)
	}

	// And a wrong token is no better.
	wrong, err := client.PostForm(srv.URL+"/models/anything/plan",
		url.Values{"action": {"deploy"}, "version": {"1"}, auth.CSRFField: {"not-the-token"}})
	if err != nil {
		t.Fatal(err)
	}
	wrong.Body.Close()
	if wrong.StatusCode != http.StatusForbidden {
		t.Fatalf("a form post with the wrong CSRF token = %d, want 403", wrong.StatusCode)
	}
}
