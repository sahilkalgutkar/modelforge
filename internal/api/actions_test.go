package api

import (
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"

	"github.com/sahilkalgutkar/modelforge/internal/auth"
	"github.com/sahilkalgutkar/modelforge/internal/routing"
)

// postForm submits an HTML form the way a browser would.
func (h *harness) postForm(t *testing.T, path, token string, form url.Values) (*http.Response, string) {
	t.Helper()
	req, err := http.NewRequest("POST", h.http.URL+path, strings.NewReader(form.Encode()))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
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

// deployable sets up a model with two versions serving v1.
func (h *harness) deployable(t *testing.T, name string) (int, int) {
	t.Helper()
	h.do("POST", "/v1/models", createModelRequest{Name: name}) //nolint:errcheck
	v1 := h.upload(name, "binary_logistic.model.json", featureNames(6))
	v2 := h.upload(name, "binary_logistic.model.json", featureNames(6))
	h.do("PUT", "/v1/models/"+name+"/policy", routing.Policy{ //nolint:errcheck
		Routes: []routing.Route{{Version: v1, Weight: 1}},
	})
	return v1, v2
}

// TestDeployFromTheDashboard is the feature: plan, see the change, apply it.
func TestDeployFromTheDashboard(t *testing.T) {
	h := newHarness(t)
	_, v2 := h.deployable(t, "shippable")

	resp, body := h.postForm(t, "/models/shippable/plan", "", url.Values{
		"action": {"deploy"}, "version": {itoa(v2)},
	})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("plan = %d: %s", resp.StatusCode, truncate(body))
	}
	// The confirmation has to say what is true now and what will be true.
	for _, want := range []string{"v1=100%", "v2=100%", "Confirm this change"} {
		if !strings.Contains(body, want) {
			t.Errorf("the confirmation does not show %q:\n%s", want, truncate(body))
		}
	}
	// And nothing has changed yet.
	if p, _ := h.router.Policy("shippable"); policyLabel(p) != "v1=100%" {
		t.Fatalf("planning changed the policy to %q", policyLabel(p))
	}

	resp, body = h.postForm(t, "/models/shippable/apply", "", url.Values{
		"action": {"deploy"}, "version": {itoa(v2)}, "expected": {"v1=100%"},
	})
	if resp.StatusCode != http.StatusSeeOther {
		t.Fatalf("apply = %d: %s", resp.StatusCode, truncate(body))
	}
	// Redirect after post, so a refresh does not re-apply.
	if loc := resp.Header.Get("Location"); loc != "/models/shippable" {
		t.Errorf("Location = %q", loc)
	}
	if p, _ := h.router.Policy("shippable"); policyLabel(p) != "v2=100%" {
		t.Fatalf("policy = %q, want v2=100%%", policyLabel(p))
	}
}

func TestCanaryFromTheDashboard(t *testing.T) {
	h := newHarness(t)
	v1, v2 := h.deployable(t, "canaried")

	_, body := h.postForm(t, "/models/canaried/plan", "", url.Values{
		"action": {"canary"}, "version": {itoa(v2)}, "stable": {itoa(v1)}, "percent": {"10"},
	})
	if !strings.Contains(body, "Send 10% of traffic to version 2") {
		t.Errorf("the plan does not describe the split:\n%s", truncate(body))
	}

	resp, _ := h.postForm(t, "/models/canaried/apply", "", url.Values{
		"action": {"canary"}, "version": {itoa(v2)}, "stable": {itoa(v1)},
		"percent": {"10"}, "expected": {"v1=100%"},
	})
	if resp.StatusCode != http.StatusSeeOther {
		t.Fatalf("apply = %d", resp.StatusCode)
	}
	if p, _ := h.router.Policy("canaried"); policyLabel(p) != "v1=90% v2=10%" {
		t.Fatalf("policy = %q", policyLabel(p))
	}
}

// TestConcurrentEditIsRefused is the lost-update case. Two people acting on the
// same model must not have the later click silently discard the earlier change.
func TestConcurrentEditIsRefused(t *testing.T) {
	h := newHarness(t)
	v1, v2 := h.deployable(t, "contended")

	// Somebody else moves the model after our confirmation was rendered.
	h.do("PUT", "/v1/models/contended/policy", routing.Policy{ //nolint:errcheck
		Routes: []routing.Route{{Version: v1, Weight: 50}, {Version: v2, Weight: 50}},
	})

	resp, body := h.postForm(t, "/models/contended/apply", "", url.Values{
		"action": {"deploy"}, "version": {itoa(v2)},
		"expected": {"v1=100%"}, // what we saw
	})
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("a stale apply = %d, want 400", resp.StatusCode)
	}
	if !strings.Contains(body, "changed while you were looking at it") {
		t.Errorf("the refusal does not explain itself:\n%s", truncate(body))
	}
	// The other person's change survives.
	if p, _ := h.router.Policy("contended"); policyLabel(p) != "v1=50% v2=50%" {
		t.Fatalf("the stale apply overwrote a concurrent change: %q", policyLabel(p))
	}
}

// TestDeployActionsNeedAdmin: the buttons are not a way around authorisation.
func TestDeployActionsNeedAdmin(t *testing.T) {
	h := newAuthedHarness(t)
	h.call(t, "POST", "/v1/models", adminToken, createModelRequest{Name: "guarded-actions"})
	v := h.uploadAs(t, "guarded-actions", "binary_logistic.model.json", featureNames(6), adminToken)

	for _, path := range []string{"/models/guarded-actions/plan", "/models/guarded-actions/apply"} {
		form := url.Values{"action": {"deploy"}, "version": {itoa(v)}}
		if resp, _ := h.postForm(t, path, readToken, form); resp.StatusCode != http.StatusForbidden {
			t.Errorf("%s with a read token = %d, want 403", path, resp.StatusCode)
		}
		if resp, _ := h.postForm(t, path, "", form); resp.StatusCode != http.StatusUnauthorized {
			t.Errorf("%s with no credential = %d, want 401", path, resp.StatusCode)
		}
	}
}

// TestActionFormsHiddenWithoutAdmin: showing buttons that always refuse is a
// worse experience than not showing them.
func TestActionFormsShownOnlyToAdmins(t *testing.T) {
	h := newAuthedHarness(t)
	h.call(t, "POST", "/v1/models", adminToken, createModelRequest{Name: "scoped-ui"})
	h.uploadAs(t, "scoped-ui", "binary_logistic.model.json", featureNames(6), adminToken)

	_, admin := h.getPage(t, "/models/scoped-ui", adminToken)
	if !strings.Contains(admin, "Change what serves traffic") {
		t.Errorf("an admin does not see the actions:\n%s", truncate(admin))
	}

	_, reader := h.getPage(t, "/models/scoped-ui", readToken)
	if strings.Contains(reader, "Change what serves traffic") {
		t.Errorf("a read-only viewer sees the actions:\n%s", truncate(reader))
	}
	if !strings.Contains(reader, "not change what it serves") {
		t.Errorf("a read-only viewer is not told why:\n%s", truncate(reader))
	}
}

// TestActionsRejectNonsense keeps a mistyped form from producing a policy
// nobody intended.
func TestActionsRejectNonsense(t *testing.T) {
	h := newHarness(t)
	v1, v2 := h.deployable(t, "validated")

	for _, tc := range []struct {
		name string
		form url.Values
		want string
	}{
		{"unknown action", url.Values{"action": {"destroy"}, "version": {"1"}}, "unknown action"},
		{"no version", url.Values{"action": {"deploy"}}, "no version"},
		{"non-numeric version", url.Values{"action": {"deploy"}, "version": {"latest"}}, "positive number"},
		{"canary at 0%", url.Values{"action": {"canary"}, "version": {itoa(v2)},
			"stable": {itoa(v1)}, "percent": {"0"}}, "between 1 and 99"},
		{"canary at 100%", url.Values{"action": {"canary"}, "version": {itoa(v2)},
			"stable": {itoa(v1)}, "percent": {"100"}}, "between 1 and 99"},
		{"canary against itself", url.Values{"action": {"canary"}, "version": {itoa(v1)},
			"stable": {itoa(v1)}, "percent": {"10"}}, "both"},
		{"unshadow with no shadow", url.Values{"action": {"unshadow"}}, "no shadow"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resp, body := h.postForm(t, "/models/validated/plan", "", tc.form)
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("= %d, want 400: %s", resp.StatusCode, truncate(body))
			}
			if !strings.Contains(body, tc.want) {
				t.Errorf("the refusal does not mention %q:\n%s", tc.want, truncate(body))
			}
		})
	}

	// Nothing was changed by any of it.
	if p, _ := h.router.Policy("validated"); policyLabel(p) != "v1=100%" {
		t.Errorf("a rejected action changed the policy to %q", policyLabel(p))
	}
}

// TestShadowActions cover the pair, including the rule that a serving version
// cannot also be the shadow.
func TestShadowActions(t *testing.T) {
	h := newHarness(t)
	v1, v2 := h.deployable(t, "shadowed-ui")

	resp, _ := h.postForm(t, "/models/shadowed-ui/apply", "", url.Values{
		"action": {"shadow"}, "version": {itoa(v2)}, "expected": {"v1=100%"},
	})
	if resp.StatusCode != http.StatusSeeOther {
		t.Fatalf("shadow apply = %d", resp.StatusCode)
	}
	p, _ := h.router.Policy("shadowed-ui")
	if p.Shadow == nil || *p.Shadow != v2 {
		t.Fatalf("shadow = %v, want v%d", p.Shadow, v2)
	}

	// Shadowing something already serving is refused rather than producing a
	// policy the router would reject.
	resp, body := h.postForm(t, "/models/shadowed-ui/plan", "", url.Values{
		"action": {"shadow"}, "version": {itoa(v1)},
	})
	if resp.StatusCode != http.StatusBadRequest || !strings.Contains(body, "already serving") {
		t.Errorf("shadowing a serving version = %d: %s", resp.StatusCode, truncate(body))
	}

	// And it can be turned off.
	resp, _ = h.postForm(t, "/models/shadowed-ui/apply", "", url.Values{
		"action": {"unshadow"}, "expected": {policyLabel(p)},
	})
	if resp.StatusCode != http.StatusSeeOther {
		t.Fatalf("unshadow apply = %d", resp.StatusCode)
	}
	if p, _ := h.router.Policy("shadowed-ui"); p.Shadow != nil {
		t.Errorf("shadow survived: %v", p.Shadow)
	}
}

// TestDeployCarriesTheShadowAcross matches what the CLI does, so the same
// operation means the same thing on both surfaces.
func TestDeployCarriesTheShadowAcross(t *testing.T) {
	h := newHarness(t)
	h.do("POST", "/v1/models", createModelRequest{Name: "carried"}) //nolint:errcheck
	v1 := h.upload("carried", "binary_logistic.model.json", featureNames(6))
	v2 := h.upload("carried", "binary_logistic.model.json", featureNames(6))
	v3 := h.upload("carried", "binary_logistic.model.json", featureNames(6))
	h.do("PUT", "/v1/models/carried/policy", routing.Policy{ //nolint:errcheck
		Routes: []routing.Route{{Version: v1, Weight: 1}}, Shadow: &v3,
	})

	before, _ := h.router.Policy("carried")
	resp, _ := h.postForm(t, "/models/carried/apply", "", url.Values{
		"action": {"deploy"}, "version": {itoa(v2)}, "expected": {policyLabel(before)},
	})
	if resp.StatusCode != http.StatusSeeOther {
		t.Fatalf("apply = %d", resp.StatusCode)
	}
	p, _ := h.router.Policy("carried")
	if p.Shadow == nil || *p.Shadow != v3 {
		t.Errorf("deploying dropped an unrelated shadow: %v", p.Shadow)
	}

	// But promoting the shadow itself clears it, since a version cannot be both.
	before, _ = h.router.Policy("carried")
	resp, _ = h.postForm(t, "/models/carried/apply", "", url.Values{
		"action": {"deploy"}, "version": {itoa(v3)}, "expected": {policyLabel(before)},
	})
	if resp.StatusCode != http.StatusSeeOther {
		t.Fatalf("apply = %d", resp.StatusCode)
	}
	if p, _ := h.router.Policy("carried"); p.Shadow != nil {
		t.Errorf("promoting the shadow left it shadowed: %v", p.Shadow)
	}
}

func itoa(v int) string { return strconv.Itoa(v) }

// TestApplyRefusesAVersionThatCannotLoad: the dashboard goes through the same
// applyPolicy as the JSON API, so it inherits the rule that traffic is never
// routed at something unloadable.
func TestApplyRefusesAVersionThatCannotLoad(t *testing.T) {
	h := newHarness(t)
	h.deployable(t, "unloadable")

	resp, body := h.postForm(t, "/models/unloadable/apply", "", url.Values{
		"action": {"deploy"}, "version": {"99"}, "expected": {"v1=100%"},
	})
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("= %d, want 400: %s", resp.StatusCode, truncate(body))
	}
	if p, _ := h.router.Policy("unloadable"); policyLabel(p) != "v1=100%" {
		t.Errorf("a failed apply changed the policy to %q", policyLabel(p))
	}
}

// TestPlanOnAModelWithNoPolicy covers the first deploy, where there is nothing
// to carry across and nothing to compare against.
func TestPlanOnAModelWithNoPolicy(t *testing.T) {
	h := newHarness(t)
	h.do("POST", "/v1/models", createModelRequest{Name: "never-deployed"}) //nolint:errcheck
	v := h.upload("never-deployed", "binary_logistic.model.json", featureNames(6))

	_, body := h.postForm(t, "/models/never-deployed/plan", "", url.Values{
		"action": {"deploy"}, "version": {itoa(v)},
	})
	if !strings.Contains(body, "not deployed") {
		t.Errorf("the plan does not show the empty starting state:\n%s", truncate(body))
	}

	resp, _ := h.postForm(t, "/models/never-deployed/apply", "", url.Values{
		"action": {"deploy"}, "version": {itoa(v)}, "expected": {"not deployed"},
	})
	if resp.StatusCode != http.StatusSeeOther {
		t.Fatalf("first deploy = %d", resp.StatusCode)
	}
	if p, _ := h.router.Policy("never-deployed"); policyLabel(p) != "v1=100%" {
		t.Errorf("policy = %q", policyLabel(p))
	}
}

// TestRollbackIsDeployUnderAnotherName, matching the CLI so the same operation
// means the same thing on both surfaces.
func TestRollbackIsDeployUnderAnotherName(t *testing.T) {
	h := newHarness(t)
	v1, v2 := h.deployable(t, "revertible")

	h.postForm(t, "/models/revertible/apply", "", url.Values{ //nolint:errcheck
		"action": {"deploy"}, "version": {itoa(v2)}, "expected": {"v1=100%"},
	})

	_, body := h.postForm(t, "/models/revertible/plan", "", url.Values{
		"action": {"rollback"}, "version": {itoa(v1)},
	})
	if !strings.Contains(body, "Roll back to version 1") {
		t.Errorf("the plan does not read as a rollback:\n%s", truncate(body))
	}

	resp, _ := h.postForm(t, "/models/revertible/apply", "", url.Values{
		"action": {"rollback"}, "version": {itoa(v1)}, "expected": {"v2=100%"},
	})
	if resp.StatusCode != http.StatusSeeOther {
		t.Fatalf("rollback = %d", resp.StatusCode)
	}
	if p, _ := h.router.Policy("revertible"); policyLabel(p) != "v1=100%" {
		t.Errorf("policy = %q after rollback", policyLabel(p))
	}
}

func TestPolicyLabelIsStableForComparison(t *testing.T) {
	// The label doubles as the concurrency token, so route order must not
	// change it — two policies that mean the same thing have to compare equal.
	a := routing.Policy{Routes: []routing.Route{{Version: 1, Weight: 90}, {Version: 2, Weight: 10}}}
	b := routing.Policy{Routes: []routing.Route{{Version: 2, Weight: 10}, {Version: 1, Weight: 90}}}
	if policyLabel(a) != policyLabel(b) {
		t.Errorf("route order changed the label: %q vs %q", policyLabel(a), policyLabel(b))
	}
	if got := policyLabel(routing.Policy{}); got != "not deployed" {
		t.Errorf("an empty policy labelled %q", got)
	}
	// And a change an operator cares about does change it.
	c := routing.Policy{Routes: []routing.Route{{Version: 1, Weight: 50}, {Version: 2, Weight: 50}}}
	if policyLabel(a) == policyLabel(c) {
		t.Error("a different split produced the same label")
	}
}

// TestCSRFValueSources: the token embedded in a rendered form has to come from
// whichever way the caller proved it holds one, or the next submit is refused.
func TestCSRFValueSources(t *testing.T) {
	// From the header, as an API client sends it.
	r := httptest.NewRequest("POST", "/models/x/plan", nil)
	r.Header.Set(auth.CSRFHeader, "from-header")
	if got := csrfValue(r); got != "from-header" {
		t.Errorf("csrfValue = %q, want the header", got)
	}

	// From the form field, as the rendered page sends it.
	r = httptest.NewRequest("POST", "/models/x/plan",
		strings.NewReader(url.Values{auth.CSRFField: {"from-form"}}.Encode()))
	r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	if err := r.ParseForm(); err != nil {
		t.Fatal(err)
	}
	if got := csrfValue(r); got != "from-form" {
		t.Errorf("csrfValue = %q, want the form field", got)
	}

	// From the cookie, which is how the first page load gets one to embed.
	r = httptest.NewRequest("GET", "/models/x", nil)
	r.AddCookie(&http.Cookie{Name: auth.CSRFCookie, Value: "from-cookie"})
	if got := csrfValue(r); got != "from-cookie" {
		t.Errorf("csrfValue = %q, want the cookie", got)
	}

	// And nothing at all is empty rather than a panic.
	if got := csrfValue(httptest.NewRequest("GET", "/", nil)); got != "" {
		t.Errorf("csrfValue with no source = %q", got)
	}
}

// TestMalformedFormBodyIsRefused rather than being read as an empty form, which
// would look like "no action given" and hide the real problem.
func TestMalformedFormBodyIsRefused(t *testing.T) {
	h := newHarness(t)
	h.deployable(t, "malformed")

	req, err := http.NewRequest("POST", h.http.URL+"/models/malformed/plan",
		strings.NewReader("action=deploy&version=%zz"))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := h.http.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusSeeOther {
		t.Fatalf("a malformed form was accepted with %d", resp.StatusCode)
	}
	if p, _ := h.router.Policy("malformed"); policyLabel(p) != "v1=100%" {
		t.Errorf("a malformed form changed the policy to %q", policyLabel(p))
	}
}
