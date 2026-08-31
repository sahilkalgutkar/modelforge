package api

import (
	"fmt"
	"net/http"
	"sort"
	"strconv"

	"github.com/sahilkalgutkar/modelforge/internal/auth"
	"github.com/sahilkalgutkar/modelforge/internal/routing"
)

// Deploy actions from the dashboard are two steps: a plan that says exactly what
// will change, and an apply that carries it out.
//
// A single button that immediately moved production traffic would be the wrong
// shape for this. These are the operations that decide which model answers every
// request, they are being triggered by a mouse rather than a reviewed script,
// and the cost of a mis-click is measured in wrong predictions rather than an
// error message. The confirmation page is where somebody sees "90/10 becomes
// 100% v2" before it is true.
//
// Both steps are POSTs, so both go through the CSRF check. The plan could be a
// GET, but then the intended change would sit in browser history and be handed
// to whatever the next page links to in a Referer.

// planAction renders what an action would do, without doing it.
func (s *Server) handlePlanAction(w http.ResponseWriter, r *http.Request) {
	tok := s.dashboardIdentity(r)
	name := r.PathValue("model")

	if err := r.ParseForm(); err != nil {
		s.dashboardError(w, "Could not read the form.", err)
		return
	}

	current, _ := s.deps.Router.Policy(name)
	next, summary, err := s.planPolicy(r, name, current)
	if err != nil {
		s.actionRefused(w, name, err.Error())
		return
	}
	if err := next.Validate(); err != nil {
		s.actionRefused(w, name, err.Error())
		return
	}

	s.render(w, confirmTemplate, pageData{
		Title:    "Confirm",
		Identity: tok.Name,
		Scopes:   scopeSummary(tok),
		CSRF:     csrfValue(r),
		Confirm: &confirmData{
			Model:   name,
			Summary: summary,
			Before:  policyLabel(current),
			After:   policyLabel(next),
			// The policy as it was when this page was rendered. Apply refuses
			// if it no longer matches, so two people changing the same model
			// at once cannot silently overwrite each other — the second one is
			// told rather than winning by arriving later.
			Expected: policyLabel(current),
			Action:   r.PostForm.Get("action"),
			Version:  r.PostForm.Get("version"),
			Percent:  r.PostForm.Get("percent"),
			Stable:   r.PostForm.Get("stable"),
		},
	})
}

// handleApplyAction carries out a planned change.
func (s *Server) handleApplyAction(w http.ResponseWriter, r *http.Request) {
	tok := s.dashboardIdentity(r)
	name := r.PathValue("model")

	if err := r.ParseForm(); err != nil {
		s.dashboardError(w, "Could not read the form.", err)
		return
	}

	current, _ := s.deps.Router.Policy(name)
	if expected := r.PostForm.Get("expected"); expected != policyLabel(current) {
		// Somebody else changed this between the plan and the apply. Applying
		// anyway would quietly discard their change, and the person who lost
		// it would have no way to know.
		s.actionRefused(w, name, fmt.Sprintf(
			"This model changed while you were looking at it. It was %q when the "+
				"confirmation was shown and is %q now, so nothing was applied. "+
				"Start again from the model page.", expected, policyLabel(current)))
		return
	}

	next, summary, err := s.planPolicy(r, name, current)
	if err != nil {
		s.actionRefused(w, name, err.Error())
		return
	}
	if err := next.Validate(); err != nil {
		s.actionRefused(w, name, err.Error())
		return
	}
	if err := s.applyPolicy(r.Context(), name, next); err != nil {
		s.actionRefused(w, name, err.Error())
		return
	}

	// Logged here as well as by the middleware's audit line, because the
	// middleware records the route and this records the effect.
	s.deps.Logger.Info("policy changed from the dashboard",
		"actor", tok.Name, "subject", tok.Subject, "model", name,
		"summary", summary, "before", policyLabel(current), "after", policyLabel(next))

	// Redirect after a successful post, so a refresh does not re-apply it.
	http.Redirect(w, r, "/models/"+name, http.StatusSeeOther)
}

// planPolicy turns a form submission into the policy it would produce, and a
// sentence describing it.
//
// Every action is expressed as a whole replacement policy rather than a mutation
// of the live one, which is what makes the confirmation honest: what is shown is
// exactly what will be installed.
func (s *Server) planPolicy(r *http.Request, model string, current routing.Policy) (routing.Policy, string, error) {
	action := r.PostForm.Get("action")

	version, verr := formVersion(r, "version")

	switch action {
	case "deploy", "rollback":
		if verr != nil {
			return routing.Policy{}, "", verr
		}
		next := routing.Policy{
			Model:  model,
			Routes: []routing.Route{{Version: version, Weight: 100}},
			Guard:  current.Guard,
		}
		// The shadow is carried across unless it is now serving, matching what
		// the CLI does. Dropping it here would mean the same operation means
		// different things depending on which surface performed it.
		next.Shadow = carryShadow(current, next.Routes)

		verb := "Deploy"
		if action == "rollback" {
			verb = "Roll back to"
		}
		return next, fmt.Sprintf("%s version %d, sending it all traffic.", verb, version), nil

	case "canary":
		if verr != nil {
			return routing.Policy{}, "", verr
		}
		stable, serr := formVersion(r, "stable")
		if serr != nil {
			return routing.Policy{}, "", serr
		}
		if stable == version {
			return routing.Policy{}, "", fmt.Errorf("the candidate and the stable version are both %d", version)
		}
		percent, perr := formPercent(r)
		if perr != nil {
			return routing.Policy{}, "", perr
		}
		next := routing.Policy{
			Model: model,
			Routes: []routing.Route{
				{Version: stable, Weight: 100 - percent},
				{Version: version, Weight: percent},
			},
			Guard: current.Guard,
		}
		next.Shadow = carryShadow(current, next.Routes)
		return next, fmt.Sprintf("Send %d%% of traffic to version %d, keeping %d%% on version %d.",
			percent, version, 100-percent, stable), nil

	case "shadow":
		if verr != nil {
			return routing.Policy{}, "", verr
		}
		if len(current.Routes) == 0 {
			return routing.Policy{}, "", fmt.Errorf("this model is not serving anything yet, so there is nothing to shadow alongside")
		}
		for _, rt := range current.Routes {
			if rt.Version == version && rt.Weight > 0 {
				return routing.Policy{}, "", fmt.Errorf(
					"version %d is already serving traffic; a version cannot be both shadow and serving", version)
			}
		}
		next := current
		next.Model = model
		next.Shadow = &version
		return next, fmt.Sprintf("Mirror traffic to version %d without serving from it.", version), nil

	case "unshadow":
		if current.Shadow == nil {
			return routing.Policy{}, "", fmt.Errorf("this model has no shadow")
		}
		was := *current.Shadow
		next := current
		next.Model = model
		next.Shadow = nil
		return next, fmt.Sprintf("Stop mirroring traffic to version %d.", was), nil
	}

	return routing.Policy{}, "", fmt.Errorf("unknown action %q", action)
}

// carryShadow keeps an existing shadow unless it is about to serve traffic.
func carryShadow(current routing.Policy, routes []routing.Route) *int {
	if current.Shadow == nil {
		return nil
	}
	for _, rt := range routes {
		if rt.Version == *current.Shadow && rt.Weight > 0 {
			return nil
		}
	}
	return current.Shadow
}

func formVersion(r *http.Request, field string) (int, error) {
	raw := r.PostForm.Get(field)
	if raw == "" {
		return 0, fmt.Errorf("no %s given", field)
	}
	v, err := strconv.Atoi(raw)
	if err != nil || v <= 0 {
		return 0, fmt.Errorf("%s must be a positive number, got %q", field, raw)
	}
	return v, nil
}

func formPercent(r *http.Request) (int, error) {
	raw := r.PostForm.Get("percent")
	p, err := strconv.Atoi(raw)
	if err != nil {
		return 0, fmt.Errorf("the percentage must be a number, got %q", raw)
	}
	// Zero would deploy nothing to the candidate while looking like a canary,
	// and 100 would be a full deploy wearing a canary's name. Both are better
	// expressed with the action that actually means them.
	if p < 1 || p > 99 {
		return 0, fmt.Errorf("a canary percentage must be between 1 and 99, got %d", p)
	}
	return p, nil
}

// policyLabel renders a policy for display and for comparison.
//
// It doubles as the optimistic-concurrency token: two policies that render the
// same are the same deployment, and any change an operator would care about
// changes the string.
func policyLabel(p routing.Policy) string {
	if len(p.Routes) == 0 {
		return "not deployed"
	}
	routes := append([]routing.Route(nil), p.Routes...)
	sort.Slice(routes, func(i, j int) bool { return routes[i].Version < routes[j].Version })
	copyOf := p
	copyOf.Routes = routes
	return copyOf.String()
}

// csrfValue returns the token to embed in a form, which is the one the caller
// already proved it holds.
func csrfValue(r *http.Request) string {
	if v := r.Header.Get(auth.CSRFHeader); v != "" {
		return v
	}
	if v := r.PostForm.Get(auth.CSRFField); v != "" {
		return v
	}
	if c, err := r.Cookie(auth.CSRFCookie); err == nil {
		return c.Value
	}
	return ""
}

func (s *Server) actionRefused(w http.ResponseWriter, model, reason string) {
	s.deps.Logger.Warn("dashboard action refused", "model", model, "reason", reason)
	page(w, http.StatusBadRequest, "Nothing was changed", reason)
}
