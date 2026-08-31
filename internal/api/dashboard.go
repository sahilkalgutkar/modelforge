package api

import (
	"embed"
	"fmt"
	"html/template"
	"net/http"
	"net/url"
	"sort"
	"strings"

	"github.com/sahilkalgutkar/modelforge/internal/auth"
	"github.com/sahilkalgutkar/modelforge/internal/drift"
	"github.com/sahilkalgutkar/modelforge/internal/registry"
	"github.com/sahilkalgutkar/modelforge/internal/routing"
)

//go:embed templates/*.html
var templateFS embed.FS

// dashboardTemplates are parsed once at startup. A parse error is a programming
// mistake in a file that ships inside the binary, so failing here rather than on
// the first request is right: it cannot be caused by anything a user does.
var (
	indexTemplate   = template.Must(template.ParseFS(templateFS, "templates/layout.html", "templates/index.html"))
	modelTemplate   = template.Must(template.ParseFS(templateFS, "templates/layout.html", "templates/model.html"))
	confirmTemplate = template.Must(template.ParseFS(templateFS, "templates/layout.html", "templates/confirm.html"))
)

// The dashboard is server-rendered HTML with no JavaScript at all, which is a
// deliberate choice rather than a shortcut.
//
// html/template gives contextual auto-escaping — it knows whether a value is
// landing in an attribute, a URL or element text, and escapes accordingly. That
// is the XSS defence, and it matters more here than it would have a week ago:
// this server now hands out session cookies, so a script injected into one of
// these pages would be running with somebody's session.
//
// No JavaScript means the Content-Security-Policy can be `script-src 'none'`,
// which turns "we escaped everything correctly" from a claim into something the
// browser enforces. It also means no build step, no bundler and no node in CI
// for a Go serving platform, and the binary stays the single self-contained
// artifact the rest of the design depends on.
//
// Deploy actions arrive as ordinary form posts, which is the reason the CSRF
// check accepts a form field as well as a header: a page with no JavaScript
// cannot set a header.

// dashboardRefresh is how often a page reloads itself, in seconds. Done with a
// meta refresh rather than a script, so the no-JavaScript policy holds.
const dashboardRefresh = 15

type pageData struct {
	Title    string
	Identity string
	Scopes   string
	Refresh  int
	Models   []modelRow
	Model    modelDetail
	Versions []versionRow
	Drift    []driftPanel

	// CanDeploy gates the action forms on the viewer's scope, so a read-only
	// credential is not shown buttons that would refuse it.
	CanDeploy bool
	HasShadow bool
	CSRF      string
	Confirm   *confirmData
}

type confirmData struct {
	Model    string
	Summary  string
	Before   string
	After    string
	Expected string
	Action   string
	Version  string
	Percent  string
	Stable   string
}

type modelRow struct {
	Name         string
	Policy       string
	VersionCount int
	Requests     int64
	Failures     int64
	DriftLabel   string
	DriftClass   string
}

type modelDetail struct {
	Name        string
	Description string
	Policy      string
}

type versionRow struct {
	Version        int
	Weight         int
	BarWidth       int
	Digest         string
	FeatureCount   int
	Requests       int64
	Failures       int64
	MeanBatch      string
	Loaded         bool
	Shadow         bool
	RemovedByGuard bool
}

type driftPanel struct {
	Version    int
	Ready      bool
	Window     string
	Samples    int64
	MinSamples int
	Features   []driftRow
}

type driftRow struct {
	Feature  string
	PSI      string
	Severity string
	Class    string
	Missing  string
}

// handleDashboard renders the model overview.
func (s *Server) handleDashboard(w http.ResponseWriter, r *http.Request) {
	tok := s.dashboardIdentity(r)

	models, err := s.deps.Registry.ListModels(r.Context())
	if err != nil {
		s.dashboardError(w, "Could not list models.", err)
		return
	}

	data := pageData{
		Title: "Models", Identity: tok.Name, Scopes: scopeSummary(tok), Refresh: dashboardRefresh,
	}
	for _, m := range models {
		data.Models = append(data.Models, s.modelRow(r, m))
	}
	s.render(w, indexTemplate, data)
}

func (s *Server) modelRow(r *http.Request, m registry.Model) modelRow {
	row := modelRow{Name: m.Name}

	versions, err := s.deps.Registry.ListVersions(r.Context(), m.Name)
	if err == nil {
		row.VersionCount = len(versions)
	}

	policy, deployed := s.deps.Router.Policy(m.Name)
	if deployed {
		row.Policy = policy.String()
	}

	// Traffic counters are per version, so the model's totals are the sum of
	// what each version has served — including versions no longer receiving
	// traffic, whose failures are exactly what somebody is looking for after a
	// rollback.
	for _, v := range versions {
		requests, failures, _ := s.deps.Router.Health(m.Name, v.Version)
		row.Requests += requests
		row.Failures += failures
	}

	// The worst drift across the model's versions, since a model is healthy
	// only if all of its serving versions are.
	worst := ""
	worstClass := ""
	for _, v := range versions {
		rep, ready, derr := s.deps.Manager.DriftReport(m.Name, v.Version)
		if derr != nil || !ready {
			continue
		}
		if feature, found := rep.Worst(); found {
			if worst == "" || severityRank(string(feature.Severity)) > severityRank(worst) {
				worst, worstClass = string(feature.Severity), severityClass(feature.Severity)
			}
		}
	}
	row.DriftLabel, row.DriftClass = worst, worstClass
	return row
}

// handleModelPage renders one model in detail.
func (s *Server) handleModelPage(w http.ResponseWriter, r *http.Request) {
	tok := s.dashboardIdentity(r)
	name := r.PathValue("model")

	model, err := s.deps.Registry.GetModel(r.Context(), name)
	if err != nil {
		s.dashboardError(w, "No such model.", err)
		return
	}
	versions, err := s.deps.Registry.ListVersions(r.Context(), name)
	if err != nil {
		s.dashboardError(w, "Could not list versions.", err)
		return
	}

	policy, deployed := s.deps.Router.Policy(name)
	data := pageData{
		Title: model.Name, Identity: tok.Name, Scopes: scopeSummary(tok), Refresh: dashboardRefresh,
		Model:     modelDetail{Name: model.Name, Description: model.Description},
		CanDeploy: tok.Allows(auth.ScopeAdmin),
		HasShadow: deployed && policy.Shadow != nil,
		CSRF:      csrfValue(r),
	}
	if deployed {
		data.Model.Policy = policy.String()
	}

	weights, total := versionWeights(policy)
	for _, v := range versions {
		row := versionRow{
			Version:      v.Version,
			Digest:       v.Digest.Short(),
			FeatureCount: len(v.Features),
			Loaded:       s.deps.Manager.IsLoaded(name, v.Version),
			Shadow:       deployed && policy.Shadow != nil && *policy.Shadow == v.Version,
		}
		if total > 0 {
			row.Weight = weights[v.Version] * 100 / total
			// A fixed scale rather than one normalised to the largest share,
			// so a 5% canary looks like 5% rather than filling the column.
			row.BarWidth = row.Weight * 2
		}
		row.Requests, row.Failures, row.RemovedByGuard = s.deps.Router.Health(name, v.Version)

		if stats, serr := s.deps.Manager.BatchStats(name, v.Version); serr == nil {
			row.MeanBatch = fmt.Sprintf("%.1f", stats.Mean())
		}
		data.Versions = append(data.Versions, row)

		if panel, has := s.driftPanel(name, v.Version); has {
			data.Drift = append(data.Drift, panel)
		}
	}
	s.render(w, modelTemplate, data)
}

func (s *Server) driftPanel(model string, version int) (driftPanel, bool) {
	rep, ready, err := s.deps.Manager.DriftReport(model, version)
	if err != nil {
		return driftPanel{}, false
	}

	panel := driftPanel{
		Version: version, Ready: ready, Window: rep.Window,
		Samples: rep.Samples, MinSamples: drift.MinSamples,
	}
	if panel.Window == "" {
		// A version with no baseline attached has nothing to show, and an
		// empty panel would read as "no drift" rather than "not monitored".
		return driftPanel{}, false
	}
	if !ready {
		return panel, true
	}

	features := append([]drift.Reading(nil), rep.Features...)
	sort.Slice(features, func(i, j int) bool { return features[i].PSI > features[j].PSI })
	for _, f := range features {
		panel.Features = append(panel.Features, driftRow{
			Feature:  f.Feature,
			PSI:      fmt.Sprintf("%.4f", f.PSI),
			Severity: string(f.Severity),
			Class:    severityClass(f.Severity),
			Missing:  fmt.Sprintf("%.1f%%", f.MissingRate*100),
		})
	}
	if rep.Prediction != nil {
		panel.Features = append(panel.Features, driftRow{
			Feature:  "(prediction)",
			PSI:      fmt.Sprintf("%.4f", rep.Prediction.PSI),
			Severity: string(rep.Prediction.Severity),
			Class:    severityClass(rep.Prediction.Severity),
			Missing:  "—",
		})
	}
	return panel, true
}

// dashboardIdentity returns who is viewing.
//
// The middleware refuses before the handler runs when authentication fails, so
// a request that arrives here without a token came through a disabled
// Authenticator — the only configuration where that happens.
func (s *Server) dashboardIdentity(r *http.Request) auth.Token {
	if tok, ok := auth.FromContext(r.Context()); ok {
		return tok
	}
	return auth.Token{Name: "anonymous", Scopes: []auth.Scope{auth.ScopeAdmin}}
}

// dashboardEntry wraps a page handler so an unauthenticated browser is sent to
// sign in instead of being refused.
//
// This is the one place the HTML surface should behave differently from the
// API. A browser handed a bare 401 has nowhere to go and no way to know a
// sign-in exists; an API client handed a redirect follows it and tries to parse
// a login page as JSON. Each gets what it can act on, which is why the redirect
// is here rather than in the shared middleware.
func (s *Server) dashboardEntry(next http.HandlerFunc) http.Handler {
	guarded := s.middleware().RequireFunc(auth.ScopeRead, next)

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.deps.Sessions != nil && s.deps.Auth != nil && !s.deps.Auth.IsDisabled() {
			_, cookieErr := r.Cookie(auth.SessionCookie)
			hasSession := cookieErr == nil
			hasBearer := r.Header.Get("Authorization") != ""
			if !hasSession && !hasBearer {
				// RequestURI rather than anything caller-supplied, and it is
				// re-validated on the way back out by safeRedirect.
				http.Redirect(w, r, "/login?next="+url.QueryEscape(r.URL.RequestURI()), http.StatusFound)
				return
			}
		}
		guarded.ServeHTTP(w, r)
	})
}

// render writes a page with the headers that make the no-JavaScript policy
// enforceable rather than merely intended.
func (s *Server) render(w http.ResponseWriter, tmpl *template.Template, data pageData) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	// script-src 'none' is the point: html/template escaping is the defence,
	// and this is the browser enforcing it independently. Styles are inline,
	// so they need the one exception.
	w.Header().Set("Content-Security-Policy",
		"default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
	w.Header().Set("X-Frame-Options", "DENY")
	w.Header().Set("Referrer-Policy", "same-origin")
	// These pages carry model names, traffic volumes and drift readings.
	w.Header().Set("Cache-Control", "no-store")

	if err := tmpl.ExecuteTemplate(w, "layout", data); err != nil {
		// The status is already written by the time a template fails partway,
		// so there is nothing useful to send; log it and let the truncated
		// page speak for itself.
		s.deps.Logger.Error("dashboard render failed", "error", err)
	}
}

func (s *Server) dashboardError(w http.ResponseWriter, msg string, err error) {
	s.deps.Logger.Warn("dashboard", "message", msg, "error", err)
	page(w, http.StatusNotFound, "Not found", msg)
}

func versionWeights(p routing.Policy) (map[int]int, int) {
	weights := make(map[int]int, len(p.Routes))
	total := 0
	for _, r := range p.Routes {
		weights[r.Version] = r.Weight
		total += r.Weight
	}
	return weights, total
}

func scopeSummary(tok auth.Token) string {
	names := make([]string, len(tok.Scopes))
	for i, sc := range tok.Scopes {
		names[i] = string(sc)
	}
	if len(names) == 0 {
		return "no scopes"
	}
	return strings.Join(names, ", ")
}

func severityClass(sev drift.Severity) string {
	switch sev {
	case drift.SeveritySignificant:
		return "bad"
	case drift.SeverityModerate:
		return "warn"
	default:
		return "ok"
	}
}

func severityRank(sev string) int {
	switch sev {
	case string(drift.SeveritySignificant):
		return 3
	case string(drift.SeverityModerate):
		return 2
	case string(drift.SeverityStable):
		return 1
	}
	return 0
}
