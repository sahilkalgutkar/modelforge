package api

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/coreos/go-oidc/v3/oidc"
	"golang.org/x/oauth2"

	"github.com/sahilkalgutkar/modelforge/internal/auth"
)

// handleBrowserLogin starts an authorization-code login for a browser.
//
// The server is the OAuth client here rather than the CLI, but it is still a
// *public* client using PKCE. It could hold a secret, being a server — but
// having one would mean every deployment needs a secret provisioned and
// rotated, for no gain over PKCE on a confidential channel. Fewer secrets is
// the better default.
func (s *Server) handleBrowserLogin(w http.ResponseWriter, r *http.Request) {
	provider, meta, ok := s.loginProvider(w, r)
	if !ok {
		return
	}

	state, err := auth.RandomToken()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	nonce, err := auth.RandomToken()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	verifier := oauth2.GenerateVerifier()

	// Where to land after signing in. Validated rather than trusted: an
	// unchecked value here is a textbook open redirect, and one on a login
	// endpoint is worth more than most — an attacker sends somebody through a
	// genuine sign-in at their real provider and bounces them to a page they
	// control, which is exactly what a convincing phish looks like.
	next := safeRedirect(r.URL.Query().Get("next"))

	loginID, err := s.deps.Logins.Start(state, nonce, verifier, next)
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, err)
		return
	}
	auth.SetCookie(w, auth.LoginCookie, loginID, time.Now().Add(10*time.Minute), s.cookieOpts())

	conf := s.oauthConfig(provider, meta)
	http.Redirect(w, r, conf.AuthCodeURL(state,
		oidc.Nonce(nonce),
		oauth2.S256ChallengeOption(verifier),
		oauth2.SetAuthURLParam("audience", meta.Audience),
	), http.StatusFound)
}

// handleBrowserCallback completes a browser login and starts a session.
func (s *Server) handleBrowserCallback(w http.ResponseWriter, r *http.Request) {
	provider, meta, ok := s.loginProvider(w, r)
	if !ok {
		return
	}

	if e := r.URL.Query().Get("error"); e != "" {
		// The provider's own message is shown, since it is the only thing that
		// explains a declined consent or a misconfigured client. It is escaped
		// on the way out.
		s.loginFailed(w, http.StatusBadRequest, "The identity provider refused the sign-in.", e)
		return
	}

	cookie, err := r.Cookie(auth.LoginCookie)
	if err != nil {
		s.loginFailed(w, http.StatusBadRequest, "This sign-in did not start here.",
			"no login is in progress in this browser")
		return
	}
	auth.ClearCookie(w, auth.LoginCookie, s.cookieOpts())

	pending, found := s.deps.Logins.Take(cookie.Value)
	if !found {
		s.loginFailed(w, http.StatusBadRequest, "This sign-in has expired.",
			"start again from the beginning")
		return
	}

	// Compared before the code is touched. Without it, anybody who can make a
	// browser load this URL completes a login of their choosing, and the
	// victim ends up holding the attacker's session.
	if r.URL.Query().Get("state") != pending.State {
		s.loginFailed(w, http.StatusBadRequest, "This response did not match the sign-in that started here.",
			"the state parameter did not match")
		return
	}

	code := r.URL.Query().Get("code")
	if code == "" {
		s.loginFailed(w, http.StatusBadRequest, "The provider returned no authorization code.", "")
		return
	}

	conf := s.oauthConfig(provider, meta)
	token, err := conf.Exchange(r.Context(), code, oauth2.VerifierOption(pending.Verifier))
	if err != nil {
		s.deps.Logger.Warn("browser login: code exchange failed", "error", err)
		s.loginFailed(w, http.StatusBadGateway, "Could not complete the sign-in.",
			"the authorization code could not be exchanged")
		return
	}

	rawID, _ := token.Extra("id_token").(string)
	if rawID == "" {
		s.loginFailed(w, http.StatusBadGateway, "Could not complete the sign-in.",
			"the provider returned no id_token")
		return
	}

	// Verified through the same path an API request takes, so a browser
	// session can never carry an identity a bearer token could not. Anything
	// else would make the cookie a second, weaker front door.
	tok, err := s.deps.Auth.VerifyIDToken(r.Context(), rawID)
	if err != nil {
		s.deps.Logger.Warn("browser login: token rejected", "error", err)
		s.loginFailed(w, http.StatusForbidden, "You are signed in, but not permitted here.", err.Error())
		return
	}

	// The nonce proves this token was minted for this login rather than
	// captured from another and replayed into it.
	if idToken, verr := provider.Verifier(&oidc.Config{SkipClientIDCheck: true}).
		Verify(r.Context(), rawID); verr != nil || idToken.Nonce != pending.Nonce {
		s.loginFailed(w, http.StatusBadRequest, "This token does not match the sign-in that started here.",
			"the nonce did not match")
		return
	}

	sess, err := s.deps.Sessions.Create(tok)
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, err)
		return
	}

	auth.SetCookie(w, auth.SessionCookie, sess.ID, sess.Expires, s.cookieOpts())
	auth.SetCookie(w, auth.CSRFCookie, sess.CSRF, sess.Expires, s.cookieOpts())

	s.deps.Logger.Info("browser session started",
		"actor", tok.Name, "subject", tok.Subject, "expires", sess.Expires.UTC().Format(time.RFC3339))

	// With nowhere particular to go, say so rather than redirecting to "/".
	// This server has no UI, so a bare redirect would land on a 404 and make a
	// successful sign-in look like a failure.
	if pending.Next == "" {
		page(w, http.StatusOK, "Signed in",
			fmt.Sprintf("You are signed in as %s. This session is for browsing the API; "+
				"close this tab and carry on.", tok.Name))
		return
	}
	http.Redirect(w, r, pending.Next, http.StatusFound)
}

// handleBrowserLogout ends a session.
//
// Accepted on GET as well as POST. A logout triggered cross-site is an
// annoyance rather than a compromise — the worst an attacker achieves is
// signing somebody out — and refusing GET would mean there is no way to log out
// of a service with no UI to POST from.
func (s *Server) handleBrowserLogout(w http.ResponseWriter, r *http.Request) {
	if cookie, err := r.Cookie(auth.SessionCookie); err == nil && s.deps.Sessions != nil {
		if s.deps.Sessions.Delete(cookie.Value) {
			s.deps.Logger.Info("browser session ended")
		}
	}
	auth.ClearCookie(w, auth.SessionCookie, s.cookieOpts())
	auth.ClearCookie(w, auth.CSRFCookie, s.cookieOpts())

	if next := safeRedirect(r.URL.Query().Get("next")); next != "" {
		http.Redirect(w, r, next, http.StatusFound)
		return
	}
	page(w, http.StatusOK, "Signed out", "Your browser session has ended.")
}

// loginProvider resolves the identity provider, or writes the reason it cannot.
func (s *Server) loginProvider(w http.ResponseWriter, r *http.Request) (*oidc.Provider, auth.PublicConfig, bool) {
	if s.deps.Sessions == nil || s.deps.Logins == nil {
		page(w, http.StatusNotFound, "Not available", "This server does not support browser sessions.")
		return nil, auth.PublicConfig{}, false
	}
	v := s.deps.Auth.OIDC()
	if v == nil || !v.LoginEnabled() {
		page(w, http.StatusNotFound, "Not available",
			"This server has no identity provider configured for interactive sign-in.")
		return nil, auth.PublicConfig{}, false
	}

	meta := v.PublicConfig()
	provider, err := oidc.NewProvider(r.Context(), meta.Issuer)
	if err != nil {
		s.deps.Logger.Error("browser login: provider unreachable", "issuer", meta.Issuer, "error", err)
		page(w, http.StatusBadGateway, "Sign-in unavailable",
			"The identity provider could not be reached. Try again shortly.")
		return nil, auth.PublicConfig{}, false
	}
	return provider, meta, true
}

func (s *Server) oauthConfig(provider *oidc.Provider, meta auth.PublicConfig) *oauth2.Config {
	endpoint := provider.Endpoint()
	// Explicit for the same reason the CLI sets it: left unset, x/oauth2 probes
	// by retrying, and an authorization code is single-use, so the probe burns
	// it and reports the wrong error.
	endpoint.AuthStyle = oauth2.AuthStyleInParams

	return &oauth2.Config{
		ClientID:    meta.ClientID,
		Endpoint:    endpoint,
		RedirectURL: strings.TrimRight(s.deps.ExternalURL, "/") + "/auth/callback",
		// No offline_access. A browser session lasts as long as its cookie and
		// then signs in again, so a refresh token would be a long-lived secret
		// held server-side for no benefit — and one more thing to leak.
		Scopes: []string{oidc.ScopeOpenID, "profile", "email", "groups"},
	}
}

func (s *Server) cookieOpts() auth.CookieOptions {
	return auth.CookieOptions{Insecure: s.deps.InsecureCookies}
}

// safeRedirect reduces a caller-supplied destination to a same-site path, or
// empty when there is no usable one.
//
// Only a rooted path is allowed, and "//host" is rejected explicitly because it
// is protocol-relative: a browser reads //evil.example as a different origin
// while a naive "starts with /" check reads it as a local path. That single
// character is the whole open-redirect bug.
//
// Anything rejected becomes empty rather than "/" so the caller renders a page
// instead of bouncing somebody to a route this server does not have.
func safeRedirect(next string) string {
	if next == "" || !strings.HasPrefix(next, "/") || strings.HasPrefix(next, "//") {
		return ""
	}
	u, err := url.Parse(next)
	if err != nil || u.Scheme != "" || u.Host != "" {
		return ""
	}
	if u.RequestURI() == "/" {
		return ""
	}
	return u.RequestURI()
}

// page renders a minimal HTML response.
//
// Nothing from the request is interpolated without escaping, because these
// pages exist at exactly the moment a browser is holding an authorization code
// and a session cookie.
func page(w http.ResponseWriter, status int, title, body string) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	// A sign-in page has no business being framed, and nothing here loads
	// anything external.
	w.Header().Set("X-Frame-Options", "DENY")
	w.Header().Set("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")
	w.WriteHeader(status)
	fmt.Fprintf(w, `<!doctype html><meta charset="utf-8"><title>modelforge</title>`+
		`<body style="font:16px system-ui;margin:4rem auto;max-width:32rem">`+
		`<h1>%s</h1><p>%s</p></body>`, escapeHTML(title), escapeHTML(body))
}

func (s *Server) loginFailed(w http.ResponseWriter, status int, title, detail string) {
	body := title
	if detail != "" {
		body = title + " (" + detail + ")"
	}
	page(w, status, "Sign-in failed", body)
}

func escapeHTML(s string) string {
	return strings.NewReplacer(
		"&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;", "'", "&#39;",
	).Replace(s)
}
