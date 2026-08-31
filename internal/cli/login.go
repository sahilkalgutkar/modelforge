package cli

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os/exec"
	"runtime"
	"strings"
	"time"

	"github.com/coreos/go-oidc/v3/oidc"
	"golang.org/x/oauth2"
)

// loginTimeout bounds how long the CLI waits for somebody to finish
// authenticating in their browser. Long enough to find a password manager and
// a second factor; short enough that an abandoned attempt does not leave a
// listener open on a laptop indefinitely.
const loginTimeout = 3 * time.Minute

// errNoBrowserRequested is returned by the browser opener when --no-browser was
// asked for, so the flow prints the URL and waits rather than reporting a
// failure that did not happen.
var errNoBrowserRequested = errors.New("no browser requested")

// Login performs an OAuth authorization-code login and stores the result.
//
// This is the native-application flow from RFC 8252, and the shape is dictated
// by what a command-line tool can safely do. It cannot keep a client secret —
// one shipped inside a binary is a string every user can read out of it — so
// the client is public and PKCE takes the secret's place. It redirects to
// 127.0.0.1 on an ephemeral port, which is the one redirect target a native
// application can prove it owns.
func (c *Client) Login(ctx context.Context, openBrowser func(string) error) error {
	meta, err := c.authConfig()
	if err != nil {
		return err
	}
	if !meta.Login {
		return fmt.Errorf("this server does not support interactive login: %s", meta.Reason)
	}

	provider, err := oidc.NewProvider(ctx, meta.Issuer)
	if err != nil {
		return fmt.Errorf("reach the identity provider at %s: %w", meta.Issuer, err)
	}

	// A listener on the loopback interface, on whatever port the OS gives us.
	// RFC 8252 requires providers to accept any port for a loopback redirect,
	// precisely so a native app does not need a fixed one.
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return fmt.Errorf("open a local callback listener: %w", err)
	}
	defer listener.Close()

	// 127.0.0.1 rather than localhost, deliberately. localhost resolves
	// through DNS, and a hostile resolver can point it somewhere else — at
	// which point the authorization code is delivered to somebody else's
	// machine. The literal address cannot be redirected.
	redirectURL := fmt.Sprintf("http://127.0.0.1:%d/callback", listener.Addr().(*net.TCPAddr).Port)

	endpoint := provider.Endpoint()
	// Send the client id in the request body rather than as HTTP Basic auth.
	//
	// This is what RFC 6749 specifies for a public client, which has no secret
	// to put in a Basic header in the first place. It also avoids a failure
	// mode that is genuinely nasty to debug: left unset, x/oauth2 probes for
	// the server's preferred style by trying one and retrying with the other
	// if the first is rejected. An authorization code is single-use, so the
	// probe burns it, and the error surfaced is the *second* attempt's
	// "invalid_grant" — which points at the code rather than at whatever
	// actually went wrong with the first request. Being explicit means one
	// request and the real error.
	endpoint.AuthStyle = oauth2.AuthStyleInParams

	conf := &oauth2.Config{
		ClientID:    meta.ClientID,
		Endpoint:    endpoint,
		RedirectURL: redirectURL,
		// offline_access is what asks for a refresh token. Providers that do
		// not implement it ignore the scope, which is why this does not fail
		// when none comes back — the session simply behaves as it did before
		// and expires.
		Scopes: []string{oidc.ScopeOpenID, "profile", "email", "groups", "offline_access"},
	}

	// state defends the callback against a forged request; nonce binds the
	// resulting ID token to this particular login, so one captured elsewhere
	// cannot be replayed into it.
	state, err := randomString()
	if err != nil {
		return err
	}
	nonce, err := randomString()
	if err != nil {
		return err
	}
	verifier := oauth2.GenerateVerifier()

	authURL := conf.AuthCodeURL(state,
		oidc.Nonce(nonce),
		oauth2.S256ChallengeOption(verifier),
		// The audience this server checks. Providers that scope tokens per
		// API need to be told which one is being asked for, or they mint a
		// token the server will refuse.
		oauth2.SetAuthURLParam("audience", meta.Audience),
	)

	result := make(chan loginResult, 1)
	srv := &http.Server{
		Handler:           callbackHandler(state, result),
		ReadHeaderTimeout: 5 * time.Second,
	}
	go srv.Serve(listener) //nolint:errcheck // Shutdown below is the exit path
	defer func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		srv.Shutdown(shutdownCtx) //nolint:errcheck // best effort
	}()

	// Printed before the browser is launched, and always, because the browser
	// may open somewhere the person cannot see it — a different display, a
	// remote session — and a URL they can copy is the fallback that always
	// works.
	fmt.Fprintf(c.Out, "to sign in, visit:\n\n  %s\n\n", authURL)

	switch err := openBrowser(authURL); {
	case errors.Is(err, errNoBrowserRequested):
		fmt.Fprintf(c.Out, "waiting for you to complete the sign-in...\n")
	case err != nil:
		// Not fatal: the URL is above, and a machine with no browser is a
		// perfectly ordinary place to run this.
		fmt.Fprintf(c.Out, "(could not open a browser automatically: %v)\n", err)
	default:
		fmt.Fprintf(c.Out, "opening your browser at %s...\n", meta.Issuer)
	}

	waitCtx, cancel := context.WithTimeout(ctx, loginTimeout)
	defer cancel()

	var res loginResult
	select {
	case res = <-result:
	case <-waitCtx.Done():
		return fmt.Errorf("timed out after %s waiting for the browser to complete the login", loginTimeout)
	}
	if res.err != nil {
		return res.err
	}

	token, err := conf.Exchange(ctx, res.code, oauth2.VerifierOption(verifier))
	if err != nil {
		return fmt.Errorf("exchange the authorization code: %w", err)
	}

	rawID, ok := token.Extra("id_token").(string)
	if !ok || rawID == "" {
		// The access token is for calling the provider's own APIs; this server
		// authenticates the *identity* token. A provider that returns no ID
		// token has not been asked for the openid scope, or is not an OIDC
		// provider at all.
		return errors.New("the provider returned no id_token; check that the openid scope is permitted for this client")
	}

	// Verified locally before it is stored, so a token this server would
	// refuse is reported now rather than on the next command. The nonce check
	// is the part that matters: it proves the token was minted for this login
	// and not captured from another.
	idToken, err := provider.Verifier(&oidc.Config{ClientID: meta.Audience}).Verify(ctx, rawID)
	if err != nil {
		return fmt.Errorf("the provider issued a token this server would reject: %w", err)
	}
	if idToken.Nonce != nonce {
		return errors.New("the returned token does not match this login attempt")
	}

	cred := Credential{
		IDToken:      rawID,
		RefreshToken: token.RefreshToken,
		Expiry:       idToken.Expiry,
		Issuer:       meta.Issuer,
		ClientID:     meta.ClientID,
	}
	if err := SaveCredential(cred); err != nil {
		return err
	}

	c.Token = rawID
	who, err := c.whoAmI()
	if err != nil {
		// Stored anyway: the token verified, so a server that cannot answer
		// right now does not make it invalid.
		fmt.Fprintf(c.Out, "signed in, but could not confirm with the server: %v\n", err)
		return nil
	}
	fmt.Fprintf(c.Out, "signed in as %s (%s)\n", who.Name, strings.Join(scopeNames(who.Scopes), ", "))
	switch {
	case cred.RefreshToken != "":
		fmt.Fprintln(c.Out, "this session renews itself; run `modelforgectl logout` to end it")
	case who.Expires != "":
		fmt.Fprintf(c.Out, "this session expires at %s and cannot be renewed "+
			"(the provider issued no refresh token)\n", who.Expires)
	}
	return nil
}

type loginResult struct {
	code string
	err  error
}

// callbackHandler receives the redirect from the identity provider.
func callbackHandler(wantState string, out chan<- loginResult) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/callback", func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()

		if e := q.Get("error"); e != "" {
			msg := e
			if d := q.Get("error_description"); d != "" {
				msg += ": " + d
			}
			finish(w, http.StatusBadRequest, "Sign-in failed", msg)
			send(out, loginResult{err: fmt.Errorf("the identity provider refused the login: %s", msg)})
			return
		}

		// Checked before the code is touched. Without it, anybody who can get
		// this browser to load a URL can complete a login of their choosing —
		// the classic session-fixation shape, where the victim ends up holding
		// the attacker's session.
		if q.Get("state") != wantState {
			finish(w, http.StatusBadRequest, "Sign-in failed",
				"This response did not come from the sign-in that was started here.")
			send(out, loginResult{err: errors.New("the callback state did not match; the login was not the one this command started")})
			return
		}

		code := q.Get("code")
		if code == "" {
			finish(w, http.StatusBadRequest, "Sign-in failed", "The provider returned no authorization code.")
			send(out, loginResult{err: errors.New("the provider returned no authorization code")})
			return
		}

		finish(w, http.StatusOK, "Signed in", "You can close this tab and return to your terminal.")
		send(out, loginResult{code: code})
	})
	return mux
}

// send never blocks, so a duplicate callback cannot wedge the handler.
func send(out chan<- loginResult, r loginResult) {
	select {
	case out <- r:
	default:
	}
}

func finish(w http.ResponseWriter, status int, title, body string) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	// Static markup with no interpolation of anything the provider sent. A
	// query parameter reflected into this page would be cross-site scripting
	// on a page that has just handled an authorization code.
	fmt.Fprintf(w, `<!doctype html><meta charset="utf-8"><title>modelforge</title>`+
		`<body style="font:16px system-ui;margin:4rem auto;max-width:30rem">`+
		`<h1>%s</h1><p>%s</p></body>`, htmlEscape(title), htmlEscape(body))
}

func htmlEscape(s string) string {
	r := strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;", "'", "&#39;")
	return r.Replace(s)
}

func randomString() (string, error) {
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		return "", fmt.Errorf("generate random value: %w", err)
	}
	return base64.RawURLEncoding.EncodeToString(buf), nil
}

// browserCommand picks how to open a URL on a given platform.
//
// Split from OpenBrowser so the choice can be tested without launching
// anything: the part worth checking is that each platform gets the right
// command and that the URL is passed as an argument rather than interpolated
// into a shell string, which is what keeps a URL containing shell
// metacharacters from becoming a command.
func browserCommand(goos, url string) (string, []string) {
	switch goos {
	case "darwin":
		return "open", []string{url}
	case "windows":
		return "rundll32", []string{"url.dll,FileProtocolHandler", url}
	default:
		return "xdg-open", []string{url}
	}
}

// OpenBrowser launches the system browser.
func OpenBrowser(url string) error {
	name, args := browserCommand(runtime.GOOS, url)
	return exec.Command(name, args...).Start()
}

// Refresh renews a stored login using its refresh token.
//
// It is the whole reason a refresh token is worth its risk: without one, an
// expiring session means signing in again in the middle of whatever you were
// doing, which in practice means people reach for a long-lived static token
// instead — trading an hour-long credential for a permanent one.
//
// A provider that rotates refresh tokens returns a new one, and it is stored.
// Missing that would work perfectly until the old token was invalidated and
// then fail with no obvious cause, so the new value is taken whenever one is
// offered and the previous one kept when it is not.
func (c *Client) Refresh(ctx context.Context, cred Credential) (Credential, error) {
	if !cred.CanRefresh() {
		return cred, errors.New("this login cannot be renewed; sign in again")
	}

	provider, err := oidc.NewProvider(ctx, cred.Issuer)
	if err != nil {
		return cred, fmt.Errorf("reach the identity provider at %s: %w", cred.Issuer, err)
	}
	endpoint := provider.Endpoint()
	endpoint.AuthStyle = oauth2.AuthStyleInParams

	conf := &oauth2.Config{ClientID: cred.ClientID, Endpoint: endpoint}
	token, err := conf.TokenSource(ctx, &oauth2.Token{RefreshToken: cred.RefreshToken}).Token()
	if err != nil {
		// The usual causes are a refresh token that expired, one that was
		// revoked, and a session an administrator ended. None of them are
		// fixable by retrying, so this says to sign in rather than looking
		// like a transient failure.
		return cred, fmt.Errorf("could not renew this session (%w); run `modelforgectl login` again", err)
	}

	rawID, _ := token.Extra("id_token").(string)
	if rawID == "" {
		// Some providers only return an id_token on the initial exchange. That
		// leaves nothing this server can authenticate with, so it is a failure
		// rather than a partial success.
		return cred, errors.New("the provider renewed the session but returned no id_token; sign in again")
	}
	if _, err := provider.Verifier(&oidc.Config{SkipClientIDCheck: true}).Verify(ctx, rawID); err != nil {
		return cred, fmt.Errorf("the renewed token did not verify: %w", err)
	}

	next := cred
	next.IDToken = rawID
	next.Expiry = token.Expiry
	if token.RefreshToken != "" {
		next.RefreshToken = token.RefreshToken
	}
	return next, nil
}

// EnsureFresh renews the stored credential if it is about to expire, and
// returns the token to use.
//
// The refresh happens under a file lock and re-reads the credential inside it,
// so a process that blocked waiting for the lock uses whatever the winner just
// wrote rather than refreshing again with a token that has since been rotated
// away.
func (c *Client) EnsureFresh(ctx context.Context) error {
	cred := LoadCredential()
	if !cred.Valid() || !cred.Stale(time.Now()) || !cred.CanRefresh() {
		return nil
	}

	return withCredentialLock(func() error {
		cred := LoadCredential()
		if !cred.Valid() || !cred.Stale(time.Now()) {
			// Somebody else refreshed while this process waited for the lock.
			if cred.Valid() {
				c.Token = cred.IDToken
			}
			return nil
		}

		next, err := c.Refresh(ctx, cred)
		if err != nil {
			return err
		}
		if err := SaveCredential(next); err != nil {
			return err
		}
		c.Token = next.IDToken
		return nil
	})
}

// revoke asks the provider to invalidate a token, if it supports RFC 7009.
//
// Deleting the local file was never revocation — the token stayed valid at the
// provider until it expired, and a refresh token can be good for weeks. Adding
// a longer-lived secret without also adding a way to withdraw it would be the
// wrong trade, so logout now tries this first.
//
// A provider with no revocation endpoint is reported rather than glossed,
// because "logged out" meaning two different things depending on the provider
// is exactly the sort of ambiguity that gets somebody in trouble.
func revoke(ctx context.Context, issuer, clientID, token string) error {
	var meta struct {
		RevocationEndpoint string `json:"revocation_endpoint"`
	}
	req, err := http.NewRequestWithContext(ctx, "GET",
		strings.TrimRight(issuer, "/")+"/.well-known/openid-configuration", nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("reach the identity provider: %w", err)
	}
	defer resp.Body.Close()
	if err := json.NewDecoder(resp.Body).Decode(&meta); err != nil {
		return fmt.Errorf("read the provider metadata: %w", err)
	}
	if meta.RevocationEndpoint == "" {
		return errNoRevocationEndpoint
	}

	form := url.Values{"token": {token}, "client_id": {clientID}}
	rreq, err := http.NewRequestWithContext(ctx, "POST", meta.RevocationEndpoint,
		strings.NewReader(form.Encode()))
	if err != nil {
		return err
	}
	rreq.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	rresp, err := http.DefaultClient.Do(rreq)
	if err != nil {
		return fmt.Errorf("call the revocation endpoint: %w", err)
	}
	defer rresp.Body.Close()
	io.Copy(io.Discard, rresp.Body) //nolint:errcheck // draining for connection reuse

	// RFC 7009 says a provider returns 200 for a token it does not recognise,
	// so anything else is a real failure rather than "already gone".
	if rresp.StatusCode != http.StatusOK {
		return fmt.Errorf("the provider returned %d from its revocation endpoint", rresp.StatusCode)
	}
	return nil
}

var errNoRevocationEndpoint = errors.New("the provider advertises no revocation endpoint")

// --- server metadata ---
// --- server metadata ---

type authMeta struct {
	Login    bool   `json:"login"`
	Reason   string `json:"reason"`
	Issuer   string `json:"issuer"`
	ClientID string `json:"client_id"`
	Audience string `json:"audience"`
}

func (c *Client) authConfig() (authMeta, error) {
	var meta authMeta
	// Deliberately not through c.do: this endpoint is unauthenticated, and a
	// client that has not logged in yet has nothing to send.
	resp, err := c.HTTP.Get(c.BaseURL + "/v1/auth/config")
	if err != nil {
		return meta, fmt.Errorf("ask the server where to sign in: %w", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 300 {
		return meta, serverError(resp.StatusCode, raw)
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		return meta, err
	}
	return meta, nil
}

type whoAmI struct {
	Name    string   `json:"name"`
	Kind    string   `json:"kind"`
	Scopes  []string `json:"scopes"`
	Subject string   `json:"subject"`
	Issuer  string   `json:"issuer"`
	Email   string   `json:"email"`
	Expires string   `json:"expires"`
}

func (c *Client) whoAmI() (whoAmI, error) {
	var who whoAmI
	err := c.get("/v1/auth/whoami", &who)
	return who, err
}

func scopeNames(scopes []string) []string {
	if len(scopes) == 0 {
		return []string{"no scopes"}
	}
	return scopes
}

// WhoAmI prints the current identity.
func (c *Client) WhoAmI() error {
	if c.Token == "" {
		fmt.Fprintln(c.Out, "not signed in (no MODELFORGE_TOKEN and no stored login)")
		return nil
	}
	who, err := c.whoAmI()
	if err != nil {
		return err
	}
	fmt.Fprintf(c.Out, "%s (%s)\n", who.Name, who.Kind)
	fmt.Fprintf(c.Out, "  scopes   %s\n", strings.Join(scopeNames(who.Scopes), ", "))
	if who.Subject != "" {
		fmt.Fprintf(c.Out, "  subject  %s\n", who.Subject)
		fmt.Fprintf(c.Out, "  issuer   %s\n", who.Issuer)
	}
	if who.Expires != "" {
		fmt.Fprintf(c.Out, "  expires  %s\n", who.Expires)
	}
	return nil
}

// Logout revokes the session at the provider, then removes it locally.
//
// Revocation is attempted first and its outcome reported honestly. Deleting the
// file alone was never logging out — the tokens stayed valid at the provider
// until they expired, and a refresh token can be good for weeks. A tool that
// says "logged out" while leaving a working credential in existence is telling
// somebody they are safe when they are not.
func (c *Client) Logout(ctx context.Context) error {
	cred := LoadCredential()

	if cred.CanRefresh() {
		// The refresh token is the one that matters: revoking it ends the
		// session, while the ID token expires on its own within the hour.
		switch err := revoke(ctx, cred.Issuer, cred.ClientID, cred.RefreshToken); {
		case err == nil:
			fmt.Fprintln(c.Out, "revoked this session at the identity provider")
		case errors.Is(err, errNoRevocationEndpoint):
			fmt.Fprintln(c.Out, "the identity provider advertises no revocation endpoint, so the session "+
				"could not be ended remotely; it stays valid until it expires")
		default:
			fmt.Fprintf(c.Out, "could not revoke at the identity provider: %v\n", err)
			fmt.Fprintln(c.Out, "the session stays valid until it expires; revoke it there if it may have been exposed")
		}
	}

	removed, err := DeleteCredential()
	if err != nil {
		return err
	}
	if !removed {
		fmt.Fprintln(c.Out, "no stored login to remove")
		return nil
	}
	fmt.Fprintln(c.Out, "removed the stored login from this machine")
	return nil
}
