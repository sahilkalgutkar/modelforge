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
	"os"
	"os/exec"
	"path/filepath"
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
		Scopes:      []string{oidc.ScopeOpenID, "profile", "email", "groups"},
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

	if err := SaveCredential(rawID); err != nil {
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
	if who.Expires != "" {
		fmt.Fprintf(c.Out, "this session expires at %s\n", who.Expires)
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

// --- credential storage ---

// CredentialPath is where a login is stored.
func CredentialPath() (string, error) {
	if p := os.Getenv("MODELFORGE_CREDENTIAL_FILE"); p != "" {
		return p, nil
	}
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", fmt.Errorf("locate a config directory: %w", err)
	}
	return filepath.Join(dir, "modelforge", "credential"), nil
}

// SaveCredential writes a token to the credential file.
//
// The directory and the file are both created 0700/0600 and the mode is set
// explicitly rather than left to the umask, because a umask of 022 would leave
// a readable credential in a home directory that other accounts on the machine
// can list.
func SaveCredential(token string) error {
	path, err := CredentialPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return fmt.Errorf("create the credential directory: %w", err)
	}
	if err := os.WriteFile(path, []byte(token+"\n"), 0o600); err != nil {
		return fmt.Errorf("write the credential: %w", err)
	}
	// MkdirAll leaves an existing directory's mode alone, so a directory that
	// predates this — or was created by something with a laxer umask — is
	// tightened here rather than assumed.
	if err := os.Chmod(filepath.Dir(path), 0o700); err != nil {
		return fmt.Errorf("secure the credential directory: %w", err)
	}
	return os.Chmod(path, 0o600)
}

// LoadCredential reads a stored login, returning "" when there is none.
func LoadCredential() string {
	path, err := CredentialPath()
	if err != nil {
		return ""
	}
	body, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(body))
}

// DeleteCredential removes a stored login.
func DeleteCredential() (bool, error) {
	path, err := CredentialPath()
	if err != nil {
		return false, err
	}
	if err := os.Remove(path); err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, fmt.Errorf("remove the credential: %w", err)
	}
	return true, nil
}

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

// Logout removes a stored login.
func (c *Client) Logout() error {
	removed, err := DeleteCredential()
	if err != nil {
		return err
	}
	if !removed {
		fmt.Fprintln(c.Out, "no stored login to remove")
		return nil
	}
	// Worth saying plainly: this deletes a local file. The token remains valid
	// at the provider until it expires, and nothing here can recall it.
	fmt.Fprintln(c.Out, "removed the stored login from this machine")
	fmt.Fprintln(c.Out, "the token itself stays valid until it expires; revoke it at your identity provider "+
		"if it may have been exposed")
	return nil
}
