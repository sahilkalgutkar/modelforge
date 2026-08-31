package auth

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/url"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coreos/go-oidc/v3/oidc"
)

// ErrNoIdentityProvider means a JWT was presented but no provider is
// configured, or the configured one is not reachable yet.
var ErrNoIdentityProvider = errors.New("auth: no identity provider is available")

// OIDCConfig configures per-user authentication against an identity provider.
type OIDCConfig struct {
	// Issuer is the provider's base URL. Its discovery document supplies the
	// signing keys.
	Issuer string

	// Audience is the value a token's `aud` claim must contain.
	//
	// This is not optional and there is no "any audience" setting, because
	// skipping it is the classic confused-deputy bug: an identity provider
	// issues tokens for many services, and a token minted for the expense tool
	// would otherwise be a valid credential for the thing that decides which
	// model scores production traffic. The user is real, their token is real,
	// and they never intended to authenticate here.
	Audience string

	// GroupsClaim names the claim carrying a user's group or role
	// memberships. Providers disagree: Okta and Auth0 commonly use "groups",
	// Keycloak "roles", Entra ID "roles" or a mapped claim.
	GroupsClaim string

	// ScopeMap maps a group value to the scopes it grants, e.g.
	// {"platform-oncall": {ScopeAdmin}}. A user in no mapped group gets no
	// scopes and is refused — authenticated is not authorised.
	ScopeMap map[string][]Scope

	// Now is injectable for tests.
	Now func() time.Time

	// SkipIssuerCheck relaxes the issuer match. It exists for tests, which run
	// a provider on a loopback URL that will not equal the issuer a token
	// claims.
	SkipIssuerCheck bool

	// verifier bypasses discovery. Tests set it to a verifier built against
	// their own key set; nothing in cmd/ or internal/app can reach it.
	verifier *oidc.IDTokenVerifier
}

// OIDCVerifier authenticates JWTs from an identity provider.
//
// Discovery happens in the background rather than at startup. An identity
// provider being briefly unreachable should not stop this server from booting:
// the static tokens still work, the serving path still serves, and only
// per-user logins are affected — degrading one authentication method is a much
// smaller failure than refusing to start.
type OIDCVerifier struct {
	cfg      OIDCConfig
	log      *slog.Logger
	verifier atomic.Pointer[oidc.IDTokenVerifier]

	readyOnce sync.Once
	ready     chan struct{}
}

// Validate checks an OIDC configuration.
func (c OIDCConfig) Validate() error {
	if c.Issuer == "" {
		return errors.New("auth: an OIDC issuer URL is required")
	}
	if !strings.HasPrefix(c.Issuer, "https://") && !strings.HasPrefix(c.Issuer, "http://") {
		return fmt.Errorf("auth: OIDC issuer %q must be a URL", c.Issuer)
	}
	if strings.HasPrefix(c.Issuer, "http://") {
		// Permitted for a local provider in development, but worth saying out
		// loud: over plain HTTP the discovery document and the signing keys
		// can be replaced in transit, which is the whole trust anchor.
		if !isLoopback(c.Issuer) {
			return fmt.Errorf("auth: OIDC issuer %q uses http; the signing keys would be fetched over an "+
				"unauthenticated channel, so this is only allowed for loopback addresses", c.Issuer)
		}
	}
	if c.Audience == "" {
		return errors.New("auth: an OIDC audience is required, so tokens minted for other services are refused")
	}
	if len(c.ScopeMap) == 0 {
		return errors.New("auth: an OIDC scope map is required, or every authenticated user would have no access")
	}
	for group, scopes := range c.ScopeMap {
		if group == "" {
			return errors.New("auth: OIDC scope map has an empty group")
		}
		for _, s := range scopes {
			if !validScope(s) {
				return fmt.Errorf("auth: OIDC group %q maps to unknown scope %q (want %s)", group, s, scopeList())
			}
		}
	}
	return nil
}

func isLoopback(rawURL string) bool {
	u, err := url.Parse(rawURL)
	if err != nil {
		return false
	}
	host := u.Hostname()
	return host == "localhost" || host == "127.0.0.1" || host == "::1"
}

// NewOIDCVerifier starts a verifier and begins discovery in the background.
func NewOIDCVerifier(ctx context.Context, cfg OIDCConfig, log *slog.Logger) (*OIDCVerifier, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if cfg.GroupsClaim == "" {
		cfg.GroupsClaim = "groups"
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	if log == nil {
		log = slog.Default()
	}

	v := &OIDCVerifier{cfg: cfg, log: log, ready: make(chan struct{})}

	// Supplied directly by tests, which run their own provider.
	if cfg.verifier != nil {
		v.verifier.Store(cfg.verifier)
		v.markReady()
		return v, nil
	}

	go v.discover(ctx)
	return v, nil
}

// discover fetches the provider's metadata, retrying until it succeeds.
//
// The retry backs off because a provider that is down stays down for minutes,
// and hammering its discovery endpoint from every replica is a good way to keep
// it that way.
func (v *OIDCVerifier) discover(ctx context.Context) {
	backoff := time.Second
	const maxBackoff = time.Minute

	for {
		provider, err := oidc.NewProvider(ctx, v.cfg.Issuer)
		if err == nil {
			v.verifier.Store(provider.Verifier(&oidc.Config{
				ClientID: v.cfg.Audience,
				// go-oidc allowlists asymmetric algorithms by default. That
				// matters more than it looks: accepting HMAC here would let
				// anybody who can read the provider's *public* key sign tokens
				// with it, the classic algorithm-confusion bypass.
				SkipIssuerCheck: v.cfg.SkipIssuerCheck,
				Now:             v.cfg.Now,
			}))
			v.markReady()
			v.log.Info("identity provider ready", "issuer", v.cfg.Issuer, "audience", v.cfg.Audience)
			return
		}

		v.log.Error("identity provider discovery failed; per-user login is unavailable until it succeeds",
			"issuer", v.cfg.Issuer, "retry_in", backoff.String(), "error", err)

		select {
		case <-ctx.Done():
			return
		case <-time.After(backoff):
		}
		if backoff *= 2; backoff > maxBackoff {
			backoff = maxBackoff
		}
	}
}

func (v *OIDCVerifier) markReady() { v.readyOnce.Do(func() { close(v.ready) }) }

// Ready reports whether discovery has completed.
func (v *OIDCVerifier) Ready() bool {
	select {
	case <-v.ready:
		return true
	default:
		return false
	}
}

// WaitReady blocks until discovery completes or ctx is done. It exists for
// tests and for a caller that wants to fail fast at startup.
func (v *OIDCVerifier) WaitReady(ctx context.Context) error {
	select {
	case <-v.ready:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// claims is the subset of an ID token this server reads.
type claims struct {
	Subject           string `json:"sub"`
	Email             string `json:"email"`
	EmailVerified     *bool  `json:"email_verified"`
	PreferredUsername string `json:"preferred_username"`
	Name              string `json:"name"`
}

// Verify checks a JWT and turns it into a Token carrying the user's identity.
func (v *OIDCVerifier) Verify(ctx context.Context, raw string) (Token, error) {
	verifier := v.verifier.Load()
	if verifier == nil {
		return Token{}, ErrNoIdentityProvider
	}

	idToken, err := verifier.Verify(ctx, raw)
	if err != nil {
		// The provider's message is not passed through. It is detailed enough
		// to be a probing oracle — which claim failed, which key was tried —
		// and none of that helps a legitimate caller, who is holding a token
		// their own provider will explain.
		v.log.Debug("rejected an OIDC token", "error", err)
		return Token{}, fmt.Errorf("%w: token failed verification", ErrBadCredential)
	}

	var c claims
	if err := idToken.Claims(&c); err != nil {
		return Token{}, fmt.Errorf("%w: token claims are unreadable", ErrBadCredential)
	}
	if c.Subject == "" {
		// Without a subject there is no identity to record, which defeats the
		// point of per-user authentication.
		return Token{}, fmt.Errorf("%w: token has no subject", ErrBadCredential)
	}

	groups, err := extractGroups(idToken, v.cfg.GroupsClaim)
	if err != nil {
		return Token{}, err
	}

	scopes := v.scopesFor(groups)
	if len(scopes) == 0 {
		// Authenticated but not authorised. Defaulting to any access for a
		// valid token would mean every employee at the company gets whatever
		// the fallback is, on a service that decides what scores production
		// traffic.
		return Token{}, fmt.Errorf("%w: %s is in no group that grants access here",
			ErrForbidden, identityOf(c))
	}

	return Token{
		Name:     identityOf(c),
		Scopes:   scopes,
		Subject:  c.Subject,
		Issuer:   idToken.Issuer,
		Email:    c.Email,
		NotAfter: idToken.Expiry,
	}, nil
}

// identityOf picks the most human-readable stable name available.
//
// The subject is always present and always unique, but it is often an opaque
// UUID that means nothing in an audit log. An email or username is what makes
// "who changed the production model" answerable without a directory lookup, so
// it is preferred when the provider supplies one — with the subject recorded
// separately either way, since that is the identifier that does not change when
// somebody's surname does.
func identityOf(c claims) string {
	switch {
	case c.Email != "":
		return c.Email
	case c.PreferredUsername != "":
		return c.PreferredUsername
	default:
		return c.Subject
	}
}

// extractGroups reads the configured claim, tolerating the shapes providers
// actually emit: a list of strings, or a single string.
func extractGroups(idToken *oidc.IDToken, claim string) ([]string, error) {
	var raw map[string]any
	if err := idToken.Claims(&raw); err != nil {
		return nil, fmt.Errorf("%w: token claims are unreadable", ErrBadCredential)
	}

	value, present := raw[claim]
	if !present {
		return nil, nil
	}

	switch v := value.(type) {
	case string:
		// Some providers emit a single group as a bare string, and others emit
		// a space-separated list in the style of the `scope` claim.
		return strings.Fields(v), nil
	case []any:
		out := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out, nil
	default:
		return nil, fmt.Errorf("%w: claim %q is not a list of groups", ErrBadCredential, claim)
	}
}

// scopesFor maps a user's groups onto the scopes they grant, taking the union.
func (v *OIDCVerifier) scopesFor(groups []string) []Scope {
	seen := make(map[Scope]struct{})
	for _, g := range groups {
		for _, s := range v.cfg.ScopeMap[g] {
			seen[s] = struct{}{}
		}
	}
	if len(seen) == 0 {
		return nil
	}
	out := make([]Scope, 0, len(seen))
	for s := range seen {
		out = append(out, s)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// ParseScopeMap reads a scope map from `group=scope[+scope]` entries.
func ParseScopeMap(entries []string) (map[string][]Scope, error) {
	out := make(map[string][]Scope, len(entries))
	for _, raw := range entries {
		entry := strings.TrimSpace(raw)
		if entry == "" {
			continue
		}
		group, granted, found := strings.Cut(entry, "=")
		if !found {
			return nil, fmt.Errorf("auth: scope map entry %q must be group=scope[+scope]", entry)
		}
		group = strings.TrimSpace(group)
		if group == "" {
			return nil, fmt.Errorf("auth: scope map entry %q has an empty group", entry)
		}

		var scopes []Scope
		for _, s := range strings.Split(granted, "+") {
			scope := Scope(strings.TrimSpace(s))
			if !validScope(scope) {
				return nil, fmt.Errorf("auth: group %q maps to unknown scope %q (want %s)",
					group, scope, scopeList())
			}
			scopes = append(scopes, scope)
		}
		if len(scopes) == 0 {
			return nil, fmt.Errorf("auth: group %q maps to no scopes", group)
		}
		out[group] = scopes
	}
	return out, nil
}

// looksLikeJWT reports whether a bearer credential is a JWT rather than one of
// this server's own tokens.
//
// A JWS has three base64url segments separated by dots, and the tokens minted
// by `modelforgectl token` are raw base64url with no dots at all — so the two
// are unambiguous without parsing either. Trying both paths for every
// credential would work too, but it would mean a mistyped static token gets a
// signature verification attempted against it, which is wasted RSA work on
// exactly the requests an attacker controls.
func looksLikeJWT(token string) bool {
	return strings.Count(token, ".") == 2
}
