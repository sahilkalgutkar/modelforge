// Package auth is bearer-token authentication and scope authorisation for the
// API.
//
// The server holds SHA-256 hashes of tokens, never the tokens themselves, so a
// leaked config file or a leaked environment does not hand anybody a working
// credential. Tokens are minted by `modelforgectl token`, printed once, and
// only their digest is ever configured.
//
// Plain SHA-256 is the right primitive here, and that is worth stating because
// the reflex is to reach for bcrypt or argon2. Those are deliberately slow to
// make brute force expensive over the small, guessable space human-chosen
// passwords occupy. A token from this package is 256 bits of cryptographic
// randomness: there is no space to brute force, so a slow hash would buy
// nothing and would put its cost on every single request. The threat a password
// hash defends against does not exist here.
package auth

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync/atomic"
	"time"
)

// Scope is a capability a token carries.
type Scope string

const (
	// ScopePredict allows calling the serving endpoint.
	ScopePredict Scope = "predict"
	// ScopeRead allows reading models, versions, policies, drift and metrics.
	ScopeRead Scope = "read"
	// ScopeAdmin allows everything, including changing what serves traffic.
	ScopeAdmin Scope = "admin"
)

// AllScopes is every scope, for validation and help text.
var AllScopes = []Scope{ScopePredict, ScopeRead, ScopeAdmin}

// Errors callers are expected to handle.
var (
	// ErrNoCredential means the request carried no usable Authorization
	// header.
	ErrNoCredential = errors.New("auth: no bearer token")
	// ErrBadCredential means a token was presented and is not recognised.
	ErrBadCredential = errors.New("auth: unrecognised token")
	// ErrForbidden means the token is valid but lacks the required scope.
	ErrForbidden = errors.New("auth: token lacks the required scope")
	// ErrExpiredCredential means the token was recognised but is past its
	// deadline.
	//
	// It is deliberately distinct from ErrBadCredential. Saying "expired"
	// tells the holder something they already know — they possess the token —
	// so it leaks nothing, and it is the difference between an operator
	// rotating a credential in a minute and hunting a phantom typo for an
	// hour.
	ErrExpiredCredential = errors.New("auth: token has expired")
)

// Token is a configured credential. It holds no secret — only the name used in
// logs and the scopes granted.
type Token struct {
	Name   string
	Scopes []Scope

	// Subject, Issuer and Email carry a human identity when the credential
	// came from an identity provider, and are empty for a static token.
	//
	// Subject is the stable identifier and Name is the readable one, kept
	// separately on purpose: an email is what makes an audit log answerable
	// without a directory lookup, but it changes when somebody's surname does,
	// and a subject does not. Recording both means the log stays readable and
	// still joins correctly a year later.
	Subject string
	Issuer  string
	Email   string

	// NotAfter is when this credential stops being accepted. The zero value
	// means it never expires.
	//
	// It is checked on every request rather than only at load time, so a token
	// stops working at its deadline whether or not anybody reloads the file.
	// That is what makes "we will clean up the old credential later" safe:
	// later happens on its own.
	NotAfter time.Time
}

// Expired reports whether the token is past its deadline.
func (t Token) Expired(now time.Time) bool {
	return !t.NotAfter.IsZero() && now.After(t.NotAfter)
}

// Allows reports whether the token carries a scope.
//
// admin implies read and predict. An operator who can replace the model that
// answers every request is not meaningfully restrained by being unable to read
// which model that is, so making admin a strict superset avoids the failure
// mode where a deploy script works until it tries to check its own result.
func (t Token) Allows(want Scope) bool {
	for _, s := range t.Scopes {
		if s == want || s == ScopeAdmin {
			return true
		}
	}
	return false
}

// Human reports whether this credential identifies a person rather than a
// service. It is what lets an audit line distinguish "the CI token did this"
// from "this person did this".
func (t Token) Human() bool { return t.Subject != "" }

// String renders a token for logs. It deliberately cannot print a secret,
// because there is none in the struct to print.
func (t Token) String() string {
	names := make([]string, len(t.Scopes))
	for i, s := range t.Scopes {
		names[i] = string(s)
	}
	return fmt.Sprintf("%s[%s]", t.Name, strings.Join(names, "+"))
}

// tokenSet is one immutable generation of the configured tokens.
//
// byDigest maps the hex SHA-256 of a token to what it grants. Looking a token
// up by the digest of what the caller presented is what makes the comparison
// timing-safe. The classic leak is comparing a secret byte by byte and
// returning early, which lets an attacker discover it one character at a time.
// Here the caller's guess is hashed first, so changing one bit of the guess
// changes the whole digest and a near-miss looks exactly like a wild miss.
type tokenSet struct {
	byDigest map[string]Token
	loadedAt time.Time
}

// Authenticator checks bearer tokens against a configured set.
//
// The set is replaced wholesale rather than mutated, and swapped through an
// atomic pointer. Two things follow from that, both of which matter on a
// serving path.
//
// The read side takes no lock, so authenticating a request costs a hash and a
// map lookup no matter how often tokens are rotated — a mutex here would put
// every request behind whatever a reload is doing.
//
// And a request sees exactly one generation of the set. Mutating a shared map
// in place would let a request arriving mid-rotation observe a state where the
// new token is present and the old one is already gone, which is precisely the
// window rotation exists to avoid.
type Authenticator struct {
	set atomic.Pointer[tokenSet]

	// disabled runs the server with no authentication at all. It exists so
	// that turning auth off is a visible, deliberate act recorded in the
	// process's own configuration, rather than something that happens by
	// default when nobody sets a token.
	disabled bool

	now func() time.Time

	// oidc authenticates JWTs from an identity provider, when one is
	// configured. Static tokens and per-user logins coexist deliberately: a
	// service calling the serving endpoint a thousand times a second wants a
	// credential it holds, not an interactive login, and a person changing
	// what serves production traffic should be named in the audit log rather
	// than sharing a token with everybody else who has it.
	oidc *OIDCVerifier
}

// WithOIDC adds per-user authentication against an identity provider.
func WithOIDC(v *OIDCVerifier) Option {
	return func(a *Authenticator) { a.oidc = v }
}

// Option configures an Authenticator.
type Option func(*Authenticator)

// WithClock replaces the clock used for expiry, for tests.
func WithClock(now func() time.Time) Option {
	return func(a *Authenticator) { a.now = now }
}

// Disabled builds an Authenticator that allows everything.
func Disabled() *Authenticator {
	a := &Authenticator{disabled: true, now: time.Now}
	a.set.Store(&tokenSet{byDigest: map[string]Token{}})
	return a
}

// VerifyIDToken authenticates a raw ID token, applying the same scope mapping
// an API request would get.
//
// The browser login uses this rather than its own path, so a cookie session can
// never carry an identity a bearer token could not. Two verification routines
// would be two places for the rules to drift apart, and the weaker one becomes
// a second front door.
func (a *Authenticator) VerifyIDToken(ctx context.Context, raw string) (Token, error) {
	if a.oidc == nil {
		return Token{}, ErrNoIdentityProvider
	}
	return a.oidc.Verify(ctx, raw)
}

// OIDC returns the identity provider verifier, or nil if none is configured.
func (a *Authenticator) OIDC() *OIDCVerifier { return a.oidc }

// IsDisabled reports whether authentication is off.
func (a *Authenticator) IsDisabled() bool { return a.disabled }

// Tokens returns the configured tokens, for startup logging. It exposes names
// and scopes only.
func (a *Authenticator) Tokens() []Token {
	current := a.set.Load()
	out := make([]Token, 0, len(current.byDigest))
	for _, t := range current.byDigest {
		out = append(out, t)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// New builds an Authenticator from configured token entries.
//
// Each entry is `name:scopes:sha256hex`, with scopes joined by `+`:
//
//	ci:admin:9f86d081...  dashboard:read+predict:3973e022...
//
// It returns an error rather than an empty Authenticator when given nothing,
// because an Authenticator with no tokens would reject every request while
// looking configured — a failure that presents as "the API is broken" rather
// than as "you did not set this up".
func New(entries []string, opts ...Option) (*Authenticator, error) {
	a := &Authenticator{now: time.Now}
	for _, o := range opts {
		o(a)
	}

	// With an identity provider configured, having no static tokens is a
	// legitimate deployment rather than a misconfiguration: every caller is a
	// person logging in. Without one it is the mistake buildSet refuses.
	if len(entries) == 0 && a.oidc != nil {
		a.set.Store(&tokenSet{byDigest: map[string]Token{}, loadedAt: a.now()})
		return a, nil
	}

	set, err := buildSet(entries, a.now())
	if err != nil {
		return nil, err
	}
	a.set.Store(set)
	return a, nil
}

// Change describes what a reload did, for logging. It carries token names and
// never any secret.
type Change struct {
	Added   []string
	Removed []string
	// Expired lists tokens that are configured but already past their
	// deadline. They are loaded and then refused, rather than dropped, so that
	// a caller presenting one gets "expired" instead of "unrecognised".
	Expired []string
	Total   int
}

// Empty reports whether the reload changed nothing.
func (c Change) Empty() bool { return len(c.Added) == 0 && len(c.Removed) == 0 }

// Reload replaces the token set, and reports what changed.
//
// It builds and validates the whole new set before swapping, so a malformed or
// empty configuration leaves the running set untouched. That is the property
// the whole feature depends on: a bad rotation should be a failed reload that
// gets logged, not an outage that locks every client out of a server that is
// otherwise healthy. The error is returned for the caller to log; the old
// credentials keep working either way.
func (a *Authenticator) Reload(entries []string) (Change, error) {
	if a.disabled {
		return Change{}, errors.New("auth: cannot reload tokens while authentication is disabled")
	}

	next, err := buildSet(entries, a.now())
	if err != nil {
		return Change{}, err
	}

	previous := a.set.Swap(next)
	return diff(previous, next, a.now()), nil
}

// LoadedAt is when the current token set was installed.
func (a *Authenticator) LoadedAt() time.Time { return a.set.Load().loadedAt }

// Count is how many tokens are configured.
func (a *Authenticator) Count() int { return len(a.set.Load().byDigest) }

func diff(previous, next *tokenSet, now time.Time) Change {
	c := Change{Total: len(next.byDigest)}

	for d, t := range next.byDigest {
		if _, existed := previous.byDigest[d]; !existed {
			c.Added = append(c.Added, t.Name)
		}
		if t.Expired(now) {
			c.Expired = append(c.Expired, t.Name)
		}
	}
	for d, t := range previous.byDigest {
		if _, kept := next.byDigest[d]; !kept {
			c.Removed = append(c.Removed, t.Name)
		}
	}
	sort.Strings(c.Added)
	sort.Strings(c.Removed)
	sort.Strings(c.Expired)
	return c
}

// buildSet parses entries into an immutable token set.
//
// It returns an error rather than an empty set when given nothing, because a
// set with no tokens would reject every request while looking configured — a
// failure that presents as "the API is broken" rather than as "you did not set
// this up", and one that a reload must never be able to install.
func buildSet(entries []string, now time.Time) (*tokenSet, error) {
	set := &tokenSet{byDigest: make(map[string]Token, len(entries)), loadedAt: now}

	for _, raw := range entries {
		entry := strings.TrimSpace(raw)
		if entry == "" || strings.HasPrefix(entry, "#") {
			continue
		}
		tok, digest, err := parseEntry(entry)
		if err != nil {
			return nil, err
		}
		if existing, dup := set.byDigest[digest]; dup {
			// Two names sharing a digest means the same secret was issued
			// twice, so revoking one would silently leave the other working.
			return nil, fmt.Errorf("auth: tokens %q and %q have the same digest", existing.Name, tok.Name)
		}
		set.byDigest[digest] = tok
	}

	if len(set.byDigest) == 0 {
		return nil, errors.New("auth: no tokens configured")
	}
	return set, nil
}

func parseEntry(entry string) (Token, string, error) {
	// SplitN with a limit of 4 so an RFC 3339 expiry keeps its own colons.
	// Splitting on every colon would turn 2026-12-01T00:00:00Z into three more
	// fields and reject every entry that carries a deadline.
	parts := strings.SplitN(entry, ":", 4)
	if len(parts) < 3 {
		return Token{}, "", fmt.Errorf("auth: token entry %q must be name:scopes:sha256hex[:expiry]", redactEntry(entry))
	}

	name := strings.TrimSpace(parts[0])
	if name == "" {
		return Token{}, "", errors.New("auth: token entry has an empty name")
	}

	var scopes []Scope
	for _, s := range strings.Split(parts[1], "+") {
		scope := Scope(strings.TrimSpace(s))
		if !validScope(scope) {
			return Token{}, "", fmt.Errorf("auth: token %q has unknown scope %q (want %s)",
				name, scope, scopeList())
		}
		scopes = append(scopes, scope)
	}
	if len(scopes) == 0 {
		return Token{}, "", fmt.Errorf("auth: token %q has no scopes", name)
	}

	digest := strings.ToLower(strings.TrimSpace(parts[2]))
	if len(digest) != sha256.Size*2 {
		return Token{}, "", fmt.Errorf("auth: token %q needs a %d-character sha256 hex digest, got %d "+
			"(configure the digest, never the token itself)", name, sha256.Size*2, len(digest))
	}
	if _, err := hex.DecodeString(digest); err != nil {
		return Token{}, "", fmt.Errorf("auth: token %q has a non-hex digest", name)
	}

	tok := Token{Name: name, Scopes: scopes}
	if len(parts) == 4 {
		raw := strings.TrimSpace(parts[3])
		if raw != "" {
			expiry, err := time.Parse(time.RFC3339, raw)
			if err != nil {
				// Refused rather than treated as "no expiry". A typo in a
				// deadline would otherwise produce a credential that never
				// expires, which is the opposite of what was written down.
				return Token{}, "", fmt.Errorf("auth: token %q has an unparseable expiry %q, want RFC 3339 "+
					"like 2026-12-01T00:00:00Z: %w", name, raw, err)
			}
			tok.NotAfter = expiry
		}
	}
	return tok, digest, nil
}

// redactEntry keeps a malformed entry out of the logs in one piece. A
// misconfigured entry is exactly the case where somebody has pasted the raw
// token where the digest belongs, and echoing it into an error message would
// then write the secret to wherever the logs go.
func redactEntry(entry string) string {
	if i := strings.Index(entry, ":"); i > 0 {
		return entry[:i] + ":<redacted>"
	}
	return "<redacted>"
}

func validScope(s Scope) bool {
	for _, known := range AllScopes {
		if s == known {
			return true
		}
	}
	return false
}

func scopeList() string {
	names := make([]string, len(AllScopes))
	for i, s := range AllScopes {
		names[i] = string(s)
	}
	return strings.Join(names, ", ")
}

// Digest returns the hex SHA-256 of a token, which is what gets configured.
func Digest(token string) string {
	sum := sha256.Sum256([]byte(token))
	return hex.EncodeToString(sum[:])
}

// GenerateToken mints a new token: 32 bytes of cryptographic randomness,
// base64url-encoded so it survives a header, a shell and a YAML file unescaped.
func GenerateToken() (string, error) {
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		return "", fmt.Errorf("auth: generate token: %w", err)
	}
	return base64.RawURLEncoding.EncodeToString(buf), nil
}

// Authenticate resolves the token on a request.
func (a *Authenticator) Authenticate(r *http.Request) (Token, error) {
	if a.disabled {
		return Token{Name: "anonymous", Scopes: []Scope{ScopeAdmin}}, nil
	}

	presented, ok := bearerToken(r)
	if !ok {
		return Token{}, ErrNoCredential
	}

	// Hashing the caller's guess before looking it up is what makes this
	// timing-safe, and it is worth being precise about why rather than
	// sprinkling a constant-time compare here for appearances.
	//
	// The attack a constant-time compare defends against is an early-exit
	// byte-by-byte comparison against a secret: an attacker sends a guess,
	// measures how long the comparison ran, and learns how many leading bytes
	// were right — recovering the secret one byte at a time. That attack needs
	// the comparison to be against the secret, in the attacker's own encoding.
	//
	// Here the guess is SHA-256'd first, so flipping one bit of it changes
	// every bit of the digest. A near-miss and a wild miss produce unrelated
	// lookup keys, there is no "how close was I" signal to measure, and the
	// only thing the timing can reveal is hit or miss — which the response
	// already says out loud. Adding subtle.ConstantTimeCompare after this
	// lookup would compare a value against itself and prove nothing.
	// A JWS has three dot-separated segments; the tokens this server mints are
	// base64url with no dots. The two are unambiguous without parsing either,
	// which keeps a mistyped static token from having an RSA signature
	// verification attempted against it — wasted work on exactly the requests
	// an attacker controls.
	if looksLikeJWT(presented) {
		if a.oidc == nil {
			return Token{}, fmt.Errorf("%w: this looks like a JWT, but no identity provider is configured",
				ErrBadCredential)
		}
		return a.oidc.Verify(r.Context(), presented)
	}

	tok, found := a.set.Load().byDigest[Digest(presented)]
	if !found {
		return Token{}, ErrBadCredential
	}
	if tok.Expired(a.now()) {
		return Token{}, fmt.Errorf("%w: %s expired at %s",
			ErrExpiredCredential, tok.Name, tok.NotAfter.UTC().Format(time.RFC3339))
	}
	return tok, nil
}

// bearerToken pulls the credential out of an Authorization header.
func bearerToken(r *http.Request) (string, bool) {
	h := r.Header.Get("Authorization")
	if h == "" {
		return "", false
	}
	// The scheme is case-insensitive per RFC 7235, and clients do send
	// "bearer" as well as "Bearer".
	const prefix = "bearer "
	if len(h) < len(prefix) || !strings.EqualFold(h[:len(prefix)], prefix) {
		return "", false
	}
	token := strings.TrimSpace(h[len(prefix):])
	if token == "" {
		return "", false
	}
	return token, true
}
