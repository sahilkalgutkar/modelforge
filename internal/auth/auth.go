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
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
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
)

// Token is a configured credential. It holds no secret — only the name used in
// logs and the scopes granted.
type Token struct {
	Name   string
	Scopes []Scope
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

// String renders a token for logs. It deliberately cannot print a secret,
// because there is none in the struct to print.
func (t Token) String() string {
	names := make([]string, len(t.Scopes))
	for i, s := range t.Scopes {
		names[i] = string(s)
	}
	return fmt.Sprintf("%s[%s]", t.Name, strings.Join(names, "+"))
}

// Authenticator checks bearer tokens against a configured set.
type Authenticator struct {
	// byDigest maps the hex SHA-256 of a token to what it grants.
	//
	// Looking a token up by the digest of what the caller presented is what
	// makes the comparison timing-safe. The classic leak is comparing a secret
	// byte by byte and returning early, which lets an attacker discover it one
	// character at a time. Here the caller's guess is hashed first, so changing
	// one bit of the guess changes the whole digest and a near-miss looks
	// exactly like a wild miss.
	byDigest map[string]Token

	// disabled runs the server with no authentication at all. It exists so
	// that turning auth off is a visible, deliberate act recorded in the
	// process's own configuration, rather than something that happens by
	// default when nobody sets a token.
	disabled bool
}

// Disabled builds an Authenticator that allows everything.
func Disabled() *Authenticator { return &Authenticator{disabled: true} }

// IsDisabled reports whether authentication is off.
func (a *Authenticator) IsDisabled() bool { return a.disabled }

// Tokens returns the configured tokens, for startup logging. It exposes names
// and scopes only.
func (a *Authenticator) Tokens() []Token {
	out := make([]Token, 0, len(a.byDigest))
	for _, t := range a.byDigest {
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
func New(entries []string) (*Authenticator, error) {
	a := &Authenticator{byDigest: make(map[string]Token, len(entries))}

	for _, raw := range entries {
		entry := strings.TrimSpace(raw)
		if entry == "" {
			continue
		}
		name, scopes, digest, err := parseEntry(entry)
		if err != nil {
			return nil, err
		}
		if existing, dup := a.byDigest[digest]; dup {
			// Two names sharing a digest means the same secret was issued
			// twice, so revoking one would silently leave the other working.
			return nil, fmt.Errorf("auth: tokens %q and %q have the same digest", existing.Name, name)
		}
		a.byDigest[digest] = Token{Name: name, Scopes: scopes}
	}

	if len(a.byDigest) == 0 {
		return nil, errors.New("auth: no tokens configured")
	}
	return a, nil
}

func parseEntry(entry string) (name string, scopes []Scope, digest string, err error) {
	parts := strings.Split(entry, ":")
	if len(parts) != 3 {
		return "", nil, "", fmt.Errorf("auth: token entry %q must be name:scopes:sha256hex", redactEntry(entry))
	}
	name = strings.TrimSpace(parts[0])
	if name == "" {
		return "", nil, "", errors.New("auth: token entry has an empty name")
	}

	for _, s := range strings.Split(parts[1], "+") {
		scope := Scope(strings.TrimSpace(s))
		if !validScope(scope) {
			return "", nil, "", fmt.Errorf("auth: token %q has unknown scope %q (want %s)",
				name, scope, scopeList())
		}
		scopes = append(scopes, scope)
	}
	if len(scopes) == 0 {
		return "", nil, "", fmt.Errorf("auth: token %q has no scopes", name)
	}

	digest = strings.ToLower(strings.TrimSpace(parts[2]))
	if len(digest) != sha256.Size*2 {
		return "", nil, "", fmt.Errorf("auth: token %q needs a %d-character sha256 hex digest, got %d "+
			"(configure the digest, never the token itself)", name, sha256.Size*2, len(digest))
	}
	if _, err := hex.DecodeString(digest); err != nil {
		return "", nil, "", fmt.Errorf("auth: token %q has a non-hex digest", name)
	}
	return name, scopes, digest, nil
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
	tok, found := a.byDigest[Digest(presented)]
	if !found {
		return Token{}, ErrBadCredential
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
