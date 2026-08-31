package auth

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
)

type contextKey struct{}

// FromContext returns the token that authenticated a request, if any.
func FromContext(ctx context.Context) (Token, bool) {
	t, ok := ctx.Value(contextKey{}).(Token)
	return t, ok
}

// Middleware enforces a scope on the handlers it wraps.
type Middleware struct {
	auth *Authenticator
	log  *slog.Logger
}

// NewMiddleware builds a Middleware.
func NewMiddleware(a *Authenticator, log *slog.Logger) *Middleware {
	if log == nil {
		log = slog.Default()
	}
	return &Middleware{auth: a, log: log}
}

// Require wraps a handler so that only tokens carrying scope reach it, and puts
// the resolved token on the request context for audit logging.
//
// The 401/403 split is the standard one and is worth keeping straight, because
// the two have opposite remediations. A missing or unrecognised token is 401:
// the caller has not proved who they are, and retrying with a credential is the
// fix. A valid token without the scope is 403: the caller has proved who they
// are and the answer is still no, so retrying is pointless and somebody has to
// grant them the scope. Collapsing both into 403 sends operators looking for a
// permissions problem when the real one is an unset environment variable.
func (m *Middleware) Require(scope Scope, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tok, err := m.auth.Authenticate(r)
		if err != nil {
			// WWW-Authenticate is what tells a client this is an
			// authentication challenge rather than a generic refusal, and RFC
			// 7235 requires it on a 401.
			w.Header().Set("WWW-Authenticate", `Bearer realm="modelforge"`)
			m.log.Warn("rejected unauthenticated request",
				"method", r.Method, "path", r.URL.Path,
				"reason", reasonOf(err), "remote", remoteHost(r))
			writeErr(w, http.StatusUnauthorized, err.Error())
			return
		}

		if !tok.Allows(scope) {
			// Logged at Warn with the token name: a credential repeatedly
			// reaching for a scope it does not have is either a misconfigured
			// deploy or somebody probing, and both are worth seeing.
			m.log.Warn("rejected request outside token scope",
				"token", tok.Name, "have", tok.Scopes, "need", scope,
				"method", r.Method, "path", r.URL.Path)
			writeErr(w, http.StatusForbidden,
				ErrForbidden.Error()+": need "+string(scope))
			return
		}

		// Control-plane changes are recorded with the name of the credential
		// that made them. Without this, "who changed what serves traffic" is
		// answerable only by whoever still has shell history, which during an
		// incident is nobody.
		//
		// The condition is the admin scope and not simply a writing method,
		// because scoring is a POST. Keying on the method alone logged a line
		// per prediction — at serving volume that is millions of audit entries
		// a day, burying the handful that record an actual change and costing
		// real money to store. The scope is the honest test: admin is exactly
		// the set of routes that alter what serves traffic.
		if isMutation(r.Method) && scope == ScopeAdmin && !m.auth.IsDisabled() {
			m.log.Info("authorised change",
				"token", tok.Name, "method", r.Method, "path", r.URL.Path)
		}

		next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), contextKey{}, tok)))
	})
}

// RequireFunc is Require for a HandlerFunc.
func (m *Middleware) RequireFunc(scope Scope, next http.HandlerFunc) http.Handler {
	return m.Require(scope, next)
}

func isMutation(method string) bool {
	switch method {
	case http.MethodPost, http.MethodPut, http.MethodPatch, http.MethodDelete:
		return true
	}
	return false
}

func reasonOf(err error) string {
	switch {
	case errors.Is(err, ErrNoCredential):
		return "no_credential"
	case errors.Is(err, ErrBadCredential):
		return "bad_credential"
	default:
		return "unknown"
	}
}

// remoteHost is logged so a burst of rejected requests can be attributed. It is
// RemoteAddr rather than a forwarded header on purpose: X-Forwarded-For is
// caller-controlled, and trusting it here would let anybody write whatever they
// liked into the audit log.
func remoteHost(r *http.Request) string { return r.RemoteAddr }

func writeErr(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	//nolint:errcheck // the response is already committed
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}
