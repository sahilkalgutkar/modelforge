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

	// limiter is optional; nil means failed authentication is not throttled.
	limiter *Limiter
	// trustForwardedFor selects which address identifies a client. See
	// clientKey for why neither choice is safe to default.
	trustForwardedFor bool

	// OnAuthenticated is called after a credential is accepted, with the kind
	// of credential it was. It exists so the server can count service versus
	// user logins without this package depending on Prometheus.
	OnAuthenticated func(kind string)
}

// NewMiddleware builds a Middleware with no rate limiting.
func NewMiddleware(a *Authenticator, log *slog.Logger) *Middleware {
	if log == nil {
		log = slog.Default()
	}
	return &Middleware{auth: a, log: log}
}

// WithLimiter returns a Middleware that throttles clients failing
// authentication. It is a separate constructor rather than a field on the first
// so that adding a limiter is a visible change at the call site.
func WithLimiter(a *Authenticator, log *slog.Logger, l *Limiter, trustForwardedFor bool) *Middleware {
	m := NewMiddleware(a, log)
	m.limiter = l
	m.trustForwardedFor = trustForwardedFor
	return m
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
		var key string
		if m.limiter != nil {
			key = clientKey(r, m.trustForwardedFor)
		}

		// Authentication happens before the rate-limit check, so a valid
		// credential is never refused for throttling. Checking the limit first
		// would be marginally cheaper — it skips a SHA-256 over a short string,
		// which is noise next to the HTTP stack already traversed to get here —
		// but it would refuse a legitimate caller whose address is shared with
		// somebody failing. Behind a NAT or a shared egress that is a real
		// outage for an innocent client, bought with a saving too small to
		// measure. The threat being controlled is the cost of *failures*, and
		// nothing about that requires punishing a request that turned out to be
		// correct.
		tok, err := m.auth.Authenticate(r)

		// A credential that verified but grants nothing is a 403, not a 401.
		// The holder proved who they are — their signature checked out and
		// their provider vouched for them — they are simply in no group this
		// server grants access to. Reporting that as 401 would send a real
		// employee to re-authenticate over and over against a provider that is
		// working perfectly, when the actual fix is somebody adding them to a
		// group. It is the same distinction as a scope failure, and for the
		// same reason it does not spend rate-limit budget: this is a
		// misconfigured permission, not somebody guessing.
		if err != nil && errors.Is(err, ErrForbidden) {
			m.log.Warn("rejected an authenticated identity with no granted scopes",
				"method", r.Method, "path", r.URL.Path, "error", err)
			writeErr(w, http.StatusForbidden, err.Error())
			return
		}

		if err != nil {
			if m.limiter != nil {
				if ok, wait := m.limiter.Allow(key); !ok {
					w.Header().Set("Retry-After", retryAfter(wait))
					// Logging is suppressed while a client is throttled.
					// Log volume is the main thing this defends, so writing a
					// line per throttled request would be self-defeating.
					if shouldLog, suppressed := m.limiter.ShouldLog(key); shouldLog {
						m.log.Warn("throttling a client that keeps failing authentication",
							"remote", key, "retry_after", wait.String(), "suppressed", suppressed)
					}
					writeErr(w, http.StatusTooManyRequests,
						"too many failed authentication attempts; retry after "+retryAfter(wait)+"s")
					return
				}
				m.limiter.RecordFailure(key)
			}

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

		if m.limiter != nil {
			// The credential was good, so whatever this client got wrong
			// before is behind it.
			m.limiter.RecordSuccess(key)
		}
		if m.OnAuthenticated != nil {
			kind := "service"
			if tok.Human() {
				kind = "user"
			}
			m.OnAuthenticated(kind)
		}

		if !tok.Allows(scope) {
			// Deliberately not recorded as a rate-limit failure. This is a
			// valid credential refused for lacking a scope — a misconfigured
			// client, not somebody guessing — and throttling it would take a
			// deploy script offline for holding the wrong role.
			//
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
			attrs := []any{"actor", tok.Name, "method", r.Method, "path", r.URL.Path}
			if tok.Human() {
				// The subject and issuer are recorded alongside the readable
				// name because the name can change — somebody's surname, an
				// email alias — and the subject cannot. A log that only kept
				// the readable form stops joining to a directory the moment
				// anybody gets married.
				attrs = append(attrs, "subject", tok.Subject, "issuer", tok.Issuer, "kind", "user")
			} else {
				attrs = append(attrs, "kind", "service")
			}
			m.log.Info("authorised change", attrs...)
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
