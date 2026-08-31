// Command modelforge serves models: a registry, a scoring API, and the traffic
// controls that make replacing a model a gradual act.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/sahilkalgutkar/modelforge/internal/app"
)

func main() {
	cfg := app.Config{}
	flag.StringVar(&cfg.Addr, "addr", envOr("MODELFORGE_ADDR", ":8080"), "address to listen on")
	flag.StringVar(&cfg.DatabaseURL, "database-url", os.Getenv("MODELFORGE_DATABASE_URL"), "Postgres connection string")
	flag.StringVar(&cfg.ArtifactDir, "artifact-dir", envOr("MODELFORGE_ARTIFACT_DIR", "./var/artifacts"), "directory for model artifacts")
	flag.DurationVar(&cfg.ShadowTimeout, "shadow-timeout", 2*time.Second, "how long a shadow request may run")
	flag.DurationVar(&cfg.DriftInterval, "drift-interval", 30*time.Second, "how often drift readings refresh")
	tokens := flag.String("tokens", os.Getenv("MODELFORGE_TOKENS"),
		"comma-separated API tokens as name:scopes:sha256hex (mint with `modelforgectl token`)")
	flag.StringVar(&cfg.TokenFile, "token-file", os.Getenv("MODELFORGE_TOKEN_FILE"),
		"path to reloadable token entries, one per line; re-read on SIGHUP and on a timer")
	flag.DurationVar(&cfg.TokenReloadInterval, "token-reload-interval", 30*time.Second,
		"how often to re-read -token-file (negative disables polling; SIGHUP always works)")
	authDisabled := flag.Bool("auth-disabled", os.Getenv("MODELFORGE_AUTH_DISABLED") == "true",
		"serve with no authentication at all")
	flag.IntVar(&cfg.AuthMaxFailures, "auth-max-failures", 10,
		"authentication failures one client may make before being throttled (negative disables throttling)")
	flag.DurationVar(&cfg.AuthFailureWindow, "auth-failure-window", time.Minute,
		"how long an exhausted authentication-failure budget takes to refill")
	flag.StringVar(&cfg.OIDCIssuer, "oidc-issuer", os.Getenv("MODELFORGE_OIDC_ISSUER"),
		"identity provider URL, enabling per-user login alongside static tokens")
	flag.StringVar(&cfg.OIDCAudience, "oidc-audience", os.Getenv("MODELFORGE_OIDC_AUDIENCE"),
		"value a token's aud claim must contain; required with -oidc-issuer")
	flag.StringVar(&cfg.OIDCClientID, "oidc-client-id", os.Getenv("MODELFORGE_OIDC_CLIENT_ID"),
		"public OAuth client id the CLI logs in as; enables `modelforgectl login`")
	flag.StringVar(&cfg.OIDCGroupsClaim, "oidc-groups-claim", envOr("MODELFORGE_OIDC_GROUPS_CLAIM", "groups"),
		"claim carrying group membership")
	scopeMap := flag.String("oidc-scope-map", os.Getenv("MODELFORGE_OIDC_SCOPE_MAP"),
		"comma-separated group=scope[+scope] entries, e.g. platform-oncall=admin,ml-eng=read")
	flag.StringVar(&cfg.ExternalURL, "external-url", os.Getenv("MODELFORGE_EXTERNAL_URL"),
		"how a browser reaches this server, e.g. https://modelforge.example.com; enables browser sessions")
	flag.DurationVar(&cfg.SessionTTL, "session-ttl", 12*time.Hour,
		"how long a browser session lasts, capped by the identity token's own expiry")
	flag.BoolVar(&cfg.InsecureCookies, "insecure-cookies", os.Getenv("MODELFORGE_INSECURE_COOKIES") == "true",
		"drop the Secure attribute from session cookies; local HTTP development only")
	flag.BoolVar(&cfg.TrustForwardedFor, "trust-forwarded-for", os.Getenv("MODELFORGE_TRUST_FORWARDED_FOR") == "true",
		"rate-limit on X-Forwarded-For instead of the socket address; only safe behind a proxy that overwrites it")
	flag.Parse()

	cfg.AuthDisabled = *authDisabled
	for _, e := range strings.Split(*scopeMap, ",") {
		if e = strings.TrimSpace(e); e != "" {
			cfg.OIDCScopeMap = append(cfg.OIDCScopeMap, e)
		}
	}
	for _, t := range strings.Split(*tokens, ",") {
		if t = strings.TrimSpace(t); t != "" {
			cfg.Tokens = append(cfg.Tokens, t)
		}
	}

	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	cfg.Logger = logger

	if cfg.DatabaseURL == "" {
		fmt.Fprintln(os.Stderr, "modelforge: -database-url (or MODELFORGE_DATABASE_URL) is required")
		os.Exit(2)
	}

	// The signal context is installed before anything is opened, so a Ctrl-C
	// during a slow startup still exits rather than being ignored until the
	// server is up.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	a, err := app.New(ctx, cfg)
	if err != nil {
		logger.Error("startup failed", "error", err)
		os.Exit(1)
	}
	if err := a.Run(ctx); err != nil {
		logger.Error("server failed", "error", err)
		os.Exit(1)
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
