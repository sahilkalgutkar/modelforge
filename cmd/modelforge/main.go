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
	flag.BoolVar(&cfg.TrustForwardedFor, "trust-forwarded-for", os.Getenv("MODELFORGE_TRUST_FORWARDED_FOR") == "true",
		"rate-limit on X-Forwarded-For instead of the socket address; only safe behind a proxy that overwrites it")
	flag.Parse()

	cfg.AuthDisabled = *authDisabled
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
