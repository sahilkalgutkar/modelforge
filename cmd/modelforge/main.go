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
	flag.Parse()

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
