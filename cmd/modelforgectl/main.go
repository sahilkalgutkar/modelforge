// Command modelforgectl drives a modelforge server from a terminal or a deploy
// script.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/sahilkalgutkar/modelforge/internal/cli"
)

func main() {
	addr := flag.String("addr", envOr("MODELFORGE_ADDR", "http://localhost:8080"), "server address")
	flag.Usage = func() { fmt.Fprint(os.Stderr, cli.Usage) }
	flag.Parse()

	os.Exit(cli.Run(flag.Args(), *addr, os.Stdout))
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
