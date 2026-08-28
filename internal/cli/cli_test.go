package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sahilkalgutkar/modelforge/internal/app"
	"github.com/sahilkalgutkar/modelforge/internal/drift"
	"github.com/sahilkalgutkar/modelforge/internal/registry"
)

// serverURL starts a real modelforge server — real Postgres, real artifact
// store, real scorer — and returns its address.
//
// The CLI is tested against the actual server rather than a mock, because
// almost everything that can be wrong with a client is a disagreement with the
// server: a wrong verb, a wrong path, a field name that does not match. A mock
// built from the same assumptions as the client agrees with all of those.
func serverURL(t *testing.T) string {
	t.Helper()

	dsn := os.Getenv("MODELFORGE_TEST_DATABASE_URL")
	if dsn == "" {
		t.Fatal("MODELFORGE_TEST_DATABASE_URL is not set; these tests require a real Postgres " +
			"(see internal/registry/postgres_test.go for why they fail rather than skip)")
	}
	ctx := context.Background()
	reg, err := registry.Open(ctx, dsn)
	if err != nil {
		t.Fatalf("connect to test Postgres: %v", err)
	}
	if err := reg.Reset(ctx); err != nil {
		t.Fatal(err)
	}
	reg.Close()

	a, err := app.New(ctx, app.Config{
		DatabaseURL: dsn,
		ArtifactDir: t.TempDir(),
		Logger:      slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatalf("start server: %v", err)
	}
	t.Cleanup(a.Close)

	ts := httptest.NewServer(a.Handler())
	t.Cleanup(ts.Close)
	return ts.URL
}

// run executes a CLI command and returns its output and exit code.
func run(t *testing.T, addr string, args ...string) (string, int) {
	t.Helper()
	var buf bytes.Buffer
	code := Run(args, addr, &buf)
	return buf.String(), code
}

func modelFile() string {
	return filepath.Join("..", "..", "testdata", "xgboost", "binary_logistic.model.json")
}

var features = []string{"f0", "f1", "f2", "f3", "f4", "f5"}

// TestRolloutWorkflow walks the sequence an operator actually performs:
// register, push two versions, deploy, canary, shadow, then roll back.
func TestRolloutWorkflow(t *testing.T) {
	addr := serverURL(t)

	out, code := run(t, addr, "create", "fraud-score", "scores a transaction")
	if code != 0 || !strings.Contains(out, "created model fraud-score") {
		t.Fatalf("create: %d %s", code, out)
	}

	push := append([]string{"push", "fraud-score", modelFile()}, features...)
	out, code = run(t, addr, push...)
	if code != 0 || !strings.Contains(out, "version 1") {
		t.Fatalf("push v1: %d %s", code, out)
	}
	// A push must not start serving on its own; that is a separate, deliberate
	// act.
	if !strings.Contains(out, "not serving") {
		t.Errorf("push output should say the version is not yet serving: %s", out)
	}
	if !strings.Contains(out, "binary:logistic") {
		t.Errorf("push output should report the objective it read from the artifact: %s", out)
	}

	out, code = run(t, addr, push...)
	if code != 0 || !strings.Contains(out, "version 2") {
		t.Fatalf("push v2: %d %s", code, out)
	}

	out, code = run(t, addr, "versions", "fraud-score")
	if code != 0 || !strings.Contains(out, "VERSION") {
		t.Fatalf("versions: %d %s", code, out)
	}

	out, code = run(t, addr, "deploy", "fraud-score", "1")
	if code != 0 || !strings.Contains(out, "v1=100%") {
		t.Fatalf("deploy: %d %s", code, out)
	}

	out, code = run(t, addr, "canary", "fraud-score", "1", "2", "10")
	if code != 0 || !strings.Contains(out, "v2=10%") {
		t.Fatalf("canary: %d %s", code, out)
	}

	// Rolling back is deploying an older version, and it must clear the
	// canary rather than leaving the candidate quietly taking traffic.
	out, code = run(t, addr, "rollback", "fraud-score", "1")
	if code != 0 || !strings.Contains(out, "v1=100%") {
		t.Fatalf("rollback: %d %s", code, out)
	}
	out, code = run(t, addr, "policy", "fraud-score")
	if code != 0 {
		t.Fatalf("policy: %d %s", code, out)
	}
	if strings.Contains(out, "v2") {
		t.Errorf("the canary is still in the policy after a rollback: %s", out)
	}
}

// TestShadowPreservesTheServingSplit is the behaviour that stops "watch this
// candidate" turning into an unannounced deploy.
func TestShadowPreservesTheServingSplit(t *testing.T) {
	addr := serverURL(t)
	run(t, addr, "create", "shadowed")                                              //nolint:errcheck
	run(t, addr, append([]string{"push", "shadowed", modelFile()}, features...)...) //nolint:errcheck
	run(t, addr, append([]string{"push", "shadowed", modelFile()}, features...)...) //nolint:errcheck
	run(t, addr, "deploy", "shadowed", "1")                                         //nolint:errcheck

	out, code := run(t, addr, "shadow", "shadowed", "2")
	if code != 0 {
		t.Fatalf("shadow: %d %s", code, out)
	}
	if !strings.Contains(out, "v1=100%") {
		t.Errorf("adding a shadow changed what is serving: %s", out)
	}
	if !strings.Contains(out, "shadow=v2") {
		t.Errorf("shadow was not recorded: %s", out)
	}
}

func TestShadowRequiresAnExistingPolicy(t *testing.T) {
	addr := serverURL(t)
	run(t, addr, "create", "unserved")                                              //nolint:errcheck
	run(t, addr, append([]string{"push", "unserved", modelFile()}, features...)...) //nolint:errcheck

	out, code := run(t, addr, "shadow", "unserved", "1")
	if code == 0 {
		t.Fatal("shadow succeeded on a model that is not serving")
	}
	if !strings.Contains(out, "before a model is serving") {
		t.Errorf("error should explain the problem: %s", out)
	}
}

func TestDriftAndStatsOutput(t *testing.T) {
	addr := serverURL(t)
	run(t, addr, "create", "observed")                                              //nolint:errcheck
	run(t, addr, append([]string{"push", "observed", modelFile()}, features...)...) //nolint:errcheck
	run(t, addr, "deploy", "observed", "1")                                         //nolint:errcheck

	out, code := run(t, addr, "stats", "observed", "1")
	if code != 0 || !strings.Contains(out, "batching") {
		t.Fatalf("stats: %d %s", code, out)
	}

	// With no traffic the drift command must say there is not enough data,
	// which is a different statement from "no drift".
	out, code = run(t, addr, "drift", "observed", "1")
	if code != 0 {
		t.Fatalf("drift: %d %s", code, out)
	}
	if !strings.Contains(out, "not enough traffic") {
		t.Errorf("drift with no traffic should not read as stable: %s", out)
	}
}

func TestModelsListing(t *testing.T) {
	addr := serverURL(t)

	out, code := run(t, addr, "models")
	if code != 0 || !strings.Contains(out, "no models registered") {
		t.Fatalf("empty models listing: %d %s", code, out)
	}

	run(t, addr, "create", "listed", "a description") //nolint:errcheck
	out, code = run(t, addr, "models")
	if code != 0 || !strings.Contains(out, "listed") || !strings.Contains(out, "a description") {
		t.Fatalf("models: %d %s", code, out)
	}
}

// TestServerErrorsAreReadable checks the client surfaces the server's message
// rather than a bare status code the operator has to go and decode.
func TestServerErrorsAreReadable(t *testing.T) {
	addr := serverURL(t)

	out, code := run(t, addr, "create", "Invalid Name")
	if code != 1 {
		t.Fatalf("expected exit code 1, got %d: %s", code, out)
	}
	if !strings.Contains(out, "lowercase") {
		t.Errorf("the server's explanation was not shown: %s", out)
	}

	run(t, addr, "create", "dupe") //nolint:errcheck
	out, code = run(t, addr, "create", "dupe")
	if code != 1 || !strings.Contains(out, "already exists") {
		t.Errorf("duplicate create: %d %s", code, out)
	}

	out, code = run(t, addr, "versions", "never-created")
	if code != 0 || !strings.Contains(out, "no versions") {
		t.Errorf("versions of an unknown model: %d %s", code, out)
	}
}

func TestPushRejectsAMissingFile(t *testing.T) {
	addr := serverURL(t)
	run(t, addr, "create", "nofile") //nolint:errcheck

	out, code := run(t, addr, "push", "nofile", "/does/not/exist.json", "f0")
	if code != 1 || !strings.Contains(out, "open model file") {
		t.Errorf("push with a missing file: %d %s", code, out)
	}
}

func TestArgumentValidation(t *testing.T) {
	addr := serverURL(t)

	for _, tc := range []struct {
		name string
		args []string
		want string
	}{
		{"deploy without a version", []string{"deploy", "m"}, "needs a model and a version"},
		{"non-numeric version", []string{"deploy", "m", "latest"}, "version must be a number"},
		{"create without a name", []string{"create"}, "needs a model name"},
		{"versions without a model", []string{"versions"}, "needs a model name"},
		{"policy without a model", []string{"policy"}, "needs a model name"},
		{"push without features", []string{"push", "m", "file.json"}, "at least one feature"},
		{"canary missing arguments", []string{"canary", "m", "1", "2"}, "canary needs"},
		{"canary with a bad stable version", []string{"canary", "m", "x", "2", "10"}, "stable version must be a number"},
		{"canary with a bad candidate", []string{"canary", "m", "1", "x", "10"}, "candidate version must be a number"},
		{"canary over 100 percent", []string{"canary", "m", "1", "2", "150"}, "between 0 and 100"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, code := run(t, addr, tc.args...)
			if code != 1 {
				t.Fatalf("exit code %d, want 1: %s", code, out)
			}
			if !strings.Contains(out, tc.want) {
				t.Errorf("output %q does not contain %q", out, tc.want)
			}
		})
	}
}

func TestUsageAndUnknownCommand(t *testing.T) {
	out, code := run(t, "http://localhost:1", "")
	if code != 2 {
		t.Errorf("empty command exit code = %d, want 2", code)
	}

	out, code = run(t, "http://localhost:1", "frobnicate")
	if code != 2 || !strings.Contains(out, `unknown command "frobnicate"`) {
		t.Errorf("unknown command: %d %s", code, out)
	}
	if !strings.Contains(out, "Commands:") {
		t.Errorf("usage was not printed alongside the error: %s", out)
	}
}

func TestUnreachableServer(t *testing.T) {
	// Port 1 is not going to be listening.
	out, code := run(t, "http://127.0.0.1:1", "models")
	if code != 1 {
		t.Errorf("exit code = %d, want 1 for an unreachable server: %s", code, out)
	}
}

func TestHumanBytes(t *testing.T) {
	for _, tc := range []struct {
		in   int64
		want string
	}{
		{512, "512B"},
		{2048, "2.0KB"},
		{5 << 20, "5.0MB"},
		{3 << 30, "3.0GB"},
	} {
		if got := humanBytes(tc.in); got != tc.want {
			t.Errorf("humanBytes(%d) = %s, want %s", tc.in, got, tc.want)
		}
	}
}

// writeBaseline creates a baseline file with `n` normal samples per feature,
// shifting one feature so the drift reading has something to find.
func writeBaseline(t *testing.T, shift float64) string {
	t.Helper()
	r := rand.New(rand.NewPCG(7, 8))
	samples := map[string][]float64{}
	for _, f := range features {
		vals := make([]float64, 4000)
		for i := range vals {
			vals[i] = r.NormFloat64() + shift
		}
		samples[f] = vals
	}
	body, err := json.Marshal(map[string]any{"samples": samples, "bins": 10})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "baseline.json")
	if err := os.WriteFile(path, body, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestBaselineAndDriftReadings walks the whole drift story through the CLI:
// attach baselines, send traffic that does not match them, and read the report.
func TestBaselineAndDriftReadings(t *testing.T) {
	addr := serverURL(t)
	run(t, addr, "create", "monitored")                                              //nolint:errcheck
	run(t, addr, append([]string{"push", "monitored", modelFile()}, features...)...) //nolint:errcheck
	run(t, addr, "deploy", "monitored", "1")                                         //nolint:errcheck

	out, code := run(t, addr, "baseline", "monitored", "1", writeBaseline(t, 0))
	if code != 0 || !strings.Contains(out, "6 features in 10 bins") {
		t.Fatalf("baseline: %d %s", code, out)
	}

	// Traffic drawn from a distribution well away from the baseline.
	client := &http.Client{}
	r := rand.New(rand.NewPCG(9, 10))
	for range drift.MinSamples * 2 {
		body := map[string]map[string]float64{"features": {}}
		for _, f := range features {
			body["features"][f] = r.NormFloat64() + 4
		}
		b, _ := json.Marshal(body)
		resp, err := client.Post(addr+"/v1/models/monitored/predict", "application/json", bytes.NewReader(b))
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
	}

	out, code = run(t, addr, "drift", "monitored", "1")
	if code != 0 {
		t.Fatalf("drift: %d %s", code, out)
	}
	if strings.Contains(out, "not enough traffic") {
		t.Fatalf("drift still reports insufficient traffic after %d requests:\n%s", drift.MinSamples*2, out)
	}
	if !strings.Contains(out, "significant") {
		t.Errorf("a four-sigma shift did not read as significant:\n%s", out)
	}
	t.Logf("drift report:\n%s", out)
}

func TestBaselineValidationErrors(t *testing.T) {
	addr := serverURL(t)
	run(t, addr, "create", "partial")                                              //nolint:errcheck
	run(t, addr, append([]string{"push", "partial", modelFile()}, features...)...) //nolint:errcheck

	if out, code := run(t, addr, "baseline", "partial", "1", "/nope.json"); code != 1 ||
		!strings.Contains(out, "open baseline file") {
		t.Errorf("missing file: %d %s", code, out)
	}

	// A baseline covering only some features would monitor those and silently
	// ignore the rest, which reads on a dashboard as "no drift".
	// f0 gets a full, valid sample set so the request fails on the *missing*
	// features rather than on f0 being too small to bin.
	f0 := make([]float64, 200)
	for i := range f0 {
		f0[i] = float64(i)
	}
	body, _ := json.Marshal(map[string]any{"samples": map[string][]float64{"f0": f0}})
	path := filepath.Join(t.TempDir(), "partial.json")
	if err := os.WriteFile(path, body, 0o600); err != nil {
		t.Fatal(err)
	}
	out, code := run(t, addr, "baseline", "partial", "1", path)
	if code != 1 || !strings.Contains(out, "no baseline samples for feature") {
		t.Errorf("partial baseline: %d %s", code, out)
	}

	if out, code := run(t, addr, "baseline", "partial", "99", writeBaseline(t, 0)); code != 1 {
		t.Errorf("baseline for an unloaded version: %d %s", code, out)
	}
}

// TestDeployKeepsAnUnrelatedShadow covers the consistency fix: `shadow`
// preserves the routes, so `deploy` and `canary` must preserve the shadow.
// Otherwise promoting a candidate silently stops whatever else was being
// observed, and the operator finds out by noticing an empty graph.
func TestDeployKeepsAnUnrelatedShadow(t *testing.T) {
	addr := serverURL(t)
	run(t, addr, "create", "keeps") //nolint:errcheck
	for range 3 {
		run(t, addr, append([]string{"push", "keeps", modelFile()}, features...)...) //nolint:errcheck
	}
	run(t, addr, "deploy", "keeps", "1") //nolint:errcheck
	run(t, addr, "shadow", "keeps", "3") //nolint:errcheck

	// Promoting v2 must leave v3 shadowed: it is unrelated to what changed.
	out, code := run(t, addr, "deploy", "keeps", "2")
	if code != 0 {
		t.Fatalf("deploy: %d %s", code, out)
	}
	if !strings.Contains(out, "shadow=v3") {
		t.Errorf("deploying v2 dropped the unrelated shadow on v3: %s", out)
	}

	out, code = run(t, addr, "canary", "keeps", "2", "1", "20")
	if code != 0 || !strings.Contains(out, "shadow=v3") {
		t.Errorf("canary dropped the shadow: %d %s", code, out)
	}
}

// TestPromotingTheShadowClearsIt is the other half. The server refuses a policy
// where a version is both shadow and serving — shadowing a version against
// itself produces a divergence rate that mixes two comparisons and means
// nothing — so promoting a shadow has to clear it rather than fail.
func TestPromotingTheShadowClearsIt(t *testing.T) {
	addr := serverURL(t)
	run(t, addr, "create", "promotes")                                              //nolint:errcheck
	run(t, addr, append([]string{"push", "promotes", modelFile()}, features...)...) //nolint:errcheck
	run(t, addr, append([]string{"push", "promotes", modelFile()}, features...)...) //nolint:errcheck
	run(t, addr, "deploy", "promotes", "1")                                         //nolint:errcheck
	run(t, addr, "shadow", "promotes", "2")                                         //nolint:errcheck

	out, code := run(t, addr, "canary", "promotes", "1", "2", "10")
	if code != 0 {
		t.Fatalf("promoting the shadow to a canary failed: %d %s", code, out)
	}
	if strings.Contains(out, "shadow=v2") {
		t.Errorf("v2 is both serving and shadowed: %s", out)
	}
	if !strings.Contains(out, "no longer shadowed") {
		t.Errorf("the change was not explained to the operator: %s", out)
	}
}
