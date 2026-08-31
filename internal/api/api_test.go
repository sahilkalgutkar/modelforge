package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/sahilkalgutkar/modelforge/internal/artifact"
	"github.com/sahilkalgutkar/modelforge/internal/auth"
	"github.com/sahilkalgutkar/modelforge/internal/registry"
	"github.com/sahilkalgutkar/modelforge/internal/routing"
	"github.com/sahilkalgutkar/modelforge/internal/runtime/xgboost"
	"github.com/sahilkalgutkar/modelforge/internal/serving"
)

// fixturePath locates the real XGBoost models the runtime tests use. The API
// tests deliberately serve those rather than a stub: an end-to-end test that
// scores a fake model proves the plumbing and nothing about whether this system
// can actually serve a model somebody trained.
func fixturePath(name string) string {
	return filepath.Join("..", "..", "testdata", "xgboost", name)
}

type harness struct {
	t       *testing.T
	server  *Server
	http    *httptest.Server
	reg     *registry.Store
	manager *serving.Manager
	router  *routing.Router
}

func newHarness(t *testing.T) *harness {
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
	t.Cleanup(reg.Close)
	if err := reg.Reset(ctx); err != nil {
		t.Fatalf("reset registry: %v", err)
	}

	blobs, err := artifact.NewFileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	manager := serving.NewManager(reg, blobs, serving.DefaultOptions())
	t.Cleanup(manager.Close)

	router := routing.NewRouter(manager, routing.Options{})

	srv := NewServer(Deps{Registry: reg, Manager: manager, Router: router})
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)

	return &harness{t: t, server: srv, http: ts, reg: reg, manager: manager, router: router}
}

func (h *harness) do(method, path string, body any) (*http.Response, []byte) {
	h.t.Helper()

	var r io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			h.t.Fatal(err)
		}
		r = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, h.http.URL+path, r)
	if err != nil {
		h.t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := h.http.Client().Do(req)
	if err != nil {
		h.t.Fatal(err)
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(resp.Body)
	return resp, out
}

// upload registers a version by POSTing the actual model file.
func (h *harness) upload(model, fixture string, features []string) int {
	h.t.Helper()

	f, err := os.Open(fixturePath(fixture))
	if err != nil {
		h.t.Fatal(err)
	}
	defer f.Close()

	q := make([]string, 0, len(features))
	for _, name := range features {
		q = append(q, "feature="+name)
	}
	url := fmt.Sprintf("%s/v1/models/%s/versions?%s", h.http.URL, model, strings.Join(q, "&"))

	resp, err := h.http.Client().Post(url, "application/octet-stream", f)
	if err != nil {
		h.t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusCreated {
		h.t.Fatalf("upload returned %d: %s", resp.StatusCode, body)
	}
	var out struct {
		Version registry.Version `json:"version"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		h.t.Fatal(err)
	}
	return out.Version.Version
}

func featureNames(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = fmt.Sprintf("f%d", i)
	}
	return out
}

func namedRow(row []float64) map[string]float64 {
	out := make(map[string]float64, len(row))
	for i, v := range row {
		out[fmt.Sprintf("f%d", i)] = v
	}
	return out
}

// TestEndToEndPredictionMatchesXGBoost is the test the whole system exists to
// pass. It uploads a model XGBoost trained, deploys it, calls the HTTP API, and
// checks the number that comes back is the number XGBoost produces for that
// row — through the registry, the artifact store, the batcher and the router.
func TestEndToEndPredictionMatchesXGBoost(t *testing.T) {
	h := newHarness(t)

	if resp, body := h.do("POST", "/v1/models", createModelRequest{
		Name: "fraud-score", Description: "end to end",
	}); resp.StatusCode != http.StatusCreated {
		t.Fatalf("create model: %d %s", resp.StatusCode, body)
	}

	const nFeatures = 6
	version := h.upload("fraud-score", "binary_logistic.model.json", featureNames(nFeatures))

	if resp, body := h.do("PUT", "/v1/models/fraud-score/policy", routing.Policy{
		Routes: []routing.Route{{Version: version, Weight: 1}},
	}); resp.StatusCode != http.StatusOK {
		t.Fatalf("set policy: %d %s", resp.StatusCode, body)
	}

	// Score the same rows directly against the model, as ground truth.
	direct, err := xgboost.LoadFile(fixturePath("binary_logistic.model.json"))
	if err != nil {
		t.Fatal(err)
	}
	rows := loadFixtureRows(t, "binary_logistic.expected.json")

	for i, row := range rows[:20] {
		want, err := direct.Predict(row)
		if err != nil {
			t.Fatal(err)
		}

		resp, body := h.do("POST", "/v1/models/fraud-score/predict", PredictRequest{
			Features: namedRow(row), Key: fmt.Sprintf("entity-%d", i),
		})
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("row %d: predict returned %d: %s", i, resp.StatusCode, body)
		}
		var got PredictResponse
		if err := json.Unmarshal(body, &got); err != nil {
			t.Fatal(err)
		}
		if got.Version != version {
			t.Errorf("row %d served version %d, want %d", i, got.Version, version)
		}
		if len(got.Prediction) != 1 || math.Abs(got.Prediction[0]-want[0]) > 1e-9 {
			t.Errorf("row %d: API returned %v, direct scoring gives %v", i, got.Prediction, want)
		}
	}
}

// loadFixtureRows reads the input matrix from a generated fixture, restoring
// NaN from the JSON nulls the generator writes.
func loadFixtureRows(t *testing.T, name string) [][]float64 {
	t.Helper()
	b, err := os.ReadFile(fixturePath(name))
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Inputs [][]*float64 `json:"inputs"`
	}
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	out := make([][]float64, len(f.Inputs))
	for i, row := range f.Inputs {
		out[i] = make([]float64, len(row))
		for j, v := range row {
			if v == nil {
				out[i][j] = math.NaN()
			} else {
				out[i][j] = *v
			}
		}
	}
	return out
}

// TestUnknownFeatureIsRejectedButAbsentFeatureIsAllowed covers the asymmetry
// that stops a typo being scored as a missing value.
func TestUnknownFeatureIsRejectedButAbsentFeatureIsAllowed(t *testing.T) {
	h := newHarness(t)
	h.do("POST", "/v1/models", createModelRequest{Name: "sparse"}) //nolint:errcheck
	version := h.upload("sparse", "binary_missing.model.json", featureNames(5))
	h.do("PUT", "/v1/models/sparse/policy", routing.Policy{ //nolint:errcheck
		Routes: []routing.Route{{Version: version, Weight: 1}},
	})

	// A misspelled name must fail loudly. Accepting it would score the request
	// with that feature missing while the caller believes it was supplied.
	resp, body := h.do("POST", "/v1/models/sparse/predict", PredictRequest{
		Features: map[string]float64{"f0": 1, "f1": 2, "amont": 3},
	})
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("unknown feature returned %d, want 400: %s", resp.StatusCode, body)
	}
	if !strings.Contains(string(body), "amont") {
		t.Errorf("error does not name the offending feature: %s", body)
	}

	// A genuinely absent feature is fine: the model learned a direction for
	// missing values.
	resp, body = h.do("POST", "/v1/models/sparse/predict", PredictRequest{
		Features: map[string]float64{"f0": 1, "f1": 2},
	})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("sparse input returned %d, want 200: %s", resp.StatusCode, body)
	}

	// And it must agree with scoring that row with explicit NaNs.
	direct, err := xgboost.LoadFile(fixturePath("binary_missing.model.json"))
	if err != nil {
		t.Fatal(err)
	}
	want, err := direct.Predict([]float64{1, 2, math.NaN(), math.NaN(), math.NaN()})
	if err != nil {
		t.Fatal(err)
	}
	var got PredictResponse
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatal(err)
	}
	if math.Abs(got.Prediction[0]-want[0]) > 1e-9 {
		t.Errorf("sparse prediction %v, want %v — absent features are not being treated as missing",
			got.Prediction, want)
	}
}

// TestCanarySplitOverHTTP checks the traffic split works through the full
// stack, and that the same key sticks to the same version.
func TestCanarySplitOverHTTP(t *testing.T) {
	h := newHarness(t)
	h.do("POST", "/v1/models", createModelRequest{Name: "canary"}) //nolint:errcheck

	v1 := h.upload("canary", "binary_logistic.model.json", featureNames(6))
	v2 := h.upload("canary", "binary_logistic.model.json", featureNames(6))

	if resp, body := h.do("PUT", "/v1/models/canary/policy", routing.Policy{
		Routes: []routing.Route{{Version: v1, Weight: 80}, {Version: v2, Weight: 20}},
	}); resp.StatusCode != http.StatusOK {
		t.Fatalf("set policy: %d %s", resp.StatusCode, body)
	}

	counts := map[int]int{}
	assigned := map[string]int{}
	for i := range 400 {
		key := fmt.Sprintf("user-%d", i)
		_, body := h.do("POST", "/v1/models/canary/predict", PredictRequest{
			Features: map[string]float64{"f0": 1}, Key: key,
		})
		var got PredictResponse
		if err := json.Unmarshal(body, &got); err != nil {
			t.Fatal(err)
		}
		counts[got.Version]++
		assigned[key] = got.Version
	}

	share := float64(counts[v2]) / 400
	if share < 0.10 || share > 0.32 {
		t.Errorf("canary received %.0f%% of traffic, want roughly 20%%", share*100)
	}

	// Every key must come back to the same version on a second pass.
	for key, want := range assigned {
		_, body := h.do("POST", "/v1/models/canary/predict", PredictRequest{
			Features: map[string]float64{"f0": 1}, Key: key,
		})
		var got PredictResponse
		if err := json.Unmarshal(body, &got); err != nil {
			t.Fatal(err)
		}
		if got.Version != want {
			t.Fatalf("key %q moved from version %d to %d between requests", key, want, got.Version)
		}
	}
}

// TestPolicyRejectsAVersionThatCannotLoad covers the ordering rule: a policy is
// only installed once every version it names is loadable, so traffic is never
// routed at a version that cannot serve it.
func TestPolicyRejectsAVersionThatCannotLoad(t *testing.T) {
	h := newHarness(t)
	h.do("POST", "/v1/models", createModelRequest{Name: "strict"}) //nolint:errcheck
	v1 := h.upload("strict", "binary_logistic.model.json", featureNames(6))

	resp, body := h.do("PUT", "/v1/models/strict/policy", routing.Policy{
		Routes: []routing.Route{{Version: v1, Weight: 1}, {Version: 999, Weight: 1}},
	})
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("policy naming a nonexistent version returned %d, want 404: %s", resp.StatusCode, body)
	}
	// And nothing was installed.
	if _, ok := h.router.Policy("strict"); ok {
		t.Error("a rejected policy was installed anyway")
	}
}

// TestVersionRejectedWhenFeatureCountDisagrees is the guard against scoring a
// caller's values in the wrong columns. The model would still return a number.
func TestVersionRejectedWhenFeatureCountDisagrees(t *testing.T) {
	h := newHarness(t)
	h.do("POST", "/v1/models", createModelRequest{Name: "mismatch"}) //nolint:errcheck

	f, err := os.Open(fixturePath("binary_logistic.model.json")) // 6 features
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	resp, err := h.http.Client().Post(
		h.http.URL+"/v1/models/mismatch/versions?feature=a&feature=b", // only 2 declared
		"application/octet-stream", f)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("mismatched feature count returned %d, want 400: %s", resp.StatusCode, body)
	}
	if !strings.Contains(string(body), "artifact expects 6") {
		t.Errorf("error should say what the artifact expects: %s", body)
	}
}

func TestUploadRejectsSomethingThatIsNotAModel(t *testing.T) {
	h := newHarness(t)
	h.do("POST", "/v1/models", createModelRequest{Name: "junk"}) //nolint:errcheck

	resp, err := h.http.Client().Post(
		h.http.URL+"/v1/models/junk/versions?feature=a",
		"application/octet-stream", strings.NewReader("this is not a model"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("uploading junk returned %d, want 400: %s", resp.StatusCode, body)
	}
	if !strings.Contains(string(body), "not a loadable model") {
		t.Errorf("unexpected error: %s", body)
	}
}

func TestUploadRequiresFeatureNames(t *testing.T) {
	h := newHarness(t)
	h.do("POST", "/v1/models", createModelRequest{Name: "nofeat"}) //nolint:errcheck

	resp, err := h.http.Client().Post(h.http.URL+"/v1/models/nofeat/versions",
		"application/octet-stream", strings.NewReader("x"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("upload without features returned %d, want 400", resp.StatusCode)
	}
}

func TestPredictOnAModelWithNoPolicy(t *testing.T) {
	h := newHarness(t)
	resp, _ := h.do("POST", "/v1/models/never-deployed/predict", PredictRequest{
		Features: map[string]float64{"f0": 1},
	})
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("predict without a policy returned %d, want 404", resp.StatusCode)
	}
}

func TestPredictRequiresFeatures(t *testing.T) {
	h := newHarness(t)
	resp, _ := h.do("POST", "/v1/models/anything/predict", PredictRequest{})
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("predict with no features returned %d, want 400", resp.StatusCode)
	}
}

// TestUnknownJSONFieldIsRejected covers the DisallowUnknownFields decision: a
// caller sending a parameter this server does not implement should be told,
// not left believing it took effect.
func TestUnknownJSONFieldIsRejected(t *testing.T) {
	h := newHarness(t)
	req, err := http.NewRequest("POST", h.http.URL+"/v1/models",
		strings.NewReader(`{"name":"x","descriptoin":"typo"}`))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := h.http.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("unknown field returned %d, want 400", resp.StatusCode)
	}
}

func TestDuplicateModelIsAConflict(t *testing.T) {
	h := newHarness(t)
	h.do("POST", "/v1/models", createModelRequest{Name: "twice"}) //nolint:errcheck
	resp, _ := h.do("POST", "/v1/models", createModelRequest{Name: "twice"})
	if resp.StatusCode != http.StatusConflict {
		t.Errorf("duplicate model returned %d, want 409", resp.StatusCode)
	}
}

func TestModelListingAndLookup(t *testing.T) {
	h := newHarness(t)
	h.do("POST", "/v1/models", createModelRequest{Name: "listed", Description: "d"}) //nolint:errcheck
	h.upload("listed", "binary_logistic.model.json", featureNames(6))

	resp, body := h.do("GET", "/v1/models", nil)
	if resp.StatusCode != http.StatusOK || !strings.Contains(string(body), "listed") {
		t.Errorf("list models: %d %s", resp.StatusCode, body)
	}
	resp, body = h.do("GET", "/v1/models/listed", nil)
	if resp.StatusCode != http.StatusOK {
		t.Errorf("get model: %d %s", resp.StatusCode, body)
	}
	resp, body = h.do("GET", "/v1/models/listed/versions", nil)
	if resp.StatusCode != http.StatusOK || !strings.Contains(string(body), `"version":1`) {
		t.Errorf("list versions: %d %s", resp.StatusCode, body)
	}
	resp, _ = h.do("GET", "/v1/models/absent", nil)
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("get missing model returned %d, want 404", resp.StatusCode)
	}
}

// TestReadinessRequiresALoadedModel is the rollout-safety check: a process that
// is alive but has nothing to serve must not be sent traffic.
func TestReadinessRequiresALoadedModel(t *testing.T) {
	h := newHarness(t)

	resp, _ := h.do("GET", "/readyz", nil)
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("readyz with no models returned %d, want 503", resp.StatusCode)
	}
	// Liveness is a different question and must still pass.
	resp, _ = h.do("GET", "/healthz", nil)
	if resp.StatusCode != http.StatusOK {
		t.Errorf("healthz returned %d, want 200", resp.StatusCode)
	}

	h.do("POST", "/v1/models", createModelRequest{Name: "ready"}) //nolint:errcheck
	h.upload("ready", "binary_logistic.model.json", featureNames(6))

	resp, _ = h.do("GET", "/readyz", nil)
	if resp.StatusCode != http.StatusOK {
		t.Errorf("readyz after loading a model returned %d, want 200", resp.StatusCode)
	}
}

func TestPolicyGetAndValidation(t *testing.T) {
	h := newHarness(t)
	h.do("POST", "/v1/models", createModelRequest{Name: "pol"}) //nolint:errcheck
	v := h.upload("pol", "binary_logistic.model.json", featureNames(6))

	resp, _ := h.do("GET", "/v1/models/pol/policy", nil)
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("policy before one is set returned %d, want 404", resp.StatusCode)
	}

	if resp, body := h.do("PUT", "/v1/models/pol/policy", routing.Policy{
		Routes: []routing.Route{{Version: v, Weight: 0}},
	}); resp.StatusCode != http.StatusBadRequest {
		t.Errorf("all-zero-weight policy returned %d, want 400: %s", resp.StatusCode, body)
	}

	h.do("PUT", "/v1/models/pol/policy", routing.Policy{ //nolint:errcheck
		Routes: []routing.Route{{Version: v, Weight: 1}},
	})
	resp, body := h.do("GET", "/v1/models/pol/policy", nil)
	if resp.StatusCode != http.StatusOK || !strings.Contains(string(body), `"version":1`) {
		t.Errorf("get policy: %d %s", resp.StatusCode, body)
	}
}

func TestStatsAndDriftEndpoints(t *testing.T) {
	h := newHarness(t)
	h.do("POST", "/v1/models", createModelRequest{Name: "obs"}) //nolint:errcheck
	v := h.upload("obs", "binary_logistic.model.json", featureNames(6))
	h.do("PUT", "/v1/models/obs/policy", routing.Policy{ //nolint:errcheck
		Routes: []routing.Route{{Version: v, Weight: 1}},
	})
	for range 5 {
		h.do("POST", "/v1/models/obs/predict", PredictRequest{ //nolint:errcheck
			Features: map[string]float64{"f0": 1},
		})
	}

	resp, body := h.do("GET", fmt.Sprintf("/v1/models/obs/versions/%d/stats", v), nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("stats: %d %s", resp.StatusCode, body)
	}
	if !strings.Contains(string(body), `"rows":5`) {
		t.Errorf("stats did not count the 5 rows: %s", body)
	}

	// No monitor is attached, so drift is reported as not ready rather than
	// as an error — a version can be perfectly servable with no baselines.
	resp, body = h.do("GET", fmt.Sprintf("/v1/models/obs/versions/%d/drift", v), nil)
	if resp.StatusCode != http.StatusOK || !strings.Contains(string(body), `"ready":false`) {
		t.Errorf("drift: %d %s", resp.StatusCode, body)
	}

	resp, _ = h.do("GET", "/v1/models/obs/versions/notanumber/drift", nil)
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("non-numeric version returned %d, want 400", resp.StatusCode)
	}
	resp, _ = h.do("GET", "/v1/models/obs/versions/77/stats", nil)
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("stats for an unloaded version returned %d, want 404", resp.StatusCode)
	}
}

// TestConcurrentPredictionsAreConsistent runs the full stack under concurrency,
// which is where the batcher, router and manager locks all interact.
func TestConcurrentPredictionsAreConsistent(t *testing.T) {
	h := newHarness(t)
	h.do("POST", "/v1/models", createModelRequest{Name: "load"}) //nolint:errcheck
	v := h.upload("load", "binary_logistic.model.json", featureNames(6))
	h.do("PUT", "/v1/models/load/policy", routing.Policy{ //nolint:errcheck
		Routes: []routing.Route{{Version: v, Weight: 1}},
	})

	direct, err := xgboost.LoadFile(fixturePath("binary_logistic.model.json"))
	if err != nil {
		t.Fatal(err)
	}
	rows := loadFixtureRows(t, "binary_logistic.expected.json")

	var wg sync.WaitGroup
	bad := make(chan string, len(rows))
	for i, row := range rows {
		wg.Add(1)
		go func() {
			defer wg.Done()
			want, err := direct.Predict(row)
			if err != nil {
				bad <- err.Error()
				return
			}
			resp, body := h.do("POST", "/v1/models/load/predict", PredictRequest{
				Features: namedRow(row), Key: fmt.Sprintf("k%d", i),
			})
			if resp.StatusCode != http.StatusOK {
				bad <- fmt.Sprintf("row %d: %d %s", i, resp.StatusCode, body)
				return
			}
			var got PredictResponse
			if err := json.Unmarshal(body, &got); err != nil {
				bad <- err.Error()
				return
			}
			if math.Abs(got.Prediction[0]-want[0]) > 1e-9 {
				bad <- fmt.Sprintf("row %d: got %v want %v", i, got.Prediction, want)
			}
		}()
	}
	wg.Wait()
	close(bad)
	for msg := range bad {
		t.Error(msg)
	}

	// The requests should have been batched rather than scored one at a time.
	stats, err := h.manager.BatchStats("load", v)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%d rows in %d batches (mean %.1f, largest %d)",
		stats.Rows, stats.Batches, stats.Mean(), stats.LargestBatch)
	if stats.LargestBatch < 2 {
		t.Errorf("largest batch was %d; concurrent requests were not batched", stats.LargestBatch)
	}
}

// --- authentication ---

// Tokens the auth harness issues, one per scope, so a test can prove that a
// credential is refused for exactly the routes it should be.
const (
	adminToken   = "harness-admin-token"
	readToken    = "harness-read-token"
	predictToken = "harness-predict-token"
)

// newAuthedHarness is newHarness with real authentication rather than the
// disabled Authenticator the other tests use.
func newAuthedHarness(t *testing.T) *harness {
	t.Helper()
	h := newHarness(t)

	authn, err := auth.New([]string{
		"admin:admin:" + auth.Digest(adminToken),
		"dash:read:" + auth.Digest(readToken),
		"edge:predict:" + auth.Digest(predictToken),
	})
	if err != nil {
		t.Fatal(err)
	}

	// Rebuild the server with auth on, against the same dependencies.
	h.server = NewServer(Deps{
		Registry: h.reg, Manager: h.manager, Router: h.router, Auth: authn,
	})
	h.http.Close()
	h.http = httptest.NewServer(h.server.Handler())
	t.Cleanup(h.http.Close)
	return h
}

// call issues a request with an optional bearer token and returns the status.
func (h *harness) call(t *testing.T, method, path, token string, body any) int {
	t.Helper()

	var r io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
		r = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, h.http.URL+path, r)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := h.http.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body) //nolint:errcheck
	return resp.StatusCode
}

// TestEveryRouteEnforcesItsScope is the table that makes the whole scheme
// checkable at a glance: for each route, which credential gets in and which
// does not. Writing it as one table rather than as scattered assertions is what
// makes a route added later without a scope visible as a missing row.
func TestEveryRouteEnforcesItsScope(t *testing.T) {
	h := newAuthedHarness(t)

	// A deployed model, so the routes reach their handlers instead of 404ing
	// before the scope check would matter.
	h.call(t, "POST", "/v1/models", adminToken, createModelRequest{Name: "guarded"})
	version := h.uploadAs(t, "guarded", "binary_logistic.model.json", featureNames(6), adminToken)
	h.call(t, "PUT", "/v1/models/guarded/policy", adminToken, routing.Policy{
		Routes: []routing.Route{{Version: version, Weight: 1}},
	})

	predictBody := PredictRequest{Features: map[string]float64{"f0": 1}}
	policyBody := routing.Policy{Routes: []routing.Route{{Version: version, Weight: 1}}}

	for _, tc := range []struct {
		name   string
		method string
		path   string
		body   any
		// allowed lists the tokens that must succeed; every other token must
		// be refused with 403.
		allowed []string
	}{
		{"predict", "POST", "/v1/models/guarded/predict", predictBody, []string{predictToken, adminToken}},

		{"list models", "GET", "/v1/models", nil, []string{readToken, adminToken}},
		{"get model", "GET", "/v1/models/guarded", nil, []string{readToken, adminToken}},
		{"list versions", "GET", "/v1/models/guarded/versions", nil, []string{readToken, adminToken}},
		{"get policy", "GET", "/v1/models/guarded/policy", nil, []string{readToken, adminToken}},
		{"drift", "GET", "/v1/models/guarded/versions/1/drift", nil, []string{readToken, adminToken}},
		{"stats", "GET", "/v1/models/guarded/versions/1/stats", nil, []string{readToken, adminToken}},

		{"create model", "POST", "/v1/models", createModelRequest{Name: "another"}, []string{adminToken}},
		{"set policy", "PUT", "/v1/models/guarded/policy", policyBody, []string{adminToken}},
		{"set baseline", "PUT", "/v1/models/guarded/versions/1/baseline", BaselineRequest{}, []string{adminToken}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// No credential at all is always 401.
			if got := h.call(t, tc.method, tc.path, "", tc.body); got != http.StatusUnauthorized {
				t.Errorf("no token = %d, want 401", got)
			}
			// An unrecognised credential is always 401, never 403.
			if got := h.call(t, tc.method, tc.path, "made-up", tc.body); got != http.StatusUnauthorized {
				t.Errorf("unknown token = %d, want 401", got)
			}

			for _, token := range []string{adminToken, readToken, predictToken} {
				permitted := slices.Contains(tc.allowed, token)
				got := h.call(t, tc.method, tc.path, token, tc.body)

				if !permitted {
					if got != http.StatusForbidden {
						t.Errorf("token %s = %d, want 403 (it lacks the scope)", nameOf(token), got)
					}
					continue
				}
				if got == http.StatusUnauthorized || got == http.StatusForbidden {
					t.Errorf("token %s = %d, but it should be permitted here", nameOf(token), got)
				}
			}
		})
	}
}

func nameOf(token string) string {
	switch token {
	case adminToken:
		return "admin"
	case readToken:
		return "read"
	case predictToken:
		return "predict"
	}
	return "unknown"
}

// uploadAs is upload with an explicit credential.
func (h *harness) uploadAs(t *testing.T, model, fixture string, features []string, token string) int {
	t.Helper()

	f, err := os.Open(fixturePath(fixture))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	q := make([]string, 0, len(features))
	for _, name := range features {
		q = append(q, "feature="+name)
	}
	url := fmt.Sprintf("%s/v1/models/%s/versions?%s", h.http.URL, model, strings.Join(q, "&"))

	req, err := http.NewRequest("POST", url, f)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := h.http.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("upload returned %d: %s", resp.StatusCode, body)
	}
	var out struct {
		Version registry.Version `json:"version"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatal(err)
	}
	return out.Version.Version
}

// TestPushingAModelRequiresAdmin is called out separately because it is the
// most consequential unauthenticated write available: replacing the artifact a
// model scores with is arbitrary control over every prediction it makes.
func TestPushingAModelRequiresAdmin(t *testing.T) {
	h := newAuthedHarness(t)
	h.call(t, "POST", "/v1/models", adminToken, createModelRequest{Name: "protected"})

	url := h.http.URL + "/v1/models/protected/versions?feature=f0"
	for token, want := range map[string]int{
		"":           http.StatusUnauthorized,
		"nonsense":   http.StatusUnauthorized,
		readToken:    http.StatusForbidden,
		predictToken: http.StatusForbidden,
	} {
		req, err := http.NewRequest("POST", url, strings.NewReader("not a model"))
		if err != nil {
			t.Fatal(err)
		}
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		resp, err := h.http.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != want {
			t.Errorf("push with token %q = %d, want %d", nameOf(token), resp.StatusCode, want)
		}
	}
}

// TestHealthEndpointsStayOpen covers the deliberate exemption. A liveness probe
// that needs a credential starts failing the moment a token is rotated or
// misconfigured, and would then restart the very process that is serving
// correctly.
func TestHealthEndpointsStayOpen(t *testing.T) {
	h := newAuthedHarness(t)

	if got := h.call(t, "GET", "/healthz", "", nil); got != http.StatusOK {
		t.Errorf("healthz without a credential = %d, want 200", got)
	}
	// readyz answers without a credential too; 503 here means "no models
	// loaded", which is the readiness answer, not an auth refusal.
	if got := h.call(t, "GET", "/readyz", "", nil); got == http.StatusUnauthorized || got == http.StatusForbidden {
		t.Errorf("readyz without a credential = %d, want an availability answer", got)
	}
}

// TestMetricsRequiresRead covers the other side of that judgement: /metrics
// carries every model name, version, request volume and drift reading, which
// together describe what a business scores and how much of it.
func TestMetricsRequiresRead(t *testing.T) {
	h := newAuthedHarness(t)
	guarded := h.server.MetricsHandler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("modelforge_predictions_total 1")) //nolint:errcheck
	}))
	ts := httptest.NewServer(guarded)
	defer ts.Close()

	get := func(token string) int {
		req, _ := http.NewRequest("GET", ts.URL, nil)
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		return resp.StatusCode
	}

	if got := get(""); got != http.StatusUnauthorized {
		t.Errorf("metrics without a credential = %d, want 401", got)
	}
	if got := get(predictToken); got != http.StatusForbidden {
		t.Errorf("metrics with a predict-only token = %d, want 403", got)
	}
	if got := get(readToken); got != http.StatusOK {
		t.Errorf("metrics with a read token = %d, want 200", got)
	}
	if got := get(adminToken); got != http.StatusOK {
		t.Errorf("metrics with an admin token = %d, want 200", got)
	}
}

// TestNilAuthMeansDisabled documents the default in Deps. It is the behaviour
// the pre-auth tests rely on, and it is safe only because app.New refuses to
// construct a server without either tokens or an explicit opt-out.
func TestNilAuthMeansDisabled(t *testing.T) {
	h := newHarness(t) // built with no Auth
	if got := h.call(t, "GET", "/v1/models", "", nil); got != http.StatusOK {
		t.Errorf("nil Auth blocked a request with %d", got)
	}
}
