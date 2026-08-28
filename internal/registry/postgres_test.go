package registry

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sahilkalgutkar/modelforge/internal/artifact"
)

// testStore connects to the Postgres this suite requires.
//
// It fails rather than skips when the database is unreachable, and that is a
// deliberate choice. A suite that skips itself when its dependency is missing
// keeps CI green while testing nothing, and the gap is invisible precisely
// because everything looks like it passed. The store is also the one component
// an in-memory double cannot say anything useful about: uniqueness constraints,
// the row lock that serialises version assignment, and what timestamptz does to
// a nanosecond timestamp all live in the database, and a fake would agree with
// whatever the code happens to do.
func testStore(t *testing.T) *Store {
	t.Helper()

	dsn := os.Getenv("MODELFORGE_TEST_DATABASE_URL")
	if dsn == "" {
		t.Fatal("MODELFORGE_TEST_DATABASE_URL is not set.\n" +
			"These tests require a real Postgres; start one with `docker compose -f deploy/docker-compose.yml up -d postgres` " +
			"and export MODELFORGE_TEST_DATABASE_URL. They fail rather than skip on purpose — " +
			"a suite that skips its own dependency keeps CI green while testing nothing.")
	}

	ctx := context.Background()
	s, err := Open(ctx, dsn)
	if err != nil {
		t.Fatalf("connect to test Postgres: %v", err)
	}
	t.Cleanup(s.Close)

	// Each test gets an empty registry. The cascade from models clears
	// versions and deployments with it.
	if err := s.Reset(ctx); err != nil {
		t.Fatalf("reset database: %v", err)
	}
	return s
}

func digestOf(s string) artifact.Digest {
	sum := sha256.Sum256([]byte(s))
	return artifact.Digest(hex.EncodeToString(sum[:]))
}

func newVersionReq(model string, seed string) NewVersion {
	return NewVersion{
		Model:     model,
		Runtime:   RuntimeXGBoost,
		Digest:    digestOf(seed),
		SizeBytes: int64(len(seed)),
		Features:  []string{"amount", "account_age", "n_prior_disputes"},
		Notes:     "trained on " + seed,
	}
}

func TestCreateAndGetModel(t *testing.T) {
	ctx := context.Background()
	s := testStore(t)

	created, err := s.CreateModel(ctx, "fraud-score", "scores a transaction")
	if err != nil {
		t.Fatalf("CreateModel: %v", err)
	}
	if created.Name != "fraud-score" || created.Description != "scores a transaction" {
		t.Errorf("CreateModel returned %+v", created)
	}

	got, err := s.GetModel(ctx, "fraud-score")
	if err != nil {
		t.Fatalf("GetModel: %v", err)
	}
	if got.Name != created.Name || got.Description != created.Description {
		t.Errorf("GetModel = %+v, want %+v", got, created)
	}

	// The timestamp must survive the round trip exactly. Go's time.Now() is
	// nanosecond-resolution and timestamptz stores microseconds, so without
	// truncation at construction the value read back differs from the one
	// returned to the caller — a difference that only shows up as an equality
	// check failing somewhere far away.
	if !got.CreatedAt.Equal(created.CreatedAt) {
		t.Errorf("CreatedAt round trip: stored %v, read back %v (difference %v)",
			created.CreatedAt, got.CreatedAt, got.CreatedAt.Sub(created.CreatedAt))
	}
}

// TestTimestampSurvivesRoundTripWithSubMicrosecondPrecision forces the failure
// the previous test can only catch by luck. time.Now() produces a value with
// sub-microsecond digits only some of the time, so asserting on a
// naturally-stamped timestamp passes or fails depending on the clock. This
// builds a timestamp that definitely has nanosecond remainder.
func TestTimestampSurvivesRoundTripWithSubMicrosecondPrecision(t *testing.T) {
	awkward := time.Date(2026, 8, 28, 12, 0, 0, 123456789, time.UTC)

	got := truncateForStore(awkward)
	if got.Nanosecond()%1000 != 0 {
		t.Fatalf("truncateForStore left %d nanoseconds, which timestamptz cannot store", got.Nanosecond())
	}
	if want := time.Date(2026, 8, 28, 12, 0, 0, 123456000, time.UTC); !got.Equal(want) {
		t.Errorf("truncateForStore = %v, want %v", got, want)
	}

	ctx := context.Background()
	s := testStore(t)
	if _, err := s.pool.Exec(ctx,
		`INSERT INTO models (name, description, created_at) VALUES ($1, '', $2)`,
		"precision-probe", got); err != nil {
		t.Fatalf("insert: %v", err)
	}
	back, err := s.GetModel(ctx, "precision-probe")
	if err != nil {
		t.Fatal(err)
	}
	if !back.CreatedAt.Equal(got) {
		t.Errorf("timestamp round trip: wrote %v, read %v", got, back.CreatedAt)
	}
}

func TestCreateModelRejectsDuplicate(t *testing.T) {
	ctx := context.Background()
	s := testStore(t)

	if _, err := s.CreateModel(ctx, "dupe", ""); err != nil {
		t.Fatal(err)
	}
	_, err := s.CreateModel(ctx, "dupe", "")
	if !errors.Is(err, ErrAlreadyExists) {
		t.Fatalf("second CreateModel error = %v, want ErrAlreadyExists", err)
	}
}

func TestGetMissingModel(t *testing.T) {
	s := testStore(t)
	if _, err := s.GetModel(context.Background(), "absent"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetModel error = %v, want ErrNotFound", err)
	}
}

func TestCreateModelValidatesName(t *testing.T) {
	ctx := context.Background()
	s := testStore(t)

	for _, name := range []string{"", "A", "Upper", "has space", "-leading", "trailing-", strings.Repeat("x", 65), "x"} {
		if _, err := s.CreateModel(ctx, name, ""); err == nil {
			t.Errorf("CreateModel(%q) succeeded, want a validation error", name)
		}
	}
	for _, name := range []string{"ab", "fraud-score", "a.b_c-1", strings.Repeat("x", 64)} {
		if _, err := s.CreateModel(ctx, name, ""); err != nil {
			t.Errorf("CreateModel(%q) = %v, want success", name, err)
		}
	}
}

func TestVersionNumbersAreDenseAndMonotonic(t *testing.T) {
	ctx := context.Background()
	s := testStore(t)
	if _, err := s.CreateModel(ctx, "fraud-score", ""); err != nil {
		t.Fatal(err)
	}

	for want := 1; want <= 4; want++ {
		v, err := s.CreateVersion(ctx, newVersionReq("fraud-score", fmt.Sprintf("run-%d", want)))
		if err != nil {
			t.Fatalf("CreateVersion: %v", err)
		}
		if v.Version != want {
			t.Fatalf("version = %d, want %d", v.Version, want)
		}
	}

	latest, err := s.LatestVersion(ctx, "fraud-score")
	if err != nil {
		t.Fatal(err)
	}
	if latest.Version != 4 {
		t.Errorf("LatestVersion = %d, want 4", latest.Version)
	}
}

// TestVersionNumbersAreUniqueUnderConcurrency is the test the row lock exists
// for. Twelve goroutines register versions of the same model at once; every one
// must come back with a distinct number, and together they must be exactly
// 1..12 with no gaps. Reading MAX(version) without the lock lets two
// transactions compute the same next number.
func TestVersionNumbersAreUniqueUnderConcurrency(t *testing.T) {
	ctx := context.Background()
	s := testStore(t)
	if _, err := s.CreateModel(ctx, "concurrent", ""); err != nil {
		t.Fatal(err)
	}

	const writers = 12
	var (
		mu   sync.Mutex
		got  []int
		errs []error
		wg   sync.WaitGroup
	)
	for i := range writers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			v, err := s.CreateVersion(ctx, newVersionReq("concurrent", fmt.Sprintf("parallel-%d", i)))
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				errs = append(errs, err)
				return
			}
			got = append(got, v.Version)
		}()
	}
	wg.Wait()

	if len(errs) != 0 {
		t.Fatalf("%d concurrent registrations failed, first: %v", len(errs), errs[0])
	}
	seen := make(map[int]bool, writers)
	for _, v := range got {
		if seen[v] {
			t.Errorf("version %d was assigned twice", v)
		}
		seen[v] = true
	}
	for want := 1; want <= writers; want++ {
		if !seen[want] {
			t.Errorf("version %d was never assigned; numbering has a gap", want)
		}
	}
}

func TestCreateVersionRequiresTheModel(t *testing.T) {
	s := testStore(t)
	_, err := s.CreateVersion(context.Background(), newVersionReq("never-created", "x"))
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("CreateVersion error = %v, want ErrNotFound", err)
	}
}

func TestVersionRoundTrip(t *testing.T) {
	ctx := context.Background()
	s := testStore(t)
	if _, err := s.CreateModel(ctx, "fraud-score", ""); err != nil {
		t.Fatal(err)
	}

	req := newVersionReq("fraud-score", "seed")
	created, err := s.CreateVersion(ctx, req)
	if err != nil {
		t.Fatal(err)
	}

	got, err := s.GetVersion(ctx, "fraud-score", created.Version)
	if err != nil {
		t.Fatalf("GetVersion: %v", err)
	}
	if got.Digest != req.Digest {
		t.Errorf("digest = %s, want %s", got.Digest, req.Digest)
	}
	if got.SizeBytes != req.SizeBytes {
		t.Errorf("size = %d, want %d", got.SizeBytes, req.SizeBytes)
	}
	if got.Notes != req.Notes {
		t.Errorf("notes = %q, want %q", got.Notes, req.Notes)
	}
	if got.Runtime != RuntimeXGBoost {
		t.Errorf("runtime = %q", got.Runtime)
	}
	// Feature order is the input contract, so it has to survive storage
	// unchanged — a set would round trip in any order and score the wrong
	// column.
	if strings.Join(got.Features, ",") != strings.Join(req.Features, ",") {
		t.Errorf("features = %v, want %v (order is part of the contract)", got.Features, req.Features)
	}
	if got.Ref() != "fraud-score:1" {
		t.Errorf("Ref() = %q, want fraud-score:1", got.Ref())
	}
}

func TestListVersionsIsOrdered(t *testing.T) {
	ctx := context.Background()
	s := testStore(t)
	if _, err := s.CreateModel(ctx, "ordered", ""); err != nil {
		t.Fatal(err)
	}
	for i := range 3 {
		if _, err := s.CreateVersion(ctx, newVersionReq("ordered", fmt.Sprintf("v%d", i))); err != nil {
			t.Fatal(err)
		}
	}

	list, err := s.ListVersions(ctx, "ordered")
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 3 {
		t.Fatalf("ListVersions returned %d entries, want 3", len(list))
	}
	for i, v := range list {
		if v.Version != i+1 {
			t.Errorf("entry %d is version %d, want %d", i, v.Version, i+1)
		}
	}

	if empty, err := s.ListVersions(ctx, "ordered-but-absent"); err != nil || len(empty) != 0 {
		t.Errorf("ListVersions for an unknown model = %v, %v; want empty, nil", empty, err)
	}
}

func TestListModels(t *testing.T) {
	ctx := context.Background()
	s := testStore(t)

	for _, n := range []string{"alpha", "beta"} {
		if _, err := s.CreateModel(ctx, n, ""); err != nil {
			t.Fatal(err)
		}
	}
	list, err := s.ListModels(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 2 {
		t.Fatalf("ListModels returned %d, want 2", len(list))
	}
}

func TestMissingVersionLookups(t *testing.T) {
	ctx := context.Background()
	s := testStore(t)
	if _, err := s.CreateModel(ctx, "empty", ""); err != nil {
		t.Fatal(err)
	}

	if _, err := s.GetVersion(ctx, "empty", 7); !errors.Is(err, ErrNotFound) {
		t.Errorf("GetVersion error = %v, want ErrNotFound", err)
	}
	if _, err := s.LatestVersion(ctx, "empty"); !errors.Is(err, ErrNotFound) {
		t.Errorf("LatestVersion on a model with no versions = %v, want ErrNotFound", err)
	}
}

func TestNewVersionValidation(t *testing.T) {
	valid := newVersionReq("fraud-score", "seed")

	tests := []struct {
		name    string
		mutate  func(*NewVersion)
		wantErr string
	}{
		{"bad model name", func(v *NewVersion) { v.Model = "Bad Name" }, "model name"},
		{"unknown runtime", func(v *NewVersion) { v.Runtime = "onnx" }, "unknown runtime"},
		{"bad digest", func(v *NewVersion) { v.Digest = "nope" }, "digest"},
		{"no features", func(v *NewVersion) { v.Features = nil }, "feature names"},
		{"empty feature name", func(v *NewVersion) { v.Features = []string{"a", ""} }, "empty name"},
		{"duplicate feature", func(v *NewVersion) { v.Features = []string{"a", "a"} }, "appears twice"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := valid
			tc.mutate(&req)
			err := req.Validate()
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("Validate() = %v, want an error containing %q", err, tc.wantErr)
			}
		})
	}
	if err := valid.Validate(); err != nil {
		t.Errorf("Validate() on a valid request = %v", err)
	}
}

type testPolicy struct {
	Primary int            `json:"primary"`
	Weights map[string]int `json:"weights"`
}

func TestPolicyRoundTrip(t *testing.T) {
	ctx := context.Background()
	s := testStore(t)
	if _, err := s.CreateModel(ctx, "routed", ""); err != nil {
		t.Fatal(err)
	}

	want := testPolicy{Primary: 2, Weights: map[string]int{"1": 10, "2": 90}}
	if err := s.SavePolicy(ctx, "routed", want); err != nil {
		t.Fatalf("SavePolicy: %v", err)
	}

	var got testPolicy
	updated, err := s.LoadPolicy(ctx, "routed", &got)
	if err != nil {
		t.Fatalf("LoadPolicy: %v", err)
	}
	if got.Primary != want.Primary || got.Weights["2"] != 90 {
		t.Errorf("policy = %+v, want %+v", got, want)
	}
	if updated.IsZero() {
		t.Error("LoadPolicy returned a zero updated_at")
	}

	// Saving again replaces rather than duplicating: the model is the key.
	want.Primary = 3
	if err := s.SavePolicy(ctx, "routed", want); err != nil {
		t.Fatal(err)
	}
	if _, err := s.LoadPolicy(ctx, "routed", &got); err != nil {
		t.Fatal(err)
	}
	if got.Primary != 3 {
		t.Errorf("policy after update = %+v, want Primary 3", got)
	}

	all, err := s.ListPolicies(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 1 {
		t.Errorf("ListPolicies returned %d entries, want 1", len(all))
	}
}

func TestPolicyForUnknownModel(t *testing.T) {
	ctx := context.Background()
	s := testStore(t)

	// The foreign key is what stops a policy existing for a model that does
	// not, which would otherwise route traffic to nothing.
	if err := s.SavePolicy(ctx, "ghost", testPolicy{}); !errors.Is(err, ErrNotFound) {
		t.Errorf("SavePolicy for an unknown model = %v, want ErrNotFound", err)
	}
	var p testPolicy
	if _, err := s.LoadPolicy(ctx, "ghost", &p); !errors.Is(err, ErrNotFound) {
		t.Errorf("LoadPolicy for an unknown model = %v, want ErrNotFound", err)
	}
}

// TestDeletingAModelCascades documents what the foreign keys do, so that a
// later schema change that drops ON DELETE CASCADE fails here rather than
// leaving orphaned versions behind.
func TestDeletingAModelCascades(t *testing.T) {
	ctx := context.Background()
	s := testStore(t)
	if _, err := s.CreateModel(ctx, "temporary", ""); err != nil {
		t.Fatal(err)
	}
	if _, err := s.CreateVersion(ctx, newVersionReq("temporary", "x")); err != nil {
		t.Fatal(err)
	}
	if err := s.SavePolicy(ctx, "temporary", testPolicy{}); err != nil {
		t.Fatal(err)
	}

	if _, err := s.pool.Exec(ctx, `DELETE FROM models WHERE name = $1`, "temporary"); err != nil {
		t.Fatal(err)
	}
	versions, err := s.ListVersions(ctx, "temporary")
	if err != nil {
		t.Fatal(err)
	}
	if len(versions) != 0 {
		t.Errorf("%d versions survived their model", len(versions))
	}
	policies, err := s.ListPolicies(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(policies) != 0 {
		t.Errorf("%d policies survived their model", len(policies))
	}
}

func TestMigrateIsIdempotent(t *testing.T) {
	s := testStore(t)
	for range 3 {
		if err := s.Migrate(context.Background()); err != nil {
			t.Fatalf("Migrate: %v", err)
		}
	}
}

func TestOpenRejectsBadDSN(t *testing.T) {
	ctx := context.Background()
	if _, err := Open(ctx, "not-a-dsn"); err == nil {
		t.Error("Open with a malformed DSN succeeded, want an error")
	}
	if _, err := Open(ctx, "postgres://nobody@127.0.0.1:1/none?sslmode=disable&connect_timeout=1"); err == nil {
		t.Error("Open against a dead address succeeded, want an error")
	}
}
