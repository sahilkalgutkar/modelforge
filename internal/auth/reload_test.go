package auth

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func writeTokenFile(t *testing.T, lines ...string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tokens")
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestRotationWithoutRestart is the whole feature: a new credential starts
// working and an old one stops, in a running process.
func TestRotationWithoutRestart(t *testing.T) {
	const oldTok, newTok = "the-old-token", "the-new-token"

	a := testAuth(t, entry("ci", "admin", oldTok))
	if _, err := a.Authenticate(request(oldTok)); err != nil {
		t.Fatalf("the old token does not work to begin with: %v", err)
	}

	// Step one of a real rotation: both valid, so clients can be moved over at
	// their own pace. This overlap is the entire point — swapping in one step
	// means every client that has not been updated yet starts failing.
	change, err := a.Reload([]string{
		entry("ci", "admin", oldTok),
		entry("ci-next", "admin", newTok),
	})
	if err != nil {
		t.Fatalf("Reload: %v", err)
	}
	if len(change.Added) != 1 || change.Added[0] != "ci-next" {
		t.Errorf("Change.Added = %v, want [ci-next]", change.Added)
	}
	if len(change.Removed) != 0 {
		t.Errorf("Change.Removed = %v, want none", change.Removed)
	}
	for _, tok := range []string{oldTok, newTok} {
		if _, err := a.Authenticate(request(tok)); err != nil {
			t.Errorf("token %q rejected during the overlap: %v", tok, err)
		}
	}

	// Step two: the old credential is withdrawn.
	change, err = a.Reload([]string{entry("ci-next", "admin", newTok)})
	if err != nil {
		t.Fatal(err)
	}
	if len(change.Removed) != 1 || change.Removed[0] != "ci" {
		t.Errorf("Change.Removed = %v, want [ci]", change.Removed)
	}
	if _, err := a.Authenticate(request(oldTok)); !errors.Is(err, ErrBadCredential) {
		t.Errorf("the withdrawn token still works: %v", err)
	}
	if _, err := a.Authenticate(request(newTok)); err != nil {
		t.Errorf("the new token stopped working: %v", err)
	}
}

// TestABadReloadKeepsTheRunningSet is the property that makes rotation safe to
// attempt. A malformed or empty configuration must be a failed reload somebody
// sees in the logs, not an outage on a server that was working a second ago.
func TestABadReloadKeepsTheRunningSet(t *testing.T) {
	a := testAuth(t, entry("ci", "admin", knownToken))

	for _, tc := range []struct {
		name    string
		entries []string
	}{
		{"malformed entry", []string{"this is not an entry"}},
		{"unknown scope", []string{entry("ci", "wizard", knownToken)}},
		{"bad digest", []string{"ci:admin:nope"}},
		{"empty set", nil},
		{"only comments", []string{"# everything is commented out"}},
		{"duplicate digest", []string{entry("a", "admin", knownToken), entry("b", "read", knownToken)}},
		{"unparseable expiry", []string{entry("ci", "admin", knownToken) + ":not-a-date"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := a.Reload(tc.entries); err == nil {
				t.Fatal("Reload accepted an invalid configuration")
			}
			// The credential that was working must still work.
			if _, err := a.Authenticate(request(knownToken)); err != nil {
				t.Fatalf("a failed reload broke the running token set: %v", err)
			}
		})
	}
}

// TestReloadIsAtomic covers the window rotation exists to avoid. A request
// arriving mid-swap must see one whole generation of the set, never a state
// where the new token is present and the old one is already gone.
func TestReloadIsAtomic(t *testing.T) {
	const oldTok, newTok = "old-token-value", "new-token-value"
	a := testAuth(t, entry("ci", "admin", oldTok))

	var (
		stop     atomic.Bool
		gaps     atomic.Int64
		wg       sync.WaitGroup
		reloaded atomic.Int64
	)

	// Readers hammer the authenticator with both credentials. At every instant
	// at least one of them must be accepted.
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for !stop.Load() {
				_, errOld := a.Authenticate(request(oldTok))
				_, errNew := a.Authenticate(request(newTok))
				if errOld != nil && errNew != nil {
					gaps.Add(1)
				}
			}
		}()
	}

	// A writer rotates back and forth, always keeping the overlap.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := range 300 {
			var entries []string
			if i%2 == 0 {
				entries = []string{entry("ci", "admin", oldTok), entry("next", "admin", newTok)}
			} else {
				entries = []string{entry("next", "admin", newTok), entry("ci", "admin", oldTok)}
			}
			if _, err := a.Reload(entries); err != nil {
				t.Error(err)
				return
			}
			reloaded.Add(1)
		}
		stop.Store(true)
	}()

	wg.Wait()
	if reloaded.Load() != 300 {
		t.Fatalf("only %d reloads completed", reloaded.Load())
	}
	if n := gaps.Load(); n != 0 {
		t.Errorf("%d requests saw neither credential accepted; the swap is not atomic", n)
	}
}

// TestExpiredTokenIsRefused covers the deadline, which is what makes "we will
// remove the old credential later" safe: later happens on its own.
func TestExpiredTokenIsRefused(t *testing.T) {
	clk := newClock()
	deadline := clk.now().Add(time.Hour)

	a, err := New([]string{
		entry("permanent", "admin", knownToken),
		entry("temporary", "read", "temp-token") + ":" + deadline.Format(time.RFC3339),
	}, WithClock(clk.now))
	if err != nil {
		t.Fatal(err)
	}

	if _, err := a.Authenticate(request("temp-token")); err != nil {
		t.Fatalf("the temporary token does not work before its deadline: %v", err)
	}

	clk.advance(2 * time.Hour)

	_, err = a.Authenticate(request("temp-token"))
	if !errors.Is(err, ErrExpiredCredential) {
		t.Fatalf("expired token gave %v, want ErrExpiredCredential", err)
	}
	// The message has to say which credential and when, or the holder is left
	// hunting a phantom typo.
	if !strings.Contains(err.Error(), "temporary") {
		t.Errorf("the error does not name the token: %v", err)
	}

	// Expiry is enforced on the request, not at load time, so no reload was
	// needed for that to happen — and unrelated credentials are unaffected.
	if _, err := a.Authenticate(request(knownToken)); err != nil {
		t.Errorf("an unrelated token expired too: %v", err)
	}
}

// TestExpiredIsDistinctFromUnrecognised matters operationally. Telling the
// holder their token expired leaks nothing they do not already know — they are
// holding it — and is the difference between a minute of rotation and an hour
// of debugging.
func TestExpiredIsDistinctFromUnrecognised(t *testing.T) {
	clk := newClock()
	past := clk.now().Add(-time.Hour).Format(time.RFC3339)

	a, err := New([]string{entry("stale", "admin", "stale-token") + ":" + past}, WithClock(clk.now))
	if err != nil {
		t.Fatal(err)
	}

	_, expiredErr := a.Authenticate(request("stale-token"))
	_, unknownErr := a.Authenticate(request("never-issued"))

	if !errors.Is(expiredErr, ErrExpiredCredential) {
		t.Errorf("expired token gave %v", expiredErr)
	}
	if !errors.Is(unknownErr, ErrBadCredential) {
		t.Errorf("unknown token gave %v", unknownErr)
	}
	if errors.Is(expiredErr, ErrBadCredential) {
		t.Error("an expired token reads as unrecognised, which sends the holder hunting a typo")
	}
}

// TestReloadReportsAlreadyExpiredTokens surfaces a configuration that will not
// do what its author expects, at the moment they install it.
func TestReloadReportsAlreadyExpiredTokens(t *testing.T) {
	clk := newClock()
	past := clk.now().Add(-time.Hour).Format(time.RFC3339)

	a, err := New([]string{entry("live", "admin", knownToken)}, WithClock(clk.now))
	if err != nil {
		t.Fatal(err)
	}

	change, err := a.Reload([]string{
		entry("live", "admin", knownToken),
		entry("already-dead", "read", "dead-token") + ":" + past,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(change.Expired) != 1 || change.Expired[0] != "already-dead" {
		t.Errorf("Change.Expired = %v, want [already-dead]", change.Expired)
	}
	if change.Total != 2 {
		t.Errorf("Change.Total = %d, want 2", change.Total)
	}
}

func TestExpiryIsOptionalAndBackwardsCompatible(t *testing.T) {
	// A three-field entry, the format that predates expiry, must still load
	// and must never expire.
	a := testAuth(t, entry("ci", "admin", knownToken))
	tok, err := a.Authenticate(request(knownToken))
	if err != nil {
		t.Fatal(err)
	}
	if !tok.NotAfter.IsZero() {
		t.Errorf("a token with no expiry got NotAfter = %v", tok.NotAfter)
	}
	if tok.Expired(time.Now().Add(100 * 365 * 24 * time.Hour)) {
		t.Error("a token with no deadline expired anyway")
	}

	// An empty fourth field means the same thing.
	a2 := testAuth(t, entry("ci", "admin", knownToken)+":")
	if _, err := a2.Authenticate(request(knownToken)); err != nil {
		t.Errorf("an entry with an empty expiry field was rejected: %v", err)
	}
}

// TestUnparseableExpiryIsRejected: a typo in a deadline must not silently
// produce a credential that never expires, which is the opposite of what was
// written down.
func TestUnparseableExpiryIsRejected(t *testing.T) {
	for _, bad := range []string{"tomorrow", "2026-13-45", "1735689600"} {
		_, err := New([]string{entry("ci", "admin", knownToken) + ":" + bad})
		if err == nil {
			t.Errorf("expiry %q was accepted", bad)
			continue
		}
		if !strings.Contains(err.Error(), "RFC 3339") {
			t.Errorf("error for %q should say what format is wanted: %v", bad, err)
		}
	}
}

func TestReloadIsRefusedWhenAuthIsDisabled(t *testing.T) {
	if _, err := Disabled().Reload([]string{entry("ci", "admin", knownToken)}); err == nil {
		t.Fatal("Reload succeeded on a disabled Authenticator")
	}
}

func TestChangeEmpty(t *testing.T) {
	if !(Change{Total: 3}).Empty() {
		t.Error("a Change with no additions or removals is not Empty")
	}
	if (Change{Added: []string{"x"}}).Empty() {
		t.Error("a Change with an addition reports Empty")
	}
}

func TestLoadedAtAndCount(t *testing.T) {
	clk := newClock()
	a, err := New([]string{entry("ci", "admin", knownToken)}, WithClock(clk.now))
	if err != nil {
		t.Fatal(err)
	}
	first := a.LoadedAt()
	if a.Count() != 1 {
		t.Errorf("Count = %d, want 1", a.Count())
	}

	clk.advance(time.Hour)
	if _, err := a.Reload([]string{
		entry("ci", "admin", knownToken),
		entry("other", "read", "other-token"),
	}); err != nil {
		t.Fatal(err)
	}
	if !a.LoadedAt().After(first) {
		t.Error("LoadedAt did not advance after a reload")
	}
	if a.Count() != 2 {
		t.Errorf("Count = %d, want 2", a.Count())
	}
}

// --- file source ---

func TestReadTokenFile(t *testing.T) {
	path := writeTokenFile(t,
		"# issued 2026-08-31 for the deploy pipeline",
		entry("ci", "admin", knownToken),
		"",
		"   ",
		"# the dashboard, read only",
		entry("dash", "read", "dash-token"),
	)

	entries, err := ReadTokenFile(path)
	if err != nil {
		t.Fatalf("ReadTokenFile: %v", err)
	}
	// Comments and blank lines are dropped, so a file can carry the notes that
	// make an old entry safe to remove later.
	if len(entries) != 2 {
		t.Fatalf("read %d entries, want 2: %v", len(entries), entries)
	}
}

func TestReadTokenFileErrors(t *testing.T) {
	if _, err := ReadTokenFile(filepath.Join(t.TempDir(), "absent")); err == nil {
		t.Error("ReadTokenFile succeeded on a missing file")
	}

	empty := writeTokenFile(t, "# nothing but a comment")
	_, err := ReadTokenFile(empty)
	if err == nil || !strings.Contains(err.Error(), "no entries") {
		t.Errorf("an effectively empty file gave %v, want a 'no entries' error", err)
	}
}

// TestReloadFromFileRotates is the mechanism as an operator uses it: rewrite
// the file, reload, done.
func TestReloadFromFileRotates(t *testing.T) {
	const oldTok, newTok = "file-old", "file-new"

	path := writeTokenFile(t, entry("ci", "admin", oldTok))
	entries, err := ReadTokenFile(path)
	if err != nil {
		t.Fatal(err)
	}
	a, err := New(entries)
	if err != nil {
		t.Fatal(err)
	}

	// Overlap, then withdraw.
	if err := os.WriteFile(path, []byte(
		entry("ci", "admin", oldTok)+"\n"+entry("ci-next", "admin", newTok)+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := a.ReloadFromFile(path); err != nil {
		t.Fatal(err)
	}
	for _, tok := range []string{oldTok, newTok} {
		if _, err := a.Authenticate(request(tok)); err != nil {
			t.Errorf("token %q rejected during overlap: %v", tok, err)
		}
	}

	if err := os.WriteFile(path, []byte(entry("ci-next", "admin", newTok)+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := a.ReloadFromFile(path); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Authenticate(request(oldTok)); err == nil {
		t.Error("the withdrawn token still works after the file was rewritten")
	}
}

// TestReloadFromAMissingFileKeepsTheRunningSet is the case a Kubernetes secret
// update actually produces: the file is briefly absent or half-written while
// the volume is being swapped.
func TestReloadFromAMissingFileKeepsTheRunningSet(t *testing.T) {
	path := writeTokenFile(t, entry("ci", "admin", knownToken))
	a := testAuth(t, entry("ci", "admin", knownToken))

	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if _, err := a.ReloadFromFile(path); err == nil {
		t.Fatal("ReloadFromFile succeeded on a missing file")
	}
	if _, err := a.Authenticate(request(knownToken)); err != nil {
		t.Fatalf("a missing token file broke the running set: %v", err)
	}

	// And a truncated file — the other thing an in-place rewrite produces.
	if err := os.WriteFile(path, []byte("ci:admin:trunc"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := a.ReloadFromFile(path); err == nil {
		t.Fatal("ReloadFromFile succeeded on a truncated entry")
	}
	if _, err := a.Authenticate(request(knownToken)); err != nil {
		t.Fatalf("a truncated token file broke the running set: %v", err)
	}
}
