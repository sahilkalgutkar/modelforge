package cli

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// loginAnd signs in against the fake provider and returns the client and the
// stored credential.
func loginAnd(t *testing.T, p *loginIDP, addr string) (*Client, Credential) {
	t.Helper()
	var out strings.Builder
	c := NewClient(addr, &out)
	if err := c.Login(context.Background(), browser(t)); err != nil {
		t.Fatalf("Login: %v", err)
	}
	cred := LoadCredential()
	if !cred.Valid() {
		t.Fatal("login stored no credential")
	}
	return c, cred
}

// TestLoginStoresARefreshToken is the precondition for everything else: the
// offline_access scope has to actually produce one.
func TestLoginStoresARefreshToken(t *testing.T) {
	p := newLoginIDP(t)
	addr := loginServer(t, p)

	var out strings.Builder
	c := NewClient(addr, &out)
	if err := c.Login(context.Background(), browser(t)); err != nil {
		t.Fatal(err)
	}

	cred := LoadCredential()
	if cred.RefreshToken == "" {
		t.Fatal("no refresh token was stored")
	}
	if !cred.CanRefresh() {
		t.Errorf("credential cannot refresh: %+v", cred)
	}
	// The issuer and client id travel with it, so renewing needs nothing but
	// this file — reaching the server to rediscover them would make refreshing
	// depend on the thing an expired token cannot reach.
	if cred.Issuer == "" || cred.ClientID == "" {
		t.Errorf("credential lacks what a refresh needs: %+v", cred)
	}
	if cred.Expiry.IsZero() {
		t.Error("credential has no recorded expiry, so nothing can tell when to renew")
	}
	if !strings.Contains(out.String(), "renews itself") {
		t.Errorf("login should say the session renews:\n%s", out.String())
	}
}

// TestExpiredSessionRenewsItself is the feature: a stale token is replaced
// without anybody signing in again.
func TestExpiredSessionRenewsItself(t *testing.T) {
	p := newLoginIDP(t)
	addr := loginServer(t, p)
	c, cred := loginAnd(t, p, addr)

	// Age the credential past its expiry, as the clock would.
	cred.Expiry = time.Now().Add(-time.Minute)
	if err := SaveCredential(cred); err != nil {
		t.Fatal(err)
	}
	if !cred.Stale(time.Now()) {
		t.Fatal("the test credential is not stale")
	}

	if err := c.EnsureFresh(context.Background()); err != nil {
		t.Fatalf("EnsureFresh: %v", err)
	}

	renewed := LoadCredential()
	if renewed.IDToken == cred.IDToken {
		t.Error("the ID token was not replaced")
	}
	if !renewed.Expiry.After(time.Now()) {
		t.Errorf("the renewed credential is still expired: %v", renewed.Expiry)
	}
	if c.Token != renewed.IDToken {
		t.Error("the client is still holding the old token")
	}

	// And the renewed token actually works against the server.
	if out, code := run(t, addr, "whoami"); code != 0 || !strings.Contains(out, "sahil@example.com") {
		t.Fatalf("whoami after a renewal: %d %s", code, out)
	}
}

// TestFreshSessionIsNotRenewed keeps the provider out of the path of every
// command. Refreshing on each invocation would put an external dependency
// between a user and a tool that had a perfectly good token.
func TestFreshSessionIsNotRenewed(t *testing.T) {
	p := newLoginIDP(t)
	addr := loginServer(t, p)
	c, _ := loginAnd(t, p, addr)

	for range 5 {
		if err := c.EnsureFresh(context.Background()); err != nil {
			t.Fatal(err)
		}
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	if p.refreshCount != 0 {
		t.Errorf("a valid session was refreshed %d times", p.refreshCount)
	}
}

// TestRefreshHappensBeforeExpiry covers the skew. A token with two seconds left
// is not usable even though it has not technically expired, so it is renewed
// early rather than allowed to fail in flight.
func TestRefreshHappensBeforeExpiry(t *testing.T) {
	cred := Credential{IDToken: "x", RefreshToken: "y", Issuer: "i", ClientID: "c"}

	cred.Expiry = time.Now().Add(10 * time.Minute)
	if cred.Stale(time.Now()) {
		t.Error("a credential with ten minutes left was called stale")
	}
	cred.Expiry = time.Now().Add(10 * time.Second)
	if !cred.Stale(time.Now()) {
		t.Error("a credential expiring in ten seconds was called fresh; it will die in flight")
	}
	cred.Expiry = time.Now().Add(-time.Hour)
	if !cred.Stale(time.Now()) {
		t.Error("an expired credential was called fresh")
	}

	// A credential with no recorded expiry — what an older version wrote — is
	// left alone rather than refreshed on every command.
	noExpiry := Credential{IDToken: "x"}
	if noExpiry.Stale(time.Now()) {
		t.Error("a credential with no expiry was treated as stale")
	}
}

// TestRotatedRefreshTokenIsStored is the failure that would otherwise appear
// days later: a provider issues a new refresh token, the old one is retired,
// and a client that kept the old one works right up until it does not.
func TestRotatedRefreshTokenIsStored(t *testing.T) {
	p := newLoginIDP(t)
	p.rotate = true
	addr := loginServer(t, p)
	c, cred := loginAnd(t, p, addr)

	original := cred.RefreshToken

	// Three renewals in a row. With rotation, each depends on the previous
	// one having been stored.
	for i := range 3 {
		stale := LoadCredential()
		stale.Expiry = time.Now().Add(-time.Minute)
		if err := SaveCredential(stale); err != nil {
			t.Fatal(err)
		}
		if err := c.EnsureFresh(context.Background()); err != nil {
			t.Fatalf("renewal %d: %v", i+1, err)
		}
	}

	final := LoadCredential()
	if final.RefreshToken == original {
		t.Error("the rotated refresh token was not stored; the next renewal would fail")
	}
	if !strings.HasSuffix(final.RefreshToken, "-rotated-rotated-rotated") {
		t.Errorf("refresh token = %q, want three rotations applied", final.RefreshToken)
	}
}

// TestRevokedRefreshTokenSaysToSignInAgain: none of the causes are fixable by
// retrying, so the message should not look like a transient failure.
func TestRevokedRefreshTokenSaysToSignInAgain(t *testing.T) {
	p := newLoginIDP(t)
	addr := loginServer(t, p)
	c, cred := loginAnd(t, p, addr)

	// The provider forgets the refresh token, as it would after a revocation
	// or an administrator ending the session.
	p.mu.Lock()
	p.refresh = map[string]bool{}
	p.mu.Unlock()

	cred.Expiry = time.Now().Add(-time.Minute)
	if err := SaveCredential(cred); err != nil {
		t.Fatal(err)
	}

	err := c.EnsureFresh(context.Background())
	if err == nil {
		t.Fatal("a revoked refresh token renewed successfully")
	}
	if !strings.Contains(err.Error(), "login") {
		t.Errorf("the error should tell the user to sign in again: %v", err)
	}
}

// TestConcurrentRefreshDoesNotBreakTheSession is the race a file lock exists
// for. Two commands at once both see a stale token; with rotation the second
// refresh invalidates the first, and whichever writes last can leave a
// credential the other already broke.
func TestConcurrentRefreshDoesNotBreakTheSession(t *testing.T) {
	p := newLoginIDP(t)
	p.rotate = true
	addr := loginServer(t, p)
	_, cred := loginAnd(t, p, addr)

	cred.Expiry = time.Now().Add(-time.Minute)
	if err := SaveCredential(cred); err != nil {
		t.Fatal(err)
	}

	const workers = 8
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var out strings.Builder
			c := NewClient(addr, &out)
			if err := c.EnsureFresh(context.Background()); err != nil {
				errs <- err
			}
		}()
	}
	wg.Wait()
	close(errs)

	for err := range errs {
		t.Errorf("a concurrent renewal failed: %v", err)
	}

	// Exactly one refresh should have happened: the others found the token
	// already renewed when they got the lock.
	p.mu.Lock()
	count := p.refreshCount
	p.mu.Unlock()
	if count != 1 {
		t.Errorf("%d refreshes for one stale credential; the lock is not preventing the stampede", count)
	}

	// And what is on disk still works.
	final := LoadCredential()
	if !final.Valid() || final.Stale(time.Now()) {
		t.Fatalf("the surviving credential is unusable: %+v", final)
	}
	if out, code := run(t, addr, "whoami"); code != 0 {
		t.Fatalf("the session is broken after concurrent renewals: %d %s", code, out)
	}
}

// TestLogoutRevokesAtTheProvider is the compensating control for storing a
// longer-lived secret. Deleting a local file was never logging out.
func TestLogoutRevokesAtTheProvider(t *testing.T) {
	p := newLoginIDP(t)
	addr := loginServer(t, p)
	c, cred := loginAnd(t, p, addr)

	var out strings.Builder
	c.Out = &out
	if err := c.Logout(context.Background()); err != nil {
		t.Fatal(err)
	}

	p.mu.Lock()
	revoked := append([]string(nil), p.revoked...)
	p.mu.Unlock()

	if len(revoked) == 0 {
		t.Fatal("logout did not revoke anything at the provider")
	}
	// The refresh token is the one that matters: it is the long-lived half.
	if revoked[0] != cred.RefreshToken {
		t.Errorf("revoked %q, want the refresh token", revoked[0])
	}
	if !strings.Contains(out.String(), "revoked this session") {
		t.Errorf("logout should report the revocation:\n%s", out.String())
	}
	if LoadCredential().Valid() {
		t.Error("the credential survived logout")
	}
}

// TestLogoutWithoutARevocationEndpointSaysSo. "Logged out" meaning two
// different things depending on the provider is how somebody ends up believing
// a credential is dead when it is not.
func TestLogoutWithoutARevocationEndpointSaysSo(t *testing.T) {
	p := newLoginIDP(t)
	p.noRevocation = true
	addr := loginServer(t, p)
	c, _ := loginAnd(t, p, addr)

	var out strings.Builder
	c.Out = &out
	if err := c.Logout(context.Background()); err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(out.String(), "no revocation endpoint") {
		t.Errorf("logout should say the session could not be ended remotely:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "stays valid until it expires") {
		t.Errorf("logout should say the token is still live:\n%s", out.String())
	}
	if LoadCredential().Valid() {
		t.Error("the credential survived logout")
	}
}

// TestCredentialIsWrittenAtomically: a crash partway through must not leave a
// truncated token where a valid one was, because that fails in a way that
// looks like a server problem.
func TestCredentialIsWrittenAtomically(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "credential")
	t.Setenv("MODELFORGE_CREDENTIAL_FILE", path)

	if err := SaveCredential(Credential{IDToken: "first", Issuer: "i", ClientID: "c"}); err != nil {
		t.Fatal(err)
	}
	if err := SaveCredential(Credential{IDToken: "second", Issuer: "i", ClientID: "c"}); err != nil {
		t.Fatal(err)
	}
	if got := LoadCredential(); got.IDToken != "second" {
		t.Errorf("credential = %q, want the second write", got.IDToken)
	}

	// No temp files left behind.
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".credential-") {
			t.Errorf("a temporary credential file was left behind: %s", e.Name())
		}
	}
}

func TestCredentialRoundTrip(t *testing.T) {
	t.Setenv("MODELFORGE_CREDENTIAL_FILE", filepath.Join(t.TempDir(), "credential"))

	want := Credential{
		IDToken: "id", RefreshToken: "refresh",
		Expiry: time.Now().Add(time.Hour).Truncate(time.Second).UTC(),
		Issuer: "https://idp.example.com", ClientID: "cli",
	}
	if err := SaveCredential(want); err != nil {
		t.Fatal(err)
	}
	got := LoadCredential()
	if got.IDToken != want.IDToken || got.RefreshToken != want.RefreshToken ||
		got.Issuer != want.Issuer || got.ClientID != want.ClientID {
		t.Errorf("round trip = %+v, want %+v", got, want)
	}
	if !got.Expiry.Equal(want.Expiry) {
		t.Errorf("expiry round trip = %v, want %v", got.Expiry, want.Expiry)
	}
}

func TestRefreshRejectsACredentialThatCannot(t *testing.T) {
	var out strings.Builder
	c := NewClient("http://localhost:1", &out)

	if _, err := c.Refresh(context.Background(), Credential{IDToken: "only-an-id-token"}); err == nil {
		t.Fatal("Refresh accepted a credential with no refresh token")
	}
}

// TestEnsureFreshIsANoOpWithoutALogin keeps every command from paying for a
// feature nobody is using.
func TestEnsureFreshIsANoOpWithoutALogin(t *testing.T) {
	t.Setenv("MODELFORGE_CREDENTIAL_FILE", filepath.Join(t.TempDir(), "absent"))

	var out strings.Builder
	c := NewClient("http://localhost:1", &out)
	if err := c.EnsureFresh(context.Background()); err != nil {
		t.Fatalf("EnsureFresh with no login = %v, want nil", err)
	}
}
