package cli

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

// Credential is a stored login.
//
// Holding a refresh token on disk is a real escalation over holding only an ID
// token, and it is worth naming rather than glossing: the ID token expires in
// an hour, the refresh token can be good for weeks, so a stolen credential file
// is worth far more than it used to be. Three things pay for that. The file is
// 0600 in a 0700 directory. Rotation is honoured, so a provider that issues a
// new refresh token on every use invalidates whatever an attacker copied the
// moment the real user runs another command. And `logout` revokes at the
// provider rather than only deleting the file, which is what makes losing a
// laptop recoverable.
type Credential struct {
	IDToken      string    `json:"id_token"`
	RefreshToken string    `json:"refresh_token,omitempty"`
	Expiry       time.Time `json:"expiry,omitempty"`

	// Issuer and ClientID are stored so a refresh needs nothing but this file.
	// Reaching the server to rediscover them would make refreshing depend on
	// the thing the credential is for, which is backwards: an expired token is
	// exactly when the server will refuse to talk.
	Issuer   string `json:"issuer,omitempty"`
	ClientID string `json:"client_id,omitempty"`
}

// Valid reports whether there is anything usable here.
func (c Credential) Valid() bool { return c.IDToken != "" }

// CanRefresh reports whether this credential can renew itself.
func (c Credential) CanRefresh() bool {
	return c.RefreshToken != "" && c.Issuer != "" && c.ClientID != ""
}

// refreshSkew is how long before expiry a token is treated as stale.
//
// Refreshing slightly early rather than on a 401 means a request never fails
// for a reason the client could have prevented. The margin covers clock skew
// between this machine and the provider, plus the time the request itself will
// spend in flight — a token with two seconds left is not usable even though it
// has not technically expired.
const refreshSkew = 60 * time.Second

// Stale reports whether the ID token is expired or close enough that it should
// be renewed before use.
func (c Credential) Stale(now time.Time) bool {
	if c.Expiry.IsZero() {
		// No recorded expiry, which is what a credential written by an older
		// version looks like. Treating it as fresh is right: it either works,
		// or the server rejects it and says so.
		return false
	}
	return now.Add(refreshSkew).After(c.Expiry)
}

// CredentialPath is where a login is stored.
func CredentialPath() (string, error) {
	if p := os.Getenv("MODELFORGE_CREDENTIAL_FILE"); p != "" {
		return p, nil
	}
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", fmt.Errorf("locate a config directory: %w", err)
	}
	return filepath.Join(dir, "modelforge", "credential"), nil
}

// LoadCredential reads a stored login. A missing or unreadable file is an empty
// Credential rather than an error, since "not signed in" is an ordinary state.
func LoadCredential() Credential {
	path, err := CredentialPath()
	if err != nil {
		return Credential{}
	}
	body, err := os.ReadFile(path)
	if err != nil {
		return Credential{}
	}

	var c Credential
	if err := json.Unmarshal(body, &c); err == nil && c.IDToken != "" {
		return c
	}

	// Older versions wrote the bare token. Reading it keeps somebody who
	// upgrades signed in, rather than logging them out for a format change
	// they did not ask for. It has no refresh token, so it simply expires as
	// it did before.
	if token := strings.TrimSpace(string(body)); token != "" && !strings.HasPrefix(token, "{") {
		return Credential{IDToken: token}
	}
	return Credential{}
}

// SaveCredential writes a login atomically.
//
// Temp file plus rename, for the same reason the artifact store does it: a
// crash or a full disk partway through must not leave a truncated credential
// where a valid one was. A half-written token is worse than no token, because
// it fails in a way that looks like a server problem.
//
// Modes are set explicitly rather than left to the umask — 022 would leave a
// readable credential in a home directory other accounts on the machine can
// list — and an existing directory is tightened rather than trusted, since
// MkdirAll leaves the mode of one that already exists alone.
func SaveCredential(c Credential) error {
	path, err := CredentialPath()
	if err != nil {
		return err
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("create the credential directory: %w", err)
	}
	if err := os.Chmod(dir, 0o700); err != nil {
		return fmt.Errorf("secure the credential directory: %w", err)
	}

	body, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return fmt.Errorf("encode the credential: %w", err)
	}

	tmp, err := os.CreateTemp(dir, ".credential-*")
	if err != nil {
		return fmt.Errorf("create a temporary credential file: %w", err)
	}
	tmpName := tmp.Name()
	defer func() {
		tmp.Close()
		os.Remove(tmpName)
	}()

	// Tightened before anything secret is written to it: CreateTemp makes the
	// file 0600 already, but saying so here means a change to that default
	// cannot silently widen it.
	if err := tmp.Chmod(0o600); err != nil {
		return fmt.Errorf("secure the temporary credential file: %w", err)
	}
	if _, err := tmp.Write(append(body, '\n')); err != nil {
		return fmt.Errorf("write the credential: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("flush the credential: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close the credential: %w", err)
	}
	return os.Rename(tmpName, path)
}

// DeleteCredential removes a stored login.
func DeleteCredential() (bool, error) {
	path, err := CredentialPath()
	if err != nil {
		return false, err
	}
	if err := os.Remove(path); err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, fmt.Errorf("remove the credential: %w", err)
	}
	return true, nil
}

// withCredentialLock runs fn holding an exclusive lock on the credential file.
//
// Two commands running at once would otherwise both notice a stale token and
// both refresh. Against a provider that rotates refresh tokens — which is what
// OAuth 2.1 tells them to do — the second refresh invalidates the first, and
// whichever process writes last leaves a credential the other has already
// broken. The symptom is a login that mysteriously stops working when somebody
// runs two commands in parallel, which is a miserable thing to debug.
//
// The lock is advisory and process-level, which is all that is needed: the
// contention being prevented is between copies of this tool on one machine.
func withCredentialLock(fn func() error) error {
	path, err := CredentialPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return fmt.Errorf("create the credential directory: %w", err)
	}

	lockPath := path + ".lock"
	lock, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return fmt.Errorf("open the credential lock: %w", err)
	}
	defer lock.Close()

	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX); err != nil {
		return fmt.Errorf("lock the credential: %w", err)
	}
	defer syscall.Flock(int(lock.Fd()), syscall.LOCK_UN) //nolint:errcheck // closing releases it too

	return fn()
}

// ErrNotSignedIn means there is no usable credential.
var ErrNotSignedIn = errors.New("not signed in")
