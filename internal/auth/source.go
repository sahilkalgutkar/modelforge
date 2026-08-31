package auth

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// ReadTokenFile reads token entries from a file, one per line.
//
// Blank lines and lines beginning with # are ignored, so a rotation can be
// annotated with who issued a credential and why — which is most of what makes
// an old entry safe to remove six weeks later.
//
// A file is the reloadable source, rather than an API for minting tokens, and
// that is a deliberate limit on what a compromised credential can do. An admin
// endpoint that issues tokens turns any admin compromise into permanent access:
// the attacker mints a credential of their own and keeps it after the one they
// stole is revoked. Keeping issuance outside the serving process means
// rotation is done by whatever already manages secrets, and this server only
// ever learns digests.
//
// It also happens to be the mechanism that works with a Kubernetes Secret or a
// Vault agent, both of which rewrite a mounted file in place and neither of
// which can call an API.
func ReadTokenFile(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("auth: open token file: %w", err)
	}
	defer f.Close()

	var entries []string
	scanner := bufio.NewScanner(f)
	// Entries are short; the default 64KB line limit is ample, but a file of
	// binary rubbish should fail cleanly rather than allocate.
	scanner.Buffer(make([]byte, 0, 4096), 64*1024)

	for line := 1; scanner.Scan(); line++ {
		text := strings.TrimSpace(scanner.Text())
		if text == "" || strings.HasPrefix(text, "#") {
			continue
		}
		entries = append(entries, text)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("auth: read token file: %w", err)
	}
	if len(entries) == 0 {
		// Distinguished from a parse failure because the remedy is different:
		// an empty file is usually a half-finished edit or a secret that has
		// not been populated yet.
		return nil, fmt.Errorf("auth: token file %s contains no entries", path)
	}
	return entries, nil
}

// ReloadFromFile re-reads a token file and installs it.
//
// The read and the parse both happen before anything is swapped, so a file that
// is missing, unreadable, empty or malformed leaves the running set alone. A
// bad rotation should be a failed reload somebody can see in the logs, not an
// outage on a server that is otherwise healthy.
func (a *Authenticator) ReloadFromFile(path string) (Change, error) {
	entries, err := ReadTokenFile(path)
	if err != nil {
		return Change{}, err
	}
	return a.Reload(entries)
}
