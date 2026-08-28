// Package registry is the control plane: it records what models exist, what
// versions each has, and which artifact each version resolves to.
//
// The organising rule is that a version is immutable. Once created, its
// artifact digest and its feature schema never change — only the deployment
// policy pointing at it does. That is what makes "roll back to version 3" a
// meaningful instruction rather than a hopeful one: version 3 today is the same
// bytes and the same input contract as version 3 last week, so rolling back
// restores a known state instead of whatever has since been written over it.
package registry

import (
	"errors"
	"fmt"
	"regexp"
	"time"

	"github.com/sahilkalgutkar/modelforge/internal/artifact"
)

// Errors the store returns for conditions callers are expected to handle.
var (
	ErrNotFound      = errors.New("registry: not found")
	ErrAlreadyExists = errors.New("registry: already exists")
)

// Model is a named prediction task. Versions belong to it, and traffic is
// routed per model rather than per version.
type Model struct {
	Name        string    `json:"name"`
	Description string    `json:"description,omitempty"`
	CreatedAt   time.Time `json:"created_at"`
}

// Version is one immutable trained model: an artifact plus the input contract
// needed to score it.
type Version struct {
	Model     string          `json:"model"`
	Version   int             `json:"version"`
	Runtime   string          `json:"runtime"`
	Digest    artifact.Digest `json:"digest"`
	SizeBytes int64           `json:"size_bytes"`
	Features  []string        `json:"features"`
	Notes     string          `json:"notes,omitempty"`
	CreatedAt time.Time       `json:"created_at"`
}

// Ref names a specific version, for logs, metric labels and API responses.
func (v Version) Ref() string { return fmt.Sprintf("%s:%d", v.Model, v.Version) }

// RuntimeXGBoost is the only runtime implemented so far. It is a field rather
// than an assumption so that adding a second one is a registry change and not a
// rewrite of everything that reads a version.
const RuntimeXGBoost = "xgboost"

// NewVersion is the request to register a trained model.
type NewVersion struct {
	Model     string
	Runtime   string
	Digest    artifact.Digest
	SizeBytes int64
	// Features is the ordered list of feature names the artifact expects. The
	// order is the contract: the runtime scores a dense row, so the names are
	// the only thing that stops a caller's "amount" from being read as the
	// model's "account_age".
	Features []string
	Notes    string
}

// nameRE is deliberately strict. Model names end up in URLs, in Prometheus
// label values and in filesystem-adjacent contexts, and a name that is safe in
// all three is easier to guarantee up front than to sanitise at each use.
var nameRE = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,62}[a-z0-9]$`)

// ValidateName checks a model name.
func ValidateName(name string) error {
	if !nameRE.MatchString(name) {
		return fmt.Errorf("registry: model name %q must be 2-64 characters of lowercase letters, digits, dot, dash or underscore, starting and ending alphanumeric", name)
	}
	return nil
}

// Validate checks a version registration request.
func (n NewVersion) Validate() error {
	if err := ValidateName(n.Model); err != nil {
		return err
	}
	if n.Runtime != RuntimeXGBoost {
		return fmt.Errorf("registry: unknown runtime %q", n.Runtime)
	}
	if _, err := artifact.ParseDigest(string(n.Digest)); err != nil {
		return err
	}
	if len(n.Features) == 0 {
		return errors.New("registry: a version must declare its feature names")
	}
	seen := make(map[string]struct{}, len(n.Features))
	for i, f := range n.Features {
		if f == "" {
			return fmt.Errorf("registry: feature %d has an empty name", i)
		}
		if _, dup := seen[f]; dup {
			// A duplicate name makes the request-to-row mapping ambiguous:
			// there would be two columns a caller could mean by one key.
			return fmt.Errorf("registry: feature %q appears twice", f)
		}
		seen[f] = struct{}{}
	}
	return nil
}

// truncateForStore rounds a timestamp down to microseconds.
//
// Go's time.Now() is nanosecond-resolution and Postgres timestamptz stores
// microseconds, so a value written and read back does not equal the one held in
// memory. The rounding happens here, at the single point where a timestamp is
// created for storage, rather than at each comparison — a check added at the
// call site only fixes the caller that remembered to add it.
func truncateForStore(t time.Time) time.Time { return t.Truncate(time.Microsecond).UTC() }

// Now returns a timestamp at the precision the store can represent.
func Now() time.Time { return truncateForStore(time.Now()) }
