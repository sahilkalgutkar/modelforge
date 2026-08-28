// Package artifact stores model files by content hash.
//
// A model version has to mean exactly one set of bytes forever. Everything the
// serving layer offers — rolling back to a previous version, shadowing a
// candidate against production, reproducing a prediction someone is disputing —
// is only true if the artifact behind a version cannot change after the fact.
// Naming blobs by the SHA-256 of their contents gets that for free: the name
// cannot be reused for different bytes, so an overwrite is either a no-op or
// impossible rather than a silent substitution.
package artifact

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// ErrNotFound is returned when no blob is stored under a digest.
var ErrNotFound = errors.New("artifact: not found")

// Digest is the lowercase hex SHA-256 of a blob's contents.
type Digest string

// String returns the digest as a string.
func (d Digest) String() string { return string(d) }

// Short returns the first 12 characters, for logs and CLI output where the full
// 64 characters are noise.
func (d Digest) Short() string {
	if len(d) <= 12 {
		return string(d)
	}
	return string(d[:12])
}

// ParseDigest validates a digest received from outside the process.
//
// This is the guard on the only place a caller-supplied string reaches the
// filesystem. Without it a digest of "../../etc/passwd" would be joined onto
// the store root and read a file that is not an artifact, so the check is for
// path traversal as much as for typos.
func ParseDigest(s string) (Digest, error) {
	if len(s) != sha256.Size*2 {
		return "", fmt.Errorf("artifact: digest must be %d hex characters, got %d", sha256.Size*2, len(s))
	}
	if _, err := hex.DecodeString(s); err != nil {
		return "", fmt.Errorf("artifact: digest is not hex: %w", err)
	}
	if s != strings.ToLower(s) {
		return "", fmt.Errorf("artifact: digest must be lowercase")
	}
	return Digest(s), nil
}

// Store is a content-addressed blob store.
type Store interface {
	// Put writes the contents of r and returns the digest they hash to.
	// Writing bytes that are already stored is a no-op that returns the same
	// digest.
	Put(r io.Reader) (Digest, int64, error)

	// Open returns a reader over a stored blob, or ErrNotFound.
	Open(d Digest) (io.ReadCloser, error)

	// Path returns the on-disk location of a blob, for callers that need a
	// filename rather than a stream.
	Path(d Digest) (string, error)

	// Exists reports whether a blob is stored.
	Exists(d Digest) (bool, error)
}

// FileStore keeps blobs on a local filesystem.
type FileStore struct {
	root string
}

// NewFileStore opens (creating if necessary) a blob store rooted at dir.
func NewFileStore(dir string) (*FileStore, error) {
	if dir == "" {
		return nil, errors.New("artifact: store directory is required")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("artifact: create store: %w", err)
	}
	return &FileStore{root: dir}, nil
}

// Root returns the directory the store writes to.
func (s *FileStore) Root() string { return s.root }

// blobPath shards blobs one level deep by the first two hex characters.
//
// Without the shard every artifact lands in one directory, and directory
// listings and lookups degrade badly on most filesystems once that reaches
// tens of thousands of entries. Two characters gives 256 buckets, which is
// enough for any number of model versions a registry will realistically hold.
func (s *FileStore) blobPath(d Digest) string {
	return filepath.Join(s.root, string(d)[:2], string(d))
}

// Put streams r to a temporary file, hashing as it goes, then renames it into
// place under its digest.
//
// The hash is computed during the write rather than by reading the file back,
// so a large artifact is not read twice. The rename is what makes the store
// safe under concurrent writes and crashes: a reader either sees no blob or a
// complete one, never a partially written file that would parse as a truncated
// model. Two callers uploading the same bytes race harmlessly, since both
// produce identical contents under identical names.
func (s *FileStore) Put(r io.Reader) (Digest, int64, error) {
	if err := os.MkdirAll(s.root, 0o755); err != nil {
		return "", 0, fmt.Errorf("artifact: create store: %w", err)
	}
	tmp, err := os.CreateTemp(s.root, ".upload-*")
	if err != nil {
		return "", 0, fmt.Errorf("artifact: create temp file: %w", err)
	}
	tmpName := tmp.Name()
	// Remove the temp file on every path that does not rename it away. Once
	// the rename succeeds this is a no-op on a name that no longer exists.
	defer func() {
		tmp.Close()
		os.Remove(tmpName)
	}()

	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(tmp, h), r)
	if err != nil {
		return "", 0, fmt.Errorf("artifact: write blob: %w", err)
	}
	// Flush to disk before the rename. Without this a crash between the two
	// can leave a correctly named blob whose contents are incomplete — which
	// is worse than no blob at all, because the name asserts the contents.
	if err := tmp.Sync(); err != nil {
		return "", 0, fmt.Errorf("artifact: sync blob: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return "", 0, fmt.Errorf("artifact: close blob: %w", err)
	}

	d := Digest(hex.EncodeToString(h.Sum(nil)))
	dst := s.blobPath(d)
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return "", 0, fmt.Errorf("artifact: create shard: %w", err)
	}
	if err := os.Rename(tmpName, dst); err != nil {
		return "", 0, fmt.Errorf("artifact: commit blob: %w", err)
	}
	return d, n, nil
}

// Open returns a reader over a stored blob.
func (s *FileStore) Open(d Digest) (io.ReadCloser, error) {
	path, err := s.Path(d)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("%w: %s", ErrNotFound, d.Short())
	}
	if err != nil {
		return nil, fmt.Errorf("artifact: open blob: %w", err)
	}
	return f, nil
}

// Path returns the on-disk location of a blob, checking that it exists.
func (s *FileStore) Path(d Digest) (string, error) {
	if _, err := ParseDigest(string(d)); err != nil {
		return "", err
	}
	path := s.blobPath(d)
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		return "", fmt.Errorf("%w: %s", ErrNotFound, d.Short())
	} else if err != nil {
		return "", fmt.Errorf("artifact: stat blob: %w", err)
	}
	return path, nil
}

// Exists reports whether a blob is stored.
func (s *FileStore) Exists(d Digest) (bool, error) {
	if _, err := ParseDigest(string(d)); err != nil {
		return false, err
	}
	_, err := os.Stat(s.blobPath(d))
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("artifact: stat blob: %w", err)
	}
	return true, nil
}
