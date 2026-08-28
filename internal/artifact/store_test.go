package artifact

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func newStore(t *testing.T) *FileStore {
	t.Helper()
	s, err := NewFileStore(t.TempDir())
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	return s
}

func TestPutReturnsContentDigest(t *testing.T) {
	s := newStore(t)
	body := []byte("a model, notionally")

	d, n, err := s.Put(strings.NewReader(string(body)))
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	if n != int64(len(body)) {
		t.Errorf("Put wrote %d bytes, want %d", n, len(body))
	}

	sum := sha256.Sum256(body)
	if want := Digest(hex.EncodeToString(sum[:])); d != want {
		t.Errorf("digest = %s, want %s", d, want)
	}

	r, err := s.Open(d)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer r.Close()
	got, _ := io.ReadAll(r)
	if string(got) != string(body) {
		t.Errorf("read back %q, want %q", got, body)
	}
}

// TestPutIsIdempotent is the property the whole design rests on: the same bytes
// are the same artifact. Two uploads of one model must not become two versions
// pointing at two copies.
func TestPutIsIdempotent(t *testing.T) {
	s := newStore(t)

	d1, _, err := s.Put(strings.NewReader("identical"))
	if err != nil {
		t.Fatal(err)
	}
	d2, _, err := s.Put(strings.NewReader("identical"))
	if err != nil {
		t.Fatal(err)
	}
	if d1 != d2 {
		t.Fatalf("same bytes produced different digests: %s vs %s", d1, d2)
	}

	// And exactly one blob on disk.
	var blobs int
	filepath.WalkDir(s.Root(), func(path string, e os.DirEntry, err error) error {
		if err == nil && !e.IsDir() {
			blobs++
		}
		return nil
	})
	if blobs != 1 {
		t.Errorf("store holds %d files, want 1", blobs)
	}
}

// TestConcurrentPutOfSameContent covers the race the temp-file-and-rename
// design exists to make safe. Without it, two writers would be appending to one
// destination file at once and a reader could see a half-written model.
func TestConcurrentPutOfSameContent(t *testing.T) {
	s := newStore(t)
	body := strings.Repeat("weights and biases ", 5000)

	const writers = 16
	digests := make([]Digest, writers)
	var wg sync.WaitGroup
	for i := range writers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			d, _, err := s.Put(strings.NewReader(body))
			if err != nil {
				t.Errorf("Put: %v", err)
				return
			}
			digests[i] = d
		}()
	}
	wg.Wait()

	for i, d := range digests {
		if d != digests[0] {
			t.Fatalf("writer %d got digest %s, writer 0 got %s", i, d, digests[0])
		}
	}

	r, err := s.Open(digests[0])
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	got, _ := io.ReadAll(r)
	if string(got) != body {
		t.Errorf("blob is %d bytes, want %d — a concurrent write was not atomic", len(got), len(body))
	}
}

// TestPutLeavesNoTempFiles guards against the store filling up with partial
// uploads. A failed read mid-stream must clean up after itself.
func TestPutLeavesNoTempFiles(t *testing.T) {
	s := newStore(t)

	if _, _, err := s.Put(failingReader{}); err == nil {
		t.Fatal("Put with a failing reader succeeded, want an error")
	}

	entries, err := os.ReadDir(s.Root())
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".upload-") {
			t.Errorf("failed upload left a temp file behind: %s", e.Name())
		}
	}
}

type failingReader struct{}

func (failingReader) Read([]byte) (int, error) { return 0, errors.New("read failed") }

func TestOpenMissingBlob(t *testing.T) {
	s := newStore(t)
	sum := sha256.Sum256([]byte("never stored"))
	d := Digest(hex.EncodeToString(sum[:]))

	_, err := s.Open(d)
	if !errors.Is(err, ErrNotFound) {
		t.Errorf("Open error = %v, want ErrNotFound", err)
	}
	if _, err := s.Path(d); !errors.Is(err, ErrNotFound) {
		t.Errorf("Path error = %v, want ErrNotFound", err)
	}
	ok, err := s.Exists(d)
	if err != nil || ok {
		t.Errorf("Exists = %v, %v; want false, nil", ok, err)
	}
}

func TestExistsAfterPut(t *testing.T) {
	s := newStore(t)
	d, _, err := s.Put(strings.NewReader("stored"))
	if err != nil {
		t.Fatal(err)
	}
	ok, err := s.Exists(d)
	if err != nil || !ok {
		t.Errorf("Exists = %v, %v; want true, nil", ok, err)
	}
	path, err := s.Path(d)
	if err != nil {
		t.Fatalf("Path: %v", err)
	}
	if !strings.HasPrefix(filepath.Base(path), string(d)) {
		t.Errorf("Path = %q, want it to be named for the digest", path)
	}
}

// TestParseDigestRejectsTraversal is the security-relevant case: a digest
// arrives from an HTTP request and is joined onto the store root, so anything
// that is not 64 lowercase hex characters must be refused before it becomes a
// path.
func TestParseDigestRejectsTraversal(t *testing.T) {
	valid := strings.Repeat("ab", sha256.Size)

	for _, tc := range []struct{ name, in string }{
		{"path traversal", "../../../etc/passwd"},
		{"too short", "abc123"},
		{"too long", valid + "ff"},
		{"not hex", strings.Repeat("zz", sha256.Size)},
		{"uppercase", strings.ToUpper(valid)},
		{"empty", ""},
		{"slash inside a correct-length string", valid[:60] + "/../"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := ParseDigest(tc.in); err == nil {
				t.Fatalf("ParseDigest(%q) succeeded, want an error", tc.in)
			}
		})
	}

	got, err := ParseDigest(valid)
	if err != nil || string(got) != valid {
		t.Fatalf("ParseDigest(valid) = %q, %v", got, err)
	}
}

// TestStoreRejectsUnparseableDigest makes sure the guard is actually on the
// filesystem path, not only on the parser.
func TestStoreRejectsUnparseableDigest(t *testing.T) {
	s := newStore(t)
	bad := Digest("../../etc/passwd")

	if _, err := s.Open(bad); err == nil || errors.Is(err, ErrNotFound) {
		t.Errorf("Open with a traversal digest = %v, want a validation error", err)
	}
	if _, err := s.Exists(bad); err == nil {
		t.Error("Exists with a traversal digest succeeded, want a validation error")
	}
}

func TestNewFileStoreRequiresADirectory(t *testing.T) {
	if _, err := NewFileStore(""); err == nil {
		t.Fatal("NewFileStore(\"\") succeeded, want an error")
	}
	// A path whose parent is a regular file cannot be created.
	f := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(f, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := NewFileStore(filepath.Join(f, "sub")); err == nil {
		t.Fatal("NewFileStore under a regular file succeeded, want an error")
	}
}

func TestDigestShort(t *testing.T) {
	long := Digest(strings.Repeat("a", 64))
	if got := long.Short(); got != strings.Repeat("a", 12) {
		t.Errorf("Short() = %q", got)
	}
	short := Digest("abc")
	if got := short.Short(); got != "abc" {
		t.Errorf("Short() on a short digest = %q, want %q", got, "abc")
	}
	if got := long.String(); len(got) != 64 {
		t.Errorf("String() length = %d, want 64", len(got))
	}
}
