package registry

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/sahilkalgutkar/modelforge/internal/artifact"
)

//go:embed schema.sql
var schemaSQL string

// Store is the registry's persistence.
type Store struct {
	pool *pgxpool.Pool
}

// Open connects to Postgres and applies the schema.
func Open(ctx context.Context, dsn string) (*Store, error) {
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("registry: connect: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("registry: ping: %w", err)
	}
	s := &Store{pool: pool}
	if err := s.Migrate(ctx); err != nil {
		pool.Close()
		return nil, err
	}
	return s, nil
}

// Migrate applies the schema. It is idempotent.
func (s *Store) Migrate(ctx context.Context) error {
	if _, err := s.pool.Exec(ctx, schemaSQL); err != nil {
		return fmt.Errorf("registry: apply schema: %w", err)
	}
	return nil
}

// Close releases the connection pool.
func (s *Store) Close() { s.pool.Close() }

// CreateModel registers a new model name.
func (s *Store) CreateModel(ctx context.Context, name, description string) (Model, error) {
	if err := ValidateName(name); err != nil {
		return Model{}, err
	}
	m := Model{Name: name, Description: description, CreatedAt: Now()}

	_, err := s.pool.Exec(ctx,
		`INSERT INTO models (name, description, created_at) VALUES ($1, $2, $3)`,
		m.Name, m.Description, m.CreatedAt)
	if isUniqueViolation(err) {
		return Model{}, fmt.Errorf("%w: model %q", ErrAlreadyExists, name)
	}
	if err != nil {
		return Model{}, fmt.Errorf("registry: create model: %w", err)
	}
	return m, nil
}

// GetModel returns a model by name.
func (s *Store) GetModel(ctx context.Context, name string) (Model, error) {
	var m Model
	err := s.pool.QueryRow(ctx,
		`SELECT name, description, created_at FROM models WHERE name = $1`, name).
		Scan(&m.Name, &m.Description, &m.CreatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return Model{}, fmt.Errorf("%w: model %q", ErrNotFound, name)
	}
	if err != nil {
		return Model{}, fmt.Errorf("registry: get model: %w", err)
	}
	m.CreatedAt = m.CreatedAt.UTC()
	return m, nil
}

// ListModels returns every model, oldest first.
func (s *Store) ListModels(ctx context.Context) ([]Model, error) {
	rows, err := s.pool.Query(ctx,
		`SELECT name, description, created_at FROM models ORDER BY created_at, name`)
	if err != nil {
		return nil, fmt.Errorf("registry: list models: %w", err)
	}
	defer rows.Close()

	var out []Model
	for rows.Next() {
		var m Model
		if err := rows.Scan(&m.Name, &m.Description, &m.CreatedAt); err != nil {
			return nil, fmt.Errorf("registry: scan model: %w", err)
		}
		m.CreatedAt = m.CreatedAt.UTC()
		out = append(out, m)
	}
	return out, rows.Err()
}

// CreateVersion registers a new version of a model and returns it with the
// version number the database assigned.
//
// The number is assigned here rather than supplied by the caller because it has
// to be dense and monotonic per model, and no caller can know what the next one
// is without racing every other caller. The transaction takes a row lock on the
// model first, so concurrent registrations of the same model serialise on that
// row and each sees the previous one's insert; registrations of *different*
// models do not contend at all, which a table-level lock or a single global
// sequence would not give.
//
// The primary key on (model_name, version) is the actual guarantee. The lock
// makes the common path efficient; the constraint is what makes it correct,
// and it is why a duplicate number surfaces as an error rather than as two
// versions quietly sharing an identity.
func (s *Store) CreateVersion(ctx context.Context, req NewVersion) (Version, error) {
	if err := req.Validate(); err != nil {
		return Version{}, err
	}

	features, err := json.Marshal(req.Features)
	if err != nil {
		return Version{}, fmt.Errorf("registry: encode features: %w", err)
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return Version{}, fmt.Errorf("registry: begin: %w", err)
	}
	defer tx.Rollback(ctx) //nolint:errcheck // no-op once committed

	var exists string
	err = tx.QueryRow(ctx, `SELECT name FROM models WHERE name = $1 FOR UPDATE`, req.Model).Scan(&exists)
	if errors.Is(err, pgx.ErrNoRows) {
		return Version{}, fmt.Errorf("%w: model %q", ErrNotFound, req.Model)
	}
	if err != nil {
		return Version{}, fmt.Errorf("registry: lock model: %w", err)
	}

	var next int
	err = tx.QueryRow(ctx,
		`SELECT COALESCE(MAX(version), 0) + 1 FROM model_versions WHERE model_name = $1`,
		req.Model).Scan(&next)
	if err != nil {
		return Version{}, fmt.Errorf("registry: next version: %w", err)
	}

	v := Version{
		Model:     req.Model,
		Version:   next,
		Runtime:   req.Runtime,
		Digest:    req.Digest,
		SizeBytes: req.SizeBytes,
		Features:  req.Features,
		Notes:     req.Notes,
		CreatedAt: Now(),
	}
	_, err = tx.Exec(ctx,
		`INSERT INTO model_versions
		   (model_name, version, runtime, digest, size_bytes, features, notes, created_at)
		 VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`,
		v.Model, v.Version, v.Runtime, string(v.Digest), v.SizeBytes, features, v.Notes, v.CreatedAt)
	if isUniqueViolation(err) {
		return Version{}, fmt.Errorf("%w: %s", ErrAlreadyExists, v.Ref())
	}
	if err != nil {
		return Version{}, fmt.Errorf("registry: insert version: %w", err)
	}

	if err := tx.Commit(ctx); err != nil {
		return Version{}, fmt.Errorf("registry: commit: %w", err)
	}
	return v, nil
}

// GetVersion returns one version of a model.
func (s *Store) GetVersion(ctx context.Context, model string, version int) (Version, error) {
	return s.scanVersion(s.pool.QueryRow(ctx,
		`SELECT model_name, version, runtime, digest, size_bytes, features, notes, created_at
		   FROM model_versions WHERE model_name = $1 AND version = $2`, model, version),
		fmt.Sprintf("%s:%d", model, version))
}

// LatestVersion returns the highest-numbered version of a model.
func (s *Store) LatestVersion(ctx context.Context, model string) (Version, error) {
	return s.scanVersion(s.pool.QueryRow(ctx,
		`SELECT model_name, version, runtime, digest, size_bytes, features, notes, created_at
		   FROM model_versions WHERE model_name = $1 ORDER BY version DESC LIMIT 1`, model),
		fmt.Sprintf("latest version of %q", model))
}

// ListVersions returns every version of a model, lowest number first.
func (s *Store) ListVersions(ctx context.Context, model string) ([]Version, error) {
	rows, err := s.pool.Query(ctx,
		`SELECT model_name, version, runtime, digest, size_bytes, features, notes, created_at
		   FROM model_versions WHERE model_name = $1 ORDER BY version`, model)
	if err != nil {
		return nil, fmt.Errorf("registry: list versions: %w", err)
	}
	defer rows.Close()

	var out []Version
	for rows.Next() {
		var (
			v        Version
			digest   string
			features []byte
		)
		if err := rows.Scan(&v.Model, &v.Version, &v.Runtime, &digest, &v.SizeBytes,
			&features, &v.Notes, &v.CreatedAt); err != nil {
			return nil, fmt.Errorf("registry: scan version: %w", err)
		}
		v.Digest = artifact.Digest(digest)
		if err := json.Unmarshal(features, &v.Features); err != nil {
			return nil, fmt.Errorf("registry: decode features: %w", err)
		}
		v.CreatedAt = v.CreatedAt.UTC()
		out = append(out, v)
	}
	return out, rows.Err()
}

func (s *Store) scanVersion(row pgx.Row, what string) (Version, error) {
	var (
		v        Version
		digest   string
		features []byte
	)
	err := row.Scan(&v.Model, &v.Version, &v.Runtime, &digest, &v.SizeBytes,
		&features, &v.Notes, &v.CreatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return Version{}, fmt.Errorf("%w: %s", ErrNotFound, what)
	}
	if err != nil {
		return Version{}, fmt.Errorf("registry: get version: %w", err)
	}
	v.Digest = artifact.Digest(digest)
	if err := json.Unmarshal(features, &v.Features); err != nil {
		return Version{}, fmt.Errorf("registry: decode features: %w", err)
	}
	v.CreatedAt = v.CreatedAt.UTC()
	return v, nil
}

// SavePolicy stores a model's traffic policy, replacing whatever was there.
func (s *Store) SavePolicy(ctx context.Context, model string, policy any) error {
	body, err := json.Marshal(policy)
	if err != nil {
		return fmt.Errorf("registry: encode policy: %w", err)
	}
	tag, err := s.pool.Exec(ctx,
		`INSERT INTO deployments (model_name, policy, updated_at) VALUES ($1, $2, $3)
		 ON CONFLICT (model_name) DO UPDATE SET policy = EXCLUDED.policy, updated_at = EXCLUDED.updated_at`,
		model, body, Now())
	if isForeignKeyViolation(err) {
		return fmt.Errorf("%w: model %q", ErrNotFound, model)
	}
	if err != nil {
		return fmt.Errorf("registry: save policy: %w", err)
	}
	if tag.RowsAffected() == 0 {
		return fmt.Errorf("registry: save policy affected no rows")
	}
	return nil
}

// LoadPolicy reads a model's traffic policy into out.
func (s *Store) LoadPolicy(ctx context.Context, model string, out any) (time.Time, error) {
	var (
		body      []byte
		updatedAt time.Time
	)
	err := s.pool.QueryRow(ctx,
		`SELECT policy, updated_at FROM deployments WHERE model_name = $1`, model).
		Scan(&body, &updatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return time.Time{}, fmt.Errorf("%w: policy for %q", ErrNotFound, model)
	}
	if err != nil {
		return time.Time{}, fmt.Errorf("registry: load policy: %w", err)
	}
	if err := json.Unmarshal(body, out); err != nil {
		return time.Time{}, fmt.Errorf("registry: decode policy: %w", err)
	}
	return updatedAt.UTC(), nil
}

// ListPolicies returns the raw policy documents for every model that has one,
// so the server can load them all at startup in a single query rather than one
// per model.
func (s *Store) ListPolicies(ctx context.Context) (map[string][]byte, error) {
	rows, err := s.pool.Query(ctx, `SELECT model_name, policy FROM deployments`)
	if err != nil {
		return nil, fmt.Errorf("registry: list policies: %w", err)
	}
	defer rows.Close()

	out := make(map[string][]byte)
	for rows.Next() {
		var (
			name string
			body []byte
		)
		if err := rows.Scan(&name, &body); err != nil {
			return nil, fmt.Errorf("registry: scan policy: %w", err)
		}
		out[name] = body
	}
	return out, rows.Err()
}

// Postgres error codes, from
// https://www.postgresql.org/docs/current/errcodes-appendix.html
const (
	pgUniqueViolation     = "23505"
	pgForeignKeyViolation = "23503"
)

func isUniqueViolation(err error) bool { return hasPGCode(err, pgUniqueViolation) }

func isForeignKeyViolation(err error) bool { return hasPGCode(err, pgForeignKeyViolation) }

func hasPGCode(err error, code string) bool {
	var pgErr *pgconn.PgError
	return errors.As(err, &pgErr) && pgErr.Code == code
}
