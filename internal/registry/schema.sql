-- modelforge control-plane schema.
--
-- Applied on startup and idempotent, so a fresh database and a restart follow
-- the same path. The registry is small and rarely changed, so a migration tool
-- would add a dependency and a failure mode for less than it gives back.

CREATE TABLE IF NOT EXISTS models (
    name        TEXT PRIMARY KEY,
    description TEXT        NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS model_versions (
    model_name TEXT        NOT NULL REFERENCES models (name) ON DELETE CASCADE,
    version    INTEGER     NOT NULL,
    runtime    TEXT        NOT NULL,
    digest     TEXT        NOT NULL,
    size_bytes BIGINT      NOT NULL,
    -- The ordered feature names the artifact expects. Stored as JSON because
    -- the order is part of the contract and a JSON array preserves it, which a
    -- side table would only do with an extra ordinal column to sort by.
    features   JSONB       NOT NULL,
    notes      TEXT        NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL,

    -- Version numbers are per model and assigned by the database. The primary
    -- key is what actually enforces that: even if two concurrent registrations
    -- somehow computed the same next number, only one row can exist.
    PRIMARY KEY (model_name, version)
);

CREATE INDEX IF NOT EXISTS model_versions_digest_idx ON model_versions (digest);

CREATE TABLE IF NOT EXISTS deployments (
    model_name TEXT PRIMARY KEY REFERENCES models (name) ON DELETE CASCADE,
    -- The full traffic policy as one document. It is read and written whole on
    -- every change, and storing it as one row makes a policy update a single
    -- atomic statement — a normalised set of per-version weight rows would
    -- need a transaction to avoid ever being observed summing to less than 1.
    policy     JSONB       NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);
