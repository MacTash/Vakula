"""Ordered, additive SQLite migrations for Vakula.

Every step raises ``PRAGMA user_version`` by exactly one and only ever adds.
Nothing here drops, renames or rewrites a table, so a database created by any
earlier release stays readable and keeps all of its rows.

New columns go through :func:`add_column`, because SQLite has no
``ADD COLUMN IF NOT EXISTS`` and a plain ALTER fails on a database that already
has the column.
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 3

_V1 = """
CREATE TABLE IF NOT EXISTS intelligence (
    id INTEGER PRIMARY KEY,
    collected_at TEXT NOT NULL,
    category TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    location TEXT NOT NULL DEFAULT '',
    latitude REAL,
    longitude REAL,
    severity TEXT NOT NULL DEFAULT 'info',
    confidence REAL NOT NULL DEFAULT 0.5,
    tags TEXT NOT NULL DEFAULT '[]',
    raw_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(category, source_url, title)
);
CREATE INDEX IF NOT EXISTS idx_intel_collected ON intelligence(collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_intel_category ON intelligence(category);
"""

_V2 = """
CREATE TABLE IF NOT EXISTS source_items (
    id INTEGER PRIMARY KEY,
    source_key TEXT NOT NULL,
    platform TEXT NOT NULL,
    author TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    published_at TEXT NOT NULL DEFAULT '',
    fetched_at TEXT NOT NULL,
    backend TEXT NOT NULL DEFAULT '',
    media_json TEXT NOT NULL DEFAULT '[]',
    raw_json TEXT NOT NULL DEFAULT '{}',
    saved INTEGER NOT NULL DEFAULT 0,
    UNIQUE(platform, source_key)
);
CREATE INDEX IF NOT EXISTS idx_source_items_fetched ON source_items(fetched_at DESC);
CREATE INDEX IF NOT EXISTS idx_source_items_platform ON source_items(platform);
"""

# v3: the intelligence-fusion foundation.
#
# source_items gains the observation columns in place. There is deliberately no
# second observations table: one physical row per retrieved post means one copy
# of the source text, so provenance cannot drift against what the user sees.
_V3_TABLES = """
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'OTHER',
    homepage TEXT NOT NULL DEFAULT '',
    reliability REAL,
    reliability_basis TEXT NOT NULL DEFAULT '',
    correction_count INTEGER NOT NULL DEFAULT 0,
    avg_latency_s REAL,
    geographic_coverage TEXT NOT NULL DEFAULT '',
    topic_specialties TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS locations (
    id INTEGER PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    country TEXT NOT NULL DEFAULT '',
    region TEXT NOT NULL DEFAULT '',
    location_type TEXT NOT NULL DEFAULT 'OTHER',
    latitude REAL,
    longitude REAL,
    radius_km REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'OTHER',
    country TEXT NOT NULL DEFAULT '',
    aliases TEXT NOT NULL DEFAULT '[]',
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    event_type TEXT NOT NULL DEFAULT 'OTHER',
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    start_time TEXT NOT NULL DEFAULT '',
    end_time TEXT NOT NULL DEFAULT '',
    location_id INTEGER,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    confidence REAL NOT NULL DEFAULT 0,
    confidence_basis TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_observations (
    event_id INTEGER NOT NULL,
    observation_id INTEGER NOT NULL,
    role TEXT NOT NULL DEFAULT 'CORROBORATES',
    weight REAL NOT NULL DEFAULT 1,
    PRIMARY KEY (event_id, observation_id)
);
CREATE INDEX IF NOT EXISTS idx_event_obs_observation ON event_observations(observation_id);

-- Entity and event edges in one table. A dedicated event_entities table would
-- duplicate the same rows for the common IN_VOLVED_IN case.
CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY,
    from_kind TEXT NOT NULL,
    from_id INTEGER NOT NULL,
    rel_type TEXT NOT NULL,
    to_kind TEXT NOT NULL,
    to_id INTEGER NOT NULL,
    observation_id INTEGER,
    confidence REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE (from_kind, from_id, rel_type, to_kind, to_id)
);
CREATE INDEX IF NOT EXISTS idx_relationships_from ON relationships(from_kind, from_id);
CREATE INDEX IF NOT EXISTS idx_relationships_to ON relationships(to_kind, to_id);

CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY,
    region_key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    window_days INTEGER NOT NULL DEFAULT 14,
    state_json TEXT NOT NULL DEFAULT '{}',
    body TEXT NOT NULL DEFAULT '',
    origin TEXT NOT NULL DEFAULT 'TEMPLATE',
    model TEXT NOT NULL DEFAULT '',
    analytic_confidence REAL NOT NULL DEFAULT 0,
    analytic_basis TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_assessments_region ON assessments(region_key, created_at DESC);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY,
    region_key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    horizon_days INTEGER NOT NULL DEFAULT 14,
    statement TEXT NOT NULL,
    probability REAL NOT NULL DEFAULT 0,
    engine TEXT NOT NULL DEFAULT '',
    basis_json TEXT NOT NULL DEFAULT '{}',
    explanation TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL DEFAULT 'OPEN',
    resolved_at TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_predictions_region ON predictions(region_key, created_at DESC);

-- Contradictions are rows rather than a computed view because a human or a
-- later review has to be able to move one to RESOLVED and keep that decision.
CREATE TABLE IF NOT EXISTS contradictions (
    id INTEGER PRIMARY KEY,
    event_id INTEGER,
    kind TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'UNRESOLVED',
    confidence REAL NOT NULL DEFAULT 0,
    detail_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    resolved_at TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_contradictions_event ON contradictions(event_id);

CREATE TABLE IF NOT EXISTS watchlists (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    region_key TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
"""

# Canonical read path for the intelligence layer. Aliases present the stored row
# in observation vocabulary. content is original_content, not body: body tracks
# the latest re-fetch for the Live Feed, while original_content is what the
# source first returned and is what an assessment may cite.
_V3_VIEW = """
CREATE VIEW IF NOT EXISTS observations AS
SELECT
    id                                       AS id,
    source_id                                AS source_id,
    source_type                              AS source_type,
    platform                                 AS platform,
    source_key                               AS source_key,
    author                                   AS author,
    published_at                             AS "timestamp",
    COALESCE(NULLIF(original_content, ''), body) AS content,
    source_url                               AS url,
    raw_json                                 AS metadata,
    location_id                              AS location,
    language                                 AS language,
    fetched_at                               AS collected_at,
    content_hash                             AS "hash",
    backend                                  AS backend,
    saved                                    AS saved,
    media_json                               AS media
FROM source_items;
"""

_V3_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_source_items_hash ON source_items(content_hash);
CREATE INDEX IF NOT EXISTS idx_source_items_source ON source_items(source_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type, start_time DESC);
CREATE INDEX IF NOT EXISTS idx_events_location ON events(location_id);
"""

# Columns added to source_items in v3, as (name, declaration).
_V3_COLUMNS = (
    ("source_id", "INTEGER"),
    ("source_type", "TEXT NOT NULL DEFAULT ''"),
    ("location_id", "INTEGER"),
    ("language", "TEXT NOT NULL DEFAULT ''"),
    ("original_content", "TEXT NOT NULL DEFAULT ''"),
    ("content_hash", "TEXT NOT NULL DEFAULT ''"),
)


def columns(db: sqlite3.Connection, table: str) -> set[str]:
    """Column names currently present on a table, or empty if it is absent."""
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def add_column(db: sqlite3.Connection, table: str, column: str, declaration: str) -> bool:
    """Add a column only when it is missing. Returns True when it was added."""
    if column in columns(db, table):
        return False
    db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
    return True


def _migrate_to_v3(db: sqlite3.Connection) -> None:
    for column, declaration in _V3_COLUMNS:
        add_column(db, "source_items", column, declaration)
    db.executescript(_V3_TABLES)
    db.executescript(_V3_VIEW)
    db.executescript(_V3_INDEXES)


def migrate(db: sqlite3.Connection) -> int:
    """Bring a database up to :data:`SCHEMA_VERSION`. Returns the version reached.

    Runs every step above the stored version, so a database at version 0 with
    only an ``intelligence`` table is brought fully forward rather than being
    assumed to already be at v2.
    """
    version = db.execute("PRAGMA user_version").fetchone()[0]
    steps = (
        (1, _V1),
        (2, _V2),
        (3, None),
    )
    for target, script in steps:
        if version >= target:
            continue
        if script is not None:
            db.executescript(script)
        else:
            _migrate_to_v3(db)
        db.execute(f"PRAGMA user_version = {target}")
        version = target
    return version
