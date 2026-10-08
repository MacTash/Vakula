"""Small, portable SQLite store owned exclusively by Vidur."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from platformdirs import user_cache_path, user_data_path

from vidur.schema import migrate

APP_NAME = "vidur"
LEGACY_APP_NAME = "geoscope"
DATABASE_FILE = "vidur.db"
LEGACY_DATABASE_FILE = "geoscope.db"


def setting(name: str, legacy: str) -> str:
    """Read a VIDUR_* setting, falling back to the pre-rename GEOSCOPE_* name."""
    return os.environ.get(name) or os.environ.get(legacy) or ""


def default_data_dir() -> Path:
    configured = setting("VIDUR_DATA_DIR", "GEOSCOPE_DATA_DIR")
    if configured:
        return Path(configured).expanduser()
    current = Path(user_data_path(APP_NAME, appauthor=False))
    legacy = Path(user_data_path(LEGACY_APP_NAME, appauthor=False))
    # A pre-rename install already has a populated geoscope directory. Keep using
    # it rather than stranding its database in a directory Vidur never reads.
    if not current.exists() and legacy.is_dir():
        return legacy
    return current


def cache_dir() -> Path:
    configured = setting("VIDUR_CACHE_DIR", "GEOSCOPE_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    current = Path(user_cache_path(APP_NAME, appauthor=False))
    legacy = Path(user_cache_path(LEGACY_APP_NAME, appauthor=False))
    if not current.exists() and legacy.is_dir():
        return legacy
    return current


def database_path() -> Path:
    folder = default_data_dir()
    if (folder / DATABASE_FILE).exists():
        return folder / DATABASE_FILE
    legacy = folder / LEGACY_DATABASE_FILE
    if legacy.is_file():
        return legacy
    return folder / DATABASE_FILE


def _copy_database(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return
    reader = sqlite3.connect(source)
    writer = sqlite3.connect(target)
    try:
        reader.backup(writer)
    finally:
        writer.close()
        reader.close()


def _migrate_project_database(target: Path) -> None:
    """Copy the checkout-local database once, preserving its contents."""
    project = Path.cwd()
    if not (project / "pyproject.toml").is_file() or not (project / APP_NAME).is_dir():
        return
    if target.exists():
        return
    data = project / "data"
    for candidate in (data / DATABASE_FILE, data / LEGACY_DATABASE_FILE):
        if candidate.is_file():
            _copy_database(candidate, target)
            return



@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _migrate_project_database(path)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    try:
        yield db
        db.commit()
    finally:
        db.close()


_gazetteer_seeded = False


def init_db() -> Path:
    """Create or forward-migrate Vidur's database. Safe to call repeatedly.

    Also seeds the curated gazetteer once per process, before any extraction can
    run, so a link in the database always refers to an entry a person curated.
    """
    global _gazetteer_seeded
    with connect() as db:
        migrate(db)
    if not _gazetteer_seeded:
        # Set before seeding, not after: upsert_location calls init_db() again,
        # so this flag doubles as the re-entrancy guard.
        _gazetteer_seeded = True
        try:
            from vidur import gazetteer
            gazetteer.seed()
        except Exception:
            _gazetteer_seeded = False
            raise
    return database_path()


def add_item(item: dict) -> bool:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    values = {
        "collected_at": item.get("collected_at", now), "category": item["category"].upper(),
        "title": item["title"], "summary": item.get("summary", ""),
        "source": item.get("source", ""), "source_url": item.get("source_url", ""),
        "location": item.get("location", ""), "latitude": item.get("latitude"),
        "longitude": item.get("longitude"), "severity": item.get("severity", "info"),
        "confidence": item.get("confidence", 0.5), "tags": json.dumps(item.get("tags", [])),
        "raw_json": json.dumps(item.get("raw", {}), default=str),
    }
    with connect() as db:
        cursor = db.execute("""
            INSERT OR IGNORE INTO intelligence
            (collected_at, category, title, summary, source, source_url, location,
             latitude, longitude, severity, confidence, tags, raw_json)
            VALUES (:collected_at, :category, :title, :summary, :source, :source_url,
                    :location, :latitude, :longitude, :severity, :confidence, :tags, :raw_json)
        """, values)
        return cursor.rowcount == 1


def list_items(category: str | None = None, query: str | None = None, limit: int = 30) -> list[dict]:
    init_db()
    where, values = [], []
    if category:
        where.append("category = ?")
        values.append(category.upper())
    if query:
        where.append("(title LIKE ? OR summary LIKE ? OR location LIKE ? OR tags LIKE ?)")
        values.extend([f"%{query}%"] * 4)
    statement = "SELECT * FROM intelligence"
    if where:
        statement += " WHERE " + " AND ".join(where)
    statement += " ORDER BY collected_at DESC LIMIT ?"
    values.append(max(1, min(limit, 500)))
    with connect() as db:
        return [dict(row) for row in db.execute(statement, values).fetchall()]


def stats() -> dict:
    init_db()
    with connect() as db:
        total = db.execute("SELECT COUNT(*) FROM intelligence").fetchone()[0]
        categories = [dict(row) for row in db.execute(
            "SELECT category, COUNT(*) AS count FROM intelligence GROUP BY category ORDER BY count DESC"
        )]
        recent = db.execute("SELECT MAX(collected_at) FROM intelligence").fetchone()[0]
        source_total = db.execute("SELECT COUNT(*) FROM source_items").fetchone()[0]
    return {"total": total, "source_items": source_total, "categories": categories,
            "latest": recent, "database": str(database_path())}


def content_digest(text: str) -> str:
    """Stable digest of source text, used to detect a publisher's later edit."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_source_item(item: dict) -> int:
    """Persist a source post without rewriting its body or media metadata.

    ``body`` follows the newest fetch so the Live Feed stays current, while
    ``original_content`` and ``content_hash`` are written once and never
    updated. That is what makes provenance durable: an assessment citing an
    observation keeps pointing at the text the source first returned, even if
    the publisher edits the post afterwards.
    """
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    platform = str(item.get("platform", "unknown")).lower()
    source_key = str(item.get("source_key") or item.get("id") or item.get("source_url") or "")
    if not source_key:
        digest = hashlib.sha256(
            f"{platform}\n{item.get('author', '')}\n{item.get('published_at', '')}\n{item.get('body', '')}"
            .encode("utf-8")
        ).hexdigest()
        source_key = digest
    body = str(item.get("body", ""))
    values = {
        "source_key": source_key,
        "platform": platform,
        "author": str(item.get("author", "")),
        "body": body,
        "original_content": body,
        "content_hash": content_digest(body),
        "source_url": str(item.get("source_url", "")),
        "published_at": str(item.get("published_at", "")),
        "fetched_at": str(item.get("fetched_at") or now),
        "backend": str(item.get("backend", "")),
        "media_json": json.dumps(item.get("media", []), ensure_ascii=False, default=str),
        "raw_json": json.dumps(item.get("raw", {}), ensure_ascii=False, default=str),
    }
    with connect() as db:
        db.execute("""
            INSERT INTO source_items
              (source_key, platform, author, body, original_content, content_hash,
               source_url, published_at, fetched_at, backend, media_json, raw_json)
            VALUES (:source_key, :platform, :author, :body, :original_content, :content_hash,
                    :source_url, :published_at, :fetched_at, :backend, :media_json, :raw_json)
            ON CONFLICT(platform, source_key) DO UPDATE SET
              author=excluded.author, body=excluded.body, source_url=excluded.source_url,
              published_at=excluded.published_at, fetched_at=excluded.fetched_at,
              backend=excluded.backend, media_json=excluded.media_json, raw_json=excluded.raw_json,
              original_content=COALESCE(NULLIF(source_items.original_content, ''), excluded.original_content),
              content_hash=COALESCE(NULLIF(source_items.content_hash, ''), excluded.content_hash)
        """, values)
        row = db.execute("SELECT id FROM source_items WHERE platform=? AND source_key=?",
                         (platform, source_key)).fetchone()
        return int(row[0])

def list_source_items(query: str | None = None, *, platform: str | None = None,
                      saved: bool | None = None, limit: int = 100) -> list[dict]:
    init_db()
    conditions, values = [], []
    if query:
        conditions.append("(body LIKE ? OR author LIKE ? OR platform LIKE ?)")
        values.extend([f"%{query}%"] * 3)
    if platform:
        conditions.append("platform = ?")
        values.append(platform.lower())
    if saved is not None:
        conditions.append("saved = ?")
        values.append(int(saved))
    statement = "SELECT * FROM source_items"
    if conditions:
        statement += " WHERE " + " AND ".join(conditions)
    statement += " ORDER BY fetched_at DESC LIMIT ?"
    values.append(max(1, min(limit, 500)))
    with connect() as db:
        rows = [dict(row) for row in db.execute(statement, values).fetchall()]
    for row in rows:
        row["media"] = json.loads(row.pop("media_json", "[]"))
        row["raw"] = json.loads(row.pop("raw_json", "{}"))
        row["saved"] = bool(row["saved"])
    return rows


def set_source_saved(item_id: int, saved: bool) -> bool:
    init_db()
    with connect() as db:
        cursor = db.execute("UPDATE source_items SET saved=? WHERE id=?", (int(saved), item_id))
        return cursor.rowcount > 0


def ensure_source(platform: str, backend: str = "", *, source_type: str = "OTHER",
                  name: str = "") -> int:
    """Return the id of the source that published this channel, creating it once.

    The key is the platform plus the backend Agent Reach reported, so two
    different backends for the same platform stay distinguishable in
    provenance while still sharing one row per publisher.
    """
    init_db()
    platform = (platform or "unknown").strip().lower()
    backend = (backend or "").strip()
    key = f"{platform}:{backend}" if backend else platform
    now = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        db.execute("""
            INSERT INTO sources (key, name, source_type, created_at, updated_at)
            VALUES (:key, :name, :source_type, :now, :now)
            ON CONFLICT(key) DO UPDATE SET updated_at=excluded.updated_at
        """, {"key": key, "name": name or backend or platform,
              "source_type": source_type or "OTHER", "now": now})
        row = db.execute("SELECT id FROM sources WHERE key=?", (key,)).fetchone()
        return int(row[0])


def insert_observation(item: dict, *, source_type: str = "OTHER", language: str = "",
                      enrich: bool = True) -> int:
    """Store one retrieved source record as an observation and return its id.

    This is the intelligence layer's write path and it goes through
    :func:`save_source_item`, so there is exactly one place where source text is
    stored. Nothing here summarises, translates or rewrites the body.

    ``enrich`` runs the deterministic extractor and links whatever the curated
    gazetteer recognises. It is pure and offline, so it never invents an entity
    and never puts model output into the intelligence graph.
    """
    observation_id = save_source_item(item)
    source_id = ensure_source(
        str(item.get("platform", "unknown")), str(item.get("backend", "")), source_type=source_type)
    with connect() as db:
        db.execute("UPDATE source_items SET source_id=?, source_type=?, language=? WHERE id=?",
                   (source_id, source_type, language, observation_id))
    if enrich:
        from vidur.extract import enrich_observation
        enrich_observation(observation_id)
    return observation_id


def _decode_observation(row: sqlite3.Row | dict) -> dict:
    """Row to dict, with attachments and metadata additionally decoded.

    ``metadata`` is kept as the raw JSON string the source returned, because
    that is what provenance has to reproduce; ``metadata_dict`` is a convenience
    for callers that would rather not parse it.
    """
    record = dict(row)
    record["saved"] = bool(record.get("saved"))
    try:
        attachments = json.loads(record.pop("media", None) or "[]")
    except (TypeError, ValueError):
        attachments = []
    record["media"] = attachments if isinstance(attachments, list) else []
    try:
        payload = json.loads(record.get("metadata") or "{}")
    except (TypeError, ValueError):
        payload = {}
    record["metadata_dict"] = payload if isinstance(payload, dict) else {}
    return record


def get_observation(observation_id: int) -> dict | None:
    """Read one observation by its numeric id, through the canonical view."""
    init_db()
    with connect() as db:
        row = db.execute("SELECT * FROM observations WHERE id=?", (int(observation_id),)).fetchone()
    return _decode_observation(row) if row else None


def list_observations(query: str | None = None, *, platform: str | None = None,
                      source_id: int | None = None, limit: int = 100) -> list[dict]:
    """List observations newest first, optionally filtered."""
    init_db()
    conditions, values = [], []
    if query:
        conditions.append("(content LIKE ? OR author LIKE ? OR platform LIKE ? OR url LIKE ?)")
        values.extend([f"%{query}%"] * 4)
    if platform:
        conditions.append("platform = ?")
        values.append(platform.lower())
    if source_id is not None:
        conditions.append("source_id = ?")
        values.append(int(source_id))
    statement = "SELECT * FROM observations"
    if conditions:
        statement += " WHERE " + " AND ".join(conditions)
    statement += " ORDER BY collected_at DESC LIMIT ?"
    values.append(max(1, min(limit, 500)))
    with connect() as db:
        return [_decode_observation(row) for row in db.execute(statement, values).fetchall()]


def list_sources() -> list[dict]:
    """Every registered source. Unrated sources are returned with None reliability."""
    init_db()
    with connect() as db:
        return [dict(row) for row in db.execute(
            "SELECT * FROM sources ORDER BY name COLLATE NOCASE").fetchall()]


def upsert_location(entry: dict) -> bool:
    """Insert or refresh one curated location, preserving its id."""
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    values = {
        "key": str(entry["key"]), "name": str(entry["name"]),
        "country": str(entry.get("country", "")), "region": str(entry.get("region", "")),
        "location_type": str(entry.get("location_type", "OTHER")),
        "latitude": entry.get("latitude"), "longitude": entry.get("longitude"),
        "radius_km": entry.get("radius_km"), "now": now,
    }
    with connect() as db:
        before = db.execute("SELECT COUNT(*) FROM locations").fetchone()[0]
        db.execute("""
            INSERT INTO locations (key, name, country, region, location_type,
                                   latitude, longitude, radius_km, created_at)
            VALUES (:key, :name, :country, :region, :location_type,
                    :latitude, :longitude, :radius_km, :now)
            ON CONFLICT(key) DO UPDATE SET
              name=excluded.name, country=excluded.country, region=excluded.region,
              location_type=excluded.location_type, latitude=excluded.latitude,
              longitude=excluded.longitude, radius_km=excluded.radius_km
        """, values)
        after = db.execute("SELECT COUNT(*) FROM locations").fetchone()[0]
        return after > before


def upsert_entity(entry: dict) -> bool:
    """Insert or refresh one curated entity, preserving its id."""
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    values = {
        "key": str(entry["key"]), "name": str(entry["name"]),
        "entity_type": str(entry.get("entity_type", "OTHER")),
        "country": str(entry.get("country", "")),
        "aliases": json.dumps(list(entry.get("aliases", ())), ensure_ascii=False),
        "now": now,
    }
    with connect() as db:
        before = db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        db.execute("""
            INSERT INTO entities (key, name, entity_type, country, aliases, first_seen, last_seen)
            VALUES (:key, :name, :entity_type, :country, :aliases, :now, :now)
            ON CONFLICT(key) DO UPDATE SET
              name=excluded.name, entity_type=excluded.entity_type,
              country=excluded.country, aliases=excluded.aliases, last_seen=excluded.last_seen
        """, values)
        after = db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        return after > before


def location_id_for(key: str) -> int | None:
    with connect() as db:
        row = db.execute("SELECT id FROM locations WHERE key=?", (key,)).fetchone()
    return int(row[0]) if row else None


def entity_id_for(key: str) -> int | None:
    with connect() as db:
        row = db.execute("SELECT id FROM entities WHERE key=?", (key,)).fetchone()
    return int(row[0]) if row else None


def link_mentions(observation_id: int, extraction) -> dict:
    """Record the curated locations and entities an observation mentions.

    Links live in ``relationships`` rather than a new table: the existing unique
    constraint makes repeated enrichment idempotent, so re-running extraction
    cannot inflate an actor's apparent prominence. Nothing about the stored
    source text is touched here.
    """
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    linked = {"locations": 0, "entities": 0, "location_id": None}
    with connect() as db:
        for mention in getattr(extraction, "locations", ()):
            location_id = location_id_for(mention.key)
            if location_id is None:
                continue
            linked["locations"] += 1
            _link(db, observation_id, "LOCATION", location_id, "LOCATED_IN", now)
        for key in getattr(extraction, "entity_keys", lambda: ())():
            entity_id = entity_id_for(key)
            if entity_id is None:
                # Account handles are extracted but not promoted to entity rows.
                # The author of a post already lives in source_items.author, and
                # creating a row per mentioned handle would let noisy text inflate
                # the actor graph without anyone curating those actors.
                continue
            linked["entities"] += 1
            _link(db, observation_id, "ENTITY", entity_id, "MENTIONS", now)
        # Only a single recognised location becomes the observation's own
        # location. With several candidates the choice would be a guess, so the
        # column stays empty and the links above carry the ambiguity.
        resolved = [location_id_for(mention.key) for mention in getattr(extraction, "locations", ())]
        resolved = [value for value in resolved if value is not None]
        if len(resolved) == 1:
            linked["location_id"] = resolved[0]
        db.execute("UPDATE source_items SET location_id=? WHERE id=?",
                   (linked["location_id"], observation_id))
    return linked


def _link(db, observation_id: int, to_kind: str, to_id: int, rel_type: str, now: str) -> None:
    db.execute("""
        INSERT INTO relationships (from_kind, from_id, rel_type, to_kind, to_id, created_at)
        VALUES ('OBSERVATION', ?, ?, ?, ?, ?)
        ON CONFLICT(from_kind, from_id, rel_type, to_kind, to_id) DO NOTHING
    """, (observation_id, rel_type, to_kind, to_id, now))


def locations_for_observation(observation_id: int) -> list[dict]:
    """Curated locations this observation mentions."""
    with connect() as db:
        rows = db.execute("""
            SELECT l.*, r.rel_type AS rel_type FROM relationships r
            JOIN locations l ON l.id = r.to_id
            WHERE r.from_kind='OBSERVATION' AND r.from_id=? AND r.to_kind='LOCATION'
            ORDER BY l.name COLLATE NOCASE
        """, (int(observation_id),)).fetchall()
    return [dict(row) for row in rows]


def entities_for_observation(observation_id: int) -> list[dict]:
    """Curated entities this observation mentions, plus any handles."""
    with connect() as db:
        rows = db.execute("""
            SELECT e.*, r.rel_type AS rel_type FROM relationships r
            JOIN entities e ON e.id = r.to_id
            WHERE r.from_kind='OBSERVATION' AND r.from_id=? AND r.to_kind='ENTITY'
            ORDER BY e.name COLLATE NOCASE
        """, (int(observation_id),)).fetchall()
    return [dict(row) for row in rows]


def list_locations() -> list[dict]:
    init_db()
    with connect() as db:
        return [dict(row) for row in db.execute(
            "SELECT * FROM locations ORDER BY name COLLATE NOCASE").fetchall()]


def list_entities() -> list[dict]:
    init_db()
    with connect() as db:
        return [dict(row) for row in db.execute(
            "SELECT * FROM entities ORDER BY name COLLATE NOCASE").fetchall()]


def save_assessment(region_key: str, body: str, *, state_payload: dict | None = None,
                    model: str = "", analytic_confidence: float = 0.0,
                    analytic_basis: str = "", window_days: int = 14,
                    origin: str = "TEMPLATE") -> int:
    """Store a generated briefing in ``assessments``.

    Kept strictly apart from source evidence: this is generated prose, recorded
    with the model that produced it and the deterministic confidence behind it, so
    a stored briefing can never be mistaken for something a source said. No
    migration is needed; the table arrived with schema version 3.
    """
    init_db()
    with connect() as db:
        cursor = db.execute("""
            INSERT INTO assessments (region_key, created_at, window_days, state_json, body,
                                    origin, model, analytic_confidence, analytic_basis)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (region_key, datetime.now(timezone.utc).isoformat(timespec="seconds"),
              int(window_days), json.dumps(state_payload or {}, ensure_ascii=False, default=str),
              body, origin, model, clamp_confidence_value(analytic_confidence), analytic_basis))
        return int(cursor.lastrowid)


def clamp_confidence_value(value) -> float:
    """Keep a stored confidence inside 0.0-1.0 without importing the domain layer."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:
        return 0.0
    return round(max(0.0, min(1.0, number)), 4)


def contradictions_for_event(event_id: int) -> list[dict]:
    """Every recorded dispute for an event, unresolved ones included."""
    init_db()
    with connect() as db:
        return [dict(row) for row in db.execute(
            "SELECT * FROM contradictions WHERE event_id=? ORDER BY id", (int(event_id),)).fetchall()]


def list_events(limit: int = 100) -> list[dict]:
    init_db()
    with connect() as db:
        return [dict(row) for row in db.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?",
            (max(1, min(int(limit), 500)),)).fetchall()]


def relationship_count() -> int:
    init_db()
    with connect() as db:
        return db.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]


def observation_stats() -> dict:
    """Counts used by the status command and the TUI overview."""
    init_db()
    with connect() as db:
        observations = db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        events = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        entities = db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        locations = db.execute("SELECT COUNT(*) FROM locations").fetchone()[0]
        sources = db.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        hashed = db.execute(
            "SELECT COUNT(*) FROM observations WHERE \"hash\" != ''").fetchone()[0]
    return {"observations": observations, "hashed_observations": hashed, "events": events,
            "entities": entities, "locations": locations, "sources": sources}
