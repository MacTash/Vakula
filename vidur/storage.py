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


def init_db() -> Path:
    """Create or forward-migrate Vidur's database. Safe to call repeatedly."""
    with connect() as db:
        migrate(db)
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


def insert_observation(item: dict, *, source_type: str = "OTHER", language: str = "") -> int:
    """Store one retrieved source record as an observation and return its id.

    This is the intelligence layer's write path and it goes through
    :func:`save_source_item`, so there is exactly one place where source text is
    stored. Nothing here summarises, translates or rewrites the body.
    """
    observation_id = save_source_item(item)
    source_id = ensure_source(
        str(item.get("platform", "unknown")), str(item.get("backend", "")), source_type=source_type)
    with connect() as db:
        db.execute("UPDATE source_items SET source_id=?, source_type=?, language=? WHERE id=?",
                   (source_id, source_type, language, observation_id))
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
