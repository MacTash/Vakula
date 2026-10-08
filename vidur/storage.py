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
    with connect() as db:
        version = db.execute("PRAGMA user_version").fetchone()[0]
        db.executescript("""
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
        """)
        if version < 2:
            db.execute("PRAGMA user_version = 2")
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


def save_source_item(item: dict) -> int:
    """Persist a source post without rewriting its body or media metadata."""
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
    values = {
        "source_key": source_key,
        "platform": platform,
        "author": str(item.get("author", "")),
        "body": str(item.get("body", "")),
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
              (source_key, platform, author, body, source_url, published_at, fetched_at,
               backend, media_json, raw_json)
            VALUES (:source_key, :platform, :author, :body, :source_url, :published_at,
                    :fetched_at, :backend, :media_json, :raw_json)
            ON CONFLICT(platform, source_key) DO UPDATE SET
              author=excluded.author, body=excluded.body, source_url=excluded.source_url,
              published_at=excluded.published_at, fetched_at=excluded.fetched_at,
              backend=excluded.backend, media_json=excluded.media_json, raw_json=excluded.raw_json
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
