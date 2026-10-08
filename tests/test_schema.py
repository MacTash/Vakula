import sqlite3

from vakula import schema, storage


def _legacy_database(path, *, version, statements):
    """Build a database the way an earlier Vakula release would have left it."""
    db = sqlite3.connect(path)
    db.executescript(statements)
    db.execute(f"PRAGMA user_version = {version}")
    db.commit()
    db.close()


def test_migration_brings_a_version_zero_database_forward(monkeypatch, tmp_path):
    # The shipped database is version 0 with only the intelligence table, which
    # is exactly what the first-launch copy produces.
    database = tmp_path / "legacy.db"
    _legacy_database(database, version=0, statements=schema._V1)
    monkeypatch.setenv("VAKULA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(storage, "database_path", lambda: database)

    assert storage.init_db() == database
    db = sqlite3.connect(database)
    assert db.execute("PRAGMA user_version").fetchone()[0] == schema.SCHEMA_VERSION
    objects = {row[0] for row in db.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    assert {"source_items", "observations", "events", "entities", "locations",
            "sources", "relationships", "assessments", "predictions", "contradictions",
            "watchlists"} <= objects
    db.close()


def test_migration_preserves_existing_rows(monkeypatch, tmp_path):
    database = tmp_path / "legacy.db"
    _legacy_database(database, version=0, statements=schema._V1)
    db = sqlite3.connect(database)
    db.execute("""INSERT INTO intelligence
                  (collected_at, category, title) VALUES ('2026-01-01', 'OSINT', 'Pre-migration row')""")
    db.commit()
    db.close()

    monkeypatch.setenv("VAKULA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(storage, "database_path", lambda: database)
    storage.init_db()

    titles = [row["title"] for row in storage.list_items()]
    assert "Pre-migration row" in titles


def test_migration_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setenv("VAKULA_DATA_DIR", str(tmp_path))
    for _ in range(3):
        storage.init_db()
    assert sqlite3.connect(storage.database_path()).execute(
        "PRAGMA user_version").fetchone()[0] == schema.SCHEMA_VERSION


def test_migration_from_version_two_adds_columns(monkeypatch, tmp_path):
    database = tmp_path / "v2.db"
    _legacy_database(database, version=2, statements=schema._V1 + schema._V2)
    monkeypatch.setenv("VAKULA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(storage, "database_path", lambda: database)

    storage.init_db()
    present = schema.columns(sqlite3.connect(database), "source_items")
    assert {"source_id", "source_type", "location_id", "language",
            "original_content", "content_hash"} <= present


def test_add_column_is_a_no_op_when_present(tmp_path):
    db = sqlite3.connect(tmp_path / "x.db")
    db.executescript(schema._V2)
    db.executescript("ALTER TABLE source_items ADD COLUMN language TEXT NOT NULL DEFAULT ''")
    assert schema.add_column(db, "source_items", "language", "TEXT") is False
    assert "language" in schema.columns(db, "source_items")
    db.close()
