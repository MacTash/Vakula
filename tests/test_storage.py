from vakula import storage


def test_store_and_filter(monkeypatch, tmp_path):
    monkeypatch.setenv("VAKULA_DATA_DIR", str(tmp_path))
    storage.init_db()
    assert storage.add_item({"category": "OSINT", "title": "Test bulletin", "source_url": "https://example.test/1"})
    assert not storage.add_item({"category": "OSINT", "title": "Test bulletin", "source_url": "https://example.test/1"})
    assert storage.list_items(query="bulletin")[0]["category"] == "OSINT"


def test_setting_walks_back_through_every_previous_name(monkeypatch):
    """Geoscope and Vidur were both real releases, so both prefixes are honoured."""
    names = ("VAKULA_DATA_DIR", "VIDUR_DATA_DIR", "GEOSCOPE_DATA_DIR")
    monkeypatch.setenv("GEOSCOPE_DATA_DIR", "/geoscope")
    assert storage.setting(*names) == "/geoscope"
    monkeypatch.setenv("VIDUR_DATA_DIR", "/vidur")
    assert storage.setting(*names) == "/vidur"
    monkeypatch.setenv("VAKULA_DATA_DIR", "/vakula")
    assert storage.setting(*names) == "/vakula"
    for name in names:
        monkeypatch.delenv(name)
    assert storage.setting(*names) == ""


def test_pre_rename_install_keeps_using_its_data_directory(monkeypatch, tmp_path):
    legacy = tmp_path / "vidur"
    legacy.mkdir()
    (legacy / "vidur.db").write_bytes(b"")
    monkeypatch.delenv("VAKULA_DATA_DIR", raising=False)
    monkeypatch.delenv("GEOSCOPE_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert storage.default_data_dir() == legacy
    assert storage.database_path() == legacy / "vidur.db"


def test_adopted_legacy_directory_still_uses_the_new_database_name(monkeypatch, tmp_path):
    # A leftover empty vidur directory must not leave Vakula writing a vidur.db;
    # it adopts the directory but names its own file.
    legacy = tmp_path / "vidur"
    legacy.mkdir()
    monkeypatch.delenv("VAKULA_DATA_DIR", raising=False)
    monkeypatch.delenv("GEOSCOPE_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert storage.default_data_dir() == legacy
    assert storage.database_path() == legacy / "vakula.db"


def test_vakula_data_directory_wins_once_it_exists(monkeypatch, tmp_path):
    (tmp_path / "geoscope").mkdir()
    current = tmp_path / "vakula"
    current.mkdir()
    monkeypatch.delenv("VAKULA_DATA_DIR", raising=False)
    monkeypatch.delenv("GEOSCOPE_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert storage.default_data_dir() == current
    assert storage.database_path() == current / "vakula.db"


def test_legacy_database_is_opened_and_preserved(monkeypatch, tmp_path):
    legacy = tmp_path / "geoscope"
    legacy.mkdir()
    monkeypatch.setenv("VAKULA_DATA_DIR", str(legacy))
    storage.init_db()
    storage.add_item({"category": "OSINT", "title": "Kept across both renames",
                      "source_url": "https://example.test/legacy"})
    # A renamed lookup must still find rows written through the legacy filename.
    assert storage.list_items(query="Kept across")[0]["title"] == "Kept across both renames"


def test_vidur_install_data_is_still_reachable(monkeypatch, tmp_path):
    """A v0.3.0 install keeps its database across the rename to Vakula."""
    monkeypatch.delenv("VAKULA_DATA_DIR", raising=False)
    monkeypatch.delenv("VIDUR_DATA_DIR", raising=False)
    monkeypatch.delenv("GEOSCOPE_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    old = tmp_path / "vidur"
    old.mkdir()
    db = old / "vidur.db"
    storage.database_path = lambda: db
    storage.init_db()
    storage.add_item({"category": "OSINT", "title": "Stored under Vidur",
                      "source_url": "https://example.test/v"})
    assert storage.list_items(query="Stored under")[0]["title"] == "Stored under Vidur"
    assert storage.database_path() == db
