from vidur import storage


def test_store_and_filter(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDUR_DATA_DIR", str(tmp_path))
    storage.init_db()
    assert storage.add_item({"category": "OSINT", "title": "Test bulletin", "source_url": "https://example.test/1"})
    assert not storage.add_item({"category": "OSINT", "title": "Test bulletin", "source_url": "https://example.test/1"})
    assert storage.list_items(query="bulletin")[0]["category"] == "OSINT"


def test_setting_prefers_vidur_then_falls_back_to_geoscope(monkeypatch):
    monkeypatch.setenv("GEOSCOPE_DATA_DIR", "/legacy")
    assert storage.setting("VIDUR_DATA_DIR", "GEOSCOPE_DATA_DIR") == "/legacy"
    monkeypatch.setenv("VIDUR_DATA_DIR", "/current")
    assert storage.setting("VIDUR_DATA_DIR", "GEOSCOPE_DATA_DIR") == "/current"
    monkeypatch.delenv("VIDUR_DATA_DIR")
    monkeypatch.delenv("GEOSCOPE_DATA_DIR")
    assert storage.setting("VIDUR_DATA_DIR", "GEOSCOPE_DATA_DIR") == ""


def test_pre_rename_install_keeps_using_its_data_directory(monkeypatch, tmp_path):
    legacy = tmp_path / "geoscope"
    legacy.mkdir()
    (legacy / "geoscope.db").write_bytes(b"")
    monkeypatch.delenv("VIDUR_DATA_DIR", raising=False)
    monkeypatch.delenv("GEOSCOPE_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert storage.default_data_dir() == legacy
    assert storage.database_path() == legacy / "geoscope.db"


def test_adopted_legacy_directory_still_uses_the_new_database_name(monkeypatch, tmp_path):
    # A leftover empty geoscope directory must not leave Vidur writing a
    # geoscope.db; it adopts the directory but names its own file.
    legacy = tmp_path / "geoscope"
    legacy.mkdir()
    monkeypatch.delenv("VIDUR_DATA_DIR", raising=False)
    monkeypatch.delenv("GEOSCOPE_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert storage.default_data_dir() == legacy
    assert storage.database_path() == legacy / "vidur.db"


def test_vidur_data_directory_wins_once_it_exists(monkeypatch, tmp_path):
    (tmp_path / "geoscope").mkdir()
    current = tmp_path / "vidur"
    current.mkdir()
    monkeypatch.delenv("VIDUR_DATA_DIR", raising=False)
    monkeypatch.delenv("GEOSCOPE_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert storage.default_data_dir() == current
    assert storage.database_path() == current / "vidur.db"


def test_legacy_database_is_opened_and_preserved(monkeypatch, tmp_path):
    legacy = tmp_path / "geoscope"
    legacy.mkdir()
    monkeypatch.setenv("VIDUR_DATA_DIR", str(legacy))
    storage.init_db()
    storage.add_item({"category": "OSINT", "title": "Kept across the rename",
                      "source_url": "https://example.test/legacy"})
    # A renamed lookup must still find rows written through the legacy filename.
    assert storage.list_items(query="Kept across")[0]["title"] == "Kept across the rename"
