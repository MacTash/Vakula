from geoscope import storage


def test_store_and_filter(monkeypatch, tmp_path):
    monkeypatch.setenv("GEOSCOPE_DATA_DIR", str(tmp_path))
    storage.init_db()
    assert storage.add_item({"category": "OSINT", "title": "Test bulletin", "source_url": "https://example.test/1"})
    assert not storage.add_item({"category": "OSINT", "title": "Test bulletin", "source_url": "https://example.test/1"})
    assert storage.list_items(query="bulletin")[0]["category"] == "OSINT"
