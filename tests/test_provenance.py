import pytest

from vakula import provenance, storage
from vakula.provenance import ProvenanceError


def _post(text="Three naval vessels departed Port X.", key="post-1"):
    return {"platform": "x", "backend": "twitter-cli", "source_key": key, "author": "@reporter",
            "body": text, "source_url": "https://x.com/reporter/status/1",
            "published_at": "2026-10-01T09:00:00Z", "raw": {"likeCount": 4}}


def test_observation_is_stored_with_provenance(monkeypatch, tmp_path):
    monkeypatch.setenv("VAKULA_DATA_DIR", str(tmp_path))
    storage.init_db()
    observation_id = storage.insert_observation(_post(), source_type="SOCIAL", language="en")

    observation = provenance.get_observation(f"OBS-{observation_id}")
    assert observation.platform == "x"
    assert observation.content == "Three naval vessels departed Port X."
    assert observation.language == "en"
    assert observation.content_hash
    assert provenance.source_name(observation) == "twitter-cli"


def test_source_content_is_immutable_once_collected(monkeypatch, tmp_path):
    # A publisher editing a post later must not rewrite the evidence an
    # assessment already cited.
    monkeypatch.setenv("VAKULA_DATA_DIR", str(tmp_path))
    storage.init_db()
    first = storage.insert_observation(_post(text="Three vessels departed."))
    storage.insert_observation(_post(text="Three vessels departed. EDITED."))

    observation = provenance.get_observation(f"OBS-{first}")
    assert observation.content == "Three vessels departed."
    assert storage.get_observation(first)["content"] == "Three vessels departed."
    # The Live Feed still tracks the newest fetch.
    assert storage.list_source_items()[0]["body"].endswith("EDITED.")


def test_observation_view_exposes_canonical_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("VAKULA_DATA_DIR", str(tmp_path))
    storage.init_db()
    observation_id = storage.insert_observation(_post())
    record = storage.get_observation(observation_id)
    for field in ("id", "source_id", "source_type", "platform", "source_key", "author",
                  "timestamp", "content", "url", "metadata", "location", "language",
                  "collected_at", "hash"):
        assert field in record, field
    assert record["metadata_dict"] == {"likeCount": 4}


def test_missing_reference_is_an_error_not_an_empty_result(monkeypatch, tmp_path):
    monkeypatch.setenv("VAKULA_DATA_DIR", str(tmp_path))
    storage.init_db()
    with pytest.raises(ProvenanceError):
        provenance.get_observation("OBS-999999")
    with pytest.raises(ProvenanceError):
        provenance.parse_reference("not-a-reference")
    with pytest.raises(ProvenanceError):
        provenance.parse_reference("")


def test_reference_parsing_accepts_loose_formats(monkeypatch, tmp_path):
    monkeypatch.setenv("VAKULA_DATA_DIR", str(tmp_path))
    storage.init_db()
    observation_id = storage.insert_observation(_post())
    assert provenance.parse_reference(f"OBS-{observation_id}") == observation_id
    assert provenance.parse_reference(f"obs-{observation_id}") == observation_id
    assert provenance.parse_reference(str(observation_id)) == observation_id
    assert provenance.format_reference(7) == "OBS-7"


def test_event_evidence_is_derived_from_stored_links(monkeypatch, tmp_path):
    monkeypatch.setenv("VAKULA_DATA_DIR", str(tmp_path))
    storage.init_db()
    first = storage.insert_observation(_post(key="a"))
    second = storage.insert_observation(_post(key="b", text="Footage shows vessels leaving."))

    with storage.connect() as db:
        db.execute("""INSERT INTO events (event_type, title, confidence, confidence_basis,
                                         created_at, updated_at)
                      VALUES ('NAVAL_DEPLOYMENT', 'Vessel deployment', 0.8,
                              'two independent reports', '2026-10-01', '2026-10-01')""")
        event_id = db.execute("SELECT MAX(id) FROM events").fetchone()[0]
        db.executemany(
            "INSERT INTO event_observations (event_id, observation_id, role, weight) VALUES (?,?,?,?)",
            [(event_id, first, "CORROBORATES", 0.9), (event_id, second, "CORROBORATES", 0.7)])

    evidence = provenance.evidence_for_event(event_id)
    assert [item.reference for item in evidence] == [f"OBS-{first}", f"OBS-{second}"]
    assert provenance.evidence_citations(evidence) == f"OBS-{first}, OBS-{second}"
    assert provenance.citation_coverage(evidence) == 1.0
    assert {event.reference for event in provenance.events_for_observation(first)} == {f"EVT-{event_id}"}


def test_citation_coverage_reflects_missing_hashes(monkeypatch, tmp_path):
    monkeypatch.setenv("VAKULA_DATA_DIR", str(tmp_path))
    storage.init_db()
    observation_id = storage.insert_observation(_post())
    with storage.connect() as db:
        db.execute("UPDATE source_items SET content_hash='' WHERE id=?", (observation_id,))
    observation = provenance.get_observation(f"OBS-{observation_id}")
    assert provenance.citation_coverage([provenance.Evidence(observation)]) == 0.0


def test_unknown_platform_never_blocks_an_insert(monkeypatch, tmp_path):
    monkeypatch.setenv("VAKULA_DATA_DIR", str(tmp_path))
    storage.init_db()
    observation_id = storage.insert_observation({"body": "no platform", "source_key": "k"})
    assert storage.get_observation(observation_id)["platform"] == "unknown"
    assert any(source["key"] == "unknown" for source in storage.list_sources())
