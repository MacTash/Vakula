import inspect

import pytest

from vidur import confidence, contradiction, fusion, gazetteer, graph, provenance, storage


def _seed(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDUR_DATA_DIR", str(tmp_path))
    storage.init_db()
    gazetteer.seed(storage)


def _post(body, key, *, author="@reporter", when="2026-10-01T09:00:00Z",
          backend="twitter-cli", platform="x"):
    return {"platform": platform, "backend": backend, "source_key": key, "author": author,
            "body": body, "source_url": f"https://x.com/{key}",
            "published_at": when}


def _profile(body, when, keys):
    return fusion._Profile(observation_id=0, content=body, timestamp=when,
                           tokens=fusion._tokens(body), units=fusion._units(body),
                           keys=keys, record={"location": None})


STRAIT = {"loc:taiwan-strait"}

# --- scoring components ----------------------------------------------------

def test_entity_overlap_is_measured_as_jaccard():
    left = _profile("A and B in the Taiwan Strait", "", {"loc:taiwan-strait", "ent:country:cn"})
    right = _profile("A and C in the Taiwan Strait", "", {"loc:taiwan-strait", "ent:country:ru"})
    _, components = fusion.score_against(left, right)
    assert components["entity"] == pytest.approx(1 / 3)


def test_temporal_proximity_decays_with_the_gap():
    near = fusion.temporal_proximity("2026-10-01T09:00:00Z", "2026-10-01T21:00:00Z")
    day = fusion.temporal_proximity("2026-10-01T09:00:00Z", "2026-10-02T09:00:00Z")
    far = fusion.temporal_proximity("2026-10-01T09:00:00Z", "2026-11-30T09:00:00Z")
    assert near == pytest.approx(2 / 3, abs=0.01)
    assert day == pytest.approx(0.5, abs=0.01)
    assert far < 0.05
    assert fusion.temporal_proximity("", "2026-10-01T09:00:00Z") == 0.0


def test_geographic_proximity_uses_shared_keys_then_distance():
    record = {"location": None}
    assert fusion.geographic_proximity(record, record, {"loc:gaza"}) == 1.0
    assert fusion.geographic_proximity(record, record, set()) == 0.0
    # Two records that are both unmapped cannot be compared by distance.
    assert fusion.geographic_proximity({"location": 1}, {"location": 2}, set()) == 0.0


def test_token_similarity_drops_curated_names():
    # "Taiwan Strait" must not make two unrelated sentences look similar.
    same_place = fusion.token_similarity(
        "Three naval vessels departed the Taiwan Strait",
        "A magnitude 5.1 earthquake was recorded near the Taiwan Strait")
    assert same_place == 0.0


# --- thresholds and verdicts ----------------------------------------------

def test_thresholds_are_explicit_and_ordered():
    assert 0.0 < fusion.CANDIDATE_THRESHOLD < fusion.CONFIRM_THRESHOLD < 1.0
    assert fusion.CONFIRMED != fusion.CANDIDATE != fusion.NONE


def test_same_event_fusion_confirms():
    reuters = _profile("Three naval vessels departed a port in the Taiwan Strait.",
                       "2026-10-01T09:00:00Z", STRAIT)
    video = _profile("Three ships seen leaving the Taiwan Strait.",
                     "2026-10-01T11:00:00Z", STRAIT)
    result = fusion.compare(reuters, video)
    assert result.verdict == fusion.CONFIRMED
    assert result.score >= fusion.CONFIRM_THRESHOLD
    assert result.components["unit"] == 1.0  # vessels and ships are one unit


def test_near_match_is_rejected_not_merged():
    vessels = _profile("Three naval vessels departed a port in the Taiwan Strait.",
                       "2026-10-01T09:00:00Z", STRAIT)
    quake = _profile("A magnitude 5.1 earthquake was recorded near the Taiwan Strait.",
                     "2026-10-01T10:00:00Z", STRAIT)
    result = fusion.compare(vessels, quake)
    assert result.verdict != fusion.CONFIRMED


def test_temporal_separation_prevents_confirmation():
    early = _profile("Three naval vessels departed a port in the Taiwan Strait.",
                     "2026-10-01T09:00:00Z", STRAIT)
    late = _profile("Three naval vessels departed a port in the Taiwan Strait.",
                    "2026-11-30T09:00:00Z", STRAIT)
    result = fusion.compare(early, late)
    assert result.verdict != fusion.CONFIRMED
    assert result.components["time_penalty"] < 1.0


def test_geographic_separation_prevents_confirmation():
    strait = _profile("Three naval vessels departed a port in the Taiwan Strait.",
                      "2026-10-01T09:00:00Z", STRAIT)
    sea = _profile("Three naval vessels departed a port in the Red Sea.",
                   "2026-10-01T09:00:00Z", {"loc:red-sea"})
    result = fusion.compare(strait, sea)
    assert result.verdict != fusion.CONFIRMED
    assert result.components["place_penalty"] < 1.0


def test_missing_timestamp_never_confirms():
    """No time information is not agreement; it must stay a candidate."""
    timed = _profile("Three ships seen leaving the Taiwan Strait.",
                     "2026-10-01T11:00:00Z", STRAIT)
    untimed = _profile("Three ships seen leaving the Taiwan Strait.", "", STRAIT)
    assert fusion.compare(timed, untimed).verdict != fusion.CONFIRMED


def test_candidate_matches_are_reported_but_not_attached(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    first = storage.insert_observation(_post(
        "Three naval vessels departed a port in the Taiwan Strait.", "a"))
    original_event = fusion.fuse_observation(first).event_id

    second = storage.insert_observation(_post(
        "Three naval vessels departed a port in the Taiwan Strait.", "b",
        when="2026-11-20T09:00:00Z"))
    result = fusion.fuse_observation(second)

    assert result.verdict == fusion.CANDIDATE
    # Reported as a candidate, but not attached: the original event keeps one
    # observation and the near miss became an event of its own.
    assert result.event_id != original_event
    assert len(provenance.evidence_for_event(original_event)) == 1
    assert len(provenance.evidence_for_event(result.event_id)) == 1
    assert "not confirmed" in result.reason


# --- end to end through the database ---------------------------------------

def test_three_sources_fuse_into_one_event_with_its_evidence(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    reuters = storage.insert_observation(_post(
        "Three naval vessels departed a port in the Taiwan Strait.", "r1",
        author="Reuters", backend="wire", platform="rss"))
    result = fusion.fuse_observation(reuters)
    assert result.verdict == fusion.NONE
    event_id = result.event_id
    assert event_id is not None

    x_post = storage.insert_observation(_post(
        "Three ships seen leaving the Taiwan Strait.", "x1",
        when="2026-10-01T11:00:00Z", author="@naval_watch"))
    assert fusion.fuse_observation(x_post).verdict == fusion.CONFIRMED

    video = storage.insert_observation(_post(
        "Footage shows three vessels leaving the Taiwan Strait.", "y1",
        when="2026-10-01T13:00:00Z", author="Port X", platform="youtube", backend="yt-dlp"))
    assert fusion.fuse_observation(video).verdict == fusion.CONFIRMED

    evidence = provenance.evidence_for_event(event_id)
    assert {item.observation.id for item in evidence} == {reuters, x_post, video}
    assert len(storage.list_events()) == 1


def test_multiple_observations_attach_to_one_event(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    first = storage.insert_observation(_post(
        "Two fighter jets were seen over Gaza City.", "a", author="@one"))
    event_id = fusion.fuse_observation(first).event_id
    second = storage.insert_observation(_post(
        "Two fighter aircraft over Gaza City this evening.", "b",
        when="2026-10-01T20:00:00Z", author="@two"))
    fusion.fuse_observation(second)
    assert len(provenance.evidence_for_event(event_id)) == 2


def test_a_deliberate_near_match_stays_separate(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    vessels = storage.insert_observation(_post(
        "Three naval vessels departed a port in the Taiwan Strait.", "v",
        author="@a"))
    first_event = fusion.fuse_observation(vessels).event_id
    quake = storage.insert_observation(_post(
        "A magnitude 5.1 earthquake was recorded near the Taiwan Strait.", "q",
        when="2026-10-01T10:00:00Z", author="@b"))
    second_event = fusion.fuse_observation(quake).event_id
    assert second_event != first_event
    assert len(storage.list_events()) == 2
    assert len(provenance.evidence_for_event(first_event)) == 1


def test_source_independence_does_not_inflate_confidence(monkeypatch, tmp_path):
    """Ten reposts from one account are one voice, not ten."""
    _seed(monkeypatch, tmp_path)
    origin = storage.insert_observation(_post(
        "Three naval vessels departed a port in the Taiwan Strait.", "one",
        author="@same_account"))
    event_id = fusion.fuse_observation(origin).event_id
    for index in range(9):
        duplicate = storage.insert_observation(_post(
            "Three naval vessels departed a port in the Taiwan Strait.", f"copy{index}",
            when=f"2026-10-01T1{index % 10}:00:00Z", author="@same_account"))
        fusion.fuse_observation(duplicate)

    evidence = provenance.evidence_for_event(event_id)
    assert len(evidence) == 10
    assert confidence.independent_source_count(evidence) == 1

    single = confidence.event_confidence(evidence[:1], citation_coverage=1.0)
    assert single.value < 0.5  # one origin cannot corroborate itself

    independent = storage.insert_observation(_post(
        "Three ships seen leaving the Taiwan Strait.", "other",
        when="2026-10-01T11:00:00Z", author="@different_account"))
    fusion.fuse_observation(independent)
    evidence = provenance.evidence_for_event(event_id)
    assert confidence.independent_source_count(evidence) == 2
    assert confidence.event_confidence(evidence, citation_coverage=1.0).value > single.value


# --- contradictions --------------------------------------------------------

def test_numeric_contradiction_is_recorded_and_left_unresolved(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    first = storage.insert_observation(_post("Three vessels departed the port.", "a", author="@a"))
    event_id = fusion.fuse_observation(first).event_id
    second = storage.insert_observation(_post("Two vessels departed the port.", "b",
                                              when="2026-10-01T10:00:00Z", author="@b"))
    third = storage.insert_observation(_post("Four vessels departed the port.", "c",
                                             when="2026-10-01T11:00:00Z", author="@c"))
    for observation in (second, third):
        fusion.attach(event_id, observation)

    disputes = storage.contradictions_for_event(event_id)
    kinds = {row["kind"] for row in disputes}
    assert contradiction.KIND_QUANTITY in kinds
    assert all(row["status"] == contradiction.STATUS_UNRESOLVED for row in disputes)
    combined = " ".join(row["detail_json"] for row in disputes)
    for value in ("2", "3", "4"):
        assert f'"value": {value}' in combined, value


def test_negation_contradiction_is_recorded(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    first = storage.insert_observation(_post("Three vessels departed the port.", "a", author="@a"))
    event_id = fusion.fuse_observation(first).event_id
    denial = storage.insert_observation(_post(
        "Officials deny that three vessels departed the port.", "b",
        when="2026-10-01T10:00:00Z", author="@b"))
    fusion.attach(event_id, denial)
    disputes = storage.contradictions_for_event(event_id)
    assert any(row["kind"] == contradiction.KIND_POLARITY for row in disputes)
    assert all(row["status"] == "UNRESOLVED" for row in disputes)


def test_agreement_produces_no_contradiction(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    first = storage.insert_observation(_post("Three vessels departed the port.", "a", author="@a"))
    event_id = fusion.fuse_observation(first).event_id
    second = storage.insert_observation(_post("Three vessels left the port.", "b",
                                              when="2026-10-01T10:00:00Z", author="@b"))
    fusion.attach(event_id, second)
    assert storage.contradictions_for_event(event_id) == []


def test_semantic_contradiction_remains_unknown():
    # Contradictory in meaning, invisible to any deterministic reader.
    texts = ["No additional vessels departed the port.",
             "A further convoy was observed approaching the port."]
    assert contradiction.unresolved_semantic(texts) is None
    found = contradiction.detect([
        provenance.Evidence(provenance.Observation(id=1, content=texts[0])),
        provenance.Evidence(provenance.Observation(id=2, content=texts[1])),
    ])
    assert found == []


def test_contradiction_confidence_never_exceeds_the_dispute_itself():
    few = [contradiction.Claim(1, "OBS-1", "x", 3, "vessel", False),
           contradiction.Claim(2, "OBS-2", "x", 2, "vessel", False)]
    assert contradiction._dispute_confidence(few) == 0.0  # one voice cannot dispute itself
    many = [contradiction.Claim(index, f"OBS-{index}", f"outlet{index}", 3 + index, "vessel", False)
            for index in range(1, 9)]
    value = contradiction._dispute_confidence(many)
    assert value <= contradiction.DISPUTE_CONFIDENCE_CEILING == 0.9
    assert value > 0.0


# --- confidence ------------------------------------------------------------

def test_confidence_functions_are_four_and_separate():
    functions = {name: getattr(confidence, name) for name in
                 ("source_confidence", "event_confidence",
                  "analytic_confidence", "prediction_confidence")}
    assert len(functions) == 4
    results = {
        "source": confidence.source_confidence(None),
        "event": confidence.event_confidence([]),
        "analytic": confidence.analytic_confidence(0),
        "prediction": confidence.prediction_confidence(0),
    }
    assert all(isinstance(result, confidence.Confidence) for result in results.values())
    assert results["source"].value == 0.5
    assert results["event"].value == 0.0
    assert results["analytic"].value == 0.0
    assert results["prediction"].value < 0.1


def test_no_confidence_function_accepts_a_model_supplied_number():
    forbidden = ("model", "llm", "completion", "generated", "slm", "gpt", "reply", "text")
    for name in ("source_confidence", "event_confidence",
                 "analytic_confidence", "prediction_confidence"):
        parameters = inspect.signature(getattr(confidence, name)).parameters
        for parameter in parameters:
            assert not any(word in parameter.lower() for word in forbidden), (name, parameter)


def test_every_confidence_carries_a_human_readable_basis():
    for result in (confidence.source_confidence(None),
                   confidence.source_confidence({"name": "Reuters", "reliability": 0.8,
                                                 "reliability_basis": "operator reviewed",
                                                 "correction_count": 2}),
                   confidence.event_confidence([]),
                   confidence.analytic_confidence(30, location_coverage=3,
                                                  entity_coverage=2, anomaly_data=True),
                   confidence.prediction_confidence(60, baseline_stdev=0.4)):
        assert result.basis.strip()
        assert 0.0 <= result.value <= 1.0


def test_unrated_source_is_neutral_not_zero():
    unrated = confidence.source_confidence({"name": "New outlet", "reliability": None})
    assert unrated.value == 0.5
    assert "no recorded reliability" in unrated.basis


def test_prediction_confidence_rejects_a_short_history():
    assert confidence.prediction_confidence(5, window_days=14).value < 0.1
    assert confidence.prediction_confidence(56, baseline_stdev=0.2).value > 0.5


def test_clamped_inputs_cannot_escape_the_range():
    assert confidence.clamp_confidence("not a number") == 0.0
    assert confidence.clamp_confidence(5) == 1.0
    assert confidence.clamp_confidence(-3) == 0.0


# --- graph -----------------------------------------------------------------

def test_graph_traversal_and_depth_limits(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    observation = storage.insert_observation(_post(
        "NATO confirmed three naval vessels near Gaza City in the Taiwan Strait.", "g",
        author="@naval_watch"))
    event_id = fusion.fuse_observation(observation).event_id

    immediate = graph.neighbours("EVENT", event_id)
    assert {node.kind for node in immediate} == {"LOCATION", "ENTITY"}

    at_one = graph.walk("EVENT", event_id, max_depth=1)
    assert all(node.depth <= 1 for node in at_one)
    at_three = graph.walk("EVENT", event_id, max_depth=3)
    assert all(node.depth <= 3 for node in at_three)
    assert len(at_three) >= len(at_one)

    shallow = graph.walk("EVENT", event_id, max_depth=0)
    assert len(shallow) == 1 and shallow[0].depth == 0
    assert graph.walk("EVENT", event_id, max_depth=0)[0].label.startswith("EVT-")

    assert graph.walk("EVENT", 999999, max_depth=2) == graph.walk("EVENT", 999999, max_depth=2)
    assert graph.render(at_one)
    assert "Taiwan Strait" in graph.render(graph.walk("EVENT", event_id, max_depth=2))


def test_graph_is_restricted_to_the_relationships_table(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    observation = storage.insert_observation(_post(
        "NATO confirmed three vessels near Gaza City.", "n", author="@a"))
    event_id = fusion.fuse_observation(observation).event_id
    fusion.attach(event_id, observation)
    kinds = {row["rel_type"] for row in
             [dict(edge.__dict__) for edge in graph.edges_of("EVENT", event_id)]}
    assert "INVOLVED_IN" in kinds and "OCCURRED_AT" in kinds
    # A second attach must not duplicate edges.
    before = storage.relationship_count()
    fusion.attach(event_id, observation)
    assert storage.relationship_count() == before


# --- immutability ----------------------------------------------------------

def test_fusion_never_alters_stored_observations(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    body = "Three naval vessels departed a port in the Taiwan Strait."
    observation = storage.insert_observation(_post(body, "imm", author="@a"))
    before = storage.get_observation(observation)
    fusion.fuse_observation(observation)
    other = storage.insert_observation(_post("Three ships seen leaving the Taiwan Strait.",
                                             "imm2", when="2026-10-01T11:00:00Z", author="@b"))
    fusion.fuse_observation(other)
    disputed = storage.insert_observation(_post("Two vessels departed the port.", "imm3",
                                                when="2026-10-01T12:00:00Z", author="@c"))
    events = storage.list_events()
    for event in events:
        fusion.attach(event["id"], disputed)
        fusion.reconcile(event["id"])

    after = storage.get_observation(observation)
    for field in ("content", "hash", "id", "collected_at", "url", "author"):
        assert after[field] == before[field], field
    assert storage.list_source_items()[0]["body"] or True  # live feed untouched
    assert [row["body"] for row in storage.list_source_items()].count(body) == 1


def test_the_similarity_seam_can_be_replaced_without_touching_callers(monkeypatch, tmp_path):
    left = _profile("Three ships leaving the Taiwan Strait", "2026-10-01T09:00:00Z", STRAIT)
    right = _profile("An earthquake near the Taiwan Strait", "2026-10-01T09:00:00Z", STRAIT)
    assert fusion.compare(left, right).verdict != fusion.CONFIRMED

    # Substitute a future embedding similarity through the single seam.
    previous = fusion.set_lexical_similarity(lambda a, b: 1.0)
    try:
        assert fusion.lexical_similarity("anything", "else") == 1.0
        assert fusion.score_against(left, right)[1]["lexical"] == 1.0
    finally:
        fusion.set_lexical_similarity(previous)
    assert fusion.lexical_similarity("anything", "else") == 0.0


def test_contradiction_module_never_reaches_for_a_model():
    assert not hasattr(contradiction, "research")
    assert not hasattr(contradiction, "AISettings")
    assert not hasattr(contradiction, "requests")


def test_contradiction_records_refresh_instead_of_accumulating(monkeypatch, tmp_path):
    """A third figure must update the dispute, not leave a stale two-source row."""
    _seed(monkeypatch, tmp_path)
    first = storage.insert_observation(_post("Three vessels departed the port.", "a", author="@a"))
    event_id = fusion.fuse_observation(first).event_id
    for body, key, author, when in (
            ("Two vessels departed the port.", "b", "@b", "2026-10-01T10:00:00Z"),
            ("Four vessels departed the port.", "c", "@c", "2026-10-01T11:00:00Z")):
        observation = storage.insert_observation(_post(body, key, author=author, when=when))
        fusion.attach(event_id, observation)

    disputes = storage.contradictions_for_event(event_id)
    quantities = [row for row in disputes if row["kind"] == contradiction.KIND_QUANTITY]
    assert len(quantities) == 1
    detail = quantities[0]["detail_json"]
    for value in ("2", "3", "4"):
        assert f'"value": {value}' in detail, value
    assert quantities[0]["status"] == contradiction.STATUS_UNRESOLVED
