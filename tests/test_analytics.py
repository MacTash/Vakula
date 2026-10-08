import json
from datetime import date, timedelta

import pytest

from vakula import analytics, confidence, forecast, fusion, state, storage

# A fixed "today" so every calculation is reproducible and no test depends on the
# wall clock.
END = date(2026, 10, 1)


# Window geometry relative to END: the window covers offsets 1..14 (the last
# complete day is END-1) and the 28-day baseline covers offsets 15..42.
WINDOW_OFFSETS = range(1, 15)
BASELINE_OFFSETS = range(15, 43)


def _fill(offsets, per_day=(), *, event_type="MILITARY_ACTIVITY", **kwargs):
    """Create events on each offset, with a repeating per-day count."""
    for offset in offsets:
        counts = per_day if isinstance(per_day, (list, tuple)) else [per_day]
        for index in range(counts[offset % len(counts)]):
            _event(END - timedelta(days=offset), title=f"{offset}-{index}",
                   event_type=event_type, **kwargs)


def _seed(monkeypatch, tmp_path):
    monkeypatch.setenv("VAKULA_DATA_DIR", str(tmp_path))
    storage.init_db()
    from vakula import gazetteer
    gazetteer.seed(storage)


def _event(day: date, *, title="Event", event_type="MILITARY_ACTIVITY",
           location_key=None, body="Three vessels moved."):
    """Create an observation and an event dated on a specific day."""
    when = f"{day.isoformat()}T09:00:00Z"
    observation = storage.insert_observation({
        "platform": "x", "backend": "twitter-cli", "source_key": f"k{day}-{title}-{event_type}",
        "author": f"@{abs(hash(title)) % 9999}", "body": body,
        "published_at": when, "fetched_at": when})
    now = when
    with storage.connect() as db:
        location_id = None
        if location_key:
            row = db.execute("SELECT id FROM locations WHERE key=?", (location_key,)).fetchone()
            location_id = row[0] if row else None
        db.execute("""
            INSERT INTO events (event_type, title, start_time, location_id, status,
                                confidence, confidence_basis, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'ACTIVE', 0.5, 'test fixture', ?, ?)
        """, (event_type, title, when, location_id, now, now))
        event_id = db.execute("SELECT MAX(id) FROM events").fetchone()[0]
        db.execute("""INSERT INTO event_observations (event_id, observation_id, role, weight)
                      VALUES (?, ?, 'ORIGIN', 1.0)""", (event_id, observation))
    return event_id


def _series(counts, *, start=END - timedelta(days=len(()) or 1), scope=analytics.ALL):
    """Build a Series directly, bypassing storage, for pure maths tests."""
    points = tuple(analytics.DayPoint(day=start + timedelta(days=index), events=count)
                   for index, count in enumerate(counts))
    return analytics.Series(scope=scope, start=points[0].day, end=points[-1].day, points=points)


# --- empty and sparse ------------------------------------------------------

def test_empty_dataset_never_invents_a_number(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    analysis = analytics.analyse(end=END)
    assert analysis.status == analytics.STATUS_INSUFFICIENT
    assert analysis.anomaly_score is None
    assert analysis.change_rate is None
    assert "needed" in analysis.basis or "baseline" in analysis.basis
    assert analysis.current_rate == 0.0


def test_sparse_data_is_flagged_rather_than_filled_in(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    _event(END - timedelta(days=3))
    analysis = analytics.analyse(end=END)
    assert analysis.status == analytics.STATUS_INSUFFICIENT
    assert analysis.anomaly_score is None
    assert "observation" in analysis.basis


def test_flat_baseline_yields_no_standard_deviation(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    _fill(BASELINE_OFFSETS, 2)
    _fill(WINDOW_OFFSETS, 2)
    analysis = analytics.analyse(end=END, window_days=14, baseline_days=28)
    assert analysis.baseline_rate == pytest.approx(2.0)
    assert analysis.status == analytics.STATUS_INSUFFICIENT   # variance is exactly zero
    assert "no variance" in analysis.basis
    assert analysis.anomaly_score is None


# --- trend, spike, acceleration -------------------------------------------

def test_stable_baseline_reports_normal():
    counts = [4] * 20
    series = _series(counts)
    assert series.mean() == 4.0
    assert series.stdev() == pytest.approx(0.0)
    assert series.slope() == pytest.approx(0.0)


def test_increasing_trend_has_a_positive_slope():
    series = _series([1, 2, 3, 4, 5, 6])
    assert series.slope() > 0.9
    assert series.acceleration() == pytest.approx(0.0, abs=1e-6)


def test_decreasing_trend_has_a_negative_slope():
    assert _series([6, 5, 4, 3, 2, 1]).slope() < -0.9


def test_acceleration_is_the_change_in_slope():
    # flat then rising: the second half is steeper than the first
    accelerating = _series([1, 1, 1, 5, 6, 7])
    # rising then falling: the second half is steeper downward
    decelerating = _series([1, 3, 5, 5, 4, 3])
    assert accelerating.acceleration() > 0
    assert decelerating.acceleration() < 0
    assert _series([1, 1, 1, 1]).acceleration() == 0.0


def test_sudden_spike_is_detected_against_a_noisy_baseline(monkeypatch, tmp_path):
    """A short window so a concentrated spike is not diluted by quiet days."""
    _seed(monkeypatch, tmp_path)
    # alternating 1/3 gives the baseline genuine variance to measure against
    _fill(range(8, 36), [1, 3])
    _fill(range(1, 6), 2)                 # quiet
    # Two loud days, which fills a single sub-window, so the surge is unusual
    # rather than sustained.
    _fill(range(6, 8), 20, event_type="AIR_ACTIVITY")

    analysis = analytics.analyse(end=END, window_days=7, baseline_days=28)
    assert analysis.current_rate > analysis.baseline_rate
    assert analysis.change_rate > 0
    assert analysis.anomaly_score is not None and analysis.anomaly_score > 3.0
    # Concentrated in one sub-window, so unusual rather than sustained.
    assert analysis.status == analytics.STATUS_UNUSUAL
    assert analysis.basis


def test_a_spike_diluted_across_a_long_window_is_not_overstated(monkeypatch, tmp_path):
    """Four loud days inside a fortnight is not a fourteen-day surge."""
    _seed(monkeypatch, tmp_path)
    _fill(BASELINE_OFFSETS, [1, 3])
    _fill(range(1, 11), 2)
    _fill(range(11, 15), 12, event_type="AIR_ACTIVITY")
    analysis = analytics.analyse(end=END, window_days=14, baseline_days=28)
    assert analysis.current_rate > analysis.baseline_rate
    assert analysis.anomaly_score is not None
    assert analysis.anomaly_score < 3.0


def test_anomaly_persistence_is_distinguished_from_a_single_spike(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    _fill(BASELINE_OFFSETS, [1, 3])
    # elevated across the whole window, not just at one end
    _fill(WINDOW_OFFSETS, [1, 3], event_type="AIR_ACTIVITY")
    _fill(WINDOW_OFFSETS, 6, event_type="PROTEST")

    sustained = analytics.analyse(end=END, window_days=14, baseline_days=28)
    assert sustained.status == analytics.STATUS_SUSTAINED
    assert sustained.is_sustained
    assert "sub-windows" in sustained.basis


def test_high_but_unremarkable_activity_is_not_an_anomaly(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    _fill(BASELINE_OFFSETS, [1, 3])
    _fill(WINDOW_OFFSETS, [1, 3])

    analysis = analytics.analyse(end=END, window_days=14, baseline_days=28)
    assert analysis.status in {analytics.STATUS_NORMAL, analytics.STATUS_ELEVATED}
    assert analysis.anomaly_score < 3.0


# --- comparisons -----------------------------------------------------------

def test_event_type_comparison_counts_each_type(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    _event(END - timedelta(days=1), event_type="NAVAL_DEPLOYMENT")
    _event(END - timedelta(days=2), event_type="NAVAL_DEPLOYMENT", title="b")
    _event(END - timedelta(days=3), event_type="EARTHQUAKE")
    counts = analytics.compare_event_types(end=END)
    assert counts["NAVAL_DEPLOYMENT"] == 2
    assert counts["EARTHQUAKE"] == 1


def test_regional_comparison_uses_curated_locations_only(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    _event(END - timedelta(days=1), location_key="loc:gaza", body="Clashes in Gaza City.")
    _event(END - timedelta(days=2), location_key="loc:gaza", title="b", body="More in Gaza City.")
    _event(END - timedelta(days=3), location_key="loc:red-sea", body="Shipping in the Red Sea.")
    places = analytics.compare_locations(end=END)
    assert places.get("Gaza") == 2
    assert places.get("Red Sea") == 1
    assert "Nowhere" not in places


def test_scoped_analytics_only_count_their_scope(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    _event(END - timedelta(days=1), location_key="loc:gaza", body="Fighting in Gaza City.")
    _event(END - timedelta(days=2), location_key="loc:red-sea", body="Shipping in the Red Sea.")
    gaza = analytics.Scope(analytics.KIND_LOCATION, "loc:gaza")
    assert len(analytics.scope_observation_ids(gaza)) == 1
    assert len(analytics.scope_observation_ids(analytics.ALL)) == 2
    assert len(analytics.scope_event_ids(gaza)) == 1


# --- determinism -----------------------------------------------------------

def test_repeated_calculations_are_identical(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    for index in range(6):
        _event(END - timedelta(days=index), title=f"d{index}")
    first = analytics.analyse(end=END)
    second = analytics.analyse(end=END)
    assert first == second
    assert forecast.forecast(end=END) == forecast.forecast(end=END)
    assert state.build_state(end=END) == state.build_state(end=END)


def test_window_end_ignores_the_partial_current_day():
    # A day is only counted once complete, so the boundary is the previous day.
    assert analytics.window_end(END) == END - timedelta(days=1)


# --- forecast --------------------------------------------------------------

def test_probability_is_always_bounded(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    _fill(BASELINE_OFFSETS, [2, 6])
    _fill(WINDOW_OFFSETS, [2, 6])
    result = forecast.forecast(end=END)
    assert 0.0 <= result.probability <= 1.0
    for horizon in (1, 7, 30, 400):
        assert 0.0 <= forecast.forecast(end=END, horizon_days=horizon).probability <= 1.0


def test_forecast_reports_its_engine_and_horizon(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    result = forecast.forecast(end=END, horizon_days=21)
    assert result.engine == "Geoscope Forecast Engine v0.1"
    assert result.horizon_days == 21
    assert str(21) in result.statement
    assert result.basis


def test_probability_and_confidence_are_separate_numbers(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    _fill(BASELINE_OFFSETS, [1, 3])
    _fill(WINDOW_OFFSETS, [1, 3])
    result = forecast.forecast(end=END)
    payload = result.to_dict()
    assert payload["probability"] != payload["prediction_confidence"]
    assert payload["prediction_confidence_basis"].strip()
    # The specification's example shape is legal.
    assert 0.0 <= payload["probability"] <= 1.0
    assert 0.0 <= payload["prediction_confidence"] <= 1.0


def test_insufficient_evidence_is_stated_not_hidden(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    result = forecast.forecast(end=END)
    assert result.insufficient_evidence is True
    assert result.probability == 0.5
    assert "insufficient evidence" in result.basis.lower()


def test_no_activity_baseline_cannot_report_high_confidence(monkeypatch, tmp_path):
    """A flat zero baseline is absent signal, not stability."""
    _seed(monkeypatch, tmp_path)
    result = forecast.forecast(end=END)
    assert result.prediction_confidence.value < 0.2
    assert "no activity" in result.prediction_confidence.basis


def test_forecast_never_consults_a_model():
    for module in (forecast, analytics, state):
        assert not hasattr(module, "requests")
        assert not hasattr(module, "research")
        assert not hasattr(module, "AISettings")
        assert not hasattr(module, "agent")


def test_forecast_probability_is_pure_arithmetic(monkeypatch, tmp_path):
    """Same inputs, same probability, with no model in the path at all."""
    _seed(monkeypatch, tmp_path)
    _fill(BASELINE_OFFSETS, [1, 3])
    _fill(WINDOW_OFFSETS, [1, 3])
    values = {forecast.forecast(end=END).probability for _ in range(5)}
    assert len(values) == 1


# --- IntelligenceState -----------------------------------------------------

def test_state_serialisation_round_trips(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    _event(END - timedelta(days=1), location_key="loc:gaza", body="NATO confirmed in Gaza City.")
    original = state.build_state(end=END)
    restored = state.IntelligenceState.from_json(original.to_json())
    assert restored == original
    assert json.loads(original.to_json())["data_quality"]


def test_state_carries_no_source_text(monkeypatch, tmp_path):
    """Observations travel as references so a claim can be traced, not quoted."""
    _seed(monkeypatch, tmp_path)
    secret = "UNIQUE-MARKER-TEXT-SHOULD-NOT-APPEAR-IN-STATE"
    _event(END - timedelta(days=1), body=f"Three vessels moved. {secret}")
    built = state.build_state(end=END)
    assert secret not in built.to_json()
    assert all(reference.startswith("OBS-") for reference in built.observation_references)


def test_state_is_reconstructable_and_deterministic(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    _event(END - timedelta(days=1), body="Three vessels in the Red Sea.")
    assert state.build_state(end=END) == state.build_state(end=END)


def test_state_reports_data_quality_grades(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    assert state.build_state(end=END).data_quality == state.DATA_INSUFFICIENT
    for index in range(6):
        _event(END - timedelta(days=index), title=f"q{index}")
    assert state.build_state(end=END).data_quality in {state.DATA_SPARSE, state.DATA_SUFFICIENT}


def test_state_carries_anomalies_trends_and_actors(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    for index in range(6):
        _event(END - timedelta(days=index), title=f"a{index}",
               body="NATO confirmed three vessels in Gaza City.")
    built = state.build_state(end=END)
    assert built.anomaly["status"] in {
        analytics.STATUS_NORMAL, analytics.STATUS_ELEVATED, analytics.STATUS_UNUSUAL,
        analytics.STATUS_SUSTAINED, analytics.STATUS_INSUFFICIENT}
    assert built.anomaly["basis"]
    assert "slope_per_day" in built.trends
    assert any(actor.name == "NATO" for actor in built.actors)
    assert built.locations.get("Gaza")


def test_state_summary_is_readable(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    assert "DATA QUALITY" in state.build_state(end=END).summary()


# --- candidate, unconfirmed, contradictions --------------------------------

def test_analytics_separates_confirmed_candidate_and_unconfirmed(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    # two independent origins -> confirmed
    first = _event(END - timedelta(days=1), location_key="loc:taiwan-strait",
                   body="Three naval vessels departed a port in the Taiwan Strait.")
    twin = storage.insert_observation({
        "platform": "youtube", "backend": "yt-dlp", "source_key": "twin",
        "author": "@outlet", "body": "Footage shows three vessels leaving the Taiwan Strait.",
        "published_at": f"{END.isoformat()}T11:00:00Z",
        "fetched_at": f"{END.isoformat()}T11:00:00Z"})
    fusion.attach(first, twin)
    # single origin -> unconfirmed
    _event(END - timedelta(days=2), location_key="loc:red-sea", title="lonely",
           body="Something unrelated near the Red Sea.")

    classification = analytics.classify_events()
    assert len(classification["confirmed"]) >= 1
    assert len(classification["confirmed"]) + len(classification["candidate"]) \
        + len(classification["unconfirmed"]) == classification["total_events"]


def test_unlinked_observations_are_counted_not_hidden(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    observation = storage.insert_observation({
        "platform": "x", "source_key": "orphan", "body": "Gaza City report.",
        "published_at": f"{END.isoformat()}T09:00:00Z",
        "fetched_at": f"{END.isoformat()}T09:00:00Z"})
    classification = analytics.classify_events()
    assert observation in classification["unlinked_observations"]


def test_candidates_are_never_silently_promoted(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    first = storage.insert_observation({
        "platform": "x", "backend": "twitter-cli", "source_key": "c1", "author": "@a",
        "body": "Three naval vessels departed a port in the Taiwan Strait.",
        "published_at": f"{END.isoformat()}T09:00:00Z",
        "fetched_at": f"{END.isoformat()}T09:00:00Z"})
    event_one = fusion.fuse_observation(first).event_id
    second = storage.insert_observation({
        "platform": "x", "backend": "twitter-cli", "source_key": "c2", "author": "@b",
        "body": "Three naval vessels departed a port in the Taiwan Strait.",
        "published_at": f"{(END - timedelta(days=40)).isoformat()}T09:00:00Z",
        "fetched_at": f"{(END - timedelta(days=40)).isoformat()}T09:00:00Z"})
    result = fusion.fuse_observation(second)
    assert result.verdict == fusion.CANDIDATE
    # still a candidate: the far-apart report did not join the original event
    assert len(fusion_evidence(event_one)) == 1


def fusion_evidence(event_id):
    from vakula import provenance
    return provenance.evidence_for_event(event_id)


def test_contradictions_propagate_into_the_state(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    first = storage.insert_observation({
        "platform": "x", "backend": "twitter-cli", "source_key": "p1", "author": "@a",
        "body": "Three vessels departed the port.",
        "published_at": f"{END.isoformat()}T09:00:00Z",
        "fetched_at": f"{END.isoformat()}T09:00:00Z"})
    event_id = fusion.fuse_observation(first).event_id
    for body, key, author in (("Two vessels departed the port.", "p2", "@b"),
                              ("Officials deny that three vessels departed the port.", "p3", "@c")):
        observation = storage.insert_observation({
            "platform": "x", "backend": "twitter-cli", "source_key": key, "author": author,
            "body": body, "published_at": f"{END.isoformat()}T10:00:00Z",
            "fetched_at": f"{END.isoformat()}T10:00:00Z"})
        fusion.attach(event_id, observation)

    assert len(analytics.unresolved_contradictions()) == 2
    built = state.build_state(end=END)
    assert len(built.contradictions) == 2
    assert all(row["status"] == "UNRESOLVED" for row in built.contradictions)
    # An open dispute lowers the event's confidence rather than being ignored.
    event = next(row for row in storage.list_events() if row["id"] == event_id)
    assert "unresolved contradiction" in event["confidence_basis"]


def test_confidence_functions_remain_four_and_unfed_by_a_model():
    for name in ("source_confidence", "event_confidence",
                 "analytic_confidence", "prediction_confidence"):
        assert callable(getattr(confidence, name))
    assert not hasattr(analytics, "Confidence")
    assert not hasattr(forecast, "Confidence")


# --- read-only guarantee ---------------------------------------------------

def test_analytics_does_not_alter_observations(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    body = "Three naval vessels departed a port in the Taiwan Strait. UNIQUE-BODY-TEXT"
    observation = storage.insert_observation({
        "platform": "x", "backend": "twitter-cli", "source_key": "imm", "author": "@a",
        "body": body, "published_at": f"{END.isoformat()}T09:00:00Z",
        "fetched_at": f"{END.isoformat()}T09:00:00Z"})
    before = storage.get_observation(observation)
    for _ in range(3):
        analytics.analyse(end=END)
        forecast.forecast(end=END)
        state.build_state(end=END)
        analytics.classify_events()
    after = storage.get_observation(observation)
    for field in ("content", "hash", "id", "collected_at", "url", "author", "metadata"):
        assert after[field] == before[field], field
    assert storage.list_source_items()[0]["body"] == body


def test_analytics_opens_no_new_events(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    for index in range(5):
        _event(END - timedelta(days=index), title=f"k{index}")
    before = len(storage.list_events())
    analytics.analyse(end=END)
    forecast.forecast(end=END)
    state.build_state(end=END)
    analytics.classify_events()
    assert len(storage.list_events()) == before


def test_classification_reports_the_true_total_not_the_traversal_cap(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    for offset in WINDOW_OFFSETS:
        for index in range(2):
            _event(END - timedelta(days=offset), title=f"m{offset}-{index}")
    classification = analytics.classify_events(limit=5)
    assert classification["considered"] == 5
    assert classification["total_events"] == 28
    assert classification["truncated"] is True


def test_forecast_says_when_no_deviation_could_be_measured(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    _fill(BASELINE_OFFSETS, 2)      # perfectly flat, so variance is exactly zero
    _fill(WINDOW_OFFSETS, 6)
    result = forecast.forecast(end=END)
    assert result.anomaly_status == analytics.STATUS_INSUFFICIENT
    assert "rests on rates alone" in result.basis
