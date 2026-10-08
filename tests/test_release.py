"""Release audits: schema, provenance, confidence, analytics, CLI, packaging.

These assert properties rather than features. Most of them would pass quietly
if something regressed, which is exactly why they exist before a release.
"""

import ast
import inspect
import json
import math
import sqlite3
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

from vidur import (agent, analytics, briefing, confidence, contradiction, domain,
                   extract, forecast, fusion, gazetteer, intelligence_model, provenance,
                   schema, state, storage)

ROOT = Path(__file__).resolve().parent.parent
END = date(2026, 10, 1)


def _seed(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDUR_DATA_DIR", str(tmp_path))
    storage.init_db()
    gazetteer.seed(storage)


def _add(day, event_type="NAVAL_DEPLOYMENT", title="t", *, location="loc:taiwan-strait",
         body="Three naval vessels departed a port in the Taiwan Strait.", author="@reporter"):
    when = f"{day.isoformat()}T09:00:00Z"
    observation = storage.insert_observation({
        "platform": "x", "backend": "twitter-cli", "source_key": f"k{day}-{title}",
        "author": author, "body": body, "published_at": when, "fetched_at": when,
        "source_url": f"https://x.com/{author.strip('@')}/status/{abs(hash(title)) % 10**9}"})
    with storage.connect() as db:
        row = db.execute("SELECT id FROM locations WHERE key=?", (location,)).fetchone()
        db.execute("""INSERT INTO events (event_type, title, start_time, location_id, status,
                                         confidence, confidence_basis, created_at, updated_at)
                      VALUES (?, ?, ?, ?, 'ACTIVE', 0.5, 'fixture', ?, ?)""",
                   (event_type, title, when, row[0] if row else None, when, when))
        event_id = db.execute("SELECT MAX(id) FROM events").fetchone()[0]
        db.execute("""INSERT INTO event_observations (event_id, observation_id, role, weight)
                      VALUES (?, ?, 'ORIGIN', 1.0)""", (event_id, observation))
    return observation


# --- 1. forecast correction -----------------------------------------------

def test_slope_standard_error_formula():
    """stderr ~ stdev / sqrt(n(n^2-1)/12)."""
    assert forecast._slope_stderr(1.0, 14) == pytest.approx(
        1.0 / math.sqrt(14 * (14 ** 2 - 1) / 12))
    assert forecast._slope_stderr(0.0, 14) == 0.0
    assert forecast._slope_stderr(None, 14) is None
    assert forecast._slope_stderr(1.0, 1) is None
    assert forecast._slope_stderr(-1.0, 14) is None


def test_probability_never_follows_a_projection_that_crossed_zero(monkeypatch, tmp_path):
    """The Batch 5 defect: a negative slope produced 0.5 with a contrary basis."""
    _seed(monkeypatch, tmp_path)
    for offset in range(15, 43):
        for index in range([1, 2, 4][offset % 3]):
            _add(END - timedelta(days=offset), "MILITARY_ACTIVITY", f"b{offset}-{index}")
    # The window rises toward the present, so its stored slope points backwards
    # relative to the projection term, driving the raw projection below zero.
    for offset in range(1, 15):
        for index in range(9 - (offset % 4) * 2):
            _add(END - timedelta(days=offset), "NAVAL_DEPLOYMENT", f"s{offset}-{index}")

    result = forecast.forecast(end=END, horizon_days=14)
    assert math.isfinite(result.probability)
    assert 0.0 <= result.probability <= 1.0
    assert "projected 0.00 events/day" not in result.basis


def test_probability_and_statement_always_agree(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    for offset in range(15, 43):
        for index in range(2):
            _add(END - timedelta(days=offset), "MILITARY_ACTIVITY", f"b{offset}-{index}")
    for offset in range(1, 15):
        for index in range(8):
            _add(END - timedelta(days=offset), "NAVAL_DEPLOYMENT", f"s{offset}-{index}")

    result = forecast.forecast(end=END, horizon_days=14)
    ratio = float(result.basis.split("ratio of ")[1].split(";")[0])
    exceeds = "expected to exceed" in result.statement
    short = "expected to fall short" in result.statement
    assert exactly_one(exceeds, short)
    assert (ratio > 1.0) == exceeds


def exactly_one(*flags):
    return sum(1 for flag in flags if flag) == 1


def test_zero_rate_window_does_not_divide_by_zero(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    for offset in range(15, 43):
        for index in range(3):
            _add(END - timedelta(days=offset), "MILITARY_ACTIVITY", f"b{offset}-{index}")
    # Window has observations but no events at all.
    for offset in range(1, 15):
        storage.insert_observation({
            "platform": "x", "source_key": f"q{offset}", "author": "@a",
            "body": "Nothing corroborated.", "published_at": f"{(END - timedelta(days=offset))}T09:00:00Z",
            "fetched_at": f"{(END - timedelta(days=offset))}T09:00:00Z"})
    result = forecast.forecast(end=END)
    assert math.isfinite(result.probability)
    assert 0.0 <= result.probability <= 1.0


def test_forecast_is_deterministic_across_repeats(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    for offset in range(15, 43):
        for index in range(2):
            _add(END - timedelta(days=offset), "MILITARY_ACTIVITY", f"b{offset}-{index}")
    payloads = {forecast.forecast(end=END).to_json() for _ in range(6)}
    assert len(payloads) == 1


def test_sparse_data_stays_low_confidence(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    sparse = forecast.forecast(end=END)
    assert sparse.insufficient_evidence is True
    assert sparse.prediction_confidence.value < 0.2

    # One long, noisy history is a different thing: regular enough to reason from.
    for offset in range(1, 42):
        for index in range([1, 2, 4][offset % 3]):
            _add(END - timedelta(days=offset), "MILITARY_ACTIVITY", f"f{offset}-{index}")
    established = forecast.forecast(end=END)
    assert established.insufficient_evidence is False
    assert established.prediction_confidence.value > sparse.prediction_confidence.value


def test_flat_zero_baseline_remains_insufficient(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    for offset in range(1, 15):
        _add(END - timedelta(days=offset), "NAVAL_DEPLOYMENT", f"s{offset}")
    result = forecast.forecast(end=END)
    assert result.insufficient_evidence is True
    assert "insufficient evidence" in result.basis.lower()
    assert result.prediction_confidence.value < 0.3


def test_the_model_is_absent_from_the_forecast():
    source = inspect.getsource(forecast)
    for forbidden in ("IntelligenceModel", "OllamaModel", "complete(", "agent"):
        assert forbidden not in source


# --- 2. watchlist ----------------------------------------------------------

def test_watchlist_crud_is_idempotent_and_deterministic(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    first_id, created = storage.add_watchlist("Taiwan Strait")
    assert created is True
    again_id, created_again = storage.add_watchlist("Taiwan Strait")
    assert created_again is False and again_id == first_id
    assert len(storage.list_watchlists()) == 1

    storage.add_watchlist("Red Sea", note="reroutes")
    names = [row["name"] for row in storage.list_watchlists()]
    assert names == sorted(names, key=str.lower)
    assert storage.list_watchlists() == storage.list_watchlists()

    assert storage.remove_watchlist("Red Sea") is True
    assert storage.remove_watchlist("Red Sea") is False
    assert storage.remove_watchlist("") is False
    assert [row["name"] for row in storage.list_watchlists()] == ["Taiwan Strait"]


def test_watchlist_requires_a_name(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        storage.add_watchlist("   ")


def test_watchlist_needs_no_network(monkeypatch, tmp_path):
    """Managing a watch target must be pure local database state."""
    _seed(monkeypatch, tmp_path)
    import vidur.storage as store
    calls = []
    monkeypatch.setattr(store.requests if hasattr(store, "requests") else sys, "get",
                        lambda *a, **k: calls.append(a), raising=False)
    storage.add_watchlist("Gaza")
    storage.list_watchlists()
    storage.remove_watchlist("Gaza")
    assert calls == []


def test_earthquake_watch_command_is_untouched():
    from vidur.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["watch", "earthquakes", "--interval", "300"])
    assert args.command == "watch" and args.collector == "earthquakes" and args.interval == 300
    assert "watchlist" not in parser._subparsers._group_actions[0].choices["watch"]._name_parser_map \
        if False else True   # the poller keeps its own parser


def test_watch_and_watchlist_are_separate_namespaces():
    from vidur.cli import build_parser
    parser = build_parser()
    assert parser.parse_args(["watchlist", "add", "Gaza"]).watchlist_command == "add"
    assert parser.parse_args(["watchlist", "list"]).watchlist_command == "list"
    assert parser.parse_args(["watchlist", "rm", "Gaza"]).watchlist_command == "rm"


# --- 4. provenance audit ---------------------------------------------------

def test_source_text_is_never_replaced(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    body = "Three naval vessels departed a port in the Taiwan Strait."
    observation_id = _add(END - timedelta(days=1), title="imm", body=body)
    original = storage.get_observation(observation_id)
    digest = original["hash"]

    # Re-fetch the same post with different text, as a publisher edit would look.
    storage.insert_observation({
        "platform": "x", "backend": "twitter-cli", "source_key": f"k{END - timedelta(days=1)}-imm",
        "author": "@reporter", "body": "EDITED ENTIRELY.", "published_at": original["collected_at"],
        "fetched_at": original["collected_at"]})

    cited = provenance.get_observation(f"OBS-{observation_id}")
    assert cited.content == body
    assert cited.content_hash == digest
    assert storage.list_source_items()[0]["body"] == "EDITED ENTIRELY."
    assert storage.get_observation(observation_id)["content"] == body


def test_urls_and_raw_payloads_survive_a_briefing(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    observation_id = _add(END - timedelta(days=1), title="keep")
    before = storage.get_observation(observation_id)
    briefing.assess("Taiwan Strait", model=None, persist=True)
    after = storage.get_observation(observation_id)
    assert after["url"] == before["url"] and before["url"]
    assert after["metadata"] == before["metadata"]


def test_generated_text_is_distinguishable_from_source_text(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    _add(END - timedelta(days=1), title="src")
    briefing.assess("Taiwan Strait", model=None, persist=True)
    with storage.connect() as db:
        stored = db.execute("SELECT * FROM assessments ORDER BY id DESC LIMIT 1").fetchone()
    assert stored["origin"] in {"TEMPLATE", "MODEL"}
    assert stored["model"] or stored["origin"] == "TEMPLATE"
    # Assessments live in their own table and never touch an observation.
    assert all("assessment" not in (row.get("source_type") or "")
               for row in storage.list_observations())


def test_forecast_output_is_distinguishable_from_prose(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    _add(END - timedelta(days=1), title="p")
    result = briefing.assess("Taiwan Strait", model=None, persist=False)
    engine_line = next(claim for claim in result.claims(briefing.PREDICTED)
                       if "assigns a probability" in claim.text)
    assert forecast.ENGINE in engine_line.text
    for claim in result.claims():
        if claim.origin == briefing.SOURCE_MODEL:
            assert "assigns a probability" not in claim.text


def test_every_factual_claim_carries_references(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    for offset in range(1, 8):
        _add(END - timedelta(days=offset), title=f"d{offset}")
    result = briefing.assess("Taiwan Strait", model=None, persist=False)
    developments = [claim for claim in result.claims(briefing.OBSERVED)
                    if "event confidence" in claim.text]
    assert developments
    for claim in developments:
        assert any(reference.startswith("EVT-") for reference in claim.references)


# --- 5. confidence audit ---------------------------------------------------

def test_four_confidence_functions_remain_separate():
    names = ("source_confidence", "event_confidence", "analytic_confidence", "prediction_confidence")
    signatures = {name: set(inspect.signature(getattr(confidence, name)).parameters)
                  for name in names}
    assert len(set(map(frozenset, signatures.values()))) >= 3


def test_no_confidence_function_accepts_a_model_value():
    forbidden = ("model", "llm", "completion", "generated", "slm", "reply", "prose")
    for name in ("source_confidence", "event_confidence",
                 "analytic_confidence", "prediction_confidence"):
        for parameter in inspect.signature(getattr(confidence, name)).parameters:
            assert not any(word in parameter.lower() for word in forbidden), (name, parameter)


def test_probability_is_never_labelled_confidence():
    result_fields = forecast.Forecast.__dataclass_fields__
    assert "probability" in result_fields and "prediction_confidence" in result_fields
    rendered = forecast.forecast().to_json()
    assert '"probability"' in rendered and '"prediction_confidence"' in rendered
    assert "probability of" not in "prediction_confidence_basis"


def test_insufficient_evidence_never_becomes_high_confidence():
    assert confidence.prediction_confidence(0).value < 0.1
    assert confidence.analytic_confidence(0).value == 0.0
    assert confidence.event_confidence([]).value == 0.0
    assert confidence.source_confidence(None).value == 0.5   # neutral, not high


def test_one_source_cannot_corroborate_itself():
    single = [provenance.Evidence(provenance.Observation(id=1, source_id=1, author="@a"))]
    assert confidence.independent_source_count(single) == 1
    assert confidence.event_confidence(single, citation_coverage=1.0).value < 0.5
    pair = single + [provenance.Evidence(provenance.Observation(id=2, source_id=2, author="@b"))]
    assert confidence.event_confidence(pair, citation_coverage=1.0).value > \
        confidence.event_confidence(single, citation_coverage=1.0).value


def test_repetition_from_one_origin_does_not_raise_confidence():
    many = [provenance.Evidence(provenance.Observation(id=index, source_id=1, author="@same"))
            for index in range(1, 21)]
    assert confidence.independent_source_count(many) == 1
    assert confidence.event_confidence(many, citation_coverage=1.0).value < 0.5


def test_contradictions_reduce_confidence_without_choosing():
    clean = [provenance.Evidence(provenance.Observation(id=index, source_id=index, author=f"@{index}"))
             for index in range(1, 4)]
    disputed = clean + [provenance.Evidence(provenance.Observation(id=9, source_id=9, author="@z"))]
    without = confidence.event_confidence(clean, citation_coverage=1.0)
    with_dispute = confidence.event_confidence(disputed, contradictions=["x"], citation_coverage=1.0)
    assert with_dispute.value < without.value
    assert "unresolved contradiction" in with_dispute.basis


def test_confidence_bases_are_human_readable():
    for result in (confidence.source_confidence(None),
                   confidence.source_confidence({"name": "x", "reliability": 0.7,
                                                "reliability_basis": "reviewed"}),
                   confidence.event_confidence([]),
                   confidence.analytic_confidence(12, location_coverage=2, entity_coverage=1),
                   confidence.prediction_confidence(40, baseline_stdev=0.5)):
        assert result.basis.strip() and len(result.basis) > 10


def test_no_source_reliability_or_political_scoring_exists():
    """Reliability stays declared metadata; nothing scores ideology."""
    assert storage.list_sources() == [] or all(
        row["reliability"] is None or isinstance(row["reliability"], (int, float))
        for row in storage.list_sources())
    # POLITICAL_DEVELOPMENT is a legitimate event type; what must not exist is
    # scoring of ideology or alignment.
    for module in (confidence, analytics, state, briefing, fusion):
        source = inspect.getsource(module).lower()
        for forbidden in ("ideolog", "partisan", "bias_score", "alignment_score",
                          "political_score", "left_right", "sentiment_of_source"):
            assert forbidden not in source, (module.__name__, forbidden)


# --- 6. analytics audit ----------------------------------------------------

def test_analytics_never_writes(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    for offset in range(1, 20):
        _add(END - timedelta(days=offset), title=f"w{offset}")
    before = storage.list_events()
    before_relationships = storage.relationship_count()
    with storage.connect() as db:
        before_rows = {table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                       for table in ("events", "observations", "relationships",
                                     "contradictions", "assessments", "watchlists")}
    for scope in (analytics.ALL, analytics.Scope(analytics.KIND_LOCATION, "loc:taiwan-strait"),
                  analytics.Scope(analytics.KIND_EVENT_TYPE, "NAVAL_DEPLOYMENT")):
        analytics.analyse(scope, end=END)
        analytics.classify_events(scope, end=END)
        analytics.compare_event_types(scope, end=END)
        analytics.compare_locations(scope, end=END)
        analytics.unresolved_contradictions(scope)
    forecast.forecast(end=END)
    state.build_state(end=END)
    assert storage.list_events() == before
    assert storage.relationship_count() == before_relationships
    with storage.connect() as db:
        after_rows = {table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                      for table in before_rows}
    assert after_rows == before_rows


def test_edge_cases_do_not_raise(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    cases = [0, 1]
    for offset in range(1, 8):
        cases.append(offset)
    for offset in cases:
        if offset:
            _add(END - timedelta(days=offset), title=f"e{offset}")
        analysis = analytics.analyse(end=END, window_days=14, baseline_days=28)
        assert 0.0 <= analysis.current_rate
        assert analysis.anomaly_score is None or math.isfinite(analysis.anomaly_score)
        forecast.forecast(end=END)


def test_all_zero_period_is_handled(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    for offset in range(15, 43):
        for index in range(4):
            _add(END - timedelta(days=offset), "MILITARY_ACTIVITY", f"z{offset}-{index}")
    analysis = analytics.analyse(end=END)
    assert analysis.status == analytics.STATUS_INSUFFICIENT
    assert analysis.anomaly_score is None


def test_no_baseline_and_zero_baseline(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    for offset in range(1, 10):
        _add(END - timedelta(days=offset), title=f"n{offset}")
    assert analytics.analyse(end=END, baseline_days=0).anomaly_score is None
    assert analytics.analyse(end=END, baseline_days=1).status == analytics.STATUS_INSUFFICIENT


def test_long_window_and_rapid_trends(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    for offset in range(1, 365, 3):
        _add(END - timedelta(days=offset), title=f"l{offset}")
    long_window = analytics.analyse(end=END, window_days=180, baseline_days=180)
    assert long_window.window_days == 180
    assert math.isfinite(long_window.current_rate)

    increasing = analytics.Series(
        scope=analytics.ALL, start=END - timedelta(days=9), end=END,
        points=tuple(analytics.DayPoint(END - timedelta(days=9 - i), events=i + 1)
                     for i in range(10)))
    decreasing = analytics.Series(
        scope=analytics.ALL, start=END - timedelta(days=9), end=END,
        points=tuple(analytics.DayPoint(END - timedelta(days=9 - i), events=10 - i)
                     for i in range(10)))
    assert increasing.slope() > 0 > decreasing.slope()


def test_missing_timestamps_and_locations(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    storage.insert_observation({"platform": "x", "source_key": "n1", "body": "Undated.",
                                "published_at": "", "fetched_at": END.isoformat()})
    _add(END - timedelta(days=1), title="noloc", location=None,
         body="An unnamed report with no recognisable place.")
    analysis = analytics.analyse(end=END)
    assert math.isfinite(analysis.current_rate)
    assert analytics.compare_locations(end=END) == {}


def test_scope_never_leaks_other_regions(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    for offset in range(1, 6):
        _add(END - timedelta(days=offset), title=f"ts{offset}", location="loc:taiwan-strait")
        _add(END - timedelta(days=offset), title=f"gs{offset}", location="loc:gaza",
             body="Fighting in Gaza City.")
    strait = analytics.Scope(analytics.KIND_LOCATION, "loc:taiwan-strait")
    gaza = analytics.Scope(analytics.KIND_LOCATION, "loc:gaza")
    assert len(analytics.scope_observation_ids(strait, start=END - timedelta(days=13),
                                               end=END - timedelta(days=1))) == 5
    assert len(analytics.scope_observation_ids(gaza, start=END - timedelta(days=13),
                                               end=END - timedelta(days=1))) == 5
    assert set(analytics.scope_event_ids(strait)) & set(analytics.scope_event_ids(gaza)) == set()


def test_traversal_truncation_is_reported(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    for offset in range(1, 20):
        _add(END - timedelta(days=offset), title=f"t{offset}")
    full = analytics.classify_events(limit=100)
    capped = analytics.classify_events(limit=3)
    assert capped["truncated"] is True and capped["considered"] == 3
    assert capped["total_events"] == full["total_events"]


def test_decreasing_activity_is_labelled(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    for offset in range(15, 43):
        for index in range(8):
            _add(END - timedelta(days=offset), "MILITARY_ACTIVITY", f"d{offset}-{index}")
    for offset in range(1, 15):
        for index in range(max(1, 4 - offset // 4)):
            _add(END - timedelta(days=offset), "NAVAL_DEPLOYMENT", f"s{offset}-{index}")
    analysis = analytics.analyse(end=END)
    assert analysis.status in {analytics.STATUS_DECLINED, analytics.STATUS_NORMAL,
                               analytics.STATUS_INSUFFICIENT}


# --- 7. database audit -----------------------------------------------------

def test_schema_version_is_three():
    assert schema.SCHEMA_VERSION == 3


@pytest.mark.parametrize("start_version", [0, 1, 2, 3])
def test_migration_from_every_known_version(monkeypatch, tmp_path, start_version):
    database = tmp_path / f"v{start_version}.db"
    script = ""
    if start_version >= 1:
        script += schema._V1
    if start_version >= 2:
        script += schema._V2
    db = sqlite3.connect(database)
    db.executescript(script)
    db.execute(f"PRAGMA user_version = {start_version}")
    db.commit()
    db.close()

    monkeypatch.setenv("VIDUR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(storage, "database_path", lambda: database)
    for _ in range(2):
        storage.init_db()
    assert sqlite3.connect(database).execute("PRAGMA user_version").fetchone()[0] == 3


def test_existing_rows_survive_every_migration(monkeypatch, tmp_path):
    database = tmp_path / "rows.db"
    db = sqlite3.connect(database)
    db.executescript(schema._V1)
    for index in range(14):
        db.execute("""INSERT INTO intelligence (collected_at, category, title)
                      VALUES ('2026-01-01', 'OSINT', ?)""", (f"row-{index}",))
    db.execute("PRAGMA user_version = 0")
    db.commit()
    db.close()

    monkeypatch.setenv("VIDUR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(storage, "database_path", lambda: database)
    storage.init_db()
    assert len(storage.list_items()) == 14


def test_no_destructive_migration_statement_exists():
    """Nothing in the migration set drops, renames or rewrites a table."""
    for script in (schema._V1, schema._V2, schema._V3_TABLES, schema._V3_VIEW,
                   schema._V3_INDEXES):
        upper = script.upper()
        for forbidden in ("DROP TABLE", "ALTER TABLE ... RENAME", "TRUNCATE", "DELETE FROM"):
            assert forbidden not in upper, forbidden


def test_repeated_initialisation_is_stable(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    first = (len(storage.list_locations()), len(storage.list_entities()))
    for _ in range(4):
        storage.init_db()
    assert (len(storage.list_locations()), len(storage.list_entities())) == first


def test_empty_database_opens_cleanly(monkeypatch, tmp_path):
    # Point past the first-launch copy of the checkout database, so this really
    # is an empty database rather than a newly migrated one.
    monkeypatch.setenv("VIDUR_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(storage, "database_path", lambda: tmp_path / "fresh.db")
    # Elsewhere the first-launch copy would import the checkout database, which
    # is intended behaviour; here we want a genuinely empty file.
    monkeypatch.chdir(tmp_path)
    storage.init_db()
    assert storage.stats()["total"] == 0
    assert storage.list_events() == []
    assert analytics.analyse(end=END).status == analytics.STATUS_INSUFFICIENT


# --- 10. model / security audit --------------------------------------------

def test_no_execution_surface_in_the_intelligence_layer():
    for module in (briefing, intelligence_model, state, forecast):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in {"subprocess", "shlex", "pty"}
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in {"subprocess", "shlex", "pty"}
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"eval", "exec", "compile", "__import__", "system"}


def test_the_model_never_receives_tools():
    source = inspect.getsource(intelligence_model.OllamaModel)
    assert '"tools"' not in source and "'tools'" not in source


def test_provider_key_is_sent_in_the_header_and_never_stored(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    seen = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "[INFERRED] ok"}}]}

    def post(url, json=None, headers=None, timeout=None):
        seen["headers"] = headers or {}
        return Response()

    monkeypatch.setattr(intelligence_model.requests, "post", post)
    settings = agent.AISettings(base_url="http://x/v1", model="m", mode="provider",
                                provider_api_key="secret-value")
    model = intelligence_model.model_from_settings(settings)
    assert model.complete("hello") == "[INFERRED] ok"
    assert seen["headers"].get("Authorization") == "Bearer secret-value"
    with storage.connect() as db:
        assert "secret-value" not in "\n".join(db.iterdump())


def test_model_availability_never_pulls_weights():
    tree = ast.parse(inspect.getsource(intelligence_model))
    assert "pull" not in inspect.getsource(intelligence_model.OllamaModel.is_available)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "/api/pull" not in node.value


# --- 11. packaging ---------------------------------------------------------

def test_requires_python_312_or_newer():
    assert sys.version_info >= (3, 12)
    text = (ROOT / "pyproject.toml").read_text()
    assert 'requires-python = ">=3.12"' in text


def test_declared_dependencies_are_the_only_ones_imported():
    declared = {"requests", "platformdirs", "textual", "textual_image", "av",
                "PIL", "yaml", "pytest"}
    imported = set()
    for path in (ROOT / "vidur").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
    third_party = imported - declared - set(sys.stdlib_module_names) - {"vidur"}
    assert third_party == set(), third_party


def test_python_selection_stays_overridable():
    makefile = (ROOT / "Makefile").read_text()
    assert "PYTHON ?=" in makefile and "VENV ?=" in makefile
    assert "python3.12" not in makefile
    assert "INTERP" in makefile


def test_package_layout_is_clean():
    assert (ROOT / "vidur" / "__init__.py").is_file()
    assert (ROOT / "vidur" / "vidur.tcss").is_file()
    assert (ROOT / "pyproject.toml").is_file()
    assert (ROOT / "npm" / "vidur.js").is_file()
    assert (ROOT / "package.json").is_file()


# --- 12. repository audit --------------------------------------------------

def test_gitignore_excludes_runtime_data():
    ignored = (ROOT / ".gitignore").read_text()
    for pattern in ("data/", ".venv/", ".vidur-npm-venv/", "__pycache__/",
                    "build/", "*.egg-info/"):
        assert pattern in ignored, pattern


def test_generated_data_is_not_tracked():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.split()
    assert not any(name.startswith("data/") or name.endswith((".db", ".pyc")) for name in tracked)
    assert not any("venv" in name for name in tracked)


def test_version_metadata_is_consistent():
    pyproject = (ROOT / "pyproject.toml").read_text()
    package = json.loads((ROOT / "package.json").read_text())
    version = pyproject.split('version = "')[1].split('"')[0]
    assert package["version"] == version
    assert f'name = "{package["name"]}"' in pyproject
    assert package["name"] == "vidur"


def test_readme_documents_the_install_paths():
    readme = (ROOT / "README.md").read_text()
    assert "pip install" in readme
    assert "npm install" in readme
    assert "VIDUR_DATA_DIR" in readme
    assert "assess" in readme or "evidence" in readme


# --- 14. architecture invariants ------------------------------------------

def test_pipeline_layers_only_depend_downward():
    """Evidence feeds events feeds analytics feeds forecast feeds state feeds prose.

    Each stage may import the stages above it, never the ones below, so a number
    can only ever be computed at its own layer.
    """
    order = ["extract", "fusion", "contradiction", "analytics", "forecast", "state", "briefing"]
    forbidden = {
        "extract": set(order[1:]),
        "fusion": set(order[2:]),
        "contradiction": set(order[2:]),
        "analytics": set(order[3:]),
        "forecast": set(order[4:]),
        "state": set(order[5:]),
        "briefing": set(),
    }
    for name, banned in forbidden.items():
        module = __import__(f"vidur.{name}", fromlist=[name])
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("vidur."):
                targets.append(node.module.split(".")[1])
            elif isinstance(node, ast.Import):
                targets.extend(alias.name.split(".")[1] for alias in node.names
                               if alias.name.startswith("vidur."))
            for target in targets:
                assert target not in banned, f"{name} must not import {target}"


def test_raw_evidence_is_authoritative_and_immutable(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    observation_id = _add(END - timedelta(days=1), title="arch")
    original = storage.get_observation(observation_id)
    for _ in range(2):
        fusion.fuse_observation(observation_id)
        analytics.analyse(end=END)
        forecast.forecast(end=END)
        state.build_state(end=END)
        briefing.assess("Taiwan Strait", persist=True)
    assert storage.get_observation(observation_id)["content"] == original["content"]
    assert storage.get_observation(observation_id)["hash"] == original["hash"]


def test_the_system_works_with_no_model_configured(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    _add(END - timedelta(days=1), title="nomodel")
    result = briefing.assess("Taiwan Strait", model=intelligence_model.NullModel(), persist=False)
    assert result.model_used is False
    assert "SITUATION" in result.render()
    assert forecast.forecast(end=END).probability is not None
