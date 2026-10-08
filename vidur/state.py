"""IntelligenceState: the structured picture handed to the analysis layer.

This is the boundary the whole architecture turns on. Everything the model will
ever see arrives through here, already measured, so the model interprets rather
than computes. It is why Batch 3 could compute confidence and Batch 4 could
compute probability without any involvement from a language model.

Two properties are enforced rather than hoped for.

It carries no source text. Observations appear as references such as ``OBS-1842``
so a claim can be traced, and the text stays in the database to be fetched on
demand. Nothing here is a free-text field a model could have written into the
structure.

It is reconstructable. :meth:`IntelligenceState.from_dict` round-trips exactly,
and rebuilding from the same database on the same day produces the same state,
so a briefing can be checked against the evidence that produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date

from vidur import analytics, confidence, forecast as forecast_module, provenance, storage

DATA_SUFFICIENT = "SUFFICIENT"
DATA_SPARSE = "SPARSE"
DATA_INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True)
class Coverage:
    """What evidence exists behind the numbers, and how far it can be trusted."""

    observations: int = 0
    events: int = 0
    sources: int = 0
    hashed_observations: int = 0
    observations_without_event: int = 0
    candidates: int = 0
    unconfirmed_events: int = 0

    @property
    def hash_ratio(self) -> float:
        return (self.hashed_observations / self.observations) if self.observations else 0.0


@dataclass(frozen=True)
class Actor:
    """An actor as a reference only. Never a biography, never source text."""

    key: str
    name: str
    entity_type: str
    observation_count: int = 0


@dataclass(frozen=True)
class IntelligenceState:
    """A deterministic, serialisable snapshot of one scope over one window."""

    scope: str
    window_days: int
    window_start: str
    window_end: str
    coverage: Coverage
    data_quality: str
    event_types: dict
    locations: dict
    event_totals: dict
    anomaly: dict
    trends: dict
    contradictions: list
    actors: list
    observation_references: list
    assessments: list
    predictions: list
    analytic_confidence: confidence.Confidence

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["analytic_confidence"] = {
            "value": self.analytic_confidence.value,
            "basis": self.analytic_confidence.basis,
        }
        return payload

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)

    @classmethod
    def from_dict(cls, payload: dict) -> "IntelligenceState":
        raw = dict(payload)
        confidence_payload = raw.pop("analytic_confidence", {}) or {}
        return cls(
            scope=raw["scope"], window_days=raw["window_days"],
            window_start=raw["window_start"], window_end=raw["window_end"],
            coverage=Coverage(**raw["coverage"]), data_quality=raw["data_quality"],
            event_types=dict(raw["event_types"]), locations=dict(raw["locations"]),
            event_totals=dict(raw["event_totals"]), anomaly=dict(raw["anomaly"]),
            trends=dict(raw["trends"]), contradictions=list(raw["contradictions"]),
            actors=[Actor(**actor) for actor in raw["actors"]],
            observation_references=list(raw["observation_references"]),
            assessments=list(raw["assessments"]), predictions=list(raw["predictions"]),
            analytic_confidence=confidence.Confidence(
                value=confidence_payload.get("value", 0.0),
                basis=confidence_payload.get("basis", "")),
        )

    @classmethod
    def from_json(cls, text: str) -> "IntelligenceState":
        return cls.from_dict(json.loads(text))

    # -- readability ------------------------------------------------------
    def summary(self) -> str:
        lines = [
            f"SCOPE          {self.scope}",
            f"WINDOW         {self.window_start} to {self.window_end} ({self.window_days}d)",
            f"DATA QUALITY   {self.data_quality}",
            f"OBSERVATIONS   {self.coverage.observations} "
            f"({self.coverage.hash_ratio:.0%} hash-verified)",
            f"EVENTS         {self.coverage.events} "
            f"({self.coverage.candidates} candidate, {self.coverage.unconfirmed_events} unconfirmed)",
            f"WITHOUT EVENT  {self.coverage.observations_without_event} observation(s)",
            f"CONTRADICTIONS {len(self.contradictions)} unresolved",
            f"ACTORS         {len(self.actors)}",
            f"ANALYTIC CONF  {self.analytic_confidence.value:.2f} "
            f"({self.analytic_confidence.basis})",
        ]
        return "\n".join(lines)


def _data_quality(coverage: Coverage, anomaly: analytics.Anomaly) -> str:
    """Grade the evidence itself.

    Deliberately independent of the anomaly status. Plenty of evidence can be
    collected while the baseline comparison stays uncomputable, and collapsing
    the two would report "insufficient data" about a window that is merely
    unmeasured against its own history.
    """
    if coverage.observations < analytics.MINIMUM_WINDOW_OBSERVATIONS:
        return DATA_INSUFFICIENT
    if coverage.hash_ratio < 0.5 or anomaly.baseline_rate is None:
        return DATA_SPARSE
    return DATA_SUFFICIENT


def build_state(scope: analytics.Scope = analytics.ALL, *,
                window_days: int = analytics.DEFAULT_WINDOW_DAYS,
                baseline_days: int = analytics.DEFAULT_BASELINE_DAYS,
                horizon_days: int = forecast_module.DEFAULT_HORIZON_DAYS,
                end: date | None = None) -> IntelligenceState:
    """Assemble the state for one scope from stored data alone.

    Read-only. Nothing here writes to an observation, an event or a source.
    """
    storage.init_db()
    analysis = analytics.analyse(scope, window_days=window_days,
                                 baseline_days=baseline_days, end=end)
    # Coverage describes the window being assessed, not the whole record. Without
    # this the briefing would report every observation ever collected as though
    # it fell inside these fourteen days.
    window_start = date.fromisoformat(analysis.window_start)
    window_end = date.fromisoformat(analysis.window_end)
    observation_ids = analytics.scope_observation_ids(scope, start=window_start,
                                                      end=window_end)
    classification = analytics.classify_events(scope, start=window_start, end=window_end)
    disputes = analytics.unresolved_contradictions(scope)

    with storage.connect() as db:
        sources = db.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    hashed = sum(1 for observation_id in observation_ids
                 if (storage.get_observation(observation_id) or {}).get("hash"))

    coverage = Coverage(
        observations=len(observation_ids),
        events=classification["total_events"],
        sources=sources,
        hashed_observations=hashed,
        observations_without_event=len(classification["unlinked_observations"]),
        candidates=len(classification["candidate"]),
        unconfirmed_events=len(classification["unconfirmed"]),
    )

    quality = _data_quality(coverage, analysis)
    metrics = analysis.metrics

    actors = _actors_for(observation_ids)
    assessments = _recent_rows("assessments", "region_key", scope.value)
    predictions = _recent_rows("predictions", "region_key", scope.value)

    analytic = confidence.analytic_confidence(
        coverage.observations,
        location_coverage=len(analysis.metrics.get("locations", {})),
        entity_coverage=len(actors),
        anomaly_data=analysis.status != analytics.STATUS_INSUFFICIENT,
    )

    return IntelligenceState(
        scope=scope.label(),
        window_days=analysis.window_days,
        window_start=analysis.window_start,
        window_end=analysis.window_end,
        coverage=coverage,
        data_quality=quality,
        event_types=dict(analysis.metrics.get("event_types", {})),
        locations=dict(analysis.metrics.get("locations", {})),
        event_totals={
            "confirmed": len(classification["confirmed"]),
            "candidate": len(classification["candidate"]),
            "unconfirmed": len(classification["unconfirmed"]),
            "in_window": int(metrics.get("window_events", 0)),
        },
        anomaly=analysis.to_dict(),
        trends={
            "slope_per_day": metrics.get("slope_per_day"),
            "acceleration": metrics.get("acceleration"),
            "moving_average_7d": metrics.get("moving_average_7d"),
            "baseline_rate": analysis.baseline_rate,
            "current_rate": analysis.current_rate,
            "change_rate": analysis.change_rate,
        },
        contradictions=disputes,
        actors=actors,
        # References only. Source text is never copied into the state.
        observation_references=[provenance.format_reference(observation_id)
                                for observation_id in observation_ids],
        assessments=assessments,
        predictions=predictions,
        analytic_confidence=analytic,
    )


def _actors_for(observation_ids: list[int]) -> list[Actor]:
    """Curated actors linked to these observations, with how often each appears."""
    if not observation_ids:
        return []
    storage.init_db()
    placeholders = ",".join("?" * len(observation_ids))
    with storage.connect() as db:
        rows = db.execute(f"""
            SELECT e.key AS key, e.name AS name, e.entity_type AS entity_type,
                   COUNT(DISTINCT r.from_id) AS observation_count
            FROM relationships r
            JOIN entities e ON e.id = r.to_id
            WHERE r.from_kind='OBSERVATION' AND r.to_kind='ENTITY'
              AND r.from_id IN ({placeholders})
            GROUP BY e.id ORDER BY observation_count DESC, e.name
        """, observation_ids).fetchall()
    return [Actor(key=row["key"], name=row["name"], entity_type=row["entity_type"],
                  observation_count=row["observation_count"]) for row in rows]


def _recent_rows(table: str, column: str, value: str, *, limit: int = 5) -> list[dict]:
    """Any assessments or predictions already recorded for this scope."""
    storage.init_db()
    with storage.connect() as db:
        if value:
            rows = db.execute(
                f"SELECT * FROM {table} WHERE {column}=? ORDER BY id DESC LIMIT ?",
                (value, limit)).fetchall()
        else:
            rows = db.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?",
                              (limit,)).fetchall()
    return [dict(row) for row in rows]
