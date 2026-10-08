"""Deciding when two observations describe the same real-world event.

Fusion is correlation, never authorship. It reads observations, scores them
against existing events and attaches them; it never writes to an observation, and
it never decides that a disagreement has been settled.

Three outcomes, stated explicitly:

``CONFIRMED``  scored at or above :data:`CONFIRM_THRESHOLD`; attached.
``CANDIDATE``  scored between the two thresholds; reported, not attached.
``NONE``       below both; a new event is opened instead.

Sharing a place and a time is not enough on its own. An earthquake and a naval
departure in the same strait on the same day are different events, so topical
agreement carries real weight and place-and-time alone can only ever produce a
candidate for a human to look at.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from vakula import confidence, contradiction, extract, provenance, storage
from vakula.domain import clamp_confidence

CONFIRMED = "CONFIRMED"
CANDIDATE = "CANDIDATE"
NONE = "NONE"

# Explicit and testable. Both are module constants on purpose: a threshold that
# is buried in an expression cannot be argued with or pinned by a test.
CONFIRM_THRESHOLD = 0.62
CANDIDATE_THRESHOLD = 0.38

WEIGHTS = {"lexical": 0.35, "unit": 0.20, "entity": 0.20, "temporal": 0.15, "geographic": 0.10}

TEMPORAL_SCALE_HOURS = 24.0
EARTH_RADIUS_KM = 6371.0

# Below this topical agreement, no amount of place-and-time overlap is treated as
# more than a coincidence worth reviewing.
LEXICAL_FLOOR = 0.08

# Two observations that positively disagree about place, or that are further
# apart in time than TEMPORAL_AGREEMENT_FLOOR allows, are almost certainly
# different events. Missing information is only worth no credit; conflicting
# information actively multiplies the score down, otherwise two identically
# worded reports of unrelated incidents would fuse on wording alone.
PLACE_CONFLICT_FACTOR = 0.55
TIME_CONFLICT_FACTOR = 0.55
TEMPORAL_AGREEMENT_FLOOR = 0.10

# Units that imply a kind of event on their own. A single observation with no
# corroboration is still recorded as OTHER rather than being upgraded on the
# strength of one noun.
UNIT_EVENT_TYPES = {
    "vessel": "NAVAL_DEPLOYMENT",
    "aircraft": "AIR_ACTIVITY",
    "soldier": "MILITARY_ACTIVITY",
    "vehicle": "MILITARY_ACTIVITY",
    "missile": "MILITARY_ACTIVITY",
    "strike": "MILITARY_ACTIVITY",
    "facility": "ECONOMIC_DISRUPTION",
}

_STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for from by with
is are was were be been being has have had do does did not no nor so as it its their
his her our your my we they he she i you who whom which what when where while about
into over under after before during said says say report reported reports according
new latest update updates
""".split())

SimilarityFn = Callable[[str, str], float]


def token_similarity(left: str, right: str) -> float:
    """Default lexical similarity: Jaccard overlap of content words.

    Curated location and entity names are removed first. They are already scored
    separately, and leaving them in would let two unrelated posts in the same
    place look topically similar purely by sharing its name.
    """
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


_LEXICAL_SIMILARITY: SimilarityFn = token_similarity


def set_lexical_similarity(function: SimilarityFn) -> SimilarityFn:
    """Swap the text-comparison function without touching any caller.

    This is the single seam an embedding model would replace later. Every score
    in this module goes through :func:`lexical_similarity`, so substituting it
    changes fusion wholesale and leaves the surrounding code alone.
    """
    global _LEXICAL_SIMILARITY
    previous, _LEXICAL_SIMILARITY = _LEXICAL_SIMILARITY, function
    return previous


def lexical_similarity(left: str, right: str) -> float:
    return clamp_confidence(_LEXICAL_SIMILARITY(left, right))


def _tokens(text: str) -> set[str]:
    plain = extract.clean_text(text).lower()
    for entry in (*extract.gazetteer.LOCATION_SEED, *extract.gazetteer.ENTITY_SEED):
        for phrase in extract.gazetteer.alias_phrases(entry):
            plain = plain.replace(phrase.lower(), " ")
    words = re.findall(r"[a-z][a-z'-]{1,}", plain)
    return {word for word in words if word not in _STOPWORDS and len(word) > 2}


def _units(text: str) -> set[str]:
    return {quantity.unit for quantity in extract.find_quantities(text)}


def _entity_keys(record: dict) -> set[str]:
    """Gazetteer keys an observation touches, from its stored links."""
    keys = set()
    for row in storage.locations_for_observation(record["id"]):
        keys.add(row["key"])
    for row in storage.entities_for_observation(record["id"]):
        keys.add(row["key"])
    return keys


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def temporal_proximity(left: str, right: str) -> float:
    """Decay with the gap between two timestamps. Unknown means no credit."""
    first = extract.parse_timestamp(left)
    second = extract.parse_timestamp(right)
    if first is None or second is None:
        return 0.0
    delta_hours = abs((first - second).total_seconds()) / 3600.0
    return clamp_confidence(1.0 / (1.0 + delta_hours / TEMPORAL_SCALE_HOURS))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi, d_lambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def geographic_proximity(left: dict, right: dict, shared_keys: set[str]) -> float:
    """Shared curated place scores full; distinct mapped places decay by radius."""
    if shared_keys:
        return 1.0
    left_place, right_place = _place(left), _place(right)
    if not left_place or not right_place:
        return 0.0
    if None in (left_place["latitude"], left_place["longitude"],
                right_place["latitude"], right_place["longitude"]):
        return 0.0
    distance = _haversine_km(left_place["latitude"], left_place["longitude"],
                             right_place["latitude"], right_place["longitude"])
    reach = (left_place["radius_km"] or 50.0) + (right_place["radius_km"] or 50.0)
    return clamp_confidence(1.0 - distance / reach)


def _place(record: dict) -> dict | None:
    location_id = record.get("location") or record.get("location_id")
    if not location_id:
        return None
    for row in storage.list_locations():
        if row["id"] == location_id:
            return row
    return None


@dataclass(frozen=True)
class MatchResult:
    """The outcome of comparing one observation against the event set."""

    verdict: str
    score: float
    event_id: int | None = None
    components: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    @property
    def matched(self) -> bool:
        return self.verdict == CONFIRMED


@dataclass(frozen=True)
class _Profile:
    """Everything fusion needs about one side of a comparison."""

    observation_id: int
    content: str
    timestamp: str
    tokens: set[str]
    units: set[str]
    keys: set[str]
    record: dict


def _profile(observation_id: int) -> _Profile:
    record = storage.get_observation(observation_id)
    if record is None:
        raise provenance.ProvenanceError(f"No observation {observation_id} is stored.")
    content = record.get("content", "")
    return _Profile(
        observation_id=observation_id,
        content=content,
        timestamp=record.get("timestamp", ""),
        tokens=_tokens(content),
        units=_units(content),
        keys=_entity_keys(record),
        record=record,
    )


def score_against(left: _Profile, right: _Profile) -> tuple[float, dict[str, float]]:
    """Weighted score with every component exposed for inspection.

    A conflict between place or time multiplies the result down rather than
    merely contributing nothing, so an identical sentence about two different
    regions cannot confirm.
    """
    shared = left.keys & right.keys
    components = {
        "lexical": lexical_similarity(left.content, right.content),
        "unit": _jaccard(left.units, right.units),
        "entity": _jaccard(left.keys, right.keys),
        "temporal": temporal_proximity(left.timestamp, right.timestamp),
        "geographic": geographic_proximity(left.record, right.record, shared),
    }
    both_parsed = (extract.parse_timestamp(left.timestamp) is not None
                   and extract.parse_timestamp(right.timestamp) is not None)
    place_conflict = bool(left.keys and right.keys and not shared)
    # An unreadable timestamp is treated as unverified rather than as agreement.
    # Two reports of the same wording, one of them undated, may be a syndication
    # or two separate incidents, and fusion cannot tell which.
    time_conflict = bool(not both_parsed
                         or components["temporal"] < TEMPORAL_AGREEMENT_FLOOR)
    components["place_penalty"] = PLACE_CONFLICT_FACTOR if place_conflict else 1.0
    components["time_penalty"] = TIME_CONFLICT_FACTOR if time_conflict else 1.0
    total = sum(components[name] * weight for name, weight in WEIGHTS.items())
    total *= components["place_penalty"] * components["time_penalty"]
    return clamp_confidence(total), components


def compare(left: _Profile, right: _Profile) -> MatchResult:
    """Classify a pair. Exposed so tests can pin the thresholds directly."""
    score, components = score_against(left, right)
    if components["lexical"] < LEXICAL_FLOOR:
        return MatchResult(verdict=NONE, score=score, components=components,
                           reason="no topical agreement; place and time alone are not the same event")
    if score >= CONFIRM_THRESHOLD:
        return MatchResult(verdict=CONFIRMED, score=score, components=components,
                           reason="above the confirm threshold")
    if score >= CANDIDATE_THRESHOLD:
        return MatchResult(verdict=CANDIDATE, score=score, components=components,
                           reason="between thresholds; not attached")
    return MatchResult(verdict=NONE, score=score, components=components,
                       reason="below the candidate threshold")


def _event_profiles(event_id: int) -> list[_Profile]:
    return [_profile(item.observation.id)
            for item in provenance.evidence_for_event(event_id)]


def best_match(profile: _Profile, *, limit: int = 40) -> MatchResult:
    """Best available event for one observation, or an explicit miss."""
    with storage.connect() as db:
        event_ids = [row[0] for row in db.execute(
            "SELECT id FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]
    best = MatchResult(NONE, 0.0, reason="no events exist yet")
    for event_id in event_ids:
        for existing in _event_profiles(event_id):
            result = compare(profile, existing)
            if result.score > best.score:
                best = MatchResult(result.verdict, result.score, event_id,
                                   result.components, result.reason)
    return best


def infer_event_type(profile: _Profile) -> str:
    """A conservative event type from the units present. Default OTHER."""
    for unit in sorted(profile.units):
        if unit in UNIT_EVENT_TYPES:
            return UNIT_EVENT_TYPES[unit]
    if re.search(r"\bearthquake|\bmagnitude\s+\d", profile.content, re.IGNORECASE):
        return "EARTHQUAKE"
    return "OTHER"


def open_event(profile: _Profile, title: str = "") -> int:
    """Create a single-observation event for evidence with nowhere else to go."""
    now = datetime.now(timezone.utc).isoformat()
    start = extract.normalise_timestamp(profile.timestamp) or now
    location_id = profile.record.get("location")
    evidence = [provenance.Evidence(
        provenance.get_observation(provenance.format_reference(profile.observation_id)))]
    score = confidence.event_confidence(evidence, citation_coverage=provenance.citation_coverage(evidence))
    with storage.connect() as db:
        db.execute("""
            INSERT INTO events (event_type, title, description, start_time, location_id,
                                status, confidence, confidence_basis, created_at, updated_at)
            VALUES (?, ?, '', ?, ?, 'ACTIVE', ?, ?, ?, ?)
        """, (infer_event_type(profile), title or _default_title(profile), start, location_id,
              score.value, score.basis, now, now))
        event_id = db.execute("SELECT MAX(id) FROM events").fetchone()[0]
    attach(event_id, profile.observation_id, role="ORIGIN", weight=1.0)
    return event_id


def _default_title(profile: _Profile) -> str:
    text = extract.clean_text(profile.content).strip()
    return text[:120] if text else f"Observation {provenance.format_reference(profile.observation_id)}"


def attach(event_id: int, observation_id: int, *, role: str = "CORROBORATES",
           weight: float = 1.0) -> bool:
    """Link an observation to an event. The only way fusion touches an event."""
    now = datetime.now(timezone.utc).isoformat()
    with storage.connect() as db:
        cursor = db.execute("""
            INSERT INTO event_observations (event_id, observation_id, role, weight)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(event_id, observation_id) DO NOTHING
        """, (event_id, int(observation_id), role, clamp_confidence(weight)))
        linked = cursor.rowcount > 0
    if linked:
        reconcile(event_id)
        link_event_entities(event_id)
    return linked


def refresh_confidence(event_id: int) -> confidence.Confidence:
    """Recompute an event's confidence and basis from its stored evidence."""
    evidence = provenance.evidence_for_event(event_id)
    disputes = [dict(row) for row in storage.contradictions_for_event(event_id)]
    score = confidence.event_confidence(
        evidence, contradictions=disputes,
        citation_coverage=provenance.citation_coverage(evidence))
    with storage.connect() as db:
        db.execute("UPDATE events SET confidence=?, confidence_basis=?, updated_at=? WHERE id=?",
                   (score.value, score.basis, datetime.now(timezone.utc).isoformat(), event_id))
    return score


def reconcile(event_id: int) -> int:
    """Store the contradictions the evidence shows, and never resolve one.

    One record per event, kind and unit, refreshed in place. Appending a fresh
    row each time would leave a stale snapshot claiming two sources disagreed
    when three later did, which is exactly the kind of quiet inaccuracy this
    layer exists to prevent.
    """
    evidence = provenance.evidence_for_event(event_id)
    found = contradiction.detect(evidence, event_id=event_id)
    with storage.connect() as db:
        now = datetime.now(timezone.utc).isoformat()
        existing: dict[tuple[str, str], int] = {}
        for row in db.execute(
                "SELECT id, kind, detail_json FROM contradictions WHERE event_id=?",
                (event_id,)).fetchall():
            try:
                unit = str(json.loads(row["detail_json"] or "{}").get("unit", ""))
            except (TypeError, ValueError):
                unit = ""
            existing[(row["kind"], unit)] = row["id"]
        for dispute in found:
            detail = _detail(dispute)
            key = (dispute.kind, dispute.unit)
            if key in existing:
                db.execute("""UPDATE contradictions
                              SET confidence=?, detail_json=?, status=?, resolved_at=''
                              WHERE id=?""",
                           (dispute.confidence, detail, dispute.status, existing[key]))
            else:
                db.execute("""
                    INSERT INTO contradictions (event_id, kind, status, confidence, detail_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (event_id, dispute.kind, dispute.status, dispute.confidence, detail, now))
    refresh_confidence(event_id)
    return len(found)


def _detail(dispute: contradiction.Contradiction) -> str:
    return json.dumps({
        "unit": dispute.unit,
        "basis": dispute.basis,
        "claims": [{"reference": claim.reference, "source": claim.source_name,
                    "value": claim.value, "unit": claim.unit, "negated": claim.negated}
                   for claim in dispute.claims],
    }, ensure_ascii=False)


def link_event_entities(event_id: int) -> int:
    """Give an event the union of its evidence's entities and locations."""
    linked = 0
    for item in provenance.evidence_for_event(event_id):
        for row in storage.entities_for_observation(item.observation.id):
            linked += _relate(event_id, "INVOLVED_IN", "ENTITY", row["id"], item.observation.id)
        for row in storage.locations_for_observation(item.observation.id):
            linked += _relate(event_id, "OCCURRED_AT", "LOCATION", row["id"], item.observation.id)
        if item.observation.location_id:
            linked += _relate(event_id, "OCCURRED_AT", "LOCATION",
                              item.observation.location_id, item.observation.id)
    return linked


def _relate(event_id: int, rel_type: str, to_kind: str, to_id: int, observation_id: int) -> int:
    """Write one event-to-actor or event-to-place edge, ignoring duplicates."""
    now = datetime.now(timezone.utc).isoformat()
    with storage.connect() as db:
        cursor = db.execute("""
            INSERT INTO relationships (from_kind, from_id, rel_type, to_kind, to_id,
                                      observation_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(from_kind, from_id, rel_type, to_kind, to_id) DO NOTHING
        """, ("EVENT", event_id, rel_type, to_kind, to_id, observation_id, now))
        return cursor.rowcount


def fuse_observation(observation_id: int, *, allow_new_event: bool = True) -> MatchResult:
    """Correlate one stored observation with the event set.

    A confirmed match is attached to its event. Anything short of confirmation
    opens a separate event instead, including a near miss: the observation is not
    asserted to be part of the existing event, and the returned verdict still
    reports that a candidate was seen so a person can merge it deliberately.
    """
    profile = _profile(observation_id)
    result = best_match(profile)
    if result.verdict == CONFIRMED and result.event_id is not None:
        attach(result.event_id, observation_id, role="CORROBORATES",
               weight=round(result.score, 4))
        return MatchResult(CONFIRMED, result.score, result.event_id, result.components,
                           f"{result.reason}; attached to EVT-{result.event_id}")
    if not allow_new_event:
        return result
    event_id = open_event(profile)
    return MatchResult(result.verdict, result.score, event_id, result.components,
                       f"{result.reason}; not confirmed, opened EVT-{event_id} for review")
