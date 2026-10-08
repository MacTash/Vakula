"""Tracing analytical claims back to the source text they rest on.

Nothing in Vakula may state a fact without an observation behind it. The
functions here are the only sanctioned path from a finding to its evidence, and
they fail loudly rather than returning an empty success: a claim that cannot cite
its observations is a bug, not a sparse result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vakula import storage
from vakula.domain import OBSERVATION_PREFIX, Event, Observation, clamp_confidence

_REFERENCE = re.compile(rf"^{OBSERVATION_PREFIX}-(\d+)$", re.IGNORECASE)


class ProvenanceError(ValueError):
    """A reference could not be resolved to a stored observation."""


@dataclass(frozen=True)
class Evidence:
    """One observation cited by an event, with how it was used."""

    observation: Observation
    weight: float = 1.0
    role: str = "CORROBORATES"

    @property
    def reference(self) -> str:
        return self.observation.reference


def format_reference(observation_id: int) -> str:
    """Render the identifier a user types, for example OBS-1842."""
    return f"{OBSERVATION_PREFIX}-{int(observation_id)}"


def parse_reference(reference: str) -> int:
    """Accept ``OBS-1842``, ``obs-1842`` or a bare number and return 1842.

    Raises :class:`ProvenanceError` on anything else, so a typo can never be
    silently read as observation 0.
    """
    text = str(reference or "").strip()
    match = _REFERENCE.match(text)
    if match:
        return int(match.group(1))
    if text.isdigit():
        return int(text)
    raise ProvenanceError(
        f"{text!r} is not an observation reference. Use {OBSERVATION_PREFIX}-<id>, "
        f"for example {OBSERVATION_PREFIX}-1842.")


def _to_observation(record: dict) -> Observation:
    return Observation(
        id=record["id"],
        source_id=record.get("source_id"),
        source_type=record.get("source_type", ""),
        platform=record.get("platform", ""),
        source_key=record.get("source_key", ""),
        author=record.get("author", ""),
        timestamp=record.get("timestamp", ""),
        content=record.get("content", ""),
        url=record.get("url", ""),
        metadata=record.get("metadata", "{}"),
        location_id=record.get("location"),
        language=record.get("language", ""),
        collected_at=record.get("collected_at", ""),
        content_hash=record.get("hash", ""),
        saved=bool(record.get("saved")),
    )


def get_observation(reference: str) -> Observation:
    """Resolve a reference to an Observation, or raise ProvenanceError."""
    observation_id = parse_reference(reference)
    record = storage.get_observation(observation_id)
    if record is None:
        raise ProvenanceError(f"No observation {format_reference(observation_id)} is stored.")
    return _to_observation(record)


def source_name(observation: Observation) -> str:
    """Best available attribution for an observation, without inventing one."""
    if observation.source_id is not None:
        for source in storage.list_sources():
            if source["id"] == observation.source_id:
                return source.get("name") or observation.platform
    return observation.platform or "unknown source"


def events_for_observation(reference: str) -> list[Event]:
    """Every event that rests on this observation as evidence."""
    observation_id = parse_reference(reference)
    with storage.connect() as db:
        rows = db.execute("""
            SELECT e.* FROM events e
            JOIN event_observations eo ON eo.event_id = e.id
            WHERE eo.observation_id = ?
            ORDER BY e.start_time DESC, e.id DESC
        """, (observation_id,)).fetchall()
    return [Event(**dict(row)) for row in rows]


def evidence_for_event(event_id: int) -> list[Evidence]:
    """The observations an event was built from, oldest evidence first.

    Derived purely from stored links, so anything returned is an observation
    that genuinely exists. A missing link is visible as a missing citation; it is
    never filled in with something plausible.
    """
    with storage.connect() as db:
        rows = db.execute("""
            SELECT o.*, eo.weight AS evidence_weight, eo.role AS evidence_role
            FROM event_observations eo
            JOIN observations o ON o.id = eo.observation_id
            WHERE eo.event_id = ?
            ORDER BY o.collected_at ASC, o.id ASC
        """, (int(event_id),)).fetchall()
    evidence = []
    for row in rows:
        record = dict(row)
        weight = clamp_confidence(record.pop("evidence_weight", 1.0))
        role = str(record.pop("evidence_role", "CORROBORATES"))
        evidence.append(Evidence(_to_observation(record), weight=weight, role=role))
    return evidence


def evidence_citations(evidence: list[Evidence]) -> str:
    """Render a citation block, for example ``OBS-1842, OBS-1901``."""
    return ", ".join(item.reference for item in evidence) or "none"


def citation_coverage(evidence: list[Evidence]) -> float:
    """Fraction of cited observations that carry a stored content hash.

    The hash is what proves the text has not changed since collection, so an
    assessment resting on unhashed observations is measurably weaker.
    """
    if not evidence:
        return 0.0
    hashed = sum(1 for item in evidence if item.observation.content_hash)
    return round(clamp_confidence(hashed / len(evidence)), 4)
