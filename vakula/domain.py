"""Plain domain objects for Vakula's intelligence model.

These are value objects only: no database handles, no network, no imports from
the rest of Vakula. Storage maps rows into them; nothing here decides anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Event types are deliberately open. Any uppercase token is a valid event_type;
# this tuple is what Vakula assigns today and what the CLI validates against.
EVENT_TYPES = (
    "MILITARY_ACTIVITY",
    "NAVAL_DEPLOYMENT",
    "AIR_ACTIVITY",
    "DIPLOMATIC_MEETING",
    "PROTEST",
    "SANCTION",
    "CYBER_INCIDENT",
    "EARTHQUAKE",
    "ECONOMIC_DISRUPTION",
    "POLITICAL_DEVELOPMENT",
    "OTHER",
)

ENTITY_TYPES = (
    "COUNTRY",
    "PERSON",
    "ORGANIZATION",
    "MILITARY_UNIT",
    "VESSEL",
    "AIRCRAFT",
    "FACILITY",
    "COMPANY",
    "POLITICAL_GROUP",
    "MEDIA_ORGANIZATION",
    "OTHER",
)

LOCATION_TYPES = ("COUNTRY", "REGION", "CITY", "WATERBODY", "SITE", "OTHER")

# Entity and event edges. An event's link to the observations it rests on lives
# in event_observations rather than here, because it carries a role and weight.
RELATIONSHIP_TYPES = (
    "INVOLVED_IN",
    "LOCATED_IN",
    "INTERACTED_WITH",
    "OCCURRED_AT",
    "RELATED_TO",
)

SOURCE_TYPES = ("WIRE", "SOCIAL", "RSS", "API", "DATASET", "MANUAL", "OTHER")

OBSERVATION_PREFIX = "OBS"


def clamp_confidence(value: Any) -> float:
    """Coerce a stored confidence into 0.0-1.0.

    Confidence is a measurement, never a free-form string. Anything unreadable
    becomes 0.0 rather than a plausible-looking guess.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:  # NaN
        return 0.0
    # Rounded so a displayed confidence never reads 0.44999999999999996.
    return round(max(0.0, min(1.0, number)), 4)


@dataclass(frozen=True)
class Source:
    """A publisher or channel that observations can be attributed to.

    ``reliability`` is declared metadata and stays None until a human records
    it. It weights evidence; it never asserts that a source is true or false.
    """

    id: int | None = None
    key: str = ""
    name: str = ""
    source_type: str = ""
    homepage: str = ""
    reliability: float | None = None
    reliability_basis: str = ""
    correction_count: int = 0
    avg_latency_s: float | None = None
    geographic_coverage: str = ""
    topic_specialties: tuple[str, ...] = ()

    @property
    def is_rated(self) -> bool:
        return self.reliability is not None

    @property
    def reliability_weight(self) -> float:
        """Evidence weight from this source. Unrated means neutral, not zero."""
        return 1.0 if self.reliability is None else clamp_confidence(self.reliability)


@dataclass(frozen=True)
class Observation:
    """Information obtained from a source, with its provenance intact.

    ``content`` is the text as the source returned it and is never replaced by a
    summary or model output. ``content_hash`` is taken once, from the first
    collection, so a later edit by the publisher cannot silently rewrite the
    evidence an assessment already cited.
    """

    id: int
    source_id: int | None = None
    source_type: str = ""
    platform: str = ""
    source_key: str = ""
    author: str = ""
    timestamp: str = ""
    content: str = ""
    url: str = ""
    metadata: str = "{}"
    location_id: int | None = None
    language: str = ""
    collected_at: str = ""
    content_hash: str = ""
    saved: bool = False

    @property
    def reference(self) -> str:
        """The identifier a user sees, for example OBS-1842."""
        return f"{OBSERVATION_PREFIX}-{self.id}"

    @property
    def metadata_dict(self) -> dict:
        try:
            loaded = json.loads(self.metadata)
        except (TypeError, ValueError):
            return {}
        return loaded if isinstance(loaded, dict) else {}


@dataclass(frozen=True)
class Location:
    """A place, possibly approximate. Exact coordinates are never required."""

    id: int | None = None
    key: str = ""
    name: str = ""
    country: str = ""
    region: str = ""
    location_type: str = ""
    latitude: float | None = None
    longitude: float | None = None
    radius_km: float | None = None

    @property
    def is_approximate(self) -> bool:
        return self.latitude is None or self.longitude is None


@dataclass(frozen=True)
class Entity:
    """A stable actor that many events can refer to."""

    id: int | None = None
    key: str = ""
    name: str = ""
    entity_type: str = "OTHER"
    country: str = ""
    aliases: tuple[str, ...] = ()
    first_seen: str = ""
    last_seen: str = ""


@dataclass(frozen=True)
class Event:
    """A real-world occurrence inferred from one or more observations.

    ``confidence`` is calculated by Vakula from evidence quality and is never
    supplied by a model. ``confidence_basis`` records the human-readable reason
    so a number can always be explained.
    """

    id: int | None = None
    event_type: str = "OTHER"
    title: str = ""
    description: str = ""
    start_time: str = ""
    end_time: str = ""
    location_id: int | None = None
    status: str = "ACTIVE"
    confidence: float = 0.0
    confidence_basis: str = ""
    created_at: str = ""
    updated_at: str = ""

    @property
    def reference(self) -> str:
        return f"EVT-{self.id}" if self.id is not None else "EVT-new"


@dataclass(frozen=True)
class Relationship:
    """A typed, weighted edge between two domain objects.

    Kind values are 'ENTITY', 'EVENT' and 'LOCATION'. This is deliberately a
    table of tuples rather than a graph database; adjacency reads are cheap at
    the scale Vakula operates at.
    """

    id: int | None = None
    from_kind: str = ""
    from_id: int = 0
    rel_type: str = ""
    to_kind: str = ""
    to_id: int = 0
    observation_id: int | None = None
    confidence: float = 0.0
    created_at: str = ""

    def connects(self, kind: str, identifier: int) -> bool:
        return ((self.from_kind == kind and self.from_id == identifier)
                or (self.to_kind == kind and self.to_id == identifier))

    def other(self, kind: str, identifier: int) -> tuple[str, int] | None:
        if self.from_kind == kind and self.from_id == identifier:
            return self.to_kind, self.to_id
        if self.to_kind == kind and self.to_id == identifier:
            return self.from_kind, self.from_id
        return None


@dataclass(frozen=True)
class Assessment:
    """A written situational assessment over one region and time window."""

    id: int | None = None
    region_key: str = ""
    created_at: str = ""
    window_days: int = 14
    state: dict = field(default_factory=dict)
    body: str = ""
    origin: str = "TEMPLATE"
    model: str = ""
    analytic_confidence: float = 0.0
    analytic_basis: str = ""


@dataclass(frozen=True)
class Prediction:
    """A probabilistic forecast produced by the forecasting layer.

    ``probability`` comes from Vakula's forecast engine. A model may only write
    ``explanation``; it never supplies the number.
    """

    id: int | None = None
    region_key: str = ""
    created_at: str = ""
    horizon_days: int = 14
    statement: str = ""
    probability: float = 0.0
    engine: str = ""
    basis: dict = field(default_factory=dict)
    explanation: str = ""
    resolved_at: str = ""
    outcome: str = "OPEN"
