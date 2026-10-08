"""Contradiction detection limited to what can be settled without a model.

Two things can be checked arithmetically: sources that state different numbers
for the same thing, and sources that affirm and deny the same thing. Everything
else is left alone.

That restraint is the point. "No additional vessels departed" and "a further
convoy was observed" are contradictory, and no rule-based reader can know that.
Vakula records no contradiction for such a pair and resolves nothing. It does
not consult the 0.6B model either: a model asked to adjudicate between sources
will pick one, and it will pick confidently. A disputed question stays disputed
until evidence settles it, and until then it is stored as unresolved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vakula import extract
from vakula.domain import clamp_confidence

KIND_QUANTITY = "QUANTITY"
KIND_POLARITY = "POLARITY"
STATUS_UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class Claim:
    """What one observation asserts about one unit of measure."""

    observation_id: int
    reference: str
    source_name: str
    value: int
    unit: str
    negated: bool
    context: str = ""

    def describe(self) -> str:
        stance = "denied" if self.negated else f"affirmed {self.value}"
        return f"{self.reference} [{self.source_name}]: {stance} ({self.unit})"


@dataclass(frozen=True)
class Contradiction:
    """A recorded disagreement. Never a verdict."""

    kind: str
    unit: str
    claims: tuple[Claim, ...]
    status: str = STATUS_UNRESOLVED
    confidence: float = 0.0
    basis: str = ""
    event_id: int | None = None

    @property
    def observation_ids(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(claim.observation_id for claim in self.claims))

    def summary(self) -> str:
        return "; ".join(claim.describe() for claim in self.claims)


def _claims_from(evidence) -> list[Claim]:
    """Pull every quantity assertion out of an event's evidence."""
    claims = []
    for item in evidence:
        observation = item.observation
        for quantity in extract.find_quantities(observation.content):
            claims.append(Claim(
                observation_id=observation.id,
                reference=observation.reference,
                source_name=observation.platform or "unknown",
                value=quantity.value,
                unit=quantity.unit,
                negated=quantity.negated,
                context=quantity.context[:160],
            ))
    return claims


def _distinct_origins(claims: list[Claim]) -> int:
    """How many different publishers are involved in a dispute."""
    return len({claim.source_name for claim in claims})


def detect(evidence, *, event_id: int | None = None) -> list[Contradiction]:
    """Compare the evidence attached to one event and report disagreements.

    Returns records to store, all with status UNRESOLVED. A contradiction is a
    finding about the evidence, never a judgement about which source is right.
    """
    claims = _claims_from(evidence)
    if len(claims) < 2:
        return []
    found: list[Contradiction] = []

    by_unit: dict[str, list[Claim]] = {}
    for claim in claims:
        by_unit.setdefault(claim.unit, []).append(claim)

    for unit, unit_claims in sorted(by_unit.items()):
        affirmed = [claim for claim in unit_claims if not claim.negated]
        denied = [claim for claim in unit_claims if claim.negated]

        values = {claim.value for claim in affirmed}
        if len(affirmed) >= 2 and len(values) >= 2:
            found.append(Contradiction(
                kind=KIND_QUANTITY, unit=unit, claims=tuple(affirmed), event_id=event_id,
                confidence=_dispute_confidence(unit_claims),
                basis=(f"{len(values)} different values stated for {unit}: "
                       f"{', '.join(str(value) for value in sorted(values))}; "
                       "no basis for preferring one"),
            ))

        if affirmed and denied:
            involved = tuple(affirmed) + tuple(denied)
            found.append(Contradiction(
                kind=KIND_POLARITY, unit=unit, claims=involved, event_id=event_id,
                confidence=_dispute_confidence(involved),
                basis=(f"{len(affirmed)} source(s) affirm {unit} while {len(denied)} "
                       f"deny it; polarity disagreement left open"),
            ))
    return found


DISPUTE_CONFIDENCE_CEILING = 0.9


def _dispute_confidence(claims: list[Claim]) -> float:
    """Confidence that a dispute exists, not in either side's claim.

    More independent origins means the disagreement is well evidenced as a
    disagreement. It is hard-capped below certainty, because a dispute can never
    be resolved from inside itself and a number of 1.0 would claim otherwise.
    """
    origins = _distinct_origins(claims)
    if origins < 2:
        return 0.0
    value = 0.4 + 0.25 * min(origins, 3)
    return round(min(value, DISPUTE_CONFIDENCE_CEILING), 4)


def unresolved_semantic(texts: list[str]) -> None:
    """Explicitly record that a semantic judgement is not being made.

    Kept as a named no-op so the decision is visible in the codebase and can be
    asserted in a test, rather than being an absence nobody notices. Vakula does
    not attempt to compare meaning, so it never returns a semantic verdict.
    """
    return None
