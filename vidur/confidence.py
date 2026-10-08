"""Four different kinds of confidence, kept deliberately apart.

These numbers are measurements, not opinions. Every one returns a value together
with a sentence explaining where it came from, because a percentage a user
cannot interrogate is worse than no percentage at all.

The four are never combined and never borrowed from one another:

``source_confidence``     how much this publisher's record is worth
``event_confidence``      how well supported this event is by its evidence
``analytic_confidence``   how much of the intended picture the data covers
``prediction_confidence`` how much the underlying series can bear a forecast

No function here takes a model, a completion or a generated number as input. A
model's opinion of its own certainty is not evidence, and letting one in would
make the whole layer unfalsifiable. ``vidur tests`` assert that by inspecting
these signatures.
"""

from __future__ import annotations

from dataclasses import dataclass

from vidur.domain import clamp_confidence

# Three independent sources is treated as full corroboration. Above that the
# curve flattens, so a hundred reposts of one wire story cannot outweigh the
# fact that they are one story.
FULL_CORROBORATION_SOURCES = 3


@dataclass(frozen=True)
class Confidence:
    """A number and the reason for it. Never return one without the other."""

    value: float
    basis: str

    def __float__(self) -> float:
        return self.value


def independence_key(source_id: int | None, author: str) -> str:
    """Identify the underlying origin of an observation.

    Two posts from the same account through the same backend are one voice, not
    two. Grouping by source_id alone would be too coarse and useless — every
    post on a platform shares one backend — so the author is part of the key.
    An unattributed item falls back to its source, which is the most generous
    reading available for it.
    """
    return f"{source_id or 0}|{(author or '').strip().lower()}"


def source_confidence(source: dict | None) -> Confidence:
    """How much weight this publisher's record carries.

    Reliability is declared metadata, never inferred. An unrated source is
    neutral rather than worthless, because refusing to score an unknown source
    is not the same as calling it unreliable.
    """
    if not source:
        return Confidence(0.5, "no source record; treated as neutral")
    recorded = source.get("reliability")
    if recorded is None:
        return Confidence(0.5, f"{source.get('name', 'source')} has no recorded reliability; neutral weight")
    value = clamp_confidence(recorded)
    basis = str(source.get("reliability_basis") or "").strip()
    corrections = source.get("correction_count")
    detail = f"; {corrections} correction(s) on record" if corrections else ""
    return Confidence(value, f"declared by operator: {basis or 'no basis recorded'}{detail}")


def independent_source_count(evidence) -> int:
    """How many distinct origins the evidence for an event comes from."""
    return len({independence_key(getattr(item.observation, "source_id", None),
                                 getattr(item.observation, "author", ""))
                for item in evidence})


def event_confidence(evidence, *, contradictions=(), citation_coverage: float = 0.0) -> Confidence:
    """How well the stored evidence supports the event it is attached to.

    Driven by three observable things: how many independent origins reported
    it, how much corroborating weight the evidence carries, and whether the
    cited text still carries a hash proving it has not changed. Unresolved
    contradictions reduce the result rather than being ignored.
    """
    total = len(evidence)
    if not total:
        return Confidence(0.0, "no evidence attached; cannot be supported")
    origins = independent_source_count(evidence)
    independence = min(1.0, origins / FULL_CORROBORATION_SOURCES)
    corroboration = sum(clamp_confidence(item.weight) for item in evidence) / total
    coverage = clamp_confidence(citation_coverage)
    base = 0.45 * independence + 0.25 * corroboration + 0.30 * coverage
    # Corroboration is bounded by independence. A single origin cannot vouch for
    # itself however clean its hash is, so one source can never clear 0.5 on its
    # own. The ceiling leaves room for doubt even in the best-supported event:
    # intelligence never reaches certainty.
    value = clamp_confidence(base * (0.55 + 0.45 * independence))
    value = min(value, 0.95)
    penalty = 0.30 * min(1.0, len(contradictions) / max(1, origins))
    value = clamp_confidence(value * (1 - penalty))
    basis = (f"{origins} independent source(s) of {total} observation(s); "
             f"mean evidence weight {corroboration:.2f}; "
             f"hash coverage {coverage:.0%}")
    if contradictions:
        basis += f"; reduced for {len(contradictions)} unresolved contradiction(s)"
    return Confidence(value, basis)


def analytic_confidence(observation_count: int, *, location_coverage: int = 0,
                        entity_coverage: int = 0, anomaly_data: bool = False,
                        expected_actors: int = 0) -> Confidence:
    """How completely the stored data covers the picture being analysed.

    This is about coverage, not truth: a rich but unverified picture and a thin
    but well-sourced one are different problems, and conflating them hides that.
    """
    observations = max(0, int(observation_count))
    if observations == 0:
        return Confidence(0.0, "no observations stored for this scope")
    volume = min(1.0, observations / 20)
    places = min(1.0, location_coverage / 5)
    actors = min(1.0, entity_coverage / 5)
    expected = min(1.0, int(expected_actors) / 10) if expected_actors else 0.5
    value = clamp_confidence(0.35 * volume + 0.25 * places + 0.25 * actors + 0.15 * expected)
    basis = (f"{observations} observation(s); {location_coverage} linked location(s); "
             f"{entity_coverage} linked actor(s)")
    basis += "; anomaly history available" if anomaly_data else "; no anomaly history yet"
    return Confidence(value, basis)


def prediction_confidence(sample_days: int, *, baseline_stdev: float | None = None,
                          window_days: int = 14, baseline_rate: float | None = None) -> Confidence:
    """Whether the recorded series is long and steady enough to forecast.

    Two problems are captured here: too few days to see a pattern, and a
    baseline that swings so much any forecast is noise. A high value here does
    not make a prediction true; it only says the arithmetic had something
    stable to work from.

    ``baseline_rate`` guards a specific trap. A baseline of exactly zero has no
    variance, which would otherwise score as perfectly stable and hand out high
    confidence for a forecast about an empty record. No activity is not
    stability, so it counts against confidence instead.
    """
    days = max(0, int(sample_days))
    window = max(1, int(window_days))
    if days < window:
        return Confidence(0.05, f"only {days} day(s) of history against a {window}-day window")
    if baseline_rate is not None and float(baseline_rate) <= 0:
        return Confidence(0.10,
                          f"{days} day(s) of history, but the baseline recorded no activity "
                          "at all, which is an absence of signal rather than stability")
    depth = min(1.0, days / (window * 4))
    if baseline_stdev is None:
        stability = 0.5
        stability_note = "no baseline variance recorded"
    else:
        spread = max(0.0, float(baseline_stdev))
        stability = clamp_confidence(1.0 - spread)
        stability_note = f"baseline standard deviation {spread:.2f}"
    value = clamp_confidence(0.6 * depth + 0.4 * stability)
    return Confidence(value, f"{days} day(s) of history over a {window}-day window; {stability_note}")
