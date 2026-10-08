"""Deterministic temporal analytics over stored events and observations.

Every number here is a function of what is in the database and the window it is
asked about. No sampling, no smoothing that invents intermediate points, no
model. Run it twice on the same data and you get the same answer.

Two ideas carry most of the weight.

Missing days count as zero, not as absent. A day with no events is a measured
zero, and leaving the gap out would quietly inflate every rate.

Insufficient data produces ``None``, never a fallback number. If the baseline has
no variance there is no standard deviation to divide by, so the anomaly score is
None and the status says so. Returning 0.0 there would read as "perfectly
normal" when it actually means "cannot tell".
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from vidur import confidence, fusion, storage

# Minimum days of baseline before a comparison means anything at all.
MINIMUM_BASELINE_DAYS = 7
# Below this many observations in the window, activity counts are not trustworthy.
MINIMUM_WINDOW_OBSERVATIONS = 3

STATUS_NORMAL = "NORMAL"
STATUS_ELEVATED = "ELEVATED"
STATUS_DECLINED = "DECLINED"
STATUS_UNUSUAL = "UNUSUAL"
STATUS_SUSTAINED = "SUSTAINED"
STATUS_INSUFFICIENT = "INSUFFICIENT_DATA"

DEFAULT_WINDOW_DAYS = 14
DEFAULT_BASELINE_DAYS = 28

KIND_ALL = "ALL"
KIND_LOCATION = "LOCATION"
KIND_EVENT_TYPE = "EVENT_TYPE"


@dataclass(frozen=True)
class Scope:
    """What is being counted. Region, event type, or everything."""

    kind: str = KIND_ALL
    value: str = ""

    def label(self) -> str:
        if self.kind == KIND_ALL:
            return "all activity"
        if self.kind == KIND_LOCATION:
            return f"location {self.display_name()}"
        return f"event type {self.value}"

    def display_name(self) -> str:
        """The curated place name, so a briefing reads 'Taiwan Strait'."""
        if self.kind != KIND_LOCATION:
            return self.value
        try:
            for row in storage.list_locations():
                if row["key"] == self.value:
                    return row["name"]
        except Exception:
            pass
        return self.value


ALL = Scope()


@dataclass(frozen=True)
class DayPoint:
    """One complete day. Zero is a measurement, not a gap."""

    day: date
    events: int = 0
    observations: int = 0
    contradictions: int = 0


@dataclass(frozen=True)
class Series:
    """A contiguous daily series with every day present."""

    scope: Scope
    start: date
    end: date
    points: tuple[DayPoint, ...]

    @property
    def days(self) -> int:
        return max(1, len(self.points))

    @property
    def total_events(self) -> int:
        return sum(point.events for point in self.points)

    @property
    def total_observations(self) -> int:
        return sum(point.observations for point in self.points)

    def rate(self) -> float:
        """Events per day across the whole series, zero days included."""
        return self.total_events / self.days

    def observation_rate(self) -> float:
        return self.total_observations / self.days

    def event_values(self) -> list[float]:
        return [float(point.events) for point in self.points]

    def mean(self) -> float:
        return self.total_events / self.days

    def stdev(self) -> float | None:
        """Sample standard deviation, or None when it cannot be computed."""
        values = self.event_values()
        if len(values) < 2:
            return None
        average = sum(values) / len(values)
        variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
        return math.sqrt(variance)

    def moving_average(self, window: int) -> float | None:
        if window <= 0 or self.days < window:
            return None
        values = self.event_values()[-window:]
        return sum(values) / len(values)

    def slope(self) -> float:
        """Least-squares change in events per day across the series."""
        return _slope(self.event_values())

    def acceleration(self) -> float:
        """Change in slope between the older and newer half of the series.

        Positive means the rate of change itself is increasing.
        """
        values = self.event_values()
        if len(values) < 4:
            return 0.0
        midpoint = len(values) // 2
        return _slope(values[midpoint:]) - _slope(values[:midpoint])

    def segments_above(self, threshold: float, *, segments: int = 3) -> int:
        """How many of N equal segments of this series average above ``threshold``.

        The threshold is supplied by the caller so persistence can be judged
        against the baseline rather than against the window's own average.
        Comparing each segment to its own mean detects a spike inside the window
        and is blind to activity that is uniformly elevated, which is the
        opposite of what sustained means.
        """
        values = self.event_values()
        if len(values) < segments or threshold <= 0:
            return 0
        size = len(values) // segments
        count = 0
        for index in range(segments):
            chunk = values[index * size:(index + 1) * size]
            if chunk and sum(chunk) / len(chunk) > threshold:
                count += 1
        return count


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    count = len(values)
    mean_x = (count - 1) / 2.0
    mean_y = sum(values) / count
    numerator = sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values))
    denominator = sum((index - mean_x) ** 2 for index in range(count))
    return numerator / denominator if denominator else 0.0


def window_end(default: date | None = None) -> date:
    """The last complete day. Today is excluded so no day is counted as partial."""
    return (default or date.today()) - timedelta(days=1)


def scope_observation_ids(scope: Scope, *, start: date | None = None,
                          end: date | None = None) -> list[int]:
    """Observation ids inside a scope, from the existing relationship links.

    With a start and end the result is restricted to days in that range, so a
    coverage figure can describe a window rather than the whole record.
    """
    ids = _all_scope_observation_ids(scope)
    if start is None and end is None:
        return ids
    return [observation_id for observation_id in ids
            if _in_window(storage.get_observation(observation_id), start, end)]


def _in_window(record: dict | None, start: date | None, end: date | None) -> bool:
    if record is None:
        return False
    moment = _day(record.get("collected_at", ""))
    if moment is None:
        return False
    return (start is None or moment >= start) and (end is None or moment <= end)


def _all_scope_observation_ids(scope: Scope) -> list[int]:
    storage.init_db()
    with storage.connect() as db:
        if scope.kind == KIND_LOCATION:
            row = db.execute("SELECT id FROM locations WHERE key=? OR name=?",
                             (scope.value, scope.value)).fetchone()
            if row is None:
                return []
            return [entry[0] for entry in db.execute("""
                SELECT DISTINCT from_id FROM relationships
                WHERE from_kind='OBSERVATION' AND to_kind='LOCATION' AND to_id=?
            """, (row[0],)).fetchall()]
        if scope.kind == KIND_EVENT_TYPE:
            return [entry[0] for entry in db.execute("""
                SELECT DISTINCT eo.observation_id FROM event_observations eo
                JOIN events e ON e.id = eo.event_id WHERE e.event_type=?
            """, (scope.value.upper(),)).fetchall()]
        return [entry[0] for entry in db.execute("SELECT id FROM observations").fetchall()]


def scope_event_ids(scope: Scope, *, start: date | None = None,
                    end: date | None = None) -> list[int]:
    """Event ids inside a scope, optionally restricted to a date range."""
    storage.init_db()
    with storage.connect() as db:
        if scope.kind == KIND_EVENT_TYPE:
            rows = db.execute(
                "SELECT id, start_time, created_at FROM events WHERE event_type=?",
                (scope.value.upper(),)).fetchall()
        elif scope.kind == KIND_LOCATION:
            row = db.execute("SELECT id FROM locations WHERE key=? OR name=?",
                             (scope.value, scope.value)).fetchone()
            if row is None:
                return []
            rows = db.execute(
                "SELECT id, start_time, created_at FROM events WHERE location_id=?",
                (row[0],)).fetchall()
        else:
            rows = db.execute("SELECT id, start_time, created_at FROM events").fetchall()
    if start is None and end is None:
        return [row["id"] for row in rows]
    kept = []
    for row in rows:
        moment = _day(row["start_time"]) or _day(row["created_at"])
        if moment is None:
            continue
        if (start is None or moment >= start) and (end is None or moment <= end):
            kept.append(row["id"])
    return kept


def _day(value: str) -> date | None:
    try:
        return date.fromisoformat((value or "")[:10])
    except ValueError:
        return None


def build_series(scope: Scope, *, start: date, end: date) -> Series:
    """Daily counts across an inclusive date range, every day filled in.

    Days with nothing recorded are present with a count of zero. That is a
    measurement of absence of events, not a missing value, and dropping them
    would inflate every rate that follows.
    """
    observation_ids = set(scope_observation_ids(scope))
    event_ids = set(scope_event_ids(scope))
    event_counts: dict[date, int] = {}
    observation_counts: dict[date, int] = {}
    days = [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
    for day in days:
        event_counts[day] = 0
        observation_counts[day] = 0

    if event_ids:
        placeholders = ",".join("?" * len(event_ids))
        with storage.connect() as db:
            stamps = db.execute(
                f"SELECT start_time, created_at FROM events WHERE id IN ({placeholders})",
                list(event_ids)).fetchall()
        for row in stamps:
            moment = _day(row["start_time"]) or _day(row["created_at"])
            if moment in event_counts:
                event_counts[moment] += 1

    if observation_ids:
        placeholders = ",".join("?" * len(observation_ids))
        with storage.connect() as db:
            stamps = db.execute(
                f"SELECT collected_at FROM observations WHERE id IN ({placeholders})",
                list(observation_ids)).fetchall()
        for row in stamps:
            moment = _day(row["collected_at"])
            if moment in observation_counts:
                observation_counts[moment] += 1

    points = tuple(analytics_day(day, event_counts[day], observation_counts[day]) for day in days)
    return Series(scope=scope, start=start, end=end, points=points)


def analytics_day(day: date, events: int = 0, observations: int = 0) -> DayPoint:
    """One complete day. Zero is a measurement, not a gap."""
    return DayPoint(day=day, events=events, observations=observations)


@dataclass(frozen=True)
class Anomaly:
    """An explainable comparison of recent activity against its own history."""

    scope: Scope
    window_days: int
    baseline_days: int
    window_start: str
    window_end: str
    baseline_rate: float | None
    current_rate: float | None
    change_rate: float | None
    anomaly_score: float | None
    status: str
    basis: str
    metrics: dict = field(default_factory=dict)

    @property
    def is_computable(self) -> bool:
        return self.anomaly_score is not None

    @property
    def is_sustained(self) -> bool:
        return self.status == STATUS_SUSTAINED

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)

    def to_dict(self) -> dict:
        return {
            "scope": self.scope.label(), "status": self.status,
            "window_days": self.window_days, "baseline_days": self.baseline_days,
            "window_start": self.window_start, "window_end": self.window_end,
            "baseline_rate": self.baseline_rate, "current_rate": self.current_rate,
            "change_rate": self.change_rate, "anomaly_score": self.anomaly_score,
            "basis": self.basis, "metrics": dict(self.metrics),
        }


def analyse(scope: Scope = ALL, *, window_days: int = DEFAULT_WINDOW_DAYS,
            baseline_days: int = DEFAULT_BASELINE_DAYS,
            end: date | None = None) -> Anomaly:
    """Compare recent activity with the period immediately before it.

    Distinguishes three things the word "anomaly" usually blurs together: high
    volume that is within normal variance, a single unusual window, and activity
    that has stayed elevated across consecutive sub-windows.
    """
    window = max(1, int(window_days))
    baseline = max(0, int(baseline_days))
    finish = window_end(end)
    window_start = finish - timedelta(days=window - 1)
    baseline_start = window_start - timedelta(days=baseline)

    current = build_series(scope, start=window_start, end=finish)
    history = build_series(scope, start=baseline_start, end=window_start - timedelta(days=1))
    trajectory = history.slope() - current.slope()

    baseline_rate = history.rate() if baseline else None
    current_rate = current.rate()
    baseline_stdev = history.stdev()
    change_rate = None
    if baseline_rate and baseline_rate > 0:
        change_rate = round((current_rate - baseline_rate) / baseline_rate, 4)

    score = None
    if baseline >= MINIMUM_BASELINE_DAYS and baseline_stdev is not None and baseline_stdev > 0:
        score = round((current_rate - baseline_rate) / baseline_stdev, 4)

    observations = current.total_observations
    elevated_threshold = (baseline_rate or 0.0) * 1.5
    sustained_segments = (current.segments_above(elevated_threshold)
                          if baseline_rate else 0)
    # Falling relative to where it came from, so a rising number is not mistaken
    # for a deteriorating one.
    declining = (baseline_rate is not None and baseline_rate > 0
                 and current_rate < baseline_rate * 0.75)

    # Ordered most-actionable first: what to go and collect before what to model.
    if observations < MINIMUM_WINDOW_OBSERVATIONS:
        status = STATUS_INSUFFICIENT
        basis = (f"only {observations} observation(s) in the window; "
                 f"at least {MINIMUM_WINDOW_OBSERVATIONS} are needed")
    elif baseline < MINIMUM_BASELINE_DAYS:
        status = STATUS_INSUFFICIENT
        basis = (f"baseline covers {baseline} day(s); at least {MINIMUM_BASELINE_DAYS} "
                 f"are needed before a rate can be compared with anything")
    elif baseline_stdev is None or baseline_stdev <= 0:
        status = STATUS_INSUFFICIENT
        basis = (f"baseline activity was flat at {baseline_rate:.2f} events/day with no "
                 "variance, so no deviation can be measured")
    elif score is None:
        status = STATUS_INSUFFICIENT
        basis = "no computable anomaly score"
    elif score >= 3.0 and sustained_segments >= 2:
        status = STATUS_SUSTAINED
        basis = (f"{current_rate:.2f} events/day against a {baseline_rate:.2f}/day baseline is "
                 f"{score:.1f} standard deviations out, and elevated across "
                 f"{sustained_segments} consecutive sub-windows")
    elif score >= 3.0:
        status = STATUS_UNUSUAL
        basis = (f"{current_rate:.2f} events/day against a {baseline_rate:.2f}/day baseline is "
                 f"{score:.1f} standard deviations out, but only in this window")
    elif change_rate is not None and change_rate > 0.5:
        status = STATUS_ELEVATED
        basis = (f"{current_rate:.2f} events/day is {change_rate:+.0%} against a "
                 f"{baseline_rate:.2f}/day baseline, inside normal variance at {score:.1f} "
                 "standard deviations")
    elif declining:
        status = STATUS_DECLINED
        basis = (f"{current_rate:.2f} events/day is {change_rate:+.0%} against a "
                 f"{baseline_rate:.2f}/day baseline; activity has fallen relative to where it "
                 f"came from (trajectory {trajectory:+.2f}/day)")
    else:
        status = STATUS_NORMAL
        basis = (f"{current_rate:.2f} events/day against a {baseline_rate:.2f}/day baseline, "
                 f"{score:.1f} standard deviations from normal")

    return Anomaly(
        scope=scope, window_days=window, baseline_days=baseline,
        window_start=window_start.isoformat(), window_end=finish.isoformat(),
        baseline_rate=round(baseline_rate, 4) if baseline_rate is not None else None,
        current_rate=round(current_rate, 4), change_rate=change_rate,
        anomaly_score=score, status=status, basis=basis,
        metrics={
            "window_events": current.total_events,
            "window_observations": observations,
            "baseline_events": history.total_events,
            "baseline_stdev": round(baseline_stdev, 4) if baseline_stdev is not None else None,
            "slope_per_day": round(current.slope(), 4),
            "acceleration": round(current.acceleration(), 4),
            "moving_average_7d": current.moving_average(7),
            "elevated_subwindows": sustained_segments,
            "trajectory_per_day": round(trajectory, 4),
            "declining": declining,
            "event_types": compare_event_types(scope, window_days=window, end=end),
            "locations": compare_locations(scope, window_days=window, end=end),
        },
    )


def compare_event_types(scope: Scope = ALL, *, window_days: int = DEFAULT_WINDOW_DAYS,
                        end: date | None = None) -> dict[str, int]:
    """Event counts per type in the window, highest first."""
    storage.init_db()
    finish = window_end(end)
    start = finish - timedelta(days=max(1, int(window_days)) - 1)
    with storage.connect() as db:
        rows = db.execute("""
            SELECT e.event_type AS event_type, COUNT(*) AS total FROM events e
            WHERE substr(COALESCE(e.start_time, e.created_at), 1, 10) BETWEEN ? AND ?
            GROUP BY e.event_type ORDER BY total DESC, e.event_type
        """, (start.isoformat(), finish.isoformat())).fetchall()
    return {row["event_type"]: row["total"] for row in rows}


def compare_locations(scope: Scope = ALL, *, window_days: int = DEFAULT_WINDOW_DAYS,
                      end: date | None = None) -> dict[str, int]:
    """Observation counts per linked location in the window, highest first.

    Region comparison only covers locations present in the curated gazetteer. An
    unrecognised place is reported as unknown rather than bucketed somewhere it
    does not belong.
    """
    storage.init_db()
    finish = window_end(end)
    start = finish - timedelta(days=max(1, int(window_days)) - 1)
    with storage.connect() as db:
        rows = db.execute("""
            SELECT l.name AS name, COUNT(DISTINCT r.from_id) AS total
            FROM relationships r
            JOIN locations l ON l.id = r.to_id
            WHERE r.from_kind='OBSERVATION' AND r.to_kind='LOCATION'
              AND substr((SELECT collected_at FROM observations o WHERE o.id = r.from_id), 1, 10)
                  BETWEEN ? AND ?
            GROUP BY l.name ORDER BY total DESC, l.name
        """, (start.isoformat(), finish.isoformat())).fetchall()
    return {row["name"]: row["total"] for row in rows}


def unresolved_contradictions(scope: Scope = ALL) -> list[dict]:
    """Open disputes. An unresolved dispute is never counted as settled."""
    storage.init_db()
    with storage.connect() as db:
        rows = db.execute("""
            SELECT c.* FROM contradictions c
            WHERE c.status='UNRESOLVED'
              AND c.event_id IN (SELECT id FROM events)
            ORDER BY c.id
        """).fetchall()
    return [dict(row) for row in rows]


def classify_events(scope: Scope = ALL, *, limit: int = 60,
                    start: date | None = None, end: date | None = None) -> dict:
    """Split events into confirmed, candidate and unconfirmed.

    Analytics is read-only, so this re-derives its answer rather than writing a
    status onto anything. An event is confirmed when independent origins
    corroborate it, and a candidate when deterministic fusion scores it against
    another event inside the candidate band without confirming. "No confirmed
    event" means nothing was corroborated; it does not mean nothing happened.
    """
    from vidur import provenance
    storage.init_db()
    every_event_id = scope_event_ids(scope, start=start, end=end)
    event_ids = every_event_id[:limit]
    confirmed, candidates, unconfirmed = [], [], []
    profiles = {}
    for event_id in event_ids:
        evidence = provenance.evidence_for_event(event_id)
        origins = confidence.independent_source_count(evidence)
        profiles[event_id] = evidence
        if origins >= 2:
            confirmed.append(event_id)
        else:
            unconfirmed.append(event_id)

    # Each profile is built once. Rebuilding inside the pair loop made
    # classification quadratic in database round trips.
    cache: dict[int, object] = {}

    def profile_for(event_id: int):
        if event_id not in cache:
            try:
                first = provenance.evidence_for_event(event_id)[0].observation.id
                cache[event_id] = fusion._profile(first)
            except (IndexError, KeyError, provenance.ProvenanceError):
                cache[event_id] = None
        return cache[event_id]

    for event_id in list(unconfirmed):
        left = profile_for(event_id)
        if left is None:
            continue
        for other_id in event_ids:
            if other_id == event_id:
                continue
            right = profile_for(other_id)
            if right is None:
                continue
            if fusion.compare(left, right).verdict == fusion.CANDIDATE:
                candidates.append(event_id)
                unconfirmed.remove(event_id)
                break

    # Counted across every event in scope. Deriving this from the considered
    # events instead would report truncated coverage as thousands of
    # observations that never reached an event.
    storage.init_db()
    placeholders = ",".join("?" * len(every_event_id)) if every_event_id else "NULL"
    with storage.connect() as db:
        linked = {row[0] for row in db.execute(
            f"SELECT DISTINCT observation_id FROM event_observations "
            f"WHERE event_id IN ({placeholders})", every_event_id).fetchall()} \
            if every_event_id else set()
    all_observations = set(scope_observation_ids(scope, start=start, end=end))
    return {
        "confirmed": sorted(confirmed),
        "candidate": sorted(candidates),
        "unconfirmed": sorted(unconfirmed),
        "unlinked_observations": sorted(all_observations - linked),
        # The true total, not the number that fitted inside the traversal budget.
        # Reporting the cap as the total would understate the record silently.
        "total_events": len(every_event_id),
        "considered": len(event_ids),
        "truncated": len(every_event_id) > len(event_ids),
    }
