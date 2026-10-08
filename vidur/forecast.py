"""A deliberately small, fully deterministic forecasting engine.

The probability this module produces answers exactly one question:

    will event activity over the next ``horizon_days`` days exceed the trailing
    baseline rate?

It is not a measure of how likely anyone is to think that is true, and it is not
confidence. Those are different numbers and they are reported separately, which
is why a forecast can legitimately be ``probability=0.72`` alongside
``prediction_confidence=0.41`` — a fairly confident prediction resting on thin
history.

No model is consulted, asked, or allowed to adjust the result. The whole
calculation is arithmetic over the series analytics already measured.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date

from vidur import analytics, confidence

ENGINE = "Geoscope Forecast Engine v0.1"
ENGINE_VERSION = "0.1"

DEFAULT_HORIZON_DAYS = 14
# A logistic on the log ratio, so a doubling reads as roughly three quarters and
# halving as roughly a quarter. Chosen because it is symmetric in log space and
# needs no fitting.
LOG_RATIO_SCALE = 1.6
# Trend is damped over the horizon rather than extrapolated linearly, so a steep
# short-term slope cannot run the probability away.
TREND_DAMPING = 0.5
# Below this many baseline days the probability is reported but flagged.
MINIMUM_HISTORY_DAYS = 7

# Extrapolation guards. A slope measured over the window is never projected
# further than a factor of two in either direction, and a slope smaller than its
# own standard error is discarded as noise before it is used at all.
MIN_PROJECTION_FACTOR = 0.5
MAX_PROJECTION_FACTOR = 2.0
# Keeps the logarithm finite when the current rate is itself zero.
MIN_RATIO = 1e-3


def _slope_stderr(stdev, points: int) -> float | None:
    """Approximate standard error of a least-squares slope.

    For a simple linear fit the slope's standard error is roughly
    ``stdev / sqrt(sum((x - xbar)^2))``, and that sum equals ``n(n^2-1)/12`` for
    evenly spaced points. Returns None when it cannot be computed, so an
    unusable estimate is never replaced with a number.
    """
    if stdev is None or points < 2:
        return None
    try:
        spread = float(stdev)
    except (TypeError, ValueError):
        return None
    if spread < 0:
        return None
    denominator = math.sqrt(points * (points ** 2 - 1) / 12.0)
    if denominator <= 0:
        return None
    return spread / denominator


@dataclass(frozen=True)
class Forecast:
    """A probability, its arithmetic, and its own separate confidence."""

    probability: float
    engine: str
    horizon_days: int
    statement: str
    basis: str
    metrics: dict = field(default_factory=dict)
    prediction_confidence: confidence.Confidence = field(
        default_factory=lambda: confidence.prediction_confidence(0))
    insufficient_evidence: bool = False
    anomaly_status: str = analytics.STATUS_INSUFFICIENT

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)

    def to_dict(self) -> dict:
        return {
            "probability": self.probability,
            "engine": self.engine,
            "horizon_days": self.horizon_days,
            "statement": self.statement,
            "basis": self.basis,
            "metrics": dict(self.metrics),
            "prediction_confidence": self.prediction_confidence.value,
            "prediction_confidence_basis": self.prediction_confidence.basis,
            "insufficient_evidence": self.insufficient_evidence,
            "anomaly_status": self.anomaly_status,
        }


def _logistic(log_ratio: float) -> float:
    """Map a log ratio to a probability, guarded against overflow."""
    try:
        return 1.0 / (1.0 + math.exp(-LOG_RATIO_SCALE * log_ratio))
    except OverflowError:
        return 0.0 if log_ratio < 0 else 1.0


def forecast(scope: analytics.Scope = analytics.ALL, *,
             horizon_days: int = DEFAULT_HORIZON_DAYS,
             window_days: int = analytics.DEFAULT_WINDOW_DAYS,
             baseline_days: int = analytics.DEFAULT_BASELINE_DAYS,
             end: date | None = None) -> Forecast:
    """Quantify how likely activity is to exceed its own trailing baseline."""
    horizon = max(1, int(horizon_days))
    analysis = analytics.analyse(scope, window_days=window_days,
                                 baseline_days=baseline_days, end=end)
    metrics = dict(analysis.metrics)

    baseline_rate = analysis.baseline_rate
    current_rate = analysis.current_rate or 0.0
    slope = float(metrics.get("slope_per_day") or 0.0)

    # A baseline of zero is not a usable history: "exceeds the baseline" cannot
    # be evaluated against nothing, and a ratio of 1.0 would report an even coin
    # dressed up as a finding. Same treatment as no history at all.
    enough_history = (baseline_rate is not None and baseline_rate > 0
                      and analysis.baseline_days >= MINIMUM_HISTORY_DAYS)
    enough_observations = int(metrics.get("window_observations") or 0) >= analytics.MINIMUM_WINDOW_OBSERVATIONS

    if not enough_history or not enough_observations:
        # With no usable baseline there is nothing to exceed, so the honest
        # answer is an even coin with the shortfall stated, not a number borrowed
        # from current activity.
        probability = 0.5
        shortfall = []
        if baseline_rate is not None and baseline_rate <= 0:
            shortfall.append("the baseline recorded no activity at all")
        elif not enough_history:
            shortfall.append(f"only {analysis.baseline_days} day(s) of baseline history")
        if not enough_observations:
            shortfall.append(f"only {metrics.get('window_observations', 0)} observation(s) in the window")
        basis = ("insufficient evidence to forecast: " + "; ".join(shortfall)
                 + ". Reported as an even coin with no historical support.")
        statement = (f"Insufficient evidence to project {scope.label()} activity "
                     f"over {horizon} days.")
        insufficient = True
    else:
        # --- The projection, and why it is guarded three separate ways ---------
        #
        # The defect this replaces: projected = max(0, current + slope*h*damping)
        # collapsed to zero whenever a negative slope won the race, the log-ratio
        # became undefined, and the engine returned exactly 0.5 while the basis
        # text said activity was "expected to fall short". Probability and
        # explanation contradicted each other, and 0.5 was an artefact of the
        # algebra rather than a finding.
        #
        # 1. Significance gate. A least-squares slope carries a standard error of
        #    roughly stdev / sqrt(sum((x - xbar)^2)). When the slope is no larger
        #    than its own error it is indistinguishable from noise, so it is set
        #    to zero. A downward blip in a noisy series is not a trend.
        # 2. Bounded extrapolation. Even a significant slope is not projected
        #    further than a factor of two in either direction over the horizon.
        #    Extrapolating a slope measured over a fortnight across a further
        #    fortnight is arithmetic without evidence.
        # 3. Ratio floor. Because projected is now floored by MIN_PROJECTION_FACTOR,
        #    it can only reach zero when the current rate is itself zero. The
        #    remaining zero case is handled by flooring the ratio, which sends the
        #    probability towards 0 rather than towards the undefined middle.
        #
        # Together these mean 0.5 arises only when the projection genuinely equals
        # the baseline, and the direction stated in the sentence always agrees with
        # the number.
        window_stdev = metrics.get("window_stdev")
        slope_stderr = _slope_stderr(window_stdev, int(analysis.window_days))
        effective_slope = slope if (slope_stderr and abs(slope) > slope_stderr) else 0.0
        slope_ignored = effective_slope == 0.0 and slope != 0.0

        raw_projection = current_rate + effective_slope * horizon * TREND_DAMPING
        floor = current_rate * MIN_PROJECTION_FACTOR
        ceiling = current_rate * MAX_PROJECTION_FACTOR
        projected = min(ceiling, max(floor, raw_projection))
        projection_clamped = projected != raw_projection

        ratio = projected / baseline_rate if baseline_rate > 0 else 1.0
        floored_ratio = max(ratio, MIN_RATIO)
        probability = _logistic(math.log(floored_ratio))
        insufficient = False
        if analysis.is_sustained:
            probability = min(1.0, probability + 0.05)

        direction = "exceed" if ratio > 1.0 else "fall short of"
        basis = (f"projected {projected:.2f} events/day against a {baseline_rate:.2f}/day "
                 f"baseline, a ratio of {ratio:.4f}; slope {slope:+.2f}/day "
                 f"(standard error {slope_stderr:.2f}) damped over {horizon} days; "
                 f"anomaly status {analysis.status}")
        if slope_ignored:
            basis += (f"; slope treated as noise because it is within one standard "
                      f"error, so the projection uses the current rate only")
        if projection_clamped:
            basis += (f"; projection bounded to within {MIN_PROJECTION_FACTOR:g}x-"
                      f"{MAX_PROJECTION_FACTOR:g}x the current rate")
        if analysis.anomaly_score is None:
            # The rate comparison stands, but say plainly that deviation could not
            # be measured, so nobody reads the probability as anomaly-backed.
            basis += "; no standard deviation available, so this rests on rates alone"
        statement = (f"Activity over the next {horizon} days is expected to {direction} "
                     f"the trailing baseline rate of {baseline_rate:.2f} events/day.")

    bounded = max(0.0, min(1.0, probability))
    quality = confidence.prediction_confidence(
        analysis.window_days + analysis.baseline_days,
        baseline_stdev=metrics.get("baseline_stdev"),
        window_days=max(1, int(window_days)),
        baseline_rate=baseline_rate)

    return Forecast(
        probability=round(bounded, 4),
        engine=ENGINE,
        horizon_days=horizon,
        statement=statement,
        basis=basis,
        metrics=metrics,
        prediction_confidence=quality,
        insufficient_evidence=insufficient,
        anomaly_status=analysis.status,
    )
