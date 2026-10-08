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

    enough_history = baseline_rate is not None and analysis.baseline_days >= MINIMUM_HISTORY_DAYS
    enough_observations = int(metrics.get("window_observations") or 0) >= analytics.MINIMUM_WINDOW_OBSERVATIONS

    if not enough_history or not enough_observations:
        # With no usable baseline there is nothing to exceed, so the honest
        # answer is an even coin with the shortfall stated, not a number borrowed
        # from current activity.
        probability = 0.5
        shortfall = []
        if not enough_history:
            shortfall.append(f"only {analysis.baseline_days} day(s) of baseline history")
        if not enough_observations:
            shortfall.append(f"only {metrics.get('window_observations', 0)} observation(s) in the window")
        basis = ("insufficient evidence to forecast: " + "; ".join(shortfall)
                 + ". Reported as an even coin with no historical support.")
        statement = (f"Insufficient evidence to project {scope.label()} activity "
                     f"over {horizon} days.")
        insufficient = True
    else:
        projected = max(0.0, current_rate + slope * horizon * TREND_DAMPING)
        ratio = projected / baseline_rate if baseline_rate > 0 else 1.0
        probability = _logistic(math.log(ratio) if ratio > 0 else 0.0)
        insufficient = False
        if analysis.is_sustained:
            probability = min(1.0, probability + 0.05)
        direction = "exceed" if ratio > 1.0 else "fall short of"
        basis = (f"projected {projected:.2f} events/day against a {baseline_rate:.2f}/day "
                 f"baseline, a ratio of {ratio:.2f}; slope {slope:+.2f}/day damped over "
                 f"{horizon} days; anomaly status {analysis.status}")
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
