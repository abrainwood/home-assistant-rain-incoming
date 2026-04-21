"""Forecast verification statistics computed from a contingency table."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass
class ContingencyTable:
    """2x2 contingency table for binary forecast verification."""

    hits: int = 0              # predicted rain, rain occurred
    misses: int = 0            # predicted no rain, rain occurred
    false_alarms: int = 0      # predicted rain, no rain occurred
    correct_negatives: int = 0  # predicted no rain, no rain occurred

    @property
    def total(self) -> int:
        return self.hits + self.misses + self.false_alarms + self.correct_negatives

    @property
    def pod(self) -> float:
        """Probability of Detection (hit rate). hits / (hits + misses)."""
        denom = self.hits + self.misses
        return self.hits / denom if denom else 0.0

    @property
    def far(self) -> float:
        """False Alarm Ratio. false_alarms / (hits + false_alarms)."""
        denom = self.hits + self.false_alarms
        return self.false_alarms / denom if denom else 0.0

    @property
    def csi(self) -> float:
        """Critical Success Index. hits / (hits + misses + false_alarms).

        Returns 1.0 when there is nothing to predict and nothing went wrong.
        """
        denom = self.hits + self.misses + self.false_alarms
        return self.hits / denom if denom else 1.0

    @property
    def bias(self) -> float:
        """Frequency Bias. (hits + false_alarms) / (hits + misses).

        >1 means over-forecast. Returns 0.0 when there are no rain events.
        """
        denom = self.hits + self.misses
        return (self.hits + self.false_alarms) / denom if denom else 0.0

    @property
    def accuracy(self) -> float:
        """Overall accuracy. (hits + correct_negatives) / total."""
        return (self.hits + self.correct_negatives) / self.total if self.total else 0.0


@dataclass
class ScoreCard:
    """Complete verification scorecard for a location."""

    location: str
    contingency: ContingencyTable
    total_windows: int
    skipped_gaps: int
    lead_time_errors: list[float]  # minutes, hits where both arrival times are known

    @property
    def pod(self) -> float:
        return self.contingency.pod

    @property
    def far(self) -> float:
        return self.contingency.far

    @property
    def csi(self) -> float:
        return self.contingency.csi

    @property
    def bias(self) -> float:
        return self.contingency.bias

    @property
    def mean_lead_time_error(self) -> float | None:
        """Mean lead time error in minutes, or None if no verified predictions."""
        return statistics.mean(self.lead_time_errors) if self.lead_time_errors else None

    @property
    def median_lead_time_error(self) -> float | None:
        """Median lead time error in minutes, or None if no verified predictions."""
        return statistics.median(self.lead_time_errors) if self.lead_time_errors else None


@dataclass(frozen=True)
class VerificationRecord:
    """One prediction paired with its verification outcome."""

    window_end_ts: int
    predicted_rain: bool
    actual_rain: bool
    predicted_arrival_minutes: float | None = None
    actual_arrival_minutes: float | None = None


def build_contingency(records: list[VerificationRecord]) -> ContingencyTable:
    """Build a contingency table from verified predictions."""
    ct = ContingencyTable()
    for r in records:
        if r.predicted_rain and r.actual_rain:
            ct.hits += 1
        elif not r.predicted_rain and r.actual_rain:
            ct.misses += 1
        elif r.predicted_rain and not r.actual_rain:
            ct.false_alarms += 1
        else:
            ct.correct_negatives += 1
    return ct


def _lead_time_errors(records: list[VerificationRecord]) -> list[float]:
    """Extract lead time errors for hits where both arrival times are known."""
    errors = []
    for r in records:
        if (
            r.predicted_rain
            and r.actual_rain
            and r.predicted_arrival_minutes is not None
            and r.actual_arrival_minutes is not None
        ):
            errors.append(r.predicted_arrival_minutes - r.actual_arrival_minutes)
    return errors


def compute_scorecard(
    location: str,
    records: list[VerificationRecord],
    total_windows: int,
    skipped_gaps: int,
) -> ScoreCard:
    """Compute a full scorecard from verification records."""
    return ScoreCard(
        location=location,
        contingency=build_contingency(records),
        total_windows=total_windows,
        skipped_gaps=skipped_gaps,
        lead_time_errors=_lead_time_errors(records),
    )
