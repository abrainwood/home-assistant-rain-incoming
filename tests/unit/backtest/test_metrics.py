"""Tests for backtest metrics - contingency table and scorecard computation."""

import pytest
from scripts.backtest.metrics import (
    ContingencyTable,
    ScoreCard,
    VerificationRecord,
    build_contingency,
    compute_scorecard,
)


# ---------------------------------------------------------------------------
# ContingencyTable property tests
# ---------------------------------------------------------------------------


class TestContingencyTableTotal:
    def test_contingency_total(self):
        ct = ContingencyTable(hits=8, misses=2, false_alarms=3, correct_negatives=7)
        assert ct.total == 20


class TestContingencyTablePerfectForecast:
    def test_pod_is_one(self):
        ct = ContingencyTable(hits=10, misses=0, false_alarms=0, correct_negatives=5)
        assert ct.pod == pytest.approx(1.0)

    def test_far_is_zero(self):
        ct = ContingencyTable(hits=10, misses=0, false_alarms=0, correct_negatives=5)
        assert ct.far == pytest.approx(0.0)

    def test_csi_is_one(self):
        ct = ContingencyTable(hits=10, misses=0, false_alarms=0, correct_negatives=5)
        assert ct.csi == pytest.approx(1.0)

    def test_accuracy_is_one(self):
        ct = ContingencyTable(hits=10, misses=0, false_alarms=0, correct_negatives=5)
        assert ct.accuracy == pytest.approx(1.0)


class TestContingencyTableAllMisses:
    def test_pod_is_zero(self):
        ct = ContingencyTable(hits=0, misses=10, false_alarms=0, correct_negatives=5)
        assert ct.pod == pytest.approx(0.0)

    def test_csi_is_zero(self):
        ct = ContingencyTable(hits=0, misses=10, false_alarms=0, correct_negatives=5)
        assert ct.csi == pytest.approx(0.0)


class TestContingencyTableAllFalseAlarms:
    def test_far_is_one(self):
        ct = ContingencyTable(hits=0, misses=0, false_alarms=10, correct_negatives=0)
        assert ct.far == pytest.approx(1.0)

    def test_csi_is_zero(self):
        ct = ContingencyTable(hits=0, misses=0, false_alarms=10, correct_negatives=0)
        assert ct.csi == pytest.approx(0.0)


class TestContingencyTableMixed:
    """hits=8, misses=2, fa=3, cn=7 - hand-computed values."""

    def setup_method(self):
        self.ct = ContingencyTable(hits=8, misses=2, false_alarms=3, correct_negatives=7)

    def test_pod(self):
        # 8 / (8 + 2) = 0.8
        assert self.ct.pod == pytest.approx(0.8)

    def test_far(self):
        # 3 / (8 + 3) = 0.2727...
        assert self.ct.far == pytest.approx(3 / 11, rel=1e-4)

    def test_csi(self):
        # 8 / (8 + 2 + 3) = 8/13 = 0.6154...
        assert self.ct.csi == pytest.approx(8 / 13, rel=1e-4)

    def test_accuracy(self):
        # (8 + 7) / 20 = 0.75
        assert self.ct.accuracy == pytest.approx(0.75)


class TestContingencyTableEmpty:
    """All zeros - no division by zero errors."""

    def setup_method(self):
        self.ct = ContingencyTable()

    def test_pod_no_error(self):
        assert self.ct.pod == pytest.approx(0.0)

    def test_far_no_error(self):
        assert self.ct.far == pytest.approx(0.0)

    def test_csi_no_error(self):
        # CSI with all zeros: hits + misses + false_alarms = 0 → 1.0 (perfect, nothing to predict)
        assert self.ct.csi == pytest.approx(1.0)

    def test_bias_no_error(self):
        assert self.ct.bias == pytest.approx(0.0)

    def test_accuracy_no_error(self):
        assert self.ct.accuracy == pytest.approx(0.0)


class TestContingencyTableBias:
    def test_bias_over_forecast(self):
        # predictions = hits + false_alarms = 12; events = hits + misses = 8 → bias > 1
        ct = ContingencyTable(hits=8, misses=0, false_alarms=4, correct_negatives=5)
        # (8 + 4) / (8 + 0) = 1.5
        assert ct.bias == pytest.approx(1.5)
        assert ct.bias > 1.0

    def test_bias_under_forecast(self):
        # predictions = 3; events = 8 → bias < 1
        ct = ContingencyTable(hits=3, misses=5, false_alarms=0, correct_negatives=7)
        # (3 + 0) / (3 + 5) = 0.375
        assert ct.bias == pytest.approx(0.375)
        assert ct.bias < 1.0


class TestContingencyTableAccuracy:
    def test_accuracy(self):
        ct = ContingencyTable(hits=6, misses=2, false_alarms=1, correct_negatives=11)
        # (6 + 11) / 20 = 0.85
        assert ct.accuracy == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# build_contingency
# ---------------------------------------------------------------------------


class TestBuildContingency:
    def test_build_contingency_from_records(self):
        records = [
            VerificationRecord(window_end_ts=1, predicted_rain=True, actual_rain=True),    # hit
            VerificationRecord(window_end_ts=2, predicted_rain=True, actual_rain=True),    # hit
            VerificationRecord(window_end_ts=3, predicted_rain=False, actual_rain=True),   # miss
            VerificationRecord(window_end_ts=4, predicted_rain=True, actual_rain=False),   # false alarm
            VerificationRecord(window_end_ts=5, predicted_rain=False, actual_rain=False),  # correct negative
            VerificationRecord(window_end_ts=6, predicted_rain=False, actual_rain=False),  # correct negative
        ]
        ct = build_contingency(records)
        assert ct.hits == 2
        assert ct.misses == 1
        assert ct.false_alarms == 1
        assert ct.correct_negatives == 2


# ---------------------------------------------------------------------------
# ScoreCard
# ---------------------------------------------------------------------------


class TestScorecardDelegatesToContingency:
    def test_pod_far_csi_bias_match_contingency(self):
        ct = ContingencyTable(hits=8, misses=2, false_alarms=3, correct_negatives=7)
        sc = ScoreCard(
            location="Sydney",
            contingency=ct,
            total_windows=100,
            skipped_gaps=5,
            lead_time_errors=[],
        )
        assert sc.pod == pytest.approx(ct.pod)
        assert sc.far == pytest.approx(ct.far)
        assert sc.csi == pytest.approx(ct.csi)
        assert sc.bias == pytest.approx(ct.bias)


class TestScorecardLeadTimeErrors:
    def test_mean_lead_time_error(self):
        # errors = [5.0, 10.0, -3.0] → mean = 4.0
        sc = ScoreCard(
            location="Melbourne",
            contingency=ContingencyTable(hits=3),
            total_windows=10,
            skipped_gaps=0,
            lead_time_errors=[5.0, 10.0, -3.0],
        )
        assert sc.mean_lead_time_error == pytest.approx(4.0)

    def test_median_lead_time_error(self):
        # sorted: [-3.0, 5.0, 10.0] → median = 5.0
        sc = ScoreCard(
            location="Melbourne",
            contingency=ContingencyTable(hits=3),
            total_windows=10,
            skipped_gaps=0,
            lead_time_errors=[5.0, 10.0, -3.0],
        )
        assert sc.median_lead_time_error == pytest.approx(5.0)


class TestScorecardNoLeadTimeData:
    def setup_method(self):
        self.sc = ScoreCard(
            location="Babinda",
            contingency=ContingencyTable(),
            total_windows=20,
            skipped_gaps=2,
            lead_time_errors=[],
        )

    def test_mean_returns_none(self):
        assert self.sc.mean_lead_time_error is None

    def test_median_returns_none(self):
        assert self.sc.median_lead_time_error is None


class TestScorecardLeadTimeFromRecords:
    """Verify lead time errors are computed correctly via compute_scorecard."""

    def test_lead_time_errors_extracted_from_records(self):
        records = [
            # hit with both arrival times → error = predicted - actual = 10 - 8 = 2.0
            VerificationRecord(
                window_end_ts=1,
                predicted_rain=True,
                actual_rain=True,
                predicted_arrival_minutes=10.0,
                actual_arrival_minutes=8.0,
            ),
            # hit with both arrival times → error = 20 - 25 = -5.0
            VerificationRecord(
                window_end_ts=2,
                predicted_rain=True,
                actual_rain=True,
                predicted_arrival_minutes=20.0,
                actual_arrival_minutes=25.0,
            ),
            # hit but missing arrival times → not included in lead time errors
            VerificationRecord(
                window_end_ts=3,
                predicted_rain=True,
                actual_rain=True,
                predicted_arrival_minutes=None,
                actual_arrival_minutes=None,
            ),
            # miss → not a hit, not included
            VerificationRecord(window_end_ts=4, predicted_rain=False, actual_rain=True),
        ]
        sc = compute_scorecard("Lake Margaret", records, total_windows=10, skipped_gaps=1)
        assert sc.lead_time_errors == pytest.approx([2.0, -5.0])
        assert sc.mean_lead_time_error == pytest.approx(-1.5)
