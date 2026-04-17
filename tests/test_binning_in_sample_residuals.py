"""
Test suite for _binning_in_sample_residuals method in ForecasterEquivalentDate.

This test suite validates the safety-critical aspects of residual binning for
conformal prediction intervals, including:
- Deterministic behavior with random_state
- Correct bin assignment and interval creation
- Edge case handling (insufficient data, empty bins, extreme n_bins)
- Sample size limits enforcement (10_000 total, 10_000//n_bins per bin)
- Reproducibility across multiple invocations
"""

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.forecaster.recursive import ForecasterEquivalentDate


class TestBinningInSampleResiduals:
    """Test suite for _binning_in_sample_residuals method."""

    def test_deterministic_behavior_with_random_state(self):
        """
        Test that random_state parameter ensures deterministic residual sampling.

        Safety-critical requirement: Predictions must be reproducible for audit trails.
        """
        np.random.seed(42)
        y = pd.Series(
            data=np.random.randn(200) * 10 + 50,
            index=pd.date_range(start="2022-01-01", periods=200, freq="D"),
        )

        # First run
        forecaster1 = ForecasterEquivalentDate(
            offset=7, binner_kwargs={"n_bins": 3, "random_state": 123}
        )
        forecaster1.fit(y=y, store_in_sample_residuals=True)
        residuals1 = forecaster1.in_sample_residuals_.copy()
        bins1 = {
            k: v.copy() for k, v in forecaster1.in_sample_residuals_by_bin_.items()
        }

        # Second run with same random_state
        forecaster2 = ForecasterEquivalentDate(
            offset=7, binner_kwargs={"n_bins": 3, "random_state": 123}
        )
        forecaster2.fit(y=y, store_in_sample_residuals=True)
        residuals2 = forecaster2.in_sample_residuals_.copy()
        bins2 = {
            k: v.copy() for k, v in forecaster2.in_sample_residuals_by_bin_.items()
        }

        # Assert exact reproducibility
        np.testing.assert_array_equal(residuals1, residuals2)
        assert bins1.keys() == bins2.keys()
        for k in bins1.keys():
            np.testing.assert_array_equal(bins1[k], bins2[k])

    def test_different_random_states_produce_different_samples(self):
        """
        Test that different random_state values produce different samples when downsampling.

        Validates that random_state parameter in fit() is actually being used.
        Need large dataset to trigger downsampling (> 10,000 residuals).
        """
        np.random.seed(42)
        # Create large dataset to trigger downsampling
        y = pd.Series(
            data=np.random.randn(12_000) * 10 + 50,
            index=pd.date_range(start="2022-01-01", periods=12_000, freq="D"),
        )

        forecaster1 = ForecasterEquivalentDate(offset=7, binner_kwargs={"n_bins": 2})
        forecaster1.fit(y=y, store_in_sample_residuals=True, random_state=123)
        residuals1 = forecaster1.in_sample_residuals_.copy()

        forecaster2 = ForecasterEquivalentDate(offset=7, binner_kwargs={"n_bins": 2})
        forecaster2.fit(y=y, store_in_sample_residuals=True, random_state=456)
        residuals2 = forecaster2.in_sample_residuals_.copy()

        # Should have different samples (with high probability)
        # Both should have exactly 10,000 samples (downsampled)
        assert len(residuals1) == 10_000
        assert len(residuals2) == 10_000
        assert not np.array_equal(residuals1, residuals2)

    def test_basic_binning_structure(self):
        """
        Test that basic binning creates correct structure with expected bin count.
        """
        y = pd.Series(
            data=np.arange(21, dtype=float),
            index=pd.date_range(start="2022-01-01", periods=21, freq="D"),
        )

        forecaster = ForecasterEquivalentDate(
            offset=7, binner_kwargs={"n_bins": 2, "random_state": 123}
        )
        forecaster.fit(y=y, store_in_sample_residuals=True)

        # Verify attributes exist
        assert hasattr(forecaster, "in_sample_residuals_")
        assert hasattr(forecaster, "in_sample_residuals_by_bin_")
        assert hasattr(forecaster, "binner_intervals_")

        # Verify bin count
        assert len(forecaster.in_sample_residuals_by_bin_) == 2
        assert len(forecaster.binner_intervals_) == 2

        # Verify all bin keys are present
        assert set(forecaster.in_sample_residuals_by_bin_.keys()) == {0, 1}
        assert set(forecaster.binner_intervals_.keys()) == {0, 1}

    def test_residuals_length_matches_expected_window(self):
        """
        Test that residuals have correct length based on offset and series length.
        """
        y = pd.Series(
            data=np.arange(50, dtype=float),
            index=pd.date_range(start="2022-01-01", periods=50, freq="D"),
        )

        offset = 10
        forecaster = ForecasterEquivalentDate(
            offset=offset, binner_kwargs={"n_bins": 2, "random_state": 123}
        )
        forecaster.fit(y=y, store_in_sample_residuals=True)

        # Expected residuals: length of y minus window_size (which equals offset)
        expected_length = len(y) - offset

        # Total residuals across all bins should equal expected length
        total_bin_residuals = sum(
            len(v) for v in forecaster.in_sample_residuals_by_bin_.values()
        )
        assert total_bin_residuals == expected_length

    def test_max_sample_per_bin_enforcement(self):
        """
        Test that residuals per bin are limited to max_sample = 10_000 // n_bins.

        Safety-critical: Prevents memory overflow in production systems.
        """
        # Create large dataset to trigger downsampling
        np.random.seed(42)
        y = pd.Series(
            data=np.random.randn(20_000) * 10 + 50,
            index=pd.date_range(start="2022-01-01", periods=20_000, freq="D"),
        )

        n_bins = 4
        max_sample_per_bin = 10_000 // n_bins  # 2500

        forecaster = ForecasterEquivalentDate(
            offset=7, binner_kwargs={"n_bins": n_bins, "random_state": 123}
        )
        forecaster.fit(y=y, store_in_sample_residuals=True)

        # Verify each bin respects max_sample limit
        for bin_idx, residuals in forecaster.in_sample_residuals_by_bin_.items():
            assert len(residuals) <= max_sample_per_bin, (
                f"Bin {bin_idx} has {len(residuals)} residuals, "
                f"exceeds limit of {max_sample_per_bin}"
            )

    def test_total_residuals_limited_to_10000(self):
        """
        Test that total in_sample_residuals_ is limited to 10,000 samples.

        Safety-critical: Prevents memory overflow in production systems.
        """
        # Create large dataset
        np.random.seed(42)
        y = pd.Series(
            data=np.random.randn(20_000) * 10 + 50,
            index=pd.date_range(start="2022-01-01", periods=20_000, freq="D"),
        )

        forecaster = ForecasterEquivalentDate(
            offset=7, binner_kwargs={"n_bins": 5, "random_state": 123}
        )
        forecaster.fit(y=y, store_in_sample_residuals=True)

        # Verify total residuals limited to 10,000
        assert len(forecaster.in_sample_residuals_) <= 10_000

    def test_empty_bins_filled_with_global_residuals(self):
        """
        Test that empty bins are filled with random samples from all residuals.

        Safety-critical: Ensures all bins have valid residuals for interval prediction.
        """
        # Create imbalanced data that may result in empty bins
        y = pd.Series(
            data=np.concatenate(
                [
                    np.full(30, 10.0),  # Constant low values
                    np.full(30, 100.0),  # Constant high values
                ]
            ),
            index=pd.date_range(start="2022-01-01", periods=60, freq="D"),
        )

        # Use many bins to increase chance of empty bins
        forecaster = ForecasterEquivalentDate(
            offset=7, binner_kwargs={"n_bins": 10, "random_state": 123}
        )
        forecaster.fit(y=y, store_in_sample_residuals=True)

        # All bins should have residuals (no empty bins)
        for bin_idx, residuals in forecaster.in_sample_residuals_by_bin_.items():
            assert len(residuals) > 0, f"Bin {bin_idx} is empty"

    def test_store_false_only_stores_intervals(self):
        """
        Test that store_in_sample_residuals=False stores intervals but not residuals.

        Safety-critical: Reduces memory footprint in production when only intervals needed.
        Note: Attributes are initialized to None in __init__, so they exist but are None.
        """
        y = pd.Series(
            data=np.arange(50, dtype=float),
            index=pd.date_range(start="2022-01-01", periods=50, freq="D"),
        )

        forecaster = ForecasterEquivalentDate(
            offset=7, binner_kwargs={"n_bins": 3, "random_state": 123}
        )
        forecaster.fit(y=y, store_in_sample_residuals=False)

        # Should have intervals (dict with 3 bins)
        assert hasattr(forecaster, "binner_intervals_")
        assert len(forecaster.binner_intervals_) == 3

        # Attributes exist but should be None when store=False
        assert forecaster.in_sample_residuals_ is None
        assert forecaster.in_sample_residuals_by_bin_ is None

    def test_single_bin_edge_case(self):
        """
        Test that n_bins=1 raises ValueError (minimum is 2 bins).

        Safety-critical: Validates input constraints prevent invalid configurations.
        """
        # QuantileBinner requires n_bins >= 2
        with pytest.raises(ValueError, match="must be an int greater than 1"):
            _ = ForecasterEquivalentDate(
                offset=7, binner_kwargs={"n_bins": 1, "random_state": 123}
            )

    def test_many_bins_edge_case(self):
        """
        Test behavior with large number of bins (n_bins=20).

        Validates that system handles high bin counts without errors.
        """
        np.random.seed(42)
        y = pd.Series(
            data=np.random.randn(500) * 20 + 100,
            index=pd.date_range(start="2022-01-01", periods=500, freq="D"),
        )

        n_bins = 20
        forecaster = ForecasterEquivalentDate(
            offset=7, binner_kwargs={"n_bins": n_bins, "random_state": 123}
        )
        forecaster.fit(y=y, store_in_sample_residuals=True)

        # Should have all bins created
        assert len(forecaster.in_sample_residuals_by_bin_) == n_bins

        # All bins should have residuals (filled if originally empty)
        for bin_idx in range(n_bins):
            assert bin_idx in forecaster.in_sample_residuals_by_bin_
            assert len(forecaster.in_sample_residuals_by_bin_[bin_idx]) > 0

    def test_minimal_data_edge_case(self):
        """
        Test behavior with minimal valid dataset (just above window size).
        """
        y = pd.Series(
            data=np.arange(15, dtype=float),
            index=pd.date_range(start="2022-01-01", periods=15, freq="D"),
        )

        forecaster = ForecasterEquivalentDate(
            offset=7, binner_kwargs={"n_bins": 2, "random_state": 123}
        )
        forecaster.fit(y=y, store_in_sample_residuals=True)

        # Should still produce valid binning
        assert hasattr(forecaster, "in_sample_residuals_")
        assert hasattr(forecaster, "in_sample_residuals_by_bin_")

        # Expected residuals: 15 - 7 = 8
        expected_length = 8
        total_residuals = sum(
            len(v) for v in forecaster.in_sample_residuals_by_bin_.values()
        )
        assert total_residuals == expected_length

    def test_binner_intervals_structure(self):
        """
        Test that binner_intervals_ contains valid interval definitions.
        """
        y = pd.Series(
            data=np.arange(50, dtype=float),
            index=pd.date_range(start="2022-01-01", periods=50, freq="D"),
        )

        forecaster = ForecasterEquivalentDate(
            offset=7, binner_kwargs={"n_bins": 3, "random_state": 123}
        )
        forecaster.fit(y=y, store_in_sample_residuals=True)

        # Verify interval structure
        assert isinstance(forecaster.binner_intervals_, dict)

        for bin_idx, interval in forecaster.binner_intervals_.items():
            # Each interval should be a tuple/list with (lower, upper) bounds
            assert hasattr(interval, "__len__")
            assert len(interval) == 2
            lower, upper = interval

            # Upper bound should be >= lower bound
            assert upper >= lower, f"Bin {bin_idx} has invalid interval: {interval}"

    def test_residuals_are_numeric(self):
        """
        Test that all residuals are numeric (not NaN or inf).

        Safety-critical: Invalid residuals could cause prediction failures.
        """
        y = pd.Series(
            data=np.arange(50, dtype=float),
            index=pd.date_range(start="2022-01-01", periods=50, freq="D"),
        )

        forecaster = ForecasterEquivalentDate(
            offset=7, binner_kwargs={"n_bins": 3, "random_state": 123}
        )
        forecaster.fit(y=y, store_in_sample_residuals=True)

        # Check in_sample_residuals_
        assert np.all(np.isfinite(forecaster.in_sample_residuals_))

        # Check in_sample_residuals_by_bin_
        for bin_idx, residuals in forecaster.in_sample_residuals_by_bin_.items():
            assert np.all(
                np.isfinite(residuals)
            ), f"Bin {bin_idx} contains non-finite residuals"

    def test_residuals_dtype_consistency(self):
        """
        Test that residuals maintain consistent numeric dtype.

        Safety-critical: Type consistency prevents numerical errors.
        """
        y = pd.Series(
            data=np.arange(50, dtype=float),
            index=pd.date_range(start="2022-01-01", periods=50, freq="D"),
        )

        forecaster = ForecasterEquivalentDate(
            offset=7, binner_kwargs={"n_bins": 2, "random_state": 123}
        )
        forecaster.fit(y=y, store_in_sample_residuals=True)

        # Verify numeric dtype
        assert np.issubdtype(forecaster.in_sample_residuals_.dtype, np.number)

        for bin_idx, residuals in forecaster.in_sample_residuals_by_bin_.items():
            assert np.issubdtype(
                residuals.dtype, np.number
            ), f"Bin {bin_idx} has non-numeric dtype: {residuals.dtype}"

    def test_with_multiple_offsets_n_offsets_parameter(self):
        """
        Test binning with n_offsets > 1 (aggregation of multiple lags).
        """
        y = pd.Series(
            data=np.arange(100, dtype=float),
            index=pd.date_range(start="2022-01-01", periods=100, freq="D"),
        )

        forecaster = ForecasterEquivalentDate(
            offset=7,
            n_offsets=3,  # Use 3 lags: 7, 14, 21 days
            agg_func=np.mean,
            binner_kwargs={"n_bins": 4, "random_state": 123},
        )
        forecaster.fit(y=y, store_in_sample_residuals=True)

        # Should still produce valid binning
        assert len(forecaster.in_sample_residuals_by_bin_) == 4

        # All bins should have residuals
        for bin_idx in range(4):
            assert len(forecaster.in_sample_residuals_by_bin_[bin_idx]) > 0

    def test_with_dateoffset_instead_of_integer(self):
        """
        Test binning with pandas DateOffset instead of integer offset.
        """
        y = pd.Series(
            data=np.arange(100, dtype=float),
            index=pd.date_range(start="2022-01-01", periods=100, freq="D"),
        )

        forecaster = ForecasterEquivalentDate(
            offset=pd.DateOffset(weeks=1),
            binner_kwargs={"n_bins": 3, "random_state": 123},
        )
        forecaster.fit(y=y, store_in_sample_residuals=True)

        # Should produce valid binning with DateOffset
        assert len(forecaster.in_sample_residuals_by_bin_) == 3
        assert hasattr(forecaster, "binner_intervals_")

        # Verify all bins populated
        for bin_idx in range(3):
            assert len(forecaster.in_sample_residuals_by_bin_[bin_idx]) > 0

    def test_residuals_distribution_across_bins(self):
        """
        Test that residuals are reasonably distributed across bins.

        For uniformly spaced predictions, expect roughly equal distribution.
        """
        # Create linearly increasing series (uniform prediction distribution)
        y = pd.Series(
            data=np.arange(200, dtype=float),
            index=pd.date_range(start="2022-01-01", periods=200, freq="D"),
        )

        n_bins = 5
        forecaster = ForecasterEquivalentDate(
            offset=7, binner_kwargs={"n_bins": n_bins, "random_state": 123}
        )
        forecaster.fit(y=y, store_in_sample_residuals=True)

        bin_counts = {
            k: len(v) for k, v in forecaster.in_sample_residuals_by_bin_.items()
        }

        # For uniform predictions, bins should be roughly balanced
        # Allow 50% deviation from perfect balance
        expected_per_bin = (len(y) - forecaster.offset) / n_bins

        for bin_idx, count in bin_counts.items():
            assert count > 0, f"Bin {bin_idx} is empty"
            # Rough balance check (not strict due to quantile boundaries)
            assert (
                count >= expected_per_bin * 0.5
            ), f"Bin {bin_idx} significantly under-represented: {count} vs expected ~{expected_per_bin}"
            assert (
                count <= expected_per_bin * 1.5
            ), f"Bin {bin_idx} significantly over-represented: {count} vs expected ~{expected_per_bin}"

    def test_reproducibility_after_refit(self):
        """
        Test that refitting with same data and random_state produces identical results.

        Safety-critical: Model behavior must be reproducible for regulatory compliance.
        """
        y = pd.Series(
            data=np.arange(100, dtype=float),
            index=pd.date_range(start="2022-01-01", periods=100, freq="D"),
        )

        forecaster = ForecasterEquivalentDate(
            offset=7, binner_kwargs={"n_bins": 3, "random_state": 123}
        )

        # First fit
        forecaster.fit(y=y, store_in_sample_residuals=True)
        residuals1 = forecaster.in_sample_residuals_.copy()
        bins1 = {k: v.copy() for k, v in forecaster.in_sample_residuals_by_bin_.items()}
        intervals1 = forecaster.binner_intervals_.copy()

        # Refit
        forecaster.fit(y=y, store_in_sample_residuals=True)
        residuals2 = forecaster.in_sample_residuals_.copy()
        bins2 = {k: v.copy() for k, v in forecaster.in_sample_residuals_by_bin_.items()}
        intervals2 = forecaster.binner_intervals_.copy()

        # Assert perfect reproducibility
        np.testing.assert_array_equal(residuals1, residuals2)
        assert bins1.keys() == bins2.keys()
        for k in bins1.keys():
            np.testing.assert_array_equal(bins1[k], bins2[k])
        assert intervals1 == intervals2
