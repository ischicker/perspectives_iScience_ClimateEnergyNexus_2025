"""Verification metrics against analytically known values."""

from __future__ import annotations

import numpy as np
import pytest

from alpinemet.verification.metrics import (
    METRIC_NAMES,
    bias,
    correlation,
    kling_gupta_efficiency,
    mean_absolute_error,
    root_mean_square_error,
    standard_deviation_ratio,
    verification_metrics,
)


def test_a_perfect_model_scores_perfectly():
    observed = np.array([1.0, 2.0, 3.0, 4.0])
    assert bias(observed, observed) == pytest.approx(0.0)
    assert root_mean_square_error(observed, observed) == pytest.approx(0.0)
    assert mean_absolute_error(observed, observed) == pytest.approx(0.0)
    assert correlation(observed, observed) == pytest.approx(1.0)


def test_bias_is_model_minus_observation():
    """A model that is 2 K too warm must report +2, not -2."""
    observed = np.array([10.0, 12.0, 14.0])
    modelled = observed + 2.0
    assert bias(observed, modelled) == pytest.approx(2.0)


def test_rmse_penalises_outliers_more_than_mae():
    observed = np.zeros(4)
    modelled = np.array([0.0, 0.0, 0.0, 4.0])
    assert mean_absolute_error(observed, modelled) == pytest.approx(1.0)
    assert root_mean_square_error(observed, modelled) == pytest.approx(2.0)


def test_rmse_is_exact_on_a_known_case():
    observed = np.array([0.0, 0.0, 0.0])
    modelled = np.array([3.0, 4.0, 0.0])
    assert root_mean_square_error(observed, modelled) == pytest.approx(
        np.sqrt(25.0 / 3.0)
    )


def test_correlation_detects_a_perfect_anticorrelation():
    observed = np.array([1.0, 2.0, 3.0, 4.0])
    assert correlation(observed, -observed) == pytest.approx(-1.0)


def test_correlation_is_undefined_for_a_constant_series():
    assert np.isnan(correlation(np.ones(5), np.arange(5.0)))


def test_correlation_needs_at_least_two_pairs():
    assert np.isnan(correlation([1.0], [1.0]))


def test_standard_deviation_ratio_flags_underdispersion():
    """A coarse product smooths variability: the signature is a ratio below 1."""
    observed = np.array([0.0, 10.0, 0.0, 10.0])
    smoothed = np.array([4.0, 6.0, 4.0, 6.0])
    assert standard_deviation_ratio(observed, smoothed) == pytest.approx(0.2)


# --------------------------------------------------------------------------
# Missing data
# --------------------------------------------------------------------------


def test_pairs_with_a_missing_value_are_dropped():
    observed = np.array([1.0, np.nan, 3.0, 4.0])
    modelled = np.array([1.0, 2.0, np.nan, 4.0])
    assert verification_metrics(observed, modelled)["n_pairs"] == 2


def test_all_missing_returns_nan_metrics_not_an_error():
    metrics = verification_metrics([np.nan, np.nan], [np.nan, np.nan])
    assert metrics["n_pairs"] == 0
    assert np.isnan(metrics["rmse"])


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        bias([1.0, 2.0], [1.0])


# --------------------------------------------------------------------------
# The combined report
# --------------------------------------------------------------------------


def test_the_metric_set_is_complete_and_ordered():
    metrics = verification_metrics([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
    assert list(metrics) == list(METRIC_NAMES)


def test_reported_metrics_agree_with_the_individual_functions():
    observed = np.array([1.0, 3.0, 5.0, 7.0])
    modelled = np.array([1.5, 2.5, 5.5, 8.0])
    metrics = verification_metrics(observed, modelled)

    assert metrics["bias"] == pytest.approx(bias(observed, modelled))
    assert metrics["rmse"] == pytest.approx(root_mean_square_error(observed, modelled))
    assert metrics["correlation"] == pytest.approx(correlation(observed, modelled))


def test_pair_count_is_reported():
    assert verification_metrics(np.arange(10.0), np.arange(10.0))["n_pairs"] == 10


# --------------------------------------------------------------------------
# Kling-Gupta efficiency and its trap
# --------------------------------------------------------------------------


def test_kge_is_one_for_a_perfect_model():
    observed = np.array([2.0, 4.0, 6.0, 8.0])
    assert kling_gupta_efficiency(observed, observed) == pytest.approx(1.0)


def test_kge_works_for_a_strictly_positive_quantity():
    """Wind speed: what KGE is designed for."""
    observed = np.array([2.0, 5.0, 8.0, 11.0])
    modelled = observed * 0.9
    assert 0.0 < kling_gupta_efficiency(observed, modelled) < 1.0


def test_kge_is_undefined_when_the_observed_mean_is_zero():
    """Temperature in degC: the observed mean passes through zero."""
    observed = np.array([-5.0, 0.0, 5.0])
    assert np.isnan(kling_gupta_efficiency(observed, observed + 1.0))


def test_kge_is_unstable_for_celsius_temperature():
    """A near-zero observed mean makes the ratio term dominate everything.

    This is why the module warns against applying KGE to degC temperature: the
    same model error yields a wildly different score depending only on where
    the observed mean happens to sit.
    """
    modelled_offset = 1.0

    near_zero = np.array([-4.0, 0.0, 4.1])
    far_from_zero = near_zero + 20.0

    unstable = kling_gupta_efficiency(near_zero, near_zero + modelled_offset)
    stable = kling_gupta_efficiency(far_from_zero, far_from_zero + modelled_offset)

    assert abs(unstable) > abs(stable) * 5
