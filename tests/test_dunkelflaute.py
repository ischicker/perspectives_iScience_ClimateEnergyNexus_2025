"""Dunkelflaute detection, checked on hand-constructed episodes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from alpinemet.indicators.dunkelflaute import (
    DUNKELFLAUTE_MIN_DURATION_HOURS,
    DUNKELFLAUTE_SOLAR_CF_THRESHOLD,
    DUNKELFLAUTE_WIND_CF_THRESHOLD,
    detect,
    detect_from_raw,
    event_statistics,
    infer_timestep_hours,
    sustained,
)


def _hourly(values, start="2021-01-01") -> xr.DataArray:
    array = np.asarray(values)
    time = pd.date_range(start, periods=array.shape[0], freq="h")
    return xr.DataArray(array, dims="time", coords={"time": time})


def _bool_series(pattern: str) -> xr.DataArray:
    """Build a boolean series from a string of 0s and 1s, one char per hour."""
    return _hourly(np.array([c == "1" for c in pattern]))


# --------------------------------------------------------------------------
# Time step inference
# --------------------------------------------------------------------------


def test_hourly_sampling_is_detected():
    assert infer_timestep_hours(_hourly(np.zeros(10))) == pytest.approx(1.0)


def test_three_hourly_sampling_is_detected():
    time = pd.date_range("2021-01-01", periods=10, freq="3h")
    data = xr.DataArray(np.zeros(10), dims="time", coords={"time": time})
    assert infer_timestep_hours(data) == pytest.approx(3.0)


def test_irregular_sampling_is_rejected():
    time = pd.DatetimeIndex(["2021-01-01 00:00", "2021-01-01 01:00", "2021-01-01 05:00"])
    data = xr.DataArray(np.zeros(3), dims="time", coords={"time": time})
    with pytest.raises(ValueError, match="irregularly spaced"):
        infer_timestep_hours(data)


def test_single_time_step_is_rejected():
    data = xr.DataArray(np.zeros(1), dims="time", coords={"time": [pd.Timestamp("2021-01-01")]})
    with pytest.raises(ValueError, match="at least two time steps"):
        infer_timestep_hours(data)


# --------------------------------------------------------------------------
# Sustained-run logic
# --------------------------------------------------------------------------


def test_runs_shorter_than_the_threshold_are_dropped():
    # Two runs: 3 hours, then 5 hours. With a 4 hour threshold only the second survives.
    condition = _bool_series("111000011111000")
    result = sustained(condition, min_duration_hours=4)
    expected = [c == "1" for c in "000000011111000"]
    np.testing.assert_array_equal(result.values, expected)


def test_a_run_of_exactly_the_threshold_length_is_kept():
    condition = _bool_series("00111100")
    result = sustained(condition, min_duration_hours=4)
    np.testing.assert_array_equal(result.values, condition.values)


def test_a_run_one_hour_short_is_dropped():
    condition = _bool_series("00111000")
    result = sustained(condition, min_duration_hours=4)
    assert not result.values.any()


def test_a_gap_of_one_hour_splits_an_episode():
    """48 hours of calm broken by a single windy hour is not one 48 hour episode."""
    pattern = "1" * 30 + "0" + "1" * 30
    result = sustained(_bool_series(pattern), min_duration_hours=48)
    assert not result.values.any()


def test_runs_at_the_series_edges_are_handled():
    condition = _bool_series("11110000000001111")
    result = sustained(condition, min_duration_hours=4)
    np.testing.assert_array_equal(result.values, condition.values)


def test_coarser_sampling_needs_fewer_steps():
    """With 3-hourly data, 48 hours is 16 steps, not 48."""
    time = pd.date_range("2021-01-01", periods=20, freq="3h")
    values = np.zeros(20, dtype=bool)
    values[2:19] = True  # 17 steps = 51 hours
    condition = xr.DataArray(values, dims="time", coords={"time": time})

    result = sustained(condition, min_duration_hours=48)
    assert result.values.sum() == 17


def test_sustained_works_on_gridded_input(small_grid):
    time = pd.date_range("2021-01-01", periods=10, freq="h")
    values = np.zeros((10, 3, 4), dtype=bool)
    values[0:6, 0, 0] = True  # long enough
    values[0:2, 1, 1] = True  # too short
    condition = xr.DataArray(
        values, dims=("time", "latitude", "longitude"), coords={"time": time, **small_grid}
    )

    result = sustained(condition, min_duration_hours=4)

    assert result.dims == condition.dims
    assert result.isel(latitude=0, longitude=0).values.sum() == 6
    assert result.isel(latitude=1, longitude=1).values.sum() == 0


# --------------------------------------------------------------------------
# Event statistics
# --------------------------------------------------------------------------


def test_event_statistics_counts_separate_episodes():
    mask = _bool_series("111100111111000")
    stats = event_statistics(mask)

    assert int(stats["event_count"]) == 2
    assert float(stats["total_hours"]) == pytest.approx(10.0)
    assert float(stats["longest_event_hours"]) == pytest.approx(6.0)
    assert float(stats["frequency"]) == pytest.approx(10 / 15)


def test_event_statistics_on_an_empty_mask():
    stats = event_statistics(_bool_series("00000"))
    assert int(stats["event_count"]) == 0
    assert float(stats["longest_event_hours"]) == 0.0
    assert float(stats["frequency"]) == 0.0


# --------------------------------------------------------------------------
# Reference detection from capacity factors
# --------------------------------------------------------------------------


def test_reference_thresholds_match_the_paper():
    assert DUNKELFLAUTE_WIND_CF_THRESHOLD == 0.10
    assert DUNKELFLAUTE_SOLAR_CF_THRESHOLD == 0.05
    assert DUNKELFLAUTE_MIN_DURATION_HOURS == 48


def test_a_60_hour_shortfall_is_detected():
    n = 100
    wind = np.full(n, 0.5)
    solar = np.full(n, 0.5)
    wind[10:70] = 0.02  # 60 hours below both thresholds
    solar[10:70] = 0.01

    result = detect(_hourly(wind), _hourly(solar))

    assert int(result["event_count"]) == 1
    assert float(result["total_hours"]) == pytest.approx(60.0)
    assert bool(result["dunkelflaute"].isel(time=40))
    assert not bool(result["dunkelflaute"].isel(time=5))


def test_a_40_hour_shortfall_is_below_the_duration_threshold():
    n = 100
    wind = np.full(n, 0.5)
    solar = np.full(n, 0.5)
    wind[10:50] = 0.02
    solar[10:50] = 0.01

    result = detect(_hourly(wind), _hourly(solar))

    assert int(result["event_count"]) == 0
    # The instantaneous condition is still there; only the episode is not.
    assert result["shortfall"].values.sum() == 40


def test_low_wind_alone_is_not_a_dunkelflaute():
    n = 100
    wind = np.full(n, 0.02)  # calm throughout
    solar = np.full(n, 0.5)  # but sunny
    result = detect(_hourly(wind), _hourly(solar))
    assert int(result["event_count"]) == 0


def test_low_solar_alone_is_not_a_dunkelflaute():
    n = 100
    wind = np.full(n, 0.5)
    solar = np.full(n, 0.0)
    result = detect(_hourly(wind), _hourly(solar))
    assert int(result["event_count"]) == 0


def test_thresholds_are_strict_inequalities():
    """A capacity factor exactly at the threshold is not a shortfall."""
    n = 100
    wind = np.full(n, DUNKELFLAUTE_WIND_CF_THRESHOLD)
    solar = np.full(n, DUNKELFLAUTE_SOLAR_CF_THRESHOLD)
    result = detect(_hourly(wind), _hourly(solar))
    assert result["shortfall"].values.sum() == 0


def test_raising_the_wind_threshold_finds_more_events():
    """Documents the sensitivity the paper warns about."""
    n = 200
    rng = np.random.default_rng(7)
    wind = rng.uniform(0.0, 0.25, n)
    solar = np.full(n, 0.01)

    strict = detect(_hourly(wind), _hourly(solar), min_duration_hours=3)
    loose = detect(_hourly(wind), _hourly(solar), wind_threshold=0.20, min_duration_hours=3)

    assert float(loose["total_hours"]) > float(strict["total_hours"])


def test_cold_dunkelflaute_requires_freezing_temperatures():
    n = 100
    wind = np.full(n, 0.02)
    solar = np.full(n, 0.01)
    temperature = np.full(n, 5.0)
    temperature[10:70] = -3.0

    result = detect(_hourly(wind), _hourly(solar), temperature=_hourly(temperature))

    assert "cold_dunkelflaute" in result
    assert float(result["cold_total_hours"]) == pytest.approx(60.0)
    assert float(result["total_hours"]) == pytest.approx(100.0)


def test_cold_variant_is_absent_without_a_temperature_field():
    n = 100
    result = detect(_hourly(np.full(n, 0.02)), _hourly(np.full(n, 0.01)))
    assert "cold_dunkelflaute" not in result


def test_detection_records_its_thresholds():
    n = 100
    result = detect(_hourly(np.full(n, 0.02)), _hourly(np.full(n, 0.01)))
    assert result.attrs["method"] == "capacity factor thresholds"
    assert result.attrs["wind_cf_threshold"] == DUNKELFLAUTE_WIND_CF_THRESHOLD
    assert result.attrs["min_duration_hours"] == DUNKELFLAUTE_MIN_DURATION_HOURS


# --------------------------------------------------------------------------
# Raw meteorological variant
# --------------------------------------------------------------------------


def test_raw_variant_uses_meteorological_thresholds():
    n = 100
    wind = np.full(n, 8.0)
    ghi = np.full(n, 400.0)
    wind[10:70] = 1.5
    ghi[10:70] = 20.0

    result = detect_from_raw(_hourly(wind), _hourly(ghi))

    assert int(result["event_count"]) == 1
    assert float(result["total_hours"]) == pytest.approx(60.0)
    assert result.attrs["method"] == "meteorological thresholds"


def test_raw_and_capacity_factor_variants_are_independent():
    """Calm but bright: a shortfall by neither definition."""
    n = 100
    wind = np.full(n, 1.0)
    ghi = np.full(n, 800.0)
    result = detect_from_raw(_hourly(wind), _hourly(ghi))
    assert int(result["event_count"]) == 0
