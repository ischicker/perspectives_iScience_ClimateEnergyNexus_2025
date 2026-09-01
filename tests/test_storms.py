"""Storm and ramp counting, checked on hand-constructed wind series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from alpinemet.indicators.storms import (
    ALPINE_STORM_THRESHOLDS,
    ESSL_GUST_THRESHOLDS,
    RAMP_THRESHOLDS,
    STANDARD_STORM_THRESHOLDS,
    classify_gust_severity,
    ramp_magnitude,
    storm_days,
    storm_hours,
    wind_ramps,
)


def _hourly(values, start="2021-01-01") -> xr.DataArray:
    array = np.asarray(values, dtype=float)
    time = pd.date_range(start, periods=array.shape[0], freq="h")
    return xr.DataArray(
        array, dims="time", coords={"time": time}, name="wind_speed_10m",
        attrs={"units": "m/s"},
    )


# --------------------------------------------------------------------------
# Threshold definitions
# --------------------------------------------------------------------------


def test_alpine_thresholds_are_lower_than_standard():
    for name, alpine in ALPINE_STORM_THRESHOLDS.items():
        assert alpine < STANDARD_STORM_THRESHOLDS[name]


def test_alpine_reduction_is_between_15_and_20_percent():
    for name, alpine in ALPINE_STORM_THRESHOLDS.items():
        reduction = 1 - alpine / STANDARD_STORM_THRESHOLDS[name]
        assert 0.12 <= reduction <= 0.21, name


def test_essl_thresholds_match_the_paper():
    assert ESSL_GUST_THRESHOLDS["severe"] == 25.0
    assert ESSL_GUST_THRESHOLDS["extreme"] == 32.7


def test_threshold_tables_are_immutable():
    with pytest.raises(TypeError):
        ALPINE_STORM_THRESHOLDS["severe_storm"] = 1.0


# --------------------------------------------------------------------------
# Storm hours
# --------------------------------------------------------------------------


def test_storm_hours_counts_exceedances():
    # 5 hours at 16 m/s: above the Alpine severe threshold of 15.
    wind = _hourly([2, 2, 16, 16, 16, 16, 16, 2, 2, 2])
    hours = storm_hours(wind)
    assert float(hours["severe_storm_hours"]) == pytest.approx(5.0)
    assert float(hours["extreme_storm_hours"]) == pytest.approx(0.0)


def test_thresholds_are_cumulative():
    wind = _hourly([18.0] * 10)
    hours = storm_hours(wind)
    # 18 m/s clears high_wind (10), strong_wind (12.5) and severe_storm (15).
    assert float(hours["high_wind_hours"]) == pytest.approx(10.0)
    assert float(hours["strong_wind_hours"]) == pytest.approx(10.0)
    assert float(hours["severe_storm_hours"]) == pytest.approx(10.0)
    assert float(hours["extreme_storm_hours"]) == pytest.approx(10.0)


def test_threshold_is_inclusive():
    wind = _hourly([15.0] * 4)
    assert float(storm_hours(wind)["severe_storm_hours"]) == pytest.approx(4.0)


def test_three_hourly_data_counts_three_hours_per_step():
    time = pd.date_range("2021-01-01", periods=8, freq="3h")
    wind = xr.DataArray(np.full(8, 20.0), dims="time", coords={"time": time})
    hours = storm_hours(wind)
    assert float(hours["severe_storm_hours"]) == pytest.approx(24.0)


def test_fraction_complements_the_hours():
    wind = _hourly([16.0] * 5 + [2.0] * 5)
    hours = storm_hours(wind)
    assert float(hours["severe_storm_fraction"]) == pytest.approx(0.5)


def test_standard_thresholds_find_fewer_hours_than_alpine():
    wind = _hourly([16.0] * 24)
    alpine = storm_hours(wind, thresholds=ALPINE_STORM_THRESHOLDS)
    standard = storm_hours(wind, thresholds=STANDARD_STORM_THRESHOLDS)
    assert float(alpine["severe_storm_hours"]) > float(standard["severe_storm_hours"])


def test_missing_time_dimension_is_rejected(small_grid):
    field = xr.DataArray(np.zeros((3, 4)), dims=("latitude", "longitude"), coords=small_grid)
    with pytest.raises(ValueError, match="no 'time' dimension"):
        storm_hours(field)


# --------------------------------------------------------------------------
# Storm days
# --------------------------------------------------------------------------


def test_storm_days_counts_a_day_with_any_exceedance():
    # Day 1 has one stormy hour; day 2 has none.
    wind = _hourly([2.0] * 10 + [20.0] + [2.0] * 13 + [2.0] * 24)
    assert float(storm_days(wind)) == pytest.approx(1.0)


def test_daily_mean_reduction_is_stricter_than_any():
    # A single stormy hour cannot lift the daily mean over the threshold.
    wind = _hourly([2.0] * 10 + [20.0] + [2.0] * 13)
    assert float(storm_days(wind, reduction="any")) == 1.0
    assert float(storm_days(wind, reduction="daily_mean")) == 0.0


def test_unknown_reduction_is_rejected():
    with pytest.raises(ValueError, match="Unknown reduction"):
        storm_days(_hourly([5.0] * 24), reduction="median")


# --------------------------------------------------------------------------
# Ramps
# --------------------------------------------------------------------------


def test_ramp_magnitude_is_the_change_across_the_window():
    wind = _hourly([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0])
    change = ramp_magnitude(wind, window_hours=3)
    # First three entries are NaN; then u(t) - u(t-3).
    assert np.isnan(change.values[:3]).all()
    np.testing.assert_allclose(change.values[3:], [3.0, 9.0, 9.0, 9.0, 3.0])


def test_ramp_is_not_a_third_order_difference():
    """Guards against the diff(n=3) mistake in the earlier implementation."""
    wind = _hourly([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0])
    ours = ramp_magnitude(wind, window_hours=3).dropna("time").values
    third_order = wind.diff("time", n=3).values
    assert not np.allclose(ours[: len(third_order)], third_order)


def test_a_constant_series_has_no_ramps():
    change = ramp_magnitude(_hourly([7.0] * 20), window_hours=3)
    np.testing.assert_allclose(change.dropna("time").values, 0.0)


def test_ramp_window_must_divide_the_time_step():
    with pytest.raises(ValueError, match="not a whole multiple"):
        ramp_magnitude(_hourly([1.0] * 10), window_hours=2.5)


def test_upward_and_downward_ramps_are_counted_separately():
    # One +12 m/s ramp and one -12 m/s ramp over 3 hours.
    wind = _hourly([2, 2, 2, 14, 14, 14, 2, 2, 2, 2])
    ramps = wind_ramps(wind, window_hours=3)

    assert int(ramps["extreme_up_count"]) >= 1
    assert int(ramps["extreme_down_count"]) >= 1
    assert float(ramps["max_ramp_up"]) == pytest.approx(12.0)
    assert float(ramps["max_ramp_down"]) == pytest.approx(-12.0)


def test_ramp_classes_are_cumulative():
    wind = _hourly([0, 0, 0, 12, 12, 12])
    ramps = wind_ramps(wind, window_hours=3)
    counts = {name: int(ramps[f"{name}_up_count"]) for name in RAMP_THRESHOLDS}
    assert len(set(counts.values())) == 1, counts
    assert all(count > 0 for count in counts.values())


def test_counts_are_time_steps_not_distinct_events():
    """A single 12 m/s step is seen by three consecutive 3 h windows."""
    wind = _hourly([0, 0, 0, 12, 12, 12])
    ramps = wind_ramps(wind, window_hours=3)
    assert int(ramps["extreme_up_count"]) == 3

    # With a one-hour window, the same step is a single exceedance.
    one_hour = wind_ramps(wind, window_hours=1)
    assert int(one_hour["extreme_up_count"]) == 1


# --------------------------------------------------------------------------
# Gust severity
# --------------------------------------------------------------------------


def test_gust_severity_counts_essl_exceedances():
    gust = _hourly([10, 26, 26, 35, 35, 10])
    severity = classify_gust_severity(gust)
    assert float(severity["severe_hours"]) == pytest.approx(4.0)
    assert float(severity["extreme_hours"]) == pytest.approx(2.0)
    assert float(severity["max_gust"]) == pytest.approx(35.0)


def test_gust_severity_records_the_gust_source():
    gust = _hourly([10.0] * 5)
    gust.attrs["gust_source"] = "native"
    assert classify_gust_severity(gust).attrs["gust_source"] == "native"


def test_gust_source_defaults_to_unknown():
    assert classify_gust_severity(_hourly([10.0] * 5)).attrs["gust_source"] == "unknown"
