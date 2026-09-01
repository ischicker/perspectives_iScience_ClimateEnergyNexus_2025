"""Threshold day and hour counting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from alpinemet.indicators.thresholds import (
    HOT_DAY_THRESHOLD,
    exceedance_days,
    exceedance_hours,
    hot_days,
)


def _hourly(values, start="2020-07-01") -> xr.DataArray:
    array = np.asarray(values, dtype=float)
    time = pd.date_range(start, periods=array.shape[0], freq="h")
    return xr.DataArray(
        array, dims="time", coords={"time": time}, attrs={"units": "degC"}
    )


def _two_days(day_one, day_two) -> xr.DataArray:
    return _hourly(list(day_one) + list(day_two))


def test_a_day_peaking_above_the_threshold_counts():
    # Day one peaks at 32 degC, day two at 25.
    warm = [20.0] * 12 + [32.0] + [20.0] * 11
    cool = [18.0] * 12 + [25.0] + [18.0] * 11
    assert float(hot_days(_two_days(warm, cool))) == 1.0


def test_the_threshold_is_inclusive():
    day = [20.0] * 12 + [HOT_DAY_THRESHOLD] + [20.0] * 11
    assert float(hot_days(_two_days(day, day))) == 2.0


def test_the_daily_reduction_changes_the_count_substantially():
    """A single hot hour makes a hot day by maximum, not by mean."""
    day = [15.0] * 23 + [35.0]
    field = _hourly(day * 3)

    by_max = float(exceedance_days(field, 30.0, reduction="max"))
    by_mean = float(exceedance_days(field, 30.0, reduction="mean"))

    assert by_max == 3.0
    assert by_mean == 0.0


def test_counting_below_a_threshold():
    field = _hourly([-5.0] * 24 + [5.0] * 24)
    assert float(exceedance_days(field, 0.0, reduction="min", below=True)) == 1.0


def test_kelvin_input_is_converted():
    warm = [293.15] * 12 + [305.15] + [293.15] * 11  # peaks at 32 degC
    field = _hourly(warm)
    field.attrs["units"] = "K"
    assert float(hot_days(field)) == 1.0


def test_an_unknown_reduction_is_rejected():
    with pytest.raises(ValueError, match="Unknown daily reduction"):
        exceedance_days(_hourly([20.0] * 24), 30.0, reduction="median")


def test_a_missing_time_dimension_is_rejected():
    field = xr.DataArray(np.zeros((2, 2)), dims=("latitude", "longitude"))
    with pytest.raises(ValueError, match="no 'time' dimension"):
        exceedance_days(field, 30.0)


def test_hours_are_counted_at_the_sampling_interval():
    field = _hourly([10.0] * 20 + [20.0] * 4)
    assert float(exceedance_hours(field, 15.0)) == pytest.approx(4.0)


def test_three_hourly_data_counts_three_hours_per_step():
    time = pd.date_range("2020-07-01", periods=8, freq="3h")
    field = xr.DataArray(np.full(8, 20.0), dims="time", coords={"time": time})
    assert float(exceedance_hours(field, 15.0)) == pytest.approx(24.0)


def test_counting_works_on_a_grid(small_grid):
    time = pd.date_range("2020-07-01", periods=48, freq="h")
    values = np.full((48, 3, 4), 20.0)
    values[12, 0, 0] = 35.0  # one hot hour in one cell
    field = xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={"time": time, **small_grid},
        attrs={"units": "degC"},
    )

    counts = hot_days(field)
    assert counts.dims == ("latitude", "longitude")
    assert float(counts.isel(latitude=0, longitude=0)) == 1.0
    assert float(counts.isel(latitude=1, longitude=1)) == 0.0


def test_the_result_records_its_definition():
    counts = hot_days(_hourly([20.0] * 24))
    assert counts.attrs["threshold"] == HOT_DAY_THRESHOLD
    assert counts.attrs["daily_reduction"] == "max"
    assert counts.attrs["units"] == "d"
