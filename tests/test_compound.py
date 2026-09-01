"""Compound events, checked on hand-constructed co-occurrences."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from alpinemet.indicators.compound import (
    combine_conditions,
    foehn_like,
    heat_drought,
    hellsturm,
    rain_on_snow,
)


def _hourly(values, start="2021-01-01") -> xr.DataArray:
    array = np.asarray(values, dtype=float)
    time = pd.date_range(start, periods=array.shape[0], freq="h")
    return xr.DataArray(array, dims="time", coords={"time": time})


def _gridded(values, start="2021-01-01") -> xr.DataArray:
    """Shape (time, 2, 2): index 0,0 gets the signal, the rest stays quiet."""
    array = np.asarray(values, dtype=float)
    time = pd.date_range(start, periods=array.shape[0], freq="h")
    field = np.zeros((array.size, 2, 2))
    field[:, 0, 0] = array
    return xr.DataArray(
        field,
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": [46.0, 47.0], "longitude": [11.0, 12.0]},
    )


# --------------------------------------------------------------------------
# Generic combiner
# --------------------------------------------------------------------------


def test_conditions_must_all_hold():
    a = _hourly([1, 1, 0, 0]) > 0.5
    b = _hourly([1, 0, 1, 0]) > 0.5
    result = combine_conditions("both", a=a, b=b)
    np.testing.assert_array_equal(result["both"].values, [True, False, False, False])


def test_components_are_preserved_for_tracing():
    a = _hourly([1, 1, 0, 0]) > 0.5
    b = _hourly([1, 0, 1, 0]) > 0.5
    result = combine_conditions("both", a=a, b=b)
    assert "a" in result
    assert "b" in result
    assert result.attrs["components"] == ["a", "b"]


def test_statistics_are_attached():
    condition = _hourly([1, 1, 1, 0, 1, 1]) > 0.5
    result = combine_conditions("event", c=condition)
    assert int(result["event_count"]) == 2
    assert float(result["total_hours"]) == pytest.approx(5.0)


def test_minimum_duration_filters_short_runs():
    condition = _hourly([1, 1, 0, 1, 1, 1, 1, 0]) > 0.5
    result = combine_conditions("event", min_duration_hours=4, c=condition)
    assert float(result["total_hours"]) == pytest.approx(4.0)


def test_no_conditions_is_rejected():
    with pytest.raises(ValueError, match="At least one condition"):
        combine_conditions("empty")


# --------------------------------------------------------------------------
# Hellsturm
# --------------------------------------------------------------------------


def test_hellsturm_needs_both_wind_and_sun():
    wind = _hourly([15, 15, 5, 5])
    solar = _hourly([600, 100, 600, 100])
    result = hellsturm(wind, solar)
    np.testing.assert_array_equal(
        result["hellsturm"].values, [True, False, False, False]
    )


def test_hellsturm_records_thresholds():
    result = hellsturm(_hourly([15.0] * 4), _hourly([600.0] * 4))
    assert result.attrs["wind_threshold_ms"] == 12.0
    assert result.attrs["solar_threshold_wm2"] == 500.0
    assert result.attrs["spatially_aggregated"] is False


def test_gridpointwise_and_aggregated_detection_differ():
    """The central aggregation-scale point: one hot cell versus the domain mean."""
    wind = _gridded([15.0] * 6)
    solar = _gridded([600.0] * 6)

    pointwise = hellsturm(wind, solar)
    aggregated = hellsturm(wind, solar, aggregate=True)

    # One of four cells is stormy and sunny; that cell is detected.
    assert bool(pointwise["hellsturm"].isel(latitude=0, longitude=0).all())
    assert not bool(pointwise["hellsturm"].isel(latitude=1, longitude=1).any())
    # The domain mean is a quarter of the signal, below both thresholds.
    assert not bool(aggregated["hellsturm"].any())


# --------------------------------------------------------------------------
# Heat and drought
# --------------------------------------------------------------------------


def test_heat_drought_needs_hot_and_dry():
    # Four days: hot+dry, hot+wet, cool+dry, cool+wet.
    temp = _hourly([30.0] * 24 + [30.0] * 24 + [10.0] * 24 + [10.0] * 24)
    precip = _hourly([0.0] * 24 + [1.0] * 24 + [0.0] * 24 + [1.0] * 24)

    result = heat_drought(temp, precip)

    np.testing.assert_array_equal(
        result["heat_drought"].values, [True, False, False, False]
    )


def test_heat_drought_output_is_daily():
    temp = _hourly([30.0] * 48)
    precip = _hourly([0.0] * 48)
    result = heat_drought(temp, precip)
    assert result.sizes["time"] == 2
    assert result.attrs["temporal_resolution"] == "daily"


def test_precipitation_is_summed_not_averaged_over_the_day():
    """0.1 mm every hour is 2.4 mm/day, which is not a dry day."""
    temp = _hourly([30.0] * 24)
    precip = _hourly([0.1] * 24)
    result = heat_drought(temp, precip)
    assert not bool(result["heat_drought"].any())


def test_minimum_spell_length_in_days():
    temp = _hourly([30.0] * 24 * 5)
    precip = _hourly([0.0] * 24 * 5)
    short = heat_drought(temp, precip, min_duration_days=7)
    long_enough = heat_drought(temp, precip, min_duration_days=3)
    assert not bool(short["heat_drought"].any())
    assert bool(long_enough["heat_drought"].any())


# --------------------------------------------------------------------------
# Rain on snow
# --------------------------------------------------------------------------


def test_rain_on_snow_requires_the_temperature_window():
    # Too cold, in window, too warm; all wet.
    temp = _hourly([-2.0, 2.0, 10.0])
    precip = _hourly([5.0, 5.0, 5.0])
    result = rain_on_snow(temp, precip)
    np.testing.assert_array_equal(
        result["rain_on_snow"].values, [False, True, False]
    )


def test_rain_on_snow_requires_enough_precipitation():
    temp = _hourly([2.0, 2.0])
    precip = _hourly([0.5, 5.0])
    result = rain_on_snow(temp, precip)
    np.testing.assert_array_equal(result["rain_on_snow"].values, [False, True])


def test_snow_depth_suppresses_false_positives():
    """In the temperature window and wet, but no snowpack to rain onto."""
    temp = _hourly([2.0, 2.0])
    precip = _hourly([5.0, 5.0])
    snow = _hourly([0.0, 0.5])

    without = rain_on_snow(temp, precip)
    with_snow = rain_on_snow(temp, precip, snow_depth=snow)

    np.testing.assert_array_equal(without["rain_on_snow"].values, [True, True])
    np.testing.assert_array_equal(with_snow["rain_on_snow"].values, [False, True])
    assert with_snow.attrs["snow_depth_used"] is True


# --------------------------------------------------------------------------
# Föhn proxy
# --------------------------------------------------------------------------


def test_foehn_proxy_needs_wind_and_warming():
    # Temperature jumps 8 K across hours 3-5 while the wind is strong.
    temp = _hourly([0, 0, 0, 8, 8, 8, 8])
    wind = _hourly([15, 15, 15, 15, 15, 15, 15])

    result = foehn_like(temp, wind, window_hours=3)

    assert bool(result["foehn_like"].isel(time=3))
    assert not bool(result["foehn_like"].isel(time=0))


def test_foehn_proxy_ignores_warming_without_wind():
    temp = _hourly([0, 0, 0, 8, 8, 8, 8])
    wind = _hourly([2.0] * 7)
    result = foehn_like(temp, wind, window_hours=3)
    assert not bool(result["foehn_like"].any())


def test_foehn_warming_is_the_window_difference_not_a_third_derivative():
    temp = _hourly([0, 0, 0, 8, 8, 8, 8])
    wind = _hourly([15.0] * 7)
    result = foehn_like(temp, wind, window_hours=3)

    warming = result["warming"].values
    np.testing.assert_allclose(warming[3:], [8.0, 8.0, 8.0, 0.0])
    third_order = temp.diff("time", n=3).values
    assert not np.allclose(warming[3:][: len(third_order)], third_order)


def test_foehn_proxy_is_labelled_as_a_proxy():
    result = foehn_like(_hourly([0.0] * 6), _hourly([1.0] * 6), window_hours=3)
    assert "proxy" in result.attrs["caveat"]
