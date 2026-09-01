"""Hydropower indicators: conservation and correct annualisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from alpinemet.energy.hydro import (
    ALPINE_RUNOFF_COEFFICIENT,
    annual_precipitation,
    runoff_from_precipitation,
    seasonal_totals,
    snow_rain_partition,
)


def _hourly(values, start="2021-01-01") -> xr.DataArray:
    array = np.asarray(values, dtype=float)
    time = pd.date_range(start, periods=array.shape[0], freq="h")
    return xr.DataArray(
        array, dims="time", coords={"time": time}, attrs={"units": "mm"}
    )


def _celsius(values, start="2021-01-01") -> xr.DataArray:
    array = np.asarray(values, dtype=float)
    time = pd.date_range(start, periods=array.shape[0], freq="h")
    return xr.DataArray(
        array, dims="time", coords={"time": time}, attrs={"units": "degC"}
    )


# --------------------------------------------------------------------------
# Snow-rain partition
# --------------------------------------------------------------------------


def test_cold_precipitation_is_all_snow():
    result = snow_rain_partition(_hourly([10.0]), _celsius([-5.0]))
    assert result["snowfall"].item() == pytest.approx(10.0)
    assert result["rainfall"].item() == pytest.approx(0.0)


def test_warm_precipitation_is_all_rain():
    result = snow_rain_partition(_hourly([10.0]), _celsius([8.0]))
    assert result["snowfall"].item() == pytest.approx(0.0)
    assert result["rainfall"].item() == pytest.approx(10.0)


def test_the_transition_window_splits_linearly():
    # 1 degC is the midpoint of the default (0, 2) window.
    result = snow_rain_partition(_hourly([10.0]), _celsius([1.0]))
    assert result["snow_fraction"].item() == pytest.approx(0.5)
    assert result["snowfall"].item() == pytest.approx(5.0)


def test_partition_conserves_total_precipitation():
    precip = _hourly([0.0, 2.0, 5.0, 10.0, 3.0])
    temp = _celsius([-5.0, 0.5, 1.0, 1.5, 6.0])
    result = snow_rain_partition(precip, temp)
    np.testing.assert_allclose(
        (result["snowfall"] + result["rainfall"]).values, precip.values
    )


def test_a_zero_width_window_is_a_hard_threshold():
    precip = _hourly([10.0, 10.0])
    temp = _celsius([-0.1, 0.1])
    result = snow_rain_partition(precip, temp, transition=(0.0, 0.0))
    np.testing.assert_allclose(result["snow_fraction"].values, [1.0, 0.0])


def test_warming_shifts_precipitation_from_snow_to_rain():
    """The mechanism the paper describes: a warmer threshold crossing."""
    precip = _hourly([10.0] * 5)
    cold = _celsius([-1.0] * 5)
    warm = _celsius([3.0] * 5)

    assert float(snow_rain_partition(precip, cold)["snowfall"].sum()) == pytest.approx(50.0)
    assert float(snow_rain_partition(precip, warm)["snowfall"].sum()) == pytest.approx(0.0)


def test_kelvin_temperature_is_converted():
    kelvin = _celsius([268.15])  # -5 degC
    kelvin.attrs["units"] = "K"
    result = snow_rain_partition(_hourly([10.0]), kelvin)
    assert result["snowfall"].item() == pytest.approx(10.0)


def test_inverted_transition_window_is_rejected():
    with pytest.raises(ValueError, match="low <= high"):
        snow_rain_partition(_hourly([1.0]), _celsius([0.0]), transition=(2.0, 0.0))


# --------------------------------------------------------------------------
# Annualisation
# --------------------------------------------------------------------------


def test_hourly_record_annualises_correctly():
    """1 mm every hour for 100 hours is about 8766 mm/year, not 365."""
    precip = _hourly([1.0] * 100)
    annual = float(annual_precipitation(precip))
    assert annual == pytest.approx(365.25 * 24, rel=1e-6)


def test_annualisation_is_not_the_old_per_step_day_assumption():
    """Guards against sum * 365 / n_steps, which understates hourly data 24x."""
    precip = _hourly([1.0] * 100)
    ours = float(annual_precipitation(precip))
    legacy = float(precip.sum()) * 365 / len(precip)
    assert ours == pytest.approx(legacy * 24, rel=1e-3)


def test_daily_and_hourly_records_of_the_same_rate_agree():
    hourly = _hourly([1.0] * 24 * 10)  # 24 mm/day for 10 days

    time = pd.date_range("2021-01-01", periods=10, freq="D")
    daily = xr.DataArray(
        np.full(10, 24.0), dims="time", coords={"time": time}, attrs={"units": "mm"}
    )

    assert float(annual_precipitation(hourly)) == pytest.approx(
        float(annual_precipitation(daily)), rel=1e-6
    )


def test_a_full_year_annualises_to_its_own_total():
    time = pd.date_range("2021-01-01", "2021-12-31 23:00", freq="h")
    precip = xr.DataArray(
        np.full(len(time), 0.1), dims="time", coords={"time": time}, attrs={"units": "mm"}
    )
    assert float(annual_precipitation(precip)) == pytest.approx(
        float(precip.sum()), rel=0.01
    )


# --------------------------------------------------------------------------
# Runoff
# --------------------------------------------------------------------------


def test_runoff_applies_the_coefficient():
    runoff = runoff_from_precipitation(_hourly([10.0, 20.0]))
    np.testing.assert_allclose(
        runoff.values, [10.0 * ALPINE_RUNOFF_COEFFICIENT, 20.0 * ALPINE_RUNOFF_COEFFICIENT]
    )


def test_runoff_carries_its_caveat():
    runoff = runoff_from_precipitation(_hourly([10.0]))
    assert "no snowpack" in runoff.attrs["caveat"]
    assert runoff.attrs["runoff_coefficient"] == ALPINE_RUNOFF_COEFFICIENT


@pytest.mark.parametrize("coefficient", [-0.1, 1.5])
def test_out_of_range_coefficients_are_rejected(coefficient):
    with pytest.raises(ValueError, match="must lie in"):
        runoff_from_precipitation(_hourly([1.0]), coefficient=coefficient)


# --------------------------------------------------------------------------
# Seasonal distribution
# --------------------------------------------------------------------------


def test_seasonal_totals_split_the_year():
    time = pd.date_range("2021-01-01", "2021-12-31", freq="D")
    precip = xr.DataArray(
        np.ones(len(time)), dims="time", coords={"time": time}, attrs={"units": "mm"}
    )
    seasonal = seasonal_totals(precip)

    assert set(seasonal["season"].values) == {"DJF", "MAM", "JJA", "SON"}
    assert float(seasonal.sum()) == pytest.approx(len(time))


def test_seasonal_totals_locate_a_winter_signal():
    time = pd.date_range("2021-01-01", "2021-12-31", freq="D")
    values = np.zeros(len(time))
    values[:31] = 5.0  # January only
    precip = xr.DataArray(values, dims="time", coords={"time": time})

    seasonal = seasonal_totals(precip)
    assert float(seasonal.sel(season="DJF")) == pytest.approx(155.0)
    assert float(seasonal.sel(season="JJA")) == pytest.approx(0.0)
