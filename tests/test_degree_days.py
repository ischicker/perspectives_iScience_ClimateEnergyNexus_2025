"""Degree days are checked against analytically known totals, not golden files."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from alpinemet.indicators.degree_days import (
    CDD_BASE_TEMPERATURE,
    HDD_BASE_TEMPERATURE,
    cooling_degree_days,
    daily_mean_temperature,
    degree_days,
    heating_degree_days,
)


def test_constant_temperature_gives_exact_annual_total(constant_temperature):
    # 10 degC against a 15 degC base is 5 degC d per day, over 365 days.
    hdd = heating_degree_days(constant_temperature)
    assert hdd.dims == ("latitude", "longitude")
    np.testing.assert_allclose(hdd.values, 365 * 5.0)


def test_constant_temperature_below_cooling_base_gives_zero(constant_temperature):
    # 10 degC never exceeds the 18 degC cooling base.
    cdd = cooling_degree_days(constant_temperature)
    np.testing.assert_allclose(cdd.values, 0.0)


def test_kelvin_and_celsius_inputs_agree(constant_temperature):
    celsius = constant_temperature - 273.15
    celsius.attrs["units"] = "degC"
    np.testing.assert_allclose(
        heating_degree_days(constant_temperature).values,
        heating_degree_days(celsius).values,
    )


def test_accumulation_uses_daily_means_not_hourly_deviations():
    """A symmetric diurnal cycle around the base must not create degree days.

    Summing hourly deviations would return a large positive total here; the
    VDI 3787 definition returns zero. This is the single most consequential
    detail of the implementation.
    """
    time = pd.date_range("2021-01-01", periods=48, freq="h")
    # Mean exactly at the heating base, swinging +/- 8 K over the day.
    values = HDD_BASE_TEMPERATURE + 8.0 * np.sin(np.arange(48) / 24 * 2 * np.pi)
    temperature = xr.DataArray(
        values, dims="time", coords={"time": time}, attrs={"units": "degC"}
    )

    hdd = heating_degree_days(temperature)
    assert float(hdd) == pytest.approx(0.0, abs=1e-9)


def test_monthly_resampling_sums_to_the_annual_total(constant_temperature):
    monthly = heating_degree_days(constant_temperature, freq="1ME")
    annual = heating_degree_days(constant_temperature)

    assert monthly.sizes["time"] == 12
    np.testing.assert_allclose(monthly.sum(dim="time").values, annual.values)


def test_base_temperature_is_honoured(constant_temperature):
    strict = heating_degree_days(constant_temperature, base=12.0)
    np.testing.assert_allclose(strict.values, 365 * 2.0)


def test_degree_days_matches_the_individual_functions(constant_temperature):
    combined = degree_days(constant_temperature)

    np.testing.assert_allclose(
        combined["hdd"].values, heating_degree_days(constant_temperature).values
    )
    np.testing.assert_allclose(
        combined["cdd"].values, cooling_degree_days(constant_temperature).values
    )
    assert combined.attrs["hdd_base_temperature_degC"] == HDD_BASE_TEMPERATURE
    assert combined.attrs["cdd_base_temperature_degC"] == CDD_BASE_TEMPERATURE


def test_already_daily_input_is_not_resampled_twice():
    time = pd.date_range("2021-01-01", periods=10, freq="D")
    temperature = xr.DataArray(
        np.full(10, 5.0), dims="time", coords={"time": time}, attrs={"units": "degC"}
    )
    hdd = heating_degree_days(temperature, already_daily=True)
    assert float(hdd) == pytest.approx(10 * 10.0)


def test_result_carries_provenance_attributes(constant_temperature):
    hdd = heating_degree_days(constant_temperature)
    assert hdd.attrs["standard"] == "VDI 3787"
    assert hdd.attrs["base_temperature_degC"] == HDD_BASE_TEMPERATURE
    assert hdd.attrs["units"] == "degC d"


def test_missing_time_dimension_is_rejected(small_grid):
    field = xr.DataArray(
        np.zeros((3, 4)),
        dims=("latitude", "longitude"),
        coords=small_grid,
        attrs={"units": "degC"},
    )
    with pytest.raises(ValueError, match="no 'time' dimension"):
        daily_mean_temperature(field)
