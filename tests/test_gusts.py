"""Gust handling: native fields take precedence, Wieringa is the fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from alpinemet.indicators.gusts import (
    MINIMUM_GUST_FACTOR,
    WIERINGA_K,
    estimate_gusts,
    gust_factor,
    native_gust_variable,
    resolve_gusts,
)


@pytest.fixture
def ramp_wind() -> xr.DataArray:
    """Wind speed rising linearly from 0 to 20 m/s."""
    time = pd.date_range("2021-01-01", periods=21, freq="h")
    return xr.DataArray(
        np.linspace(0.0, 20.0, 21),
        dims="time",
        coords={"time": time},
        name="wind_speed_10m",
        attrs={"units": "m/s"},
    )


# --------------------------------------------------------------------------
# Wieringa estimation
# --------------------------------------------------------------------------


def test_wieringa_adds_k_times_the_temporal_standard_deviation(ramp_wind):
    gust = estimate_gusts(ramp_wind)
    expected = ramp_wind.values + WIERINGA_K * ramp_wind.values.std()
    np.testing.assert_allclose(gust.values, expected)


def test_estimated_gusts_never_fall_below_the_mean_wind(wind_speed_field):
    gust = estimate_gusts(wind_speed_field)
    assert bool((gust >= wind_speed_field - 1e-12).all())


def test_floor_binds_when_k_would_undercut_the_mean_wind(ramp_wind):
    """A negative k is unphysical; the floor keeps the output usable."""
    gust = estimate_gusts(ramp_wind, k=-1.0)
    assert bool((gust >= ramp_wind - 1e-12).all())
    # Where the raw estimate undercuts, the floor value is used exactly.
    raw = ramp_wind.values + (-1.0) * ramp_wind.values.std()
    undercuts = raw < ramp_wind.values
    np.testing.assert_allclose(
        gust.values[undercuts], ramp_wind.values[undercuts] * MINIMUM_GUST_FACTOR
    )


def test_k_is_configurable_for_sensitivity_analysis(ramp_wind):
    doubled = estimate_gusts(ramp_wind, k=2 * WIERINGA_K)
    baseline = estimate_gusts(ramp_wind)
    increment = (doubled - baseline).values
    np.testing.assert_allclose(increment, WIERINGA_K * ramp_wind.values.std())


def test_wieringa_without_time_dimension_uses_the_global_spread(small_grid):
    field = xr.DataArray(
        np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]),
        dims=("latitude", "longitude"),
        coords=small_grid,
        attrs={"units": "m/s"},
    )
    gust = estimate_gusts(field)
    np.testing.assert_allclose(gust.values, field.values + WIERINGA_K * field.values.std())


def test_estimate_records_its_provenance(ramp_wind):
    gust = estimate_gusts(ramp_wind)
    assert gust.attrs["gust_source"] == "estimated"
    assert gust.attrs["method"] == "Wieringa (1973)"
    assert gust.attrs["k_factor"] == WIERINGA_K
    assert gust.attrs["units"] == "m/s"
    assert gust.name == "gust_estimated"


# --------------------------------------------------------------------------
# Native gust detection and resolution
# --------------------------------------------------------------------------


def _dataset_with(**variables) -> xr.Dataset:
    time = pd.date_range("2021-01-01", periods=5, freq="h")
    return xr.Dataset(
        {
            name: xr.DataArray(
                np.asarray(values, dtype=float),
                dims="time",
                coords={"time": time},
                attrs={"units": "m/s"},
            )
            for name, values in variables.items()
        }
    )


def test_native_gust_variable_finds_i10fg():
    ds = _dataset_with(wind_speed_10m=[1, 2, 3, 4, 5], i10fg=[2, 4, 6, 8, 10])
    assert native_gust_variable(ds) == "i10fg"


def test_native_gust_variable_returns_none_when_absent():
    ds = _dataset_with(wind_speed_10m=[1, 2, 3, 4, 5])
    assert native_gust_variable(ds) is None


def test_i10fg_wins_over_other_candidates():
    ds = _dataset_with(
        wind_speed_10m=[1, 2, 3, 4, 5], fg10=[9, 9, 9, 9, 9], i10fg=[2, 4, 6, 8, 10]
    )
    assert native_gust_variable(ds) == "i10fg"


def test_resolve_gusts_uses_the_native_field_unchanged():
    ds = _dataset_with(wind_speed_10m=[1, 2, 3, 4, 5], i10fg=[2, 4, 6, 8, 10])
    gust = resolve_gusts(ds)
    np.testing.assert_allclose(gust.values, [2, 4, 6, 8, 10])
    assert gust.attrs["gust_source"] == "native"
    assert gust.attrs["native_variable"] == "i10fg"


def test_resolve_gusts_falls_back_to_estimation():
    ds = _dataset_with(wind_speed_10m=[1, 2, 3, 4, 5])
    gust = resolve_gusts(ds)
    assert gust.attrs["gust_source"] == "estimated"
    expected = ds["wind_speed_10m"].values + WIERINGA_K * ds["wind_speed_10m"].values.std()
    np.testing.assert_allclose(gust.values, expected)


def test_resolve_gusts_accepts_an_explicit_wind_field():
    ds = _dataset_with(u_dummy=[0, 0, 0, 0, 0])
    wind = xr.DataArray(np.array([1.0, 2, 3, 4, 5]), dims="time", name="wind_speed_10m")
    gust = resolve_gusts(ds, wind_speed=wind)
    assert gust.attrs["gust_source"] == "estimated"


def test_resolve_gusts_reports_what_is_missing():
    ds = _dataset_with(t2m=[1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="no native gust field"):
        resolve_gusts(ds)


# --------------------------------------------------------------------------
# Gust factor
# --------------------------------------------------------------------------


def test_gust_factor_is_the_ratio_and_masks_calm_conditions():
    wind = xr.DataArray([0.0, 5.0, 10.0], dims="time", attrs={"units": "m/s"})
    gust = xr.DataArray([1.0, 10.0, 15.0], dims="time", attrs={"units": "m/s"})

    factor = gust_factor(gust, wind)

    assert np.isnan(factor.values[0])
    np.testing.assert_allclose(factor.values[1:], [2.0, 1.5])


def test_gust_factor_does_not_produce_infinities():
    wind = xr.DataArray([0.0, 0.0], dims="time")
    gust = xr.DataArray([3.0, 4.0], dims="time")
    assert not np.isinf(gust_factor(gust, wind).values).any()
