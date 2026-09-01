"""Solar conversion: physical invariants rather than hard-coded yields."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from alpinemet.energy.solar import (
    REFERENCE_PV_SYSTEM,
    STANDARD_TEST_IRRADIANCE,
    PVSystemSpec,
    fixed_tilt_capacity_factor,
    plane_of_array_ratio_capacity_factor,
)

pvlib = pytest.importorskip("pvlib", reason="solar chain requires the 'energy' extra")

# A clear-sky-ish summer day in the Inn valley, hourly.
INN_VALLEY_LAT = 47.27
INN_VALLEY_LON = 11.39


@pytest.fixture
def summer_day_ghi() -> xr.DataArray:
    times = pd.date_range("2021-06-21", periods=24, freq="h", tz="UTC")
    hours = np.arange(24)
    # Crude diurnal bell peaking at 12 UTC; negative lobes clipped away.
    values = np.clip(900.0 * np.sin((hours - 5) / 14 * np.pi), 0.0, None)
    return xr.DataArray(
        values,
        dims="time",
        coords={"time": times},
        name="ssrd",
        attrs={"units": "W/m2"},
    )


# --------------------------------------------------------------------------
# Reference system definition
# --------------------------------------------------------------------------


def test_reference_system_matches_the_published_specification():
    assert REFERENCE_PV_SYSTEM.dc_capacity_w == 1000.0
    assert REFERENCE_PV_SYSTEM.tilt_deg == 30.0
    assert REFERENCE_PV_SYSTEM.azimuth_deg == 180.0


@pytest.mark.parametrize(
    ("capacity", "tilt", "azimuth"),
    [(0.0, 30.0, 180.0), (1000.0, 120.0, 180.0), (1000.0, 30.0, 400.0)],
)
def test_invalid_system_specs_are_rejected(capacity, tilt, azimuth):
    with pytest.raises(ValueError):
        PVSystemSpec(name="broken", dc_capacity_w=capacity, tilt_deg=tilt, azimuth_deg=azimuth)


# --------------------------------------------------------------------------
# Ratio method
# --------------------------------------------------------------------------


def test_ratio_method_is_the_irradiance_ratio(summer_day_ghi):
    cf = plane_of_array_ratio_capacity_factor(summer_day_ghi)
    np.testing.assert_allclose(cf.values, summer_day_ghi.values / STANDARD_TEST_IRRADIANCE)


def test_ratio_method_clips_to_the_unit_interval():
    ghi = xr.DataArray([-50.0, 500.0, 1500.0], dims="time")
    cf = plane_of_array_ratio_capacity_factor(ghi)
    np.testing.assert_allclose(cf.values, [0.0, 0.5, 1.0])


def test_ratio_method_rejects_non_positive_reference(summer_day_ghi):
    with pytest.raises(ValueError, match="must be positive"):
        plane_of_array_ratio_capacity_factor(summer_day_ghi, reference_irradiance=0.0)


# --------------------------------------------------------------------------
# pvlib fixed-tilt chain
# --------------------------------------------------------------------------


def test_fixed_tilt_stays_within_the_unit_interval(summer_day_ghi):
    cf = fixed_tilt_capacity_factor(
        summer_day_ghi, latitude=INN_VALLEY_LAT, longitude=INN_VALLEY_LON
    )
    assert float(cf.min()) >= 0.0
    assert float(cf.max()) <= 1.0


def test_zero_irradiance_yields_zero_output(summer_day_ghi):
    dark = xr.zeros_like(summer_day_ghi)
    cf = fixed_tilt_capacity_factor(dark, latitude=INN_VALLEY_LAT, longitude=INN_VALLEY_LON)
    np.testing.assert_allclose(cf.values, 0.0)


def test_night_hours_produce_nothing(summer_day_ghi):
    cf = fixed_tilt_capacity_factor(
        summer_day_ghi, latitude=INN_VALLEY_LAT, longitude=INN_VALLEY_LON
    )
    night = summer_day_ghi.values == 0.0
    assert night.any()
    np.testing.assert_allclose(cf.values[night], 0.0)


def test_output_is_monotonic_in_irradiance(summer_day_ghi):
    """Doubling the irradiance cannot reduce the yield."""
    base = fixed_tilt_capacity_factor(
        summer_day_ghi, latitude=INN_VALLEY_LAT, longitude=INN_VALLEY_LON
    )
    brighter = fixed_tilt_capacity_factor(
        summer_day_ghi * 1.5, latitude=INN_VALLEY_LAT, longitude=INN_VALLEY_LON
    )
    assert bool((brighter >= base - 1e-9).all())


def test_temperature_correction_reduces_yield_when_hot(summer_day_ghi):
    """A negative power coefficient must cost yield at high air temperature."""
    warm = xr.full_like(summer_day_ghi, 35.0)
    cold = xr.full_like(summer_day_ghi, 0.0)

    cf_warm = fixed_tilt_capacity_factor(
        summer_day_ghi, latitude=INN_VALLEY_LAT, longitude=INN_VALLEY_LON, temp_air=warm
    )
    cf_cold = fixed_tilt_capacity_factor(
        summer_day_ghi, latitude=INN_VALLEY_LAT, longitude=INN_VALLEY_LON, temp_air=cold
    )
    assert float(cf_warm.sum()) < float(cf_cold.sum())


def test_temperature_correction_is_recorded_in_metadata(summer_day_ghi):
    without = fixed_tilt_capacity_factor(
        summer_day_ghi, latitude=INN_VALLEY_LAT, longitude=INN_VALLEY_LON
    )
    with_temp = fixed_tilt_capacity_factor(
        summer_day_ghi,
        latitude=INN_VALLEY_LAT,
        longitude=INN_VALLEY_LON,
        temp_air=xr.full_like(summer_day_ghi, 20.0),
    )
    assert without.attrs["temperature_correction"] == "not applied"
    assert with_temp.attrs["temperature_correction"] == "applied"
    assert with_temp.attrs["system"] == REFERENCE_PV_SYSTEM.name


def test_tilted_plane_beats_horizontal_ratio_in_winter():
    """The central argument for using the pvlib chain in Alpine winter.

    At high solar zenith a 30 degree south-facing plane collects considerably
    more than the horizontal, so the ratio method understates winter yield.
    """
    times = pd.date_range("2021-12-21", periods=24, freq="h", tz="UTC")
    hours = np.arange(24)
    values = np.clip(300.0 * np.sin((hours - 8) / 8 * np.pi), 0.0, None)
    ghi = xr.DataArray(values, dims="time", coords={"time": times}, attrs={"units": "W/m2"})

    tilted = fixed_tilt_capacity_factor(
        ghi, latitude=INN_VALLEY_LAT, longitude=INN_VALLEY_LON
    )
    horizontal = plane_of_array_ratio_capacity_factor(ghi)

    assert float(tilted.sum()) > float(horizontal.sum())


def test_gridded_input_is_processed_per_grid_point(small_grid):
    times = pd.date_range("2021-06-21", periods=12, freq="h", tz="UTC")
    shape = (len(times), len(small_grid["latitude"]), len(small_grid["longitude"]))
    ghi = xr.DataArray(
        np.full(shape, 600.0),
        dims=("time", "latitude", "longitude"),
        coords={"time": times, **small_grid},
        attrs={"units": "W/m2"},
    )

    cf = fixed_tilt_capacity_factor(ghi)

    assert cf.dims == ghi.dims
    assert cf.shape == ghi.shape
    assert float(cf.max()) <= 1.0
    # Solar geometry differs with latitude, so rows must not be identical.
    per_latitude = cf.mean(dim=("time", "longitude")).values
    assert len(np.unique(np.round(per_latitude, 6))) > 1


def test_missing_time_coordinate_is_rejected():
    ghi = xr.DataArray(np.full(5, 500.0), dims="time")
    with pytest.raises(ValueError, match="no 'time' coordinate"):
        fixed_tilt_capacity_factor(ghi, latitude=47.0, longitude=11.0)


def test_missing_coordinates_are_reported(summer_day_ghi):
    with pytest.raises(ValueError, match="no 'latitude' coordinate"):
        fixed_tilt_capacity_factor(summer_day_ghi)
