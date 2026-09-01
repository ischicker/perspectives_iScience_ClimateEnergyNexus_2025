"""Subsetting across the grid conventions the evaluated products actually use."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from alpinemet.io.subset import (
    ALPINE_DOMAIN,
    BoundingBox,
    normalise_longitude,
    subset_spatial,
    subset_temporal,
)


def _grid(latitudes, longitudes, *, lat_name="latitude", lon_name="longitude") -> xr.Dataset:
    time = pd.date_range("2021-01-01", periods=4, freq="D")
    shape = (len(time), len(latitudes), len(longitudes))
    return xr.Dataset(
        {
            "temperature_2m": xr.DataArray(
                np.zeros(shape),
                dims=("time", lat_name, lon_name),
                coords={"time": time, lat_name: latitudes, lon_name: longitudes},
            )
        }
    )


# --------------------------------------------------------------------------
# Bounding box
# --------------------------------------------------------------------------


def test_alpine_domain_matches_the_published_methods():
    assert (ALPINE_DOMAIN.lon_min, ALPINE_DOMAIN.lon_max) == (4.0, 18.0)
    assert (ALPINE_DOMAIN.lat_min, ALPINE_DOMAIN.lat_max) == (43.0, 49.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lon_min": 18.0, "lon_max": 4.0, "lat_min": 43.0, "lat_max": 49.0},
        {"lon_min": 4.0, "lon_max": 18.0, "lat_min": 49.0, "lat_max": 43.0},
        {"lon_min": 4.0, "lon_max": 18.0, "lat_min": -95.0, "lat_max": 49.0},
    ],
)
def test_invalid_boxes_are_rejected(kwargs):
    with pytest.raises(ValueError):
        BoundingBox(**kwargs)


# --------------------------------------------------------------------------
# Latitude direction: the ERA5 trap
# --------------------------------------------------------------------------


def test_ascending_latitude_is_subset():
    ds = _grid(np.arange(40.0, 55.1, 1.0), np.arange(0.0, 25.1, 1.0))
    subset = subset_spatial(ds)
    assert float(subset["latitude"].min()) >= 43.0
    assert float(subset["latitude"].max()) <= 49.0


def test_descending_latitude_is_subset_not_silently_emptied():
    """ERA5 stores latitude north to south; a naive slice returns nothing."""
    latitudes = np.arange(55.0, 39.9, -1.0)
    ds = _grid(latitudes, np.arange(0.0, 25.1, 1.0))

    naive = ds.sel(latitude=slice(43.0, 49.0))
    assert naive.sizes["latitude"] == 0, "fixture must reproduce the trap"

    subset = subset_spatial(ds)
    assert subset.sizes["latitude"] == 7
    assert float(subset["latitude"].min()) >= 43.0


def test_descending_latitude_keeps_its_direction():
    latitudes = np.arange(55.0, 39.9, -1.0)
    ds = _grid(latitudes, np.arange(0.0, 25.1, 1.0))
    subset = subset_spatial(ds)
    values = subset["latitude"].values
    assert values[0] > values[-1]


def test_both_axis_directions_select_the_same_cells():
    ascending = _grid(np.arange(40.0, 55.1, 1.0), np.arange(0.0, 25.1, 1.0))
    descending = _grid(np.arange(55.0, 39.9, -1.0), np.arange(0.0, 25.1, 1.0))

    up = subset_spatial(ascending)
    down = subset_spatial(descending)

    np.testing.assert_array_equal(
        np.sort(up["latitude"].values), np.sort(down["latitude"].values)
    )


def test_coordinate_aliases_are_resolved():
    ds = _grid(np.arange(40.0, 55.1, 1.0), np.arange(0.0, 25.1, 1.0),
               lat_name="lat", lon_name="lon")
    subset = subset_spatial(ds)
    assert subset.sizes["lat"] == 7


def test_a_non_overlapping_box_is_an_error_not_an_empty_result():
    ds = _grid(np.arange(-40.0, -30.0, 1.0), np.arange(150.0, 160.0, 1.0))
    with pytest.raises(ValueError, match="does not overlap"):
        subset_spatial(ds)


def test_the_subset_records_the_domain_name():
    ds = _grid(np.arange(40.0, 55.1, 1.0), np.arange(0.0, 25.1, 1.0))
    assert subset_spatial(ds).attrs["spatial_subset"] == "Alpine domain"


# --------------------------------------------------------------------------
# Longitude convention
# --------------------------------------------------------------------------


def test_longitudes_are_converted_to_the_signed_range():
    ds = _grid([47.0], np.array([0.0, 90.0, 200.0, 350.0]))
    converted = normalise_longitude(ds)
    np.testing.assert_allclose(
        np.sort(converted["longitude"].values), [-160.0, -10.0, 0.0, 90.0]
    )


def test_already_signed_longitudes_are_left_alone():
    ds = _grid([47.0], np.array([-10.0, 0.0, 10.0]))
    converted = normalise_longitude(ds)
    np.testing.assert_allclose(converted["longitude"].values, [-10.0, 0.0, 10.0])


def test_conversion_to_the_unsigned_range():
    ds = _grid([47.0], np.array([-10.0, 0.0, 10.0]))
    converted = normalise_longitude(ds, to_180=False)
    np.testing.assert_allclose(
        np.sort(converted["longitude"].values), [0.0, 10.0, 350.0]
    )


def test_alpine_longitudes_survive_a_round_trip():
    """The Alpine domain sits east of the meridian and must be unaffected."""
    original = np.arange(4.0, 18.1, 1.0)
    ds = _grid([47.0], original)
    round_tripped = normalise_longitude(normalise_longitude(ds, to_180=False))
    np.testing.assert_allclose(np.sort(round_tripped["longitude"].values), original)


# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------


def test_time_range_is_selected_inclusively():
    ds = _grid([47.0], [11.0])
    subset = subset_temporal(ds, "2021-01-02", "2021-01-03")
    assert subset.sizes["time"] == 2


def test_open_ended_ranges_work():
    ds = _grid([47.0], [11.0])
    assert subset_temporal(ds, "2021-01-03", None).sizes["time"] == 2
    assert subset_temporal(ds, None, "2021-01-02").sizes["time"] == 2


def test_no_bounds_returns_the_input():
    ds = _grid([47.0], [11.0])
    assert subset_temporal(ds).sizes["time"] == 4


def test_a_range_outside_the_record_is_an_error():
    ds = _grid([47.0], [11.0])
    with pytest.raises(ValueError, match="selects no data"):
        subset_temporal(ds, "2030-01-01", "2030-02-01")
