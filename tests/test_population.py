"""Population regridding must conserve totals *and* preserve where people are."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from alpinemet.indicators.population import (
    aggregate_population_to_grid,
    load_population_raster,
    person_days,
    population_weighted_mean,
)


@pytest.fixture
def fine_raster_with_a_city() -> xr.DataArray:
    """A 400x400 fine grid: one compact city plus thin rural scatter.

    The city sits in the upper-right quadrant, so a 4x4 aggregation must place
    almost everyone in a single target cell.
    """
    rng = np.random.default_rng(0)
    values = rng.uniform(0.0, 1.0, (400, 400))
    values[110:130, 110:130] += 500.0  # city block

    lats = np.linspace(46.0, 48.0, 400)
    lons = np.linspace(10.0, 12.0, 400)
    return xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={"latitude": lats, "longitude": lons},
        name="population",
    )


@pytest.fixture
def coarse_target(fine_raster_with_a_city) -> xr.DataArray:
    """A 4x4 target grid spanning the same domain."""
    lats = np.linspace(46.25, 47.75, 4)
    lons = np.linspace(10.25, 11.75, 4)
    return xr.DataArray(
        np.zeros((4, 4)),
        dims=("latitude", "longitude"),
        coords={"latitude": lats, "longitude": lons},
    )


# --------------------------------------------------------------------------
# Conservative aggregation
# --------------------------------------------------------------------------


def test_aggregation_conserves_total_population(fine_raster_with_a_city, coarse_target):
    aggregated = aggregate_population_to_grid(fine_raster_with_a_city, coarse_target)
    assert float(aggregated.sum()) == pytest.approx(float(fine_raster_with_a_city.sum()))


def test_aggregation_puts_the_city_in_one_cell(fine_raster_with_a_city, coarse_target):
    aggregated = aggregate_population_to_grid(fine_raster_with_a_city, coarse_target)
    share = float(aggregated.max()) / float(aggregated.sum())
    assert share > 0.5, "the city block must dominate a single target cell"


def test_nearest_normalised_conserves_the_total_but_not_the_pattern(
    fine_raster_with_a_city, coarse_target
):
    """Documents exactly why the legacy method must not be used.

    Both methods return the same domain total. Only the aggregation puts the
    population where it belongs; nearest-neighbour sampling scatters it.
    """
    correct = aggregate_population_to_grid(fine_raster_with_a_city, coarse_target)
    legacy = aggregate_population_to_grid(
        fine_raster_with_a_city, coarse_target, method="nearest_normalised"
    )

    assert float(legacy.sum()) == pytest.approx(float(correct.sum()))

    city = np.unravel_index(int(np.argmax(correct.values)), correct.shape)
    correct_share = float(correct.values[city]) / float(correct.sum())
    legacy_share = float(legacy.values[city]) / float(legacy.sum())

    assert correct_share > 0.5
    assert legacy_share < 0.25, "legacy method loses the city"


def test_legacy_method_is_flagged_in_metadata(fine_raster_with_a_city, coarse_target):
    legacy = aggregate_population_to_grid(
        fine_raster_with_a_city, coarse_target, method="nearest_normalised"
    )
    assert "warning" in legacy.attrs
    assert legacy.attrs["regrid_method"] == "nearest_normalised"


def test_aggregation_records_totals(fine_raster_with_a_city, coarse_target):
    aggregated = aggregate_population_to_grid(fine_raster_with_a_city, coarse_target)
    assert aggregated.attrs["regrid_method"] == "sum"
    assert aggregated.attrs["source_total"] == pytest.approx(
        aggregated.attrs["regridded_total"]
    )


def test_descending_target_latitudes_are_handled(fine_raster_with_a_city):
    """ERA5 stores latitude in descending order; the result must follow suit."""
    ascending = xr.DataArray(
        np.zeros((4, 4)),
        dims=("latitude", "longitude"),
        coords={
            "latitude": np.linspace(46.25, 47.75, 4),
            "longitude": np.linspace(10.25, 11.75, 4),
        },
    )
    descending = xr.DataArray(
        np.zeros((4, 4)),
        dims=("latitude", "longitude"),
        coords={
            "latitude": np.linspace(46.25, 47.75, 4)[::-1],
            "longitude": np.linspace(10.25, 11.75, 4),
        },
    )

    up = aggregate_population_to_grid(fine_raster_with_a_city, ascending)
    down = aggregate_population_to_grid(fine_raster_with_a_city, descending)

    np.testing.assert_allclose(up.values, np.flip(down.values, axis=0))
    assert float(down.sum()) == pytest.approx(float(up.sum()))


def test_unknown_method_is_rejected(fine_raster_with_a_city, coarse_target):
    with pytest.raises(ValueError, match="Unknown regrid method"):
        aggregate_population_to_grid(
            fine_raster_with_a_city, coarse_target, method="bilinear"
        )


def test_missing_coordinates_are_reported(coarse_target):
    bare = xr.DataArray(np.zeros((4, 4)), dims=("y", "x"))
    with pytest.raises(ValueError, match="no 'latitude' coordinate"):
        aggregate_population_to_grid(bare, coarse_target)


# --------------------------------------------------------------------------
# Weighted statistics
# --------------------------------------------------------------------------


def test_weighted_mean_follows_the_population():
    field = xr.DataArray(
        np.array([[0.0, 10.0]]),
        dims=("latitude", "longitude"),
        coords={"latitude": [47.0], "longitude": [11.0, 12.0]},
    )
    population = xr.DataArray(
        np.array([[1.0, 9.0]]),
        dims=("latitude", "longitude"),
        coords={"latitude": [47.0], "longitude": [11.0, 12.0]},
    )
    assert float(population_weighted_mean(field, population)) == pytest.approx(9.0)


def test_weighted_mean_equals_plain_mean_for_uniform_population():
    field = xr.DataArray(
        np.array([[1.0, 3.0], [5.0, 7.0]]),
        dims=("latitude", "longitude"),
        coords={"latitude": [46.0, 47.0], "longitude": [11.0, 12.0]},
    )
    population = xr.ones_like(field)
    assert float(population_weighted_mean(field, population)) == pytest.approx(4.0)


def test_weighted_mean_preserves_the_time_dimension():
    time = pd.date_range("2021-01-01", periods=3, freq="D")
    field = xr.DataArray(
        np.arange(12.0).reshape(3, 2, 2),
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": [46.0, 47.0], "longitude": [11.0, 12.0]},
    )
    population = xr.DataArray(
        np.ones((2, 2)),
        dims=("latitude", "longitude"),
        coords={"latitude": [46.0, 47.0], "longitude": [11.0, 12.0]},
    )
    weighted = population_weighted_mean(field, population)
    assert weighted.dims == ("time",)
    assert weighted.sizes["time"] == 3


def test_zero_population_is_rejected():
    field = xr.DataArray(
        np.ones((1, 1)),
        dims=("latitude", "longitude"),
        coords={"latitude": [47.0], "longitude": [11.0]},
    )
    with pytest.raises(ValueError, match="Total population is zero"):
        population_weighted_mean(field, xr.zeros_like(field))


def test_person_days_multiplies_exposure_by_population():
    time = pd.date_range("2021-07-01", periods=4, freq="D")
    condition = xr.DataArray(
        np.array([[[True]], [[True]], [[False]], [[False]]]),
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": [47.0], "longitude": [11.0]},
    )
    population = xr.DataArray(
        np.array([[1000.0]]),
        dims=("latitude", "longitude"),
        coords={"latitude": [47.0], "longitude": [11.0]},
    )
    # Two days of exposure for 1000 people.
    assert float(person_days(condition, population)) == pytest.approx(2000.0)


def test_person_days_scales_with_the_time_step():
    time = pd.date_range("2021-07-01", periods=48, freq="h")
    condition = xr.DataArray(
        np.ones((48, 1, 1), dtype=bool),
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": [47.0], "longitude": [11.0]},
    )
    population = xr.DataArray(
        np.array([[100.0]]),
        dims=("latitude", "longitude"),
        coords={"latitude": [47.0], "longitude": [11.0]},
    )
    # 48 hourly steps = 2 days for 100 people.
    assert float(person_days(condition, population)) == pytest.approx(200.0)


# --------------------------------------------------------------------------
# Raster loading
# --------------------------------------------------------------------------


def test_raster_round_trips_through_geotiff(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    values = np.arange(12, dtype="float32").reshape(3, 4)
    path = tmp_path / "pop.tif"
    # North-up raster: origin at the top-left, latitude descending.
    transform = from_origin(10.0, 48.0, 0.5, 0.5)
    with rasterio.open(
        path, "w", driver="GTiff", height=3, width=4, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(values, 1)

    raster = load_population_raster(path)

    assert float(raster.sum()) == pytest.approx(values.sum())
    # Loader must return ascending latitude, so the rows are flipped.
    assert raster["latitude"].values[0] < raster["latitude"].values[-1]
    np.testing.assert_allclose(raster.values, np.flip(values, axis=0))


def test_non_wgs84_raster_is_rejected(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path = tmp_path / "pop_3035.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=2, width=2, count=1,
        dtype="float32", crs="EPSG:3035", transform=from_origin(0, 0, 100, 100),
    ) as dst:
        dst.write(np.zeros((2, 2), dtype="float32"), 1)

    with pytest.raises(ValueError, match="must be in EPSG:4326"):
        load_population_raster(path)
