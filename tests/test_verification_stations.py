"""Station matching, extraction and elevation-resolved verification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from alpinemet.verification.stations import (
    DEFAULT_ELEVATION_BANDS,
    ENVIRONMENTAL_LAPSE_RATE,
    elevation_band_metrics,
    extract_at_stations,
    lapse_rate_adjustment,
    match_stations_to_grid,
    station_metrics,
)


@pytest.fixture
def grid() -> xr.Dataset:
    """A 2.5 km-ish grid over the eastern Alps with an orography field."""
    time = pd.date_range("2020-01-01", periods=200, freq="h")
    lats = np.arange(46.0, 48.01, 0.025)
    lons = np.arange(10.0, 14.01, 0.025)
    shape = (time.size, lats.size, lons.size)
    dims = ("time", "latitude", "longitude")
    coords = {"time": time, "latitude": lats, "longitude": lons}

    return xr.Dataset(
        {
            "temperature_2m": xr.DataArray(np.zeros(shape), dims=dims, coords=coords),
            "orography": xr.DataArray(
                np.full((lats.size, lons.size), 1200.0),
                dims=("latitude", "longitude"),
                coords={"latitude": lats, "longitude": lons},
            ),
        }
    )


@pytest.fixture
def stations() -> pd.DataFrame:
    """Three of the Austrian sites used in the paper's supplement."""
    return pd.DataFrame(
        {
            "synnr": [11399, 11200, 11111],
            "name": ["TAMSWEG", "KALS", "TANNHEIM"],
            "lon": [13.808333, 12.646389, 10.505834],
            "lat": [47.133057, 47.004723, 47.500278],
            "alt": [1025.0, 1352.0, 1100.0],
        }
    )


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def test_every_station_gets_a_grid_cell(grid, stations):
    matched = match_stations_to_grid(grid, stations)
    assert len(matched) == len(stations)
    assert matched["lat_index"].notna().all()


def test_the_matched_cell_is_close_to_the_station(grid, stations):
    matched = match_stations_to_grid(grid, stations)
    # A 0.025 degree grid puts the nearest centre within about 2 km.
    assert float(matched["distance_km"].max()) < 2.5


def test_matched_coordinates_come_from_the_grid(grid, stations):
    matched = match_stations_to_grid(grid, stations)
    grid_lats = set(np.round(grid["latitude"].values, 6))
    assert set(np.round(matched["grid_lat"].to_numpy(), 6)) <= grid_lats


def test_elevation_difference_is_computed_when_orography_is_available(grid, stations):
    matched = match_stations_to_grid(grid, stations, elevation_variable="orography")
    assert "elevation_difference" in matched.columns
    # Flat 1200 m orography against the station altitudes.
    np.testing.assert_allclose(
        matched["elevation_difference"].to_numpy(), [175.0, -152.0, 100.0]
    )


def test_elevation_difference_is_omitted_without_orography(grid, stations):
    matched = match_stations_to_grid(grid, stations)
    assert "elevation_difference" not in matched.columns


def test_missing_coordinate_columns_are_reported(grid, stations):
    with pytest.raises(ValueError, match="no 'latitude' column"):
        match_stations_to_grid(grid, stations, lat_column="latitude")


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def test_extraction_returns_one_series_per_station(grid, stations):
    matched = match_stations_to_grid(grid, stations)
    frame = extract_at_stations(grid["temperature_2m"], matched)

    assert set(frame.columns) == {"station_id", "time", "value"}
    assert frame["station_id"].nunique() == 3
    assert len(frame) == 3 * grid.sizes["time"]


def test_extraction_is_pointwise_not_an_outer_product(grid, stations):
    """Three stations must give three series, not nine."""
    matched = match_stations_to_grid(grid, stations)
    frame = extract_at_stations(grid["temperature_2m"], matched)
    assert len(frame) == 3 * grid.sizes["time"]


def test_extraction_picks_up_the_right_cell(grid, stations):
    field = grid["temperature_2m"].copy()
    values = field.values.copy()
    matched = match_stations_to_grid(grid, stations)
    # Stamp a unique value into the first station's cell.
    first = matched.iloc[0]
    values[:, int(first["lat_index"]), int(first["lon_index"])] = 42.0
    field = field.copy(data=values)

    frame = extract_at_stations(field, matched)
    station_values = frame[frame["station_id"] == first["synnr"]]["value"]
    np.testing.assert_allclose(station_values.to_numpy(), 42.0)


def test_unmatched_stations_are_rejected(grid, stations):
    with pytest.raises(ValueError, match="call match_stations_to_grid first"):
        extract_at_stations(grid["temperature_2m"], stations)


# --------------------------------------------------------------------------
# Lapse rate adjustment
# --------------------------------------------------------------------------


def test_a_model_cell_above_its_station_is_warmed():
    adjusted = lapse_rate_adjustment([0.0], [200.0])
    assert adjusted[0] == pytest.approx(200.0 * ENVIRONMENTAL_LAPSE_RATE)


def test_a_model_cell_below_its_station_is_cooled():
    adjusted = lapse_rate_adjustment([0.0], [-200.0])
    assert adjusted[0] < 0.0


def test_no_elevation_difference_leaves_the_value_alone():
    np.testing.assert_allclose(lapse_rate_adjustment([5.0, 7.0], [0.0, 0.0]), [5.0, 7.0])


def test_the_adjustment_reduces_bias_when_the_offset_is_the_cause():
    """An elevation-driven cold bias should largely disappear after correction."""
    from alpinemet.verification.metrics import bias

    observed = np.full(50, 5.0)
    elevation_difference = np.full(50, 300.0)  # model cell 300 m too high
    modelled = observed - ENVIRONMENTAL_LAPSE_RATE * elevation_difference

    raw = abs(bias(observed, modelled))
    corrected = abs(bias(observed, lapse_rate_adjustment(modelled, elevation_difference)))
    assert corrected < raw * 0.01


# --------------------------------------------------------------------------
# Per-station and per-band statistics
# --------------------------------------------------------------------------


def _paired_frame(station_ids, n=200, offsets=None) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    offsets = offsets or dict.fromkeys(station_ids, 0.0)
    rows = []
    for station_id in station_ids:
        observed = rng.normal(5.0, 4.0, n)
        rows.append(
            pd.DataFrame(
                {
                    "station_id": station_id,
                    "observed": observed,
                    "modelled": observed + offsets[station_id],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_metrics_are_computed_per_station():
    paired = _paired_frame([1, 2], offsets={1: 1.0, 2: -2.0})
    result = station_metrics(paired)

    assert len(result) == 2
    by_station = result.set_index("station_id")["bias"]
    assert by_station[1] == pytest.approx(1.0)
    assert by_station[2] == pytest.approx(-2.0)


def test_stations_with_too_few_pairs_are_excluded():
    paired = pd.concat(
        [_paired_frame([1], n=200), _paired_frame([2], n=10)], ignore_index=True
    )
    result = station_metrics(paired, minimum_pairs=100)

    assert list(result["station_id"]) == [1]
    assert result.attrs["stations_excluded"] == 1


def test_station_metadata_is_merged_in(stations):
    paired = _paired_frame([11399, 11200])
    result = station_metrics(paired, station_metadata=stations)
    assert "alt" in result.columns
    assert "name" in result.columns


# --------------------------------------------------------------------------
# Elevation bands
# --------------------------------------------------------------------------


def test_bands_split_the_station_sample():
    statistics = pd.DataFrame(
        {
            "station_id": [1, 2, 3, 4],
            "alt": [300.0, 1200.0, 1800.0, 2600.0],
            "bias": [0.5, 1.0, 2.0, 3.0],
            "rmse": [1.0, 1.5, 2.5, 3.5],
            "correlation": [0.95, 0.9, 0.85, 0.8],
        }
    )
    bands = elevation_band_metrics(statistics)

    assert list(bands["n_stations"]) == [1, 0, 1, 1, 1]
    lowest = bands.loc[bands["elevation_band"] == "0-500 m", "bias"].iloc[0]
    assert float(lowest) == pytest.approx(0.5)


def test_empty_bands_are_reported_not_dropped():
    """A high-elevation band with no stations is a finding about coverage."""
    statistics = pd.DataFrame(
        {"station_id": [1], "alt": [400.0], "bias": [0.5], "rmse": [1.0],
         "correlation": [0.9]}
    )
    bands = elevation_band_metrics(statistics)

    assert len(bands) == len(DEFAULT_ELEVATION_BANDS) - 1
    high = bands[bands["elevation_band"] == "2500-4000 m"]
    assert int(high["n_stations"].iloc[0]) == 0
    assert np.isnan(float(high["bias"].iloc[0]))


def test_the_band_edges_bracket_the_station_density_break():
    """1,500 m is where roughly 90 % of Alpine stations lie below."""
    assert 1500.0 in DEFAULT_ELEVATION_BANDS


def test_missing_elevation_column_is_reported():
    statistics = pd.DataFrame({"station_id": [1], "bias": [0.5]})
    with pytest.raises(ValueError, match="no 'alt' column"):
        elevation_band_metrics(statistics)
