"""The standardisation pipeline, end to end on synthetic products."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from alpinemet.attrs import decode_attribute
from alpinemet.io.accumulation import AccumulationKind
from alpinemet.io.datasets import DATASETS, get_dataset_spec
from alpinemet.io.loaders import open_product, standardise
from alpinemet.io.subset import ALPINE_DOMAIN

DIURNAL_FLUX = np.clip(800.0 * np.sin((np.arange(24) - 6) / 12 * np.pi), 0.0, None)


def _era5_like(days: int = 2, *, descending_latitude: bool = True) -> xr.Dataset:
    """An ERA5-shaped product: ECMWF short names, descending latitude, K, J/m2."""
    time = pd.date_range("2021-06-21", periods=24 * days, freq="h")
    latitudes = np.arange(52.0, 39.9, -1.0) if descending_latitude else np.arange(40.0, 52.1, 1.0)
    longitudes = np.arange(0.0, 25.1, 1.0)
    shape = (time.size, latitudes.size, longitudes.size)
    dims = ("time", "latitude", "longitude")
    coords = {"time": time, "latitude": latitudes, "longitude": longitudes}

    flux = np.tile(DIURNAL_FLUX, days)[:, None, None] * np.ones(shape[1:])

    return xr.Dataset(
        {
            "t2m": xr.DataArray(np.full(shape, 288.15), dims=dims, coords=coords,
                                attrs={"units": "K"}),
            "u10": xr.DataArray(np.full(shape, 3.0), dims=dims, coords=coords),
            "v10": xr.DataArray(np.full(shape, 4.0), dims=dims, coords=coords),
            "ssrd": xr.DataArray(flux * 3600.0, dims=dims, coords=coords,
                                 attrs={"units": "J m**-2"}),
        }
    )


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_every_evaluated_product_is_registered():
    for key in ("era5", "era5_land", "cerra", "ara", "climate_dt", "extremes_dt",
                "ifs", "aifs"):
        assert key in DATASETS


def test_resolutions_match_the_paper():
    assert get_dataset_spec("era5").resolution_km == 31.0
    assert get_dataset_spec("era5_land").resolution_km == 9.0
    assert get_dataset_spec("cerra").resolution_km == 5.5
    assert get_dataset_spec("ara").resolution_km == 2.5
    assert get_dataset_spec("climate_dt").resolution_km == 4.4


def test_lookup_tolerates_hyphens_and_case():
    assert get_dataset_spec("ERA5-Land").key == "era5_land"


def test_unknown_key_lists_the_alternatives():
    with pytest.raises(KeyError) as excinfo:
        get_dataset_spec("era6")
    assert "era5" in str(excinfo.value)


def test_the_two_accumulation_conventions_are_distinguished():
    """ERA5 accumulates per step; ERA5-Land accumulates from the daily reset."""
    assert get_dataset_spec("era5").accumulation_kind("solar_radiation") is (
        AccumulationKind.PERIOD
    )
    assert get_dataset_spec("era5_land").accumulation_kind("solar_radiation") is (
        AccumulationKind.RUNNING
    )


def test_unlisted_variables_default_to_instantaneous():
    assert get_dataset_spec("era5").accumulation_kind("temperature_2m") is (
        AccumulationKind.INSTANTANEOUS
    )


def test_gust_availability_is_declared():
    assert get_dataset_spec("ara").has_native_gust is True
    assert get_dataset_spec("era5").has_native_gust is False
    assert get_dataset_spec("aifs").has_native_gust is False


# --------------------------------------------------------------------------
# Standardisation pipeline
# --------------------------------------------------------------------------


def test_variables_are_renamed_to_canonical_names():
    result = standardise(_era5_like(), "era5")
    for name in ("temperature_2m", "u_wind_10m", "v_wind_10m", "solar_radiation"):
        assert name in result.data_vars


def test_radiation_is_converted_to_watts():
    result = standardise(_era5_like(), "era5")
    solar = result["solar_radiation"]
    assert float(solar.max()) == pytest.approx(800.0)
    assert solar.attrs["units"] == "W m-2"
    assert solar.attrs["accumulation_converted"] == "true"


def test_wind_speed_is_derived():
    result = standardise(_era5_like(), "era5")
    assert float(result["wind_speed_10m"].max()) == pytest.approx(5.0)


def test_descending_latitude_is_subset_correctly():
    result = standardise(_era5_like(descending_latitude=True), "era5", domain=ALPINE_DOMAIN)
    assert result.sizes["latitude"] > 0
    assert float(result["latitude"].min()) >= ALPINE_DOMAIN.lat_min
    assert float(result["latitude"].max()) <= ALPINE_DOMAIN.lat_max


def test_both_latitude_directions_give_the_same_extent():
    down = standardise(_era5_like(descending_latitude=True), "era5", domain=ALPINE_DOMAIN)
    up = standardise(_era5_like(descending_latitude=False), "era5", domain=ALPINE_DOMAIN)
    assert down.sizes["latitude"] == up.sizes["latitude"]


def test_time_subsetting_is_applied():
    result = standardise(_era5_like(days=3), "era5", start="2021-06-22", end="2021-06-22")
    assert result.sizes["time"] == 24


def test_conversion_happens_before_subsetting():
    """A running accumulator differenced after subsetting would lose an edge step.

    The full-record and subset conversions must agree on the overlapping hours.
    """
    full = standardise(_era5_like(days=3), "era5_land")
    subset = standardise(_era5_like(days=3), "era5_land", start="2021-06-22", end="2021-06-22")

    overlap = full["solar_radiation"].sel(time=subset["time"])
    np.testing.assert_allclose(
        overlap.values, subset["solar_radiation"].values, atol=1e-9
    )


def test_duplicate_timestamps_are_removed():
    ds = _era5_like(days=1)
    doubled = xr.concat([ds, ds.isel(time=slice(0, 3))], dim="time")
    result = standardise(doubled, "era5")
    assert result.sizes["time"] == 24
    assert result.attrs["duplicate_times_removed"] == 3


def test_product_metadata_is_recorded():
    result = standardise(_era5_like(), "era5")
    assert result.attrs["dataset_key"] == "era5"
    assert result.attrs["resolution_km"] == 31.0
    assert decode_attribute(result.attrs["has_native_gust"]) is False
    assert "descending" in result.attrs["dataset_notes"]


def test_quality_control_runs_by_default():
    result = standardise(_era5_like(), "era5")
    assert result.attrs["quality_control_action"] == "mask"
    # 15 degC and 5 m/s are entirely plausible, so nothing should be flagged.
    assert decode_attribute(result.attrs["quality_control"]) == {}


def test_quality_control_can_be_skipped():
    result = standardise(_era5_like(), "era5", quality_control=None)
    assert "quality_control_action" not in result.attrs


def test_unknown_variables_can_be_dropped():
    ds = _era5_like()
    ds["some_diagnostic"] = ds["t2m"]
    lean = standardise(ds, "era5", keep_unknown=False)
    assert "some_diagnostic" not in lean.data_vars


def test_era5_land_running_accumulation_is_handled():
    ds = _era5_like(days=2)
    ds["ssrd"] = ds["ssrd"].copy(data=np.cumsum(ds["ssrd"].values, axis=0))
    result = standardise(ds, "era5_land")
    # Differencing recovers the per-hour flux; the first step has no predecessor.
    assert float(result["solar_radiation"].max()) == pytest.approx(800.0, rel=1e-6)
    assert float(result["solar_radiation"].min()) >= 0.0


# --------------------------------------------------------------------------
# File opening
# --------------------------------------------------------------------------


def test_open_product_reads_and_standardises(tmp_path):
    path = tmp_path / "era5_2021.nc"
    _era5_like().to_netcdf(path)

    result = open_product(path, "era5", domain=ALPINE_DOMAIN)

    assert "temperature_2m" in result.data_vars
    assert result.attrs["dataset_key"] == "era5"
    assert float(result["solar_radiation"].max()) == pytest.approx(800.0)


def test_open_product_combines_multiple_files(tmp_path):
    first = _era5_like(days=1)
    second = first.assign_coords(time=first["time"] + pd.Timedelta(days=1))
    first.to_netcdf(tmp_path / "part_a.nc")
    second.to_netcdf(tmp_path / "part_b.nc")

    result = open_product(
        [tmp_path / "part_a.nc", tmp_path / "part_b.nc"], "era5"
    )
    assert result.sizes["time"] == 48


def test_open_product_expands_a_glob(tmp_path):
    _era5_like(days=1).to_netcdf(tmp_path / "era5_a.nc")
    result = open_product(str(tmp_path / "era5_*.nc"), "era5")
    assert result.sizes["time"] == 24


def test_a_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="No such file"):
        open_product(tmp_path / "absent.nc", "era5")


def test_a_glob_matching_nothing_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="matched no files"):
        open_product(str(tmp_path / "nothing_*.nc"), "era5")


def test_ara_maps_its_bare_t_to_2m_temperature():
    """'t' is ARA's 2 m temperature: the only one in its surface product.

    It carries a pronounced radiation-dependent diurnal bias in the 2020 Alpine
    subset (see docs/differences.md), but that is a data-quality question, not a
    reason to refuse to read the field.
    """
    spec = get_dataset_spec("ara")
    assert "t" in spec.extra_aliases["temperature_2m"]

    time = pd.date_range("2020-07-01", periods=6, freq="h")
    subset = xr.Dataset(
        {
            "t": xr.DataArray(
                np.full(6, 300.0), dims="time", coords={"time": time},
                attrs={"units": "K", "standard_name": "air_temperature"},
            ),
            "u10m": xr.DataArray(np.full(6, 3.0), dims="time", coords={"time": time}),
            "v10m": xr.DataArray(np.full(6, 4.0), dims="time", coords={"time": time}),
        }
    )

    standardised = standardise(subset, "ara", quality_control=None)
    assert "temperature_2m" in standardised.data_vars
    # Kelvin converted on the way through.
    assert float(standardised["temperature_2m"].max()) == pytest.approx(26.85)


def test_ara_wind_and_gust_aliases_still_resolve():
    time = pd.date_range("2020-07-01", periods=6, freq="h")
    subset = xr.Dataset(
        {
            name: xr.DataArray(np.full(6, 5.0), dims="time", coords={"time": time})
            for name in ("u10m", "v10m", "u100m", "v100m", "gust10m")
        }
    )
    standardised = standardise(subset, "ara", quality_control=None)

    for name in ("u_wind_10m", "v_wind_10m", "wind_speed_10m", "wind_speed_100m",
                 "wind_gust_10m"):
        assert name in standardised.data_vars, name
