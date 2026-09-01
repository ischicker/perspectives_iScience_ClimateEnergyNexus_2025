"""Canonical naming across the products evaluated in the study."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from alpinemet.attrs import decode_attribute
from alpinemet.io.naming import (
    CANONICAL_VARIABLES,
    find_coordinate,
    find_variable,
    rename_to_canonical,
    require_variable,
)


def _dataset(names, *, coords=("time", "latitude", "longitude")) -> xr.Dataset:
    time = pd.date_range("2021-01-01", periods=3, freq="h")
    coord_values = {
        coords[0]: time,
        coords[1]: [46.0, 47.0],
        coords[2]: [11.0, 12.0],
    }
    return xr.Dataset(
        {
            name: xr.DataArray(
                np.zeros((3, 2, 2)), dims=coords, coords=coord_values
            )
            for name in names
        }
    )


# --------------------------------------------------------------------------
# Variable lookup
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("t2m", "temperature_2m"),
        ("2t", "temperature_2m"),
        ("\\2t", "temperature_2m"),
        ("tas", "temperature_2m"),
        ("u10", "u_wind_10m"),
        ("10u", "u_wind_10m"),
        ("i10fg", "wind_gust_10m"),
        ("ssrd", "solar_radiation"),
        ("tp", "precipitation"),
        ("sp", "surface_pressure"),
    ],
)
def test_product_specific_aliases_resolve(alias, canonical):
    assert find_variable(_dataset([alias]), canonical) == alias


def test_missing_variable_returns_none():
    assert find_variable(_dataset(["t2m"]), "precipitation") is None


def test_preferred_alias_wins():
    ds = _dataset(["temperature", "t2m"])
    # 't2m' ranks above the generic 'temperature'.
    assert find_variable(ds, "temperature_2m") == "t2m"


def test_extra_aliases_take_precedence():
    ds = _dataset(["t2m", "my_special_temp"])
    assert (
        find_variable(ds, "temperature_2m", extra_aliases=("my_special_temp",))
        == "my_special_temp"
    )


def test_passing_an_alias_as_canonical_is_rejected():
    with pytest.raises(KeyError, match="not a canonical variable"):
        find_variable(_dataset(["t2m"]), "t2m")


def test_require_variable_returns_the_array():
    ds = _dataset(["t2m"])
    assert require_variable(ds, "temperature_2m").shape == (3, 2, 2)


def test_require_variable_lists_what_is_available():
    ds = _dataset(["t2m", "u10"])
    with pytest.raises(ValueError) as excinfo:
        require_variable(ds, "precipitation")
    message = str(excinfo.value)
    assert "precipitation" in message
    assert "t2m" in message and "u10" in message


# --------------------------------------------------------------------------
# Coordinate lookup
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("coords", "canonical", "expected"),
    [
        (("time", "latitude", "longitude"), "latitude", "latitude"),
        (("time", "lat", "lon"), "latitude", "lat"),
        (("time", "lat", "lon"), "longitude", "lon"),
        (("valid_time", "lat", "lon"), "time", "valid_time"),
        (("time", "y", "x"), "latitude", "y"),
    ],
)
def test_coordinate_aliases_resolve(coords, canonical, expected):
    ds = _dataset(["t2m"], coords=coords)
    assert find_coordinate(ds, canonical) == expected


def test_unknown_coordinate_is_rejected():
    with pytest.raises(KeyError, match="not a known coordinate"):
        find_coordinate(_dataset(["t2m"]), "level")


# --------------------------------------------------------------------------
# Wholesale renaming
# --------------------------------------------------------------------------


def test_era5_style_dataset_is_renamed():
    ds = _dataset(["t2m", "u10", "v10", "ssrd", "tp"])
    renamed = rename_to_canonical(ds)

    for name in ("temperature_2m", "u_wind_10m", "v_wind_10m", "solar_radiation",
                 "precipitation"):
        assert name in renamed.data_vars


def test_grib_style_dataset_is_renamed():
    ds = _dataset(["2t", "10u", "10v"], coords=("valid_time", "lat", "lon"))
    renamed = rename_to_canonical(ds)

    assert "temperature_2m" in renamed.data_vars
    assert "u_wind_10m" in renamed.data_vars
    assert "time" in renamed.dims
    assert "latitude" in renamed.coords


def test_renaming_is_idempotent():
    ds = _dataset(["t2m", "u10"])
    once = rename_to_canonical(ds)
    twice = rename_to_canonical(once)
    assert set(once.data_vars) == set(twice.data_vars)


def test_a_canonical_target_is_claimed_only_once():
    """Two aliases for one quantity must not collide into a single name."""
    ds = _dataset(["t2m", "temperature"])
    renamed = rename_to_canonical(ds)

    assert "temperature_2m" in renamed.data_vars
    # The loser keeps its original name rather than being overwritten.
    assert len(renamed.data_vars) == 2


def test_unknown_variables_are_kept_by_default():
    ds = _dataset(["t2m", "some_diagnostic"])
    renamed = rename_to_canonical(ds)
    assert "some_diagnostic" in renamed.data_vars


def test_unknown_variables_can_be_dropped():
    ds = _dataset(["t2m", "some_diagnostic"])
    renamed = rename_to_canonical(ds, keep_unknown=False)
    assert "some_diagnostic" not in renamed.data_vars
    assert "temperature_2m" in renamed.data_vars


def test_applied_renames_are_recorded():
    ds = _dataset(["t2m"])
    renamed = rename_to_canonical(ds)
    assert decode_attribute(renamed.attrs["canonical_renames"])["t2m"] == "temperature_2m"


def test_every_canonical_name_is_its_own_first_alias():
    """Guards the vocabulary against a typo making a name unresolvable."""
    for canonical in CANONICAL_VARIABLES:
        assert find_variable(_dataset([canonical]), canonical) == canonical
