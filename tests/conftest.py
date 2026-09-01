"""Shared synthetic fixtures.

Every fixture is generated in memory. The test suite deliberately requires no
model output, no credentials and no network access, so that it runs in CI and
for anyone who clones the repository without the underlying datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

RNG_SEED = 20260831


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator, so failures are reproducible."""
    return np.random.default_rng(RNG_SEED)


@pytest.fixture
def hourly_time() -> pd.DatetimeIndex:
    """One non-leap year of hourly timestamps."""
    return pd.date_range("2021-01-01", "2021-12-31 23:00", freq="h")


@pytest.fixture
def small_grid() -> dict[str, np.ndarray]:
    """A 3x4 latitude/longitude grid inside the Alpine domain."""
    return {
        "latitude": np.array([46.5, 47.0, 47.5]),
        "longitude": np.array([11.0, 12.0, 13.0, 14.0]),
    }


@pytest.fixture
def constant_temperature(hourly_time, small_grid) -> xr.DataArray:
    """Temperature field held at exactly 10 degC everywhere, in kelvin.

    Constant input makes degree-day totals analytically exact: with a 15 degC
    heating base, every day contributes exactly 5 degC d.
    """
    shape = (len(hourly_time), len(small_grid["latitude"]), len(small_grid["longitude"]))
    values = np.full(shape, 10.0 + 273.15)
    return xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={"time": hourly_time, **small_grid},
        name="t2m",
        attrs={"units": "K", "long_name": "2 metre temperature"},
    )


@pytest.fixture
def wind_speed_field(hourly_time, small_grid, rng) -> xr.DataArray:
    """A plausible positive wind speed field in m/s."""
    shape = (len(hourly_time), len(small_grid["latitude"]), len(small_grid["longitude"]))
    values = np.abs(rng.weibull(2.0, size=shape) * 6.0)
    return xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={"time": hourly_time, **small_grid},
        name="wind_speed_10m",
        attrs={"units": "m/s", "long_name": "10 m wind speed"},
    )
