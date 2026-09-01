"""Canonical variable and coordinate names.

The datasets evaluated here disagree on almost every name. ERA5 and ERA5-Land
use CF-ish long names or ECMWF short names (``t2m``, ``ssrd``); GRIB-derived
products use bare shortNames that start with a digit (``2t``, ``10u``), which
cfgrib sometimes escapes as ``\\2t``; CERRA and ARA add their own variants. This
module maps all of them onto one vocabulary so that every downstream indicator
can be written once.

Canonical names are lower-case, spelled out, and carry the measurement height
where it matters:

===========================  ===================================
Canonical name               Quantity
===========================  ===================================
``temperature_2m``           2 m air temperature
``u_wind_10m``               10 m zonal wind
``v_wind_10m``               10 m meridional wind
``u_wind_100m``              100 m zonal wind
``v_wind_100m``              100 m meridional wind
``wind_speed_10m``           10 m scalar wind speed
``wind_speed_100m``          100 m scalar wind speed
``wind_gust_10m``            10 m instantaneous gust
``surface_pressure``         Surface pressure
``solar_radiation``          Surface downwelling shortwave
``precipitation``            Total precipitation
``snow_depth``               Snow depth
===========================  ===================================
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

import xarray as xr

from alpinemet.attrs import as_attribute

__all__ = [
    "CANONICAL_VARIABLES",
    "COORDINATE_ALIASES",
    "VARIABLE_ALIASES",
    "find_coordinate",
    "find_variable",
    "rename_to_canonical",
    "require_variable",
]

#: Aliases for each canonical variable, in decreasing order of preference.
VARIABLE_ALIASES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "temperature_2m": (
            "temperature_2m", "t2m", "2t", "\\2t", "2m_temperature", "tas",
            "temperature", "temp", "air_temperature",
        ),
        "u_wind_10m": (
            "u_wind_10m", "u10", "10u", "\\10u", "10m_u_component_of_wind",
            "eastward_wind", "uas",
        ),
        "v_wind_10m": (
            "v_wind_10m", "v10", "10v", "\\10v", "10m_v_component_of_wind",
            "northward_wind", "vas",
        ),
        "u_wind_100m": (
            "u_wind_100m", "u100", "100u", "\\100u", "100m_u_component_of_wind",
        ),
        "v_wind_100m": (
            "v_wind_100m", "v100", "100v", "\\100v", "100m_v_component_of_wind",
        ),
        "wind_speed_10m": (
            "wind_speed_10m", "ws10", "si10", "wind_speed", "wspd", "sfcWind",
        ),
        "wind_speed_100m": ("wind_speed_100m", "ws100", "si100"),
        "wind_gust_10m": (
            "wind_gust_10m", "i10fg", "fg10", "10fg", "\\10fg", "wind_gust", "gust",
        ),
        "surface_pressure": ("surface_pressure", "sp", "ps", "psfc"),
        "solar_radiation": (
            "solar_radiation", "surface_solar_radiation_downwards", "ssrd",
            "rsds", "ghi", "surface_net_solar_radiation", "ssr", "avg_sdswrf",
        ),
        "precipitation": (
            "precipitation", "total_precipitation", "tp", "pr", "precip", "rr",
        ),
        "snow_depth": ("snow_depth", "sd", "snowdepth", "snd"),
    }
)

#: The canonical variable vocabulary.
CANONICAL_VARIABLES: tuple[str, ...] = tuple(VARIABLE_ALIASES)

#: Aliases for spatial and temporal coordinates.
COORDINATE_ALIASES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "time": ("time", "valid_time", "forecast_time", "t"),
        "latitude": ("latitude", "lat", "y", "nav_lat"),
        "longitude": ("longitude", "lon", "x", "nav_lon"),
    }
)


def find_variable(
    dataset: xr.Dataset,
    canonical: str,
    *,
    extra_aliases: Iterable[str] = (),
) -> str | None:
    """Return the name under which ``canonical`` appears in ``dataset``.

    Parameters
    ----------
    dataset
        Dataset to inspect.
    canonical
        Canonical variable name, one of :data:`CANONICAL_VARIABLES`.
    extra_aliases
        Additional names to try first, for datasets with unusual conventions.

    Returns
    -------
    str or None
        The matching variable name, or ``None`` if the dataset has none.

    Raises
    ------
    KeyError
        If ``canonical`` is not part of the vocabulary. Passing an alias here
        instead of a canonical name is the usual cause.
    """
    if canonical not in VARIABLE_ALIASES:
        raise KeyError(
            f"{canonical!r} is not a canonical variable; expected one of "
            f"{list(CANONICAL_VARIABLES)}"
        )

    for name in (*extra_aliases, *VARIABLE_ALIASES[canonical]):
        if name in dataset.data_vars:
            return name
    return None


def require_variable(
    dataset: xr.Dataset,
    canonical: str,
    *,
    extra_aliases: Iterable[str] = (),
) -> xr.DataArray:
    """Return the array for ``canonical``, or explain what is missing.

    Parameters
    ----------
    dataset
        Dataset to read from.
    canonical
        Canonical variable name.
    extra_aliases
        Additional names to try first.

    Returns
    -------
    xarray.DataArray
        The matching variable.

    Raises
    ------
    ValueError
        If the variable is absent. The message lists both the aliases tried and
        the variables actually present, which is what one needs to fix a
        dataset-specific naming gap.
    """
    name = find_variable(dataset, canonical, extra_aliases=extra_aliases)
    if name is None:
        tried = [*extra_aliases, *VARIABLE_ALIASES[canonical]]
        raise ValueError(
            f"Dataset has no {canonical!r} variable. Tried {tried}; "
            f"available: {sorted(dataset.data_vars)}"
        )
    return dataset[name]


def find_coordinate(
    dataset: xr.Dataset | xr.DataArray,
    canonical: str,
) -> str | None:
    """Return the name under which a coordinate appears.

    Parameters
    ----------
    dataset
        Object to inspect.
    canonical
        One of ``"time"``, ``"latitude"`` or ``"longitude"``.

    Returns
    -------
    str or None
        The matching coordinate or dimension name, or ``None``.

    Raises
    ------
    KeyError
        If ``canonical`` is not a known coordinate.
    """
    if canonical not in COORDINATE_ALIASES:
        raise KeyError(
            f"{canonical!r} is not a known coordinate; expected one of "
            f"{list(COORDINATE_ALIASES)}"
        )

    for name in COORDINATE_ALIASES[canonical]:
        if name in dataset.coords or name in dataset.dims:
            return name
    return None


def rename_to_canonical(
    dataset: xr.Dataset,
    *,
    extra_aliases: Mapping[str, Iterable[str]] | None = None,
    keep_unknown: bool = True,
) -> xr.Dataset:
    """Rename a dataset's variables and coordinates to the canonical vocabulary.

    Only names that actually differ are renamed, and each canonical target is
    claimed at most once -- if a dataset carries both ``t2m`` and
    ``temperature``, the higher-preference alias wins and the other is left
    under its original name rather than silently overwriting.

    Parameters
    ----------
    dataset
        Dataset to rename.
    extra_aliases
        Per-canonical extra names to try first, e.g.
        ``{"solar_radiation": ("avg_sdswrf",)}``.
    keep_unknown
        Keep variables that match no canonical name. When ``False`` they are
        dropped, which is useful for producing a lean standardised dataset.

    Returns
    -------
    xarray.Dataset
        Dataset with canonical names, recording the applied mapping in
        ``attrs["canonical_renames"]``.
    """
    extra_aliases = extra_aliases or {}
    renames: dict[str, str] = {}

    for canonical in CANONICAL_VARIABLES:
        found = find_variable(
            dataset, canonical, extra_aliases=extra_aliases.get(canonical, ())
        )
        if found is not None and found not in renames and found != canonical:
            renames[found] = canonical

    for canonical in COORDINATE_ALIASES:
        found = find_coordinate(dataset, canonical)
        if found is not None and found not in renames and found != canonical:
            renames[found] = canonical

    renamed = dataset.rename(renames) if renames else dataset.copy()

    if not keep_unknown:
        keep = [name for name in renamed.data_vars if name in CANONICAL_VARIABLES]
        renamed = renamed[keep]

    renamed.attrs = {**dataset.attrs, "canonical_renames": as_attribute(dict(renames))}
    return renamed
