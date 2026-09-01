"""Unit harmonisation helpers.

The datasets evaluated in this study arrive in mixed conventions: ERA5 and ARA
report 2 m temperature in kelvin, some derived products already in degrees
Celsius; radiation is accumulated in some products and instantaneous in others.
Every downstream indicator assumes the harmonised conventions documented here:

===================  ==========
Quantity             Unit
===================  ==========
temperature          degC
wind speed           m/s
irradiance           W/m2
precipitation        mm
===================  ==========
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = [
    "KELVIN_OFFSET",
    "MILLIMETRES_PER_METRE",
    "to_celsius",
    "to_millimetres",
    "wind_speed_from_components",
]

KELVIN_OFFSET = 273.15

MILLIMETRES_PER_METRE = 1000.0

_METRE_UNITS = frozenset({"m", "metre", "meter", "metres", "meters", "mofwaterequivalent"})
_MILLIMETRE_UNITS = frozenset({"mm", "millimetre", "millimeter", "kgm-2", "kgm**-2", "kgm^-2"})

# ECMWF reports precipitation in metres of water equivalent; an hourly total
# above this is not physically plausible anywhere on Earth, so a sample
# exceeding it cannot be in metres.
_METRE_PLAUSIBILITY_LIMIT = 0.5

_KELVIN_UNITS = frozenset({"k", "kelvin", "degk", "deg_k"})
_CELSIUS_UNITS = frozenset({"c", "degc", "deg_c", "celsius", "degreec", "degrees_celsius"})

# 2 m air temperature never reaches this value in degrees Celsius, so a sample
# exceeding it is unambiguous evidence that the data are still in kelvin.
_CELSIUS_PLAUSIBILITY_LIMIT = 100.0

# Upper bound on the number of values inspected by the unit heuristic. Keeps the
# check cheap for lazily loaded (dask-backed) arrays.
_HEURISTIC_SAMPLE_SIZE = 1000


def _normalise_unit(unit: str) -> str:
    return unit.strip().lower().replace("°", "").replace(" ", "")


def _sample_values(data: xr.DataArray) -> np.ndarray:
    """Return a small, cheaply computed sample of finite values."""
    subset = data
    for dim in data.dims:
        if subset.size <= _HEURISTIC_SAMPLE_SIZE:
            break
        subset = subset.isel({dim: slice(0, max(1, _HEURISTIC_SAMPLE_SIZE // 10))})
    values = np.asarray(subset.values).ravel()
    return values[np.isfinite(values)]


def to_celsius(temperature: xr.DataArray, *, assume: str | None = None) -> xr.DataArray:
    """Convert a temperature field to degrees Celsius.

    The unit is taken from the ``units`` attribute when it is recognised. When
    the attribute is missing or unknown, a value-range heuristic is used: a
    sample exceeding :data:`_CELSIUS_PLAUSIBILITY_LIMIT` is treated as kelvin.

    Parameters
    ----------
    temperature
        Temperature field in kelvin or degrees Celsius.
    assume
        Force the input unit (``"K"`` or ``"degC"``) instead of inferring it.
        Use this when the metadata is known to be wrong.

    Returns
    -------
    xarray.DataArray
        Temperature in degrees Celsius, with ``units`` set to ``"degC"``.

    Raises
    ------
    ValueError
        If ``assume`` is given but is not a recognised temperature unit, or if
        the array contains no finite values to run the heuristic on.
    """
    if assume is not None:
        unit = _normalise_unit(assume)
        if unit in _KELVIN_UNITS:
            is_kelvin = True
        elif unit in _CELSIUS_UNITS:
            is_kelvin = False
        else:
            raise ValueError(
                f"Cannot interpret {assume!r} as a temperature unit; "
                f"expected one of {sorted(_KELVIN_UNITS | _CELSIUS_UNITS)}"
            )
    else:
        declared = _normalise_unit(str(temperature.attrs.get("units", "")))
        if declared in _KELVIN_UNITS:
            is_kelvin = True
        elif declared in _CELSIUS_UNITS:
            is_kelvin = False
        else:
            sample = _sample_values(temperature)
            if sample.size == 0:
                raise ValueError(
                    "Cannot determine the temperature unit: the array has no finite "
                    "values and no recognised 'units' attribute. Pass assume='K' or "
                    "assume='degC' explicitly."
                )
            is_kelvin = bool(np.nanmax(sample) > _CELSIUS_PLAUSIBILITY_LIMIT)

    converted = temperature - KELVIN_OFFSET if is_kelvin else temperature.copy()
    converted.attrs = dict(temperature.attrs)
    converted.attrs["units"] = "degC"
    if is_kelvin:
        converted.attrs["unit_conversion"] = "K -> degC"
    return converted


def to_millimetres(
    precipitation: xr.DataArray, *, assume: str | None = None
) -> xr.DataArray:
    """Convert a precipitation field to millimetres.

    ECMWF products report precipitation in **metres** of water equivalent,
    which is easy to miss: a field in metres passes any plausibility check
    stated in millimetres, so the error is silent and the totals come out a
    thousand times too small.

    The unit is taken from the ``units`` attribute when recognised. Otherwise a
    range heuristic is used: an hourly total above
    :data:`_METRE_PLAUSIBILITY_LIMIT` metres is impossible, so such data must
    already be in millimetres.

    Parameters
    ----------
    precipitation
        Precipitation field in metres or millimetres. ``kg m-2`` is treated as
        millimetres, which it equals for water.
    assume
        Force the input unit (``"m"`` or ``"mm"``) instead of inferring it.

    Returns
    -------
    xarray.DataArray
        Precipitation in millimetres, with ``units`` set to ``"mm"``.

    Raises
    ------
    ValueError
        If ``assume`` is not a recognised precipitation unit, or if the array
        has no finite values to run the heuristic on.
    """
    if assume is not None:
        unit = _normalise_unit(assume)
        if unit in _METRE_UNITS:
            is_metres = True
        elif unit in _MILLIMETRE_UNITS:
            is_metres = False
        else:
            raise ValueError(
                f"Cannot interpret {assume!r} as a precipitation unit; expected one of "
                f"{sorted(_METRE_UNITS | _MILLIMETRE_UNITS)}"
            )
    else:
        declared = _normalise_unit(str(precipitation.attrs.get("units", "")))
        if declared in _METRE_UNITS:
            is_metres = True
        elif declared in _MILLIMETRE_UNITS:
            is_metres = False
        else:
            sample = _sample_values(precipitation)
            if sample.size == 0:
                raise ValueError(
                    "Cannot determine the precipitation unit: the array has no finite "
                    "values and no recognised 'units' attribute. Pass assume='m' or "
                    "assume='mm' explicitly."
                )
            is_metres = bool(np.nanmax(sample) < _METRE_PLAUSIBILITY_LIMIT)

    converted = (
        precipitation * MILLIMETRES_PER_METRE if is_metres else precipitation.copy()
    )
    converted.attrs = dict(precipitation.attrs)
    converted.attrs["units"] = "mm"
    if is_metres:
        converted.attrs["unit_conversion"] = "m -> mm"
    return converted


def wind_speed_from_components(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    name: str = "wind_speed",
) -> xr.DataArray:
    """Compute scalar wind speed from orthogonal wind components.

    Parameters
    ----------
    u, v
        Zonal and meridional wind components in m/s, on a common grid.
    name
        Name assigned to the resulting array.

    Returns
    -------
    xarray.DataArray
        Wind speed ``sqrt(u**2 + v**2)`` in m/s.
    """
    speed = np.sqrt(u**2 + v**2)
    speed = speed.rename(name)
    speed.attrs = {
        "long_name": "Wind speed",
        "units": "m/s",
        "description": f"sqrt({u.name or 'u'}**2 + {v.name or 'v'}**2)",
    }
    return speed
