"""NetCDF-safe metadata attributes.

Provenance is worth carrying in the file rather than in a separate note, so
this package records thresholds, applied renames and quality-control counts as
dataset attributes. NetCDF only accepts strings, numbers, sequences and bytes,
so structured values are stored as JSON strings.

Encoding at the point where the attribute is set, rather than sanitising just
before writing, means any dataset this package produces can be handed straight
to ``to_netcdf`` without a preparation step.
"""

from __future__ import annotations

import json
from typing import Any

import xarray as xr

__all__ = ["as_attribute", "decode_attribute", "netcdf_safe"]


def as_attribute(value: Any) -> Any:
    """Convert a value into something NetCDF can store.

    Mappings, ``None`` and booleans become JSON strings; everything else passes
    through unchanged.

    Booleans need converting even though Python treats ``bool`` as a number:
    the NetCDF-4 attribute types are the numeric widths plus ``S1``, and a
    ``b1`` is rejected by the writer. They round-trip as the JSON literals
    ``true`` and ``false``.

    Parameters
    ----------
    value
        Value destined for an ``attrs`` entry.

    Returns
    -------
    Any
        A NetCDF-storable value.
    """
    if value is None or isinstance(value, (bool, dict)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def decode_attribute(value: Any) -> Any:
    """Reverse :func:`as_attribute` where possible.

    Parameters
    ----------
    value
        Attribute value read back from a file.

    Returns
    -------
    Any
        The decoded object when ``value`` is a JSON string, otherwise the input
        unchanged. Strings that merely happen not to be JSON are returned as-is
        rather than raising.
    """
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def netcdf_safe(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Return a copy whose attributes can all be written to NetCDF.

    A safety net for datasets assembled outside this package, or combined from
    several sources. Applies :func:`as_attribute` to every attribute of the
    object and of each of its variables.

    Parameters
    ----------
    obj
        Dataset or data array to sanitise.

    Returns
    -------
    Same type as the input
        A copy with NetCDF-storable attributes.
    """
    result = obj.copy()
    result.attrs = {key: as_attribute(value) for key, value in result.attrs.items()}

    if isinstance(result, xr.Dataset):
        for name in result.variables:
            result[name].attrs = {
                key: as_attribute(value) for key, value in result[name].attrs.items()
            }
    return result
