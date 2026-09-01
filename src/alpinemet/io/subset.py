"""Spatial and temporal subsetting across differing grid conventions.

Two conventions cause most of the trouble when subsetting the products used
here:

**Latitude direction.** ERA5 and ERA5-Land store latitude *descending*
(north to south); ARA and most regional products store it ascending. A plain
``sel(latitude=slice(46, 49))`` silently returns an empty selection on a
descending axis -- no error, just no data. :func:`subset_spatial` orders the
slice to match the axis.

**Longitude range.** Some products use 0-360, others -180-180. The Alpine
domain sits near 5-18 degE and is unaffected, but the helper is applied anyway
so that the code transfers to other regions.

The Alpine domain used throughout the study is 4-18 degE, 43-49 degN, as stated
in the paper's methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from alpinemet.io.naming import find_coordinate

__all__ = [
    "ALPINE_DOMAIN",
    "BoundingBox",
    "normalise_longitude",
    "subset_spatial",
    "subset_temporal",
]


@dataclass(frozen=True)
class BoundingBox:
    """A geographic bounding box in degrees.

    Attributes
    ----------
    lon_min, lon_max
        Longitude bounds in degrees east.
    lat_min, lat_max
        Latitude bounds in degrees north.
    name
        Label recorded in output metadata.
    """

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    name: str = "domain"

    def __post_init__(self) -> None:
        if self.lon_min >= self.lon_max:
            raise ValueError(
                f"lon_min must be less than lon_max; got {self.lon_min}, {self.lon_max}"
            )
        if self.lat_min >= self.lat_max:
            raise ValueError(
                f"lat_min must be less than lat_max; got {self.lat_min}, {self.lat_max}"
            )
        if not -90.0 <= self.lat_min < self.lat_max <= 90.0:
            raise ValueError(f"Latitudes must lie within [-90, 90]; got {self}")


#: The Alpine evaluation domain used throughout the study.
ALPINE_DOMAIN = BoundingBox(
    lon_min=4.0, lon_max=18.0, lat_min=43.0, lat_max=49.0, name="Alpine domain"
)


def normalise_longitude(
    dataset: xr.Dataset | xr.DataArray,
    *,
    lon_name: str | None = None,
    to_180: bool = True,
) -> xr.Dataset | xr.DataArray:
    """Convert the longitude coordinate between the 0-360 and -180-180 ranges.

    Parameters
    ----------
    dataset
        Object with a longitude coordinate.
    lon_name
        Longitude coordinate name. Resolved from the usual aliases when
        omitted.
    to_180
        Target ``-180`` to ``180`` when true, ``0`` to ``360`` otherwise.

    Returns
    -------
    Same type as the input
        Object with the longitude coordinate converted and sorted ascending.
        Returned unchanged when the coordinate is already in the target range.

    Raises
    ------
    ValueError
        If no longitude coordinate can be found.
    """
    if lon_name is None:
        lon_name = find_coordinate(dataset, "longitude")
    if lon_name is None:
        raise ValueError("No longitude coordinate found")

    values = np.asarray(dataset[lon_name].values, dtype=float)

    if to_180:
        if values.max() <= 180.0:
            return dataset
        converted = ((values + 180.0) % 360.0) - 180.0
    else:
        if values.min() >= 0.0:
            return dataset
        converted = values % 360.0

    return dataset.assign_coords({lon_name: converted}).sortby(lon_name)


def subset_spatial(
    dataset: xr.Dataset | xr.DataArray,
    box: BoundingBox = ALPINE_DOMAIN,
    *,
    lat_name: str | None = None,
    lon_name: str | None = None,
) -> xr.Dataset | xr.DataArray:
    """Cut a dataset down to a bounding box, whatever its axis directions.

    Parameters
    ----------
    dataset
        Object with latitude and longitude coordinates.
    box
        Target bounding box. Defaults to :data:`ALPINE_DOMAIN`.
    lat_name, lon_name
        Coordinate names. Resolved from the usual aliases when omitted.

    Returns
    -------
    Same type as the input
        The subset.

    Raises
    ------
    ValueError
        If a coordinate cannot be found, or if the box does not overlap the
        data. An empty selection is treated as an error rather than returned,
        because a silently empty result is the classic symptom of a descending
        latitude axis and propagates far before it is noticed.
    """
    if lat_name is None:
        lat_name = find_coordinate(dataset, "latitude")
    if lon_name is None:
        lon_name = find_coordinate(dataset, "longitude")
    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Could not resolve spatial coordinates; found latitude={lat_name!r}, "
            f"longitude={lon_name!r}"
        )

    latitudes = np.asarray(dataset[lat_name].values, dtype=float)
    longitudes = np.asarray(dataset[lon_name].values, dtype=float)

    lat_descending = latitudes.size > 1 and latitudes[0] > latitudes[-1]
    lon_descending = longitudes.size > 1 and longitudes[0] > longitudes[-1]

    lat_slice = (
        slice(box.lat_max, box.lat_min) if lat_descending else slice(box.lat_min, box.lat_max)
    )
    lon_slice = (
        slice(box.lon_max, box.lon_min) if lon_descending else slice(box.lon_min, box.lon_max)
    )

    subset = dataset.sel({lat_name: lat_slice, lon_name: lon_slice})

    if subset.sizes[lat_name] == 0 or subset.sizes[lon_name] == 0:
        raise ValueError(
            f"Bounding box {box.name} ({box.lon_min}-{box.lon_max} degE, "
            f"{box.lat_min}-{box.lat_max} degN) does not overlap the data, which spans "
            f"{longitudes.min():.2f}-{longitudes.max():.2f} degE, "
            f"{latitudes.min():.2f}-{latitudes.max():.2f} degN"
        )

    subset.attrs = {**dataset.attrs, "spatial_subset": box.name}
    return subset


def subset_temporal(
    dataset: xr.Dataset | xr.DataArray,
    start: str | np.datetime64 | None = None,
    end: str | np.datetime64 | None = None,
    *,
    time_name: str | None = None,
) -> xr.Dataset | xr.DataArray:
    """Cut a dataset down to a time range.

    Parameters
    ----------
    dataset
        Object with a time coordinate.
    start, end
        Inclusive bounds. ``None`` leaves that end open.
    time_name
        Time coordinate name. Resolved from the usual aliases when omitted.

    Returns
    -------
    Same type as the input
        The subset.

    Raises
    ------
    ValueError
        If no time coordinate is found, or if the range selects nothing.
    """
    if time_name is None:
        time_name = find_coordinate(dataset, "time")
    if time_name is None:
        raise ValueError("No time coordinate found")

    if start is None and end is None:
        return dataset

    subset = dataset.sel({time_name: slice(start, end)})

    if subset.sizes[time_name] == 0:
        available = dataset[time_name].values
        raise ValueError(
            f"Time range {start} to {end} selects no data; the record spans "
            f"{available.min()} to {available.max()}"
        )

    subset.attrs = {
        **dataset.attrs,
        "temporal_subset": f"{start} to {end}",
    }
    return subset
