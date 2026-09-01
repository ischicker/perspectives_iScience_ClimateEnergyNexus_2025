"""Population weighting of gridded fields.

Area-averaged statistics understate what matters for energy demand in
mountainous terrain, because population is concentrated on valley floors --
precisely where inversions trap cold air and where distributed PV sits below
the fog layer. Weighting by population moves the statistic to where the people,
and therefore the load, actually are.

Getting the regridding right
----------------------------

A population raster holds **counts per cell**, not a density. Moving it to a
coarser model grid is therefore an *aggregation* (sum the people falling inside
each target cell), not an interpolation. :func:`aggregate_population_to_grid`
does this, and conserves the total exactly.

.. warning::

   The ``"nearest_normalised"`` method reproduces an earlier implementation
   that sampled the raster with nearest-neighbour interpolation and then
   rescaled the result by a single global factor to restore the domain total.
   It conserves the total but **destroys the spatial distribution**: each
   target cell takes the value of one arbitrarily chosen source cell. Going
   from a 100 m raster to a 31 km grid, that is one cell in roughly 96,000.

   The distortion is severe and, worse, resolution-dependent -- a 2.5 km target
   grid samples one cell in 625, a 31 km grid one in 96,000 -- so it does not
   affect compared datasets equally. It is retained solely to reproduce earlier
   output and must not be used for new analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

__all__ = [
    "RegridMethod",
    "aggregate_population_to_grid",
    "load_population_raster",
    "person_days",
    "population_weighted_mean",
]

RegridMethod = Literal["sum", "nearest_normalised"]


def _cell_edges(centers: np.ndarray) -> np.ndarray:
    """Infer cell edges from monotonically increasing cell centers."""
    centers = np.asarray(centers, dtype=float)
    if centers.size < 2:
        raise ValueError("Need at least two cell centers to infer edges")

    midpoints = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (midpoints[0] - centers[0])
    last = centers[-1] + (centers[-1] - midpoints[-1])
    return np.concatenate(([first], midpoints, [last]))


def load_population_raster(
    path: str | Path,
    *,
    bounds: tuple[float, float, float, float] | None = None,
    band: int = 1,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> xr.DataArray:
    """Load a population count raster into a labelled array.

    Parameters
    ----------
    path
        Path to a GeoTIFF holding population counts per cell, in EPSG:4326.
    bounds
        Optional ``(min_lon, min_lat, max_lon, max_lat)`` subset, in degrees.
        Subsetting before aggregation keeps memory manageable: a pan-Alpine
        100 m raster is several gigabytes.
    band
        Raster band to read.
    lat_name, lon_name
        Names given to the output coordinates.

    Returns
    -------
    xarray.DataArray
        Population counts with ascending latitude and longitude coordinates.
        Nodata values are replaced by zero, since "no data" for population
        counts means "nobody recorded here".

    Raises
    ------
    ValueError
        If the raster is not in EPSG:4326. Reproject it first; doing so inside
        this function would silently resample counts.
    """
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.open(path) as src:
        if src.crs is not None and src.crs.to_epsg() != 4326:
            raise ValueError(
                f"Population raster must be in EPSG:4326; got {src.crs}. Reproject it "
                "with a sum-preserving method before loading."
            )

        if bounds is None:
            window = None
            transform = src.transform
        else:
            min_lon, min_lat, max_lon, max_lat = bounds
            window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
            transform = src.window_transform(window)

        data = src.read(band, window=window, masked=True)
        values = np.asarray(data.filled(0.0), dtype=float)

    height, width = values.shape
    lons = transform.c + transform.a * (np.arange(width) + 0.5)
    lats = transform.f + transform.e * (np.arange(height) + 0.5)

    # Rasters are usually stored north-up, i.e. descending latitude.
    if lats.size > 1 and lats[1] < lats[0]:
        lats = lats[::-1]
        values = np.flip(values, axis=0)
    if lons.size > 1 and lons[1] < lons[0]:
        lons = lons[::-1]
        values = np.flip(values, axis=1)

    raster = xr.DataArray(
        values,
        dims=(lat_name, lon_name),
        coords={lat_name: lats, lon_name: lons},
        name="population",
    )
    raster.attrs = {
        "long_name": "Population count per cell",
        "units": "persons",
        "source_file": str(path),
    }
    return raster


def aggregate_population_to_grid(
    population: xr.DataArray,
    target: xr.DataArray | xr.Dataset,
    *,
    method: RegridMethod = "sum",
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> xr.DataArray:
    """Aggregate a fine population raster onto a coarser model grid.

    Each source cell is assigned in full to the target cell containing its
    centre, and the counts are summed. This conserves the total population
    exactly and preserves where people are. It assumes the source raster is
    substantially finer than the target grid, which holds for every combination
    used in this study (100 m or 1 km source against 2.5-31 km targets).

    Parameters
    ----------
    population
        Population counts on a fine grid, with latitude and longitude
        coordinates.
    target
        Object carrying the target grid coordinates.
    method
        ``"sum"`` for conservative aggregation (the default and the only method
        suitable for analysis), or ``"nearest_normalised"`` to reproduce the
        earlier implementation; see the module warning.
    lat_name, lon_name
        Coordinate names, used on both inputs.

    Returns
    -------
    xarray.DataArray
        Population counts on the target grid.

    Raises
    ------
    ValueError
        If either input lacks the named coordinates, or if ``method`` is not
        recognised.
    """
    for name, obj in (("population", population), ("target", target)):
        for coord in (lat_name, lon_name):
            if coord not in obj.coords:
                raise ValueError(f"{name} has no {coord!r} coordinate")

    target_lat = np.asarray(target[lat_name].values, dtype=float)
    target_lon = np.asarray(target[lon_name].values, dtype=float)

    # Work on ascending axes, then restore the caller's ordering.
    lat_descending = target_lat.size > 1 and target_lat[1] < target_lat[0]
    lon_descending = target_lon.size > 1 and target_lon[1] < target_lon[0]
    lat_sorted = target_lat[::-1] if lat_descending else target_lat
    lon_sorted = target_lon[::-1] if lon_descending else target_lon

    source_lat = np.asarray(population[lat_name].values, dtype=float)
    source_lon = np.asarray(population[lon_name].values, dtype=float)
    values = np.asarray(population.transpose(lat_name, lon_name).values, dtype=float)

    if method == "sum":
        lat_edges = _cell_edges(lat_sorted)
        lon_edges = _cell_edges(lon_sorted)
        lat_grid, lon_grid = np.meshgrid(source_lat, source_lon, indexing="ij")
        aggregated, _, _ = np.histogram2d(
            lat_grid.ravel(),
            lon_grid.ravel(),
            bins=[lat_edges, lon_edges],
            weights=np.nan_to_num(values).ravel(),
        )
    elif method == "nearest_normalised":
        from scipy.interpolate import RegularGridInterpolator

        interpolator = RegularGridInterpolator(
            (source_lat, source_lon),
            np.nan_to_num(values),
            method="nearest",
            bounds_error=False,
            fill_value=0.0,
        )
        lat_grid, lon_grid = np.meshgrid(lat_sorted, lon_sorted, indexing="ij")
        sampled = interpolator(
            np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        ).reshape(lat_grid.shape)

        total_source = float(np.nansum(values))
        total_sampled = float(np.nansum(sampled))
        aggregated = (
            sampled * (total_source / total_sampled) if total_sampled > 0 else sampled
        )
    else:
        raise ValueError(
            f"Unknown regrid method {method!r}; expected 'sum' or 'nearest_normalised'"
        )

    if lat_descending:
        aggregated = np.flip(aggregated, axis=0)
    if lon_descending:
        aggregated = np.flip(aggregated, axis=1)

    result = xr.DataArray(
        aggregated,
        dims=(lat_name, lon_name),
        coords={lat_name: target_lat, lon_name: target_lon},
        name="population",
    )
    result.attrs = {
        "long_name": "Population count per grid cell",
        "units": "persons",
        "regrid_method": method,
        "source_total": float(np.nansum(values)),
        "regridded_total": float(np.nansum(aggregated)),
    }
    if method == "nearest_normalised":
        result.attrs["warning"] = (
            "spatial distribution not preserved; for reproducing earlier output only"
        )
    return result


def population_weighted_mean(
    field: xr.DataArray,
    population: xr.DataArray,
    *,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> xr.DataArray:
    """Population-weighted spatial mean of a field.

    Parameters
    ----------
    field
        Field to average, on the same grid as ``population``.
    population
        Population counts per grid cell.
    lat_name, lon_name
        Spatial coordinate names to reduce over.

    Returns
    -------
    xarray.DataArray
        Weighted mean, with the spatial dimensions removed. Any other
        dimensions, such as time, are preserved.

    Raises
    ------
    ValueError
        If the total population is zero, which would make the mean undefined.
    """
    total = float(population.sum())
    if not total > 0:
        raise ValueError(
            "Total population is zero; a population-weighted mean is undefined. "
            "Check that the raster overlaps the model domain."
        )

    weighted = (field * population).sum(dim=[lat_name, lon_name]) / total
    weighted = weighted.rename(f"{field.name or 'field'}_population_weighted")
    weighted.attrs = {
        "long_name": f"Population-weighted mean of {field.attrs.get('long_name', 'field')}",
        "units": field.attrs.get("units", ""),
        "total_population": total,
    }
    return weighted


def person_days(
    condition: xr.DataArray,
    population: xr.DataArray,
    *,
    time_dim: str = "time",
    hours_per_step: float | None = None,
) -> xr.DataArray:
    """Population exposure to a condition, in person-days.

    Multiplies the time each grid cell spends in the condition by that cell's
    population and sums over the domain. This is the exposure metric behind the
    heat and cold statistics reported in the paper.

    Parameters
    ----------
    condition
        Boolean field, e.g. daily maximum temperature above 30 degC.
    population
        Population counts per grid cell.
    time_dim
        Name of the time dimension.
    hours_per_step
        Length of one time step in hours. Inferred from the time coordinate
        when omitted; pass 24 explicitly for daily input without a usable
        coordinate.

    Returns
    -------
    xarray.DataArray
        Total exposure in person-days, reduced over all dimensions.
    """
    if hours_per_step is None:
        from alpinemet.indicators.dunkelflaute import infer_timestep_hours

        hours_per_step = infer_timestep_hours(condition, time_dim=time_dim)

    days_per_step = hours_per_step / 24.0
    exposure = (condition * population).sum() * days_per_step
    exposure = exposure.rename("person_days")
    exposure.attrs = {
        "long_name": "Population exposure",
        "units": "person days",
        "hours_per_step": hours_per_step,
    }
    return exposure
