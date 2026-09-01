"""Map panels for multi-resolution comparison.

The recurring figure in this study is a grid of maps: one row per indicator, one
column per product, every panel on a **shared** colour scale so that differences
between panels are differences in the data rather than in the scaling.

Cartopy is used when available and degraded gracefully when it is not, since it
is the one dependency most likely to be missing or to lack its Natural Earth
cache on a fresh machine. Without it the panels are plain pcolormesh plots on
lon/lat axes: less pretty, identical numbers.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from alpinemet.io.naming import find_coordinate
from alpinemet.io.subset import ALPINE_DOMAIN, BoundingBox
from alpinemet.plotting.style import apply_style, panel_label, shared_limits

__all__ = ["cartopy_available", "comparison_grid", "plot_field"]


def cartopy_available() -> bool:
    """Whether cartopy and its Natural Earth features can be used.

    Returns
    -------
    bool
        True when cartopy imports and its coastline geometries load. A cartopy
        installation without its downloaded data is treated as unavailable,
        because the failure otherwise surfaces mid-plot.
    """
    try:
        import cartopy.crs  # noqa: F401
        import cartopy.feature as cfeature
    except ImportError:
        return False

    try:
        list(cfeature.BORDERS.with_scale("110m").geometries())
    except Exception:
        return False
    return True


def _coordinates(field: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    lat_name = find_coordinate(field, "latitude")
    lon_name = find_coordinate(field, "longitude")
    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Field has no latitude/longitude coordinates; got {tuple(field.dims)}"
        )
    return np.asarray(field[lon_name].values), np.asarray(field[lat_name].values)


def plot_field(
    axis,
    field: xr.DataArray,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    domain: BoundingBox | None = ALPINE_DOMAIN,
    add_features: bool = True,
):
    """Draw one two-dimensional field onto an axis.

    Parameters
    ----------
    axis
        Matplotlib axes. Pass a cartopy GeoAxes to get borders and coastlines.
    field
        Two-dimensional field with latitude and longitude coordinates.
    cmap
        Colormap name.
    vmin, vmax
        Colour limits. Pass the same values to every panel of a comparison; see
        :func:`alpinemet.plotting.style.shared_limits`.
    domain
        Extent to display. ``None`` uses the field's own extent.
    add_features
        Draw coastlines and borders when the axis supports them.

    Returns
    -------
    matplotlib.collections.QuadMesh
        The mesh, for attaching a colorbar.

    Raises
    ------
    ValueError
        If the field is not two-dimensional, or lacks coordinates.
    """
    if field.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional field, got {field.ndim} dimensions "
            f"{tuple(field.dims)}. Reduce over time first."
        )

    lons, lats = _coordinates(field)
    kwargs = {"cmap": cmap, "vmin": vmin, "vmax": vmax, "shading": "auto"}

    is_geoaxes = hasattr(axis, "coastlines")
    if is_geoaxes:
        import cartopy.crs as ccrs

        kwargs["transform"] = ccrs.PlateCarree()

    mesh = axis.pcolormesh(lons, lats, field.values, **kwargs)

    if is_geoaxes:
        if add_features:
            import cartopy.feature as cfeature

            axis.coastlines(resolution="50m", linewidth=0.4)
            axis.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="0.35")
        if domain is not None:
            import cartopy.crs as ccrs

            axis.set_extent(
                [domain.lon_min, domain.lon_max, domain.lat_min, domain.lat_max],
                crs=ccrs.PlateCarree(),
            )
    else:
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        if domain is not None:
            axis.set_xlim(domain.lon_min, domain.lon_max)
            axis.set_ylim(domain.lat_min, domain.lat_max)

    return mesh


def comparison_grid(
    rows: Sequence[dict],
    *,
    column_titles: Sequence[str],
    domain: BoundingBox | None = ALPINE_DOMAIN,
    figsize: tuple[float, float] | None = None,
    use_cartopy: bool | None = None,
):
    """Build a rows-by-columns grid of maps sharing one colour scale per row.

    Parameters
    ----------
    rows
        One mapping per row, each with:

        ``fields``
            Sequence of two-dimensional fields, one per column.
        ``label``
            Row label, used on the colorbar.
        ``cmap``
            Colormap name. Optional, defaults to viridis.
        ``limits``
            Explicit ``(vmin, vmax)``. Optional; derived from the row's fields
            when omitted.
        ``minimum_at_zero``
            Force the scale to start at zero. Optional.
    column_titles
        Titles for the columns, typically the product names and resolutions.
    domain
        Extent to display.
    figsize
        Figure size in inches. Derived from the grid shape when omitted.
    use_cartopy
        Force cartopy on or off. Detected when omitted.

    Returns
    -------
    tuple
        ``(figure, axes)`` where ``axes`` is a nested list indexed
        ``[row][column]``.

    Raises
    ------
    ValueError
        If a row's field count does not match the number of columns.
    """
    import matplotlib.pyplot as plt

    apply_style()

    n_rows = len(rows)
    n_cols = len(column_titles)

    for index, row in enumerate(rows):
        if len(row["fields"]) != n_cols:
            raise ValueError(
                f"Row {index} has {len(row['fields'])} fields but there are "
                f"{n_cols} columns"
            )

    if use_cartopy is None:
        use_cartopy = cartopy_available()

    subplot_kw = {}
    if use_cartopy:
        import cartopy.crs as ccrs

        subplot_kw["projection"] = ccrs.PlateCarree()

    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize or (3.4 * n_cols, 2.9 * n_rows),
        subplot_kw=subplot_kw,
        squeeze=False,
    )

    panel = 0
    for row_index, row in enumerate(rows):
        fields = row["fields"]
        limits = row.get("limits") or shared_limits(
            fields, minimum_at_zero=row.get("minimum_at_zero", False)
        )
        cmap = row.get("cmap", "viridis")

        mesh = None
        for column_index, field in enumerate(fields):
            axis = axes[row_index][column_index]
            mesh = plot_field(
                axis,
                field,
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
                domain=domain,
            )
            if row_index == 0:
                axis.set_title(column_titles[column_index])
            panel_label(axis, panel)
            panel += 1

        # One colorbar per row, because the scale is shared along the row and
        # differs between rows.
        figure.colorbar(
            mesh,
            ax=axes[row_index],
            orientation="vertical",
            fraction=0.025,
            pad=0.02,
            label=row.get("label", ""),
        )

    return figure, [list(row) for row in axes]
