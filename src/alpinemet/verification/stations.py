"""Point verification of gridded products against station observations.

The station comparison behind Figure S1 and Table S1 of the paper: 54 Austrian
sites excluded from ARA's assimilation, making them quasi-independent.

Two things deserve emphasis when reading any result produced here.

**Representativeness.** A grid cell is an area average; a station is a point.
At 2.5 km over Alpine terrain the two can differ by hundreds of metres in
elevation, and the resulting temperature offset is often larger than the model
error one is trying to measure. :func:`match_stations_to_grid` records the
elevation difference for every station so that this can be inspected rather
than absorbed silently into the bias, and :func:`lapse_rate_adjustment` offers a
first-order correction.

**Coverage.** As the paper stresses, roughly 90 % of Alpine stations lie below
1,500 m, exactly where the terrain is least demanding. Aggregate statistics
over such a network flatter every product. :func:`elevation_band_metrics`
splits the verification by altitude so the sparse high-elevation sample is
visible instead of being averaged away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from alpinemet.io.naming import find_coordinate
from alpinemet.verification.metrics import METRIC_NAMES, verification_metrics

__all__ = [
    "DEFAULT_ELEVATION_BANDS",
    "DRY_ADIABATIC_LAPSE_RATE",
    "ENVIRONMENTAL_LAPSE_RATE",
    "elevation_band_metrics",
    "extract_at_stations",
    "lapse_rate_adjustment",
    "match_stations_to_grid",
    "station_metrics",
]

#: Standard atmosphere temperature decrease with height, in K per metre.
ENVIRONMENTAL_LAPSE_RATE = 0.0065

#: Dry adiabatic lapse rate, in K per metre. Appropriate for well-mixed
#: conditions; badly wrong during the valley inversions this study cares about.
DRY_ADIABATIC_LAPSE_RATE = 0.0098

#: Elevation bands in metres, chosen around the 1,500 m station-density break
#: and the 1,500-2,500 m band where Alpine wind farms are increasingly sited.
DEFAULT_ELEVATION_BANDS: tuple[float, ...] = (0.0, 500.0, 1000.0, 1500.0, 2500.0, 4000.0)


def match_stations_to_grid(
    dataset: xr.Dataset | xr.DataArray,
    stations: pd.DataFrame,
    *,
    lat_column: str = "lat",
    lon_column: str = "lon",
    elevation_column: str | None = "alt",
    elevation_variable: str | None = None,
) -> pd.DataFrame:
    """Find the nearest grid cell for each station.

    Parameters
    ----------
    dataset
        Gridded product with latitude and longitude coordinates.
    stations
        Station metadata, one row per site.
    lat_column, lon_column
        Column names holding station coordinates in degrees.
    elevation_column
        Column holding station elevation in metres, or ``None`` if unavailable.
    elevation_variable
        Name of a model orography variable in ``dataset``. When both this and
        ``elevation_column`` are given, the elevation difference is computed.

    Returns
    -------
    pandas.DataFrame
        The station table with ``lat_index``, ``lon_index``, ``grid_lat``,
        ``grid_lon`` and ``distance_km`` added, plus ``grid_elevation`` and
        ``elevation_difference`` when orography is available.

    Raises
    ------
    ValueError
        If coordinates cannot be resolved or required columns are missing.
    """
    lat_name = find_coordinate(dataset, "latitude")
    lon_name = find_coordinate(dataset, "longitude")
    if lat_name is None or lon_name is None:
        raise ValueError("Could not resolve latitude and longitude coordinates")

    for column in (lat_column, lon_column):
        if column not in stations.columns:
            raise ValueError(
                f"Station table has no {column!r} column; got {list(stations.columns)}"
            )

    grid_lats = np.asarray(dataset[lat_name].values, dtype=float)
    grid_lons = np.asarray(dataset[lon_name].values, dtype=float)

    matched = stations.copy()
    station_lats = matched[lat_column].to_numpy(dtype=float)
    station_lons = matched[lon_column].to_numpy(dtype=float)

    lat_indices = np.abs(grid_lats[None, :] - station_lats[:, None]).argmin(axis=1)
    lon_indices = np.abs(grid_lons[None, :] - station_lons[:, None]).argmin(axis=1)

    matched["lat_index"] = lat_indices
    matched["lon_index"] = lon_indices
    matched["grid_lat"] = grid_lats[lat_indices]
    matched["grid_lon"] = grid_lons[lon_indices]

    # Great-circle distance is overkill at these separations; a local planar
    # approximation is accurate to well under a metre over a few kilometres.
    mean_lat = np.radians(0.5 * (station_lats + matched["grid_lat"].to_numpy()))
    dy = (matched["grid_lat"].to_numpy() - station_lats) * 111.32
    dx = (matched["grid_lon"].to_numpy() - station_lons) * 111.32 * np.cos(mean_lat)
    matched["distance_km"] = np.sqrt(dx**2 + dy**2)

    if elevation_variable is not None and elevation_column is not None:
        if elevation_column not in matched.columns:
            raise ValueError(f"Station table has no {elevation_column!r} column")
        orography = dataset[elevation_variable]
        matched["grid_elevation"] = [
            float(orography.isel({lat_name: int(i), lon_name: int(j)}).values)
            for i, j in zip(lat_indices, lon_indices, strict=True)
        ]
        matched["elevation_difference"] = (
            matched["grid_elevation"] - matched[elevation_column]
        )

    return matched


def extract_at_stations(
    data: xr.DataArray,
    matched_stations: pd.DataFrame,
    *,
    station_id_column: str = "synnr",
    time_dim: str = "time",
) -> pd.DataFrame:
    """Pull a gridded field out at the matched station locations.

    Parameters
    ----------
    data
        Gridded field with a time dimension.
    matched_stations
        Output of :func:`match_stations_to_grid`.
    station_id_column
        Column identifying each station.
    time_dim
        Name of the time dimension.

    Returns
    -------
    pandas.DataFrame
        Long-format frame with ``station_id``, ``time`` and ``value``.

    Raises
    ------
    ValueError
        If the station table has not been matched to a grid.
    """
    required = {"lat_index", "lon_index", station_id_column}
    missing = required - set(matched_stations.columns)
    if missing:
        raise ValueError(
            f"Station table is missing {sorted(missing)}; call match_stations_to_grid first"
        )

    lat_name = find_coordinate(data, "latitude")
    lon_name = find_coordinate(data, "longitude")

    lat_indices = xr.DataArray(
        matched_stations["lat_index"].to_numpy(), dims="station"
    )
    lon_indices = xr.DataArray(
        matched_stations["lon_index"].to_numpy(), dims="station"
    )
    # Pointwise (vectorised) selection: one series per station, not the outer
    # product of the two index arrays.
    extracted = data.isel({lat_name: lat_indices, lon_name: lon_indices})
    extracted = extracted.assign_coords(
        station=matched_stations[station_id_column].to_numpy()
    )

    frame = extracted.to_dataframe(name="value").reset_index()
    return frame.rename(columns={"station": "station_id", time_dim: "time"})[
        ["station_id", "time", "value"]
    ]


def lapse_rate_adjustment(
    modelled: pd.Series | np.ndarray,
    elevation_difference: pd.Series | np.ndarray,
    *,
    lapse_rate: float = ENVIRONMENTAL_LAPSE_RATE,
) -> np.ndarray:
    """Adjust modelled temperature for the station-grid elevation offset.

    ``adjusted = modelled + lapse_rate * (model_elevation - station_elevation)``

    so a model cell sitting above its station is warmed towards the station's
    level.

    .. warning::

       A constant lapse rate assumes a well-mixed atmosphere. During the
       persistent winter valley inversions that this study is largely about,
       temperature *increases* with height and the correction has the wrong
       sign. Report adjusted and unadjusted statistics side by side rather than
       replacing one with the other.

    Parameters
    ----------
    modelled
        Modelled temperature in degrees Celsius.
    elevation_difference
        Model elevation minus station elevation, in metres.
    lapse_rate
        Temperature decrease per metre of ascent, in K/m.

    Returns
    -------
    numpy.ndarray
        Adjusted temperature in degrees Celsius.
    """
    model_values = np.asarray(modelled, dtype=float)
    difference = np.asarray(elevation_difference, dtype=float)
    return model_values + lapse_rate * difference


def station_metrics(
    paired: pd.DataFrame,
    *,
    observed_column: str = "observed",
    modelled_column: str = "modelled",
    station_id_column: str = "station_id",
    minimum_pairs: int = 100,
    station_metadata: pd.DataFrame | None = None,
    metadata_id_column: str = "synnr",
) -> pd.DataFrame:
    """Verification metrics per station.

    Parameters
    ----------
    paired
        Long-format frame with one row per station and time, holding the
        observed and modelled values.
    observed_column, modelled_column, station_id_column
        Column names.
    minimum_pairs
        Stations with fewer valid pairs are excluded. Reported in the result
        attributes so that the exclusion is visible rather than silent.
    station_metadata
        Optional station table to merge in, for elevation and names.
    metadata_id_column
        Station identifier column in ``station_metadata``.

    Returns
    -------
    pandas.DataFrame
        One row per retained station with the metrics from
        :data:`alpinemet.verification.metrics.METRIC_NAMES`. The number of
        stations dropped is recorded in ``attrs["stations_excluded"]``.
    """
    rows: list[dict[str, float]] = []
    excluded = 0

    for station_id, group in paired.groupby(station_id_column, sort=True):
        valid = group[[observed_column, modelled_column]].dropna()
        if len(valid) < minimum_pairs:
            excluded += 1
            continue

        metrics = verification_metrics(
            valid[observed_column].to_numpy(), valid[modelled_column].to_numpy()
        )
        rows.append({station_id_column: station_id, **metrics})

    result = pd.DataFrame(rows, columns=[station_id_column, *METRIC_NAMES])

    if station_metadata is not None and not result.empty:
        result = result.merge(
            station_metadata,
            left_on=station_id_column,
            right_on=metadata_id_column,
            how="left",
        )

    result.attrs["stations_excluded"] = excluded
    result.attrs["minimum_pairs"] = minimum_pairs
    return result


def elevation_band_metrics(
    station_statistics: pd.DataFrame,
    *,
    elevation_column: str = "alt",
    bands: tuple[float, ...] = DEFAULT_ELEVATION_BANDS,
    metrics: tuple[str, ...] = ("bias", "rmse", "correlation"),
) -> pd.DataFrame:
    """Aggregate per-station metrics into elevation bands.

    Parameters
    ----------
    station_statistics
        Output of :func:`station_metrics`, including station elevation.
    elevation_column
        Column holding station elevation in metres.
    bands
        Band edges in metres.
    metrics
        Metrics to average within each band.

    Returns
    -------
    pandas.DataFrame
        One row per band with the station count and the mean of each metric.
        Bands containing no stations are retained with a count of zero -- an
        empty high-elevation band is a finding, not something to hide.

    Raises
    ------
    ValueError
        If the elevation column is absent.
    """
    if elevation_column not in station_statistics.columns:
        raise ValueError(
            f"Station statistics have no {elevation_column!r} column; pass "
            "station_metadata to station_metrics so elevation is carried through"
        )

    labels = [f"{low:.0f}-{high:.0f} m" for low, high in zip(bands[:-1], bands[1:], strict=True)]
    binned = pd.cut(
        station_statistics[elevation_column],
        bins=list(bands),
        labels=labels,
        include_lowest=True,
    )

    rows = []
    for label in labels:
        subset = station_statistics[binned == label]
        row: dict[str, float | str] = {
            "elevation_band": label,
            "n_stations": len(subset),
        }
        for metric in metrics:
            row[metric] = float(subset[metric].mean()) if len(subset) else float("nan")
        rows.append(row)

    result = pd.DataFrame(rows)
    result.attrs["bands_m"] = list(bands)
    return result
