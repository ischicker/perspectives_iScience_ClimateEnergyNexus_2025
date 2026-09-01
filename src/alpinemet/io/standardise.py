"""Dataset standardisation: duplicates, derived wind speed, quality control.

Applied after :mod:`alpinemet.io.naming` has established canonical names and
:mod:`alpinemet.io.accumulation` has converted fluxes to rates.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import xarray as xr

from alpinemet.attrs import as_attribute
from alpinemet.io.naming import find_coordinate, find_variable
from alpinemet.units import wind_speed_from_components

__all__ = [
    "PHYSICAL_RANGES",
    "apply_quality_control",
    "derive_wind_speeds",
    "remove_duplicate_times",
]

#: Plausible ranges for canonical variables, used by :func:`apply_quality_control`.
#: Bounds are generous: they exist to catch unit errors and corrupt records, not
#: to trim genuine extremes.
PHYSICAL_RANGES: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        "temperature_2m": (-90.0, 60.0),
        "u_wind_10m": (-120.0, 120.0),
        "v_wind_10m": (-120.0, 120.0),
        "u_wind_100m": (-150.0, 150.0),
        "v_wind_100m": (-150.0, 150.0),
        "wind_speed_10m": (0.0, 120.0),
        "wind_speed_100m": (0.0, 150.0),
        "wind_gust_10m": (0.0, 150.0),
        "surface_pressure": (40_000.0, 110_000.0),
        "solar_radiation": (0.0, 1_500.0),
        "precipitation": (0.0, 1_000.0),
        "snow_depth": (0.0, 100.0),
    }
)


def remove_duplicate_times(
    dataset: xr.Dataset,
    *,
    time_name: str | None = None,
) -> xr.Dataset:
    """Drop repeated timestamps, keeping the first occurrence of each.

    Duplicates arise when yearly or monthly files overlap at their boundaries.
    They are not harmless: a duplicated step makes ``diff`` return zero there,
    which silently corrupts every accumulation conversion and ramp calculation
    downstream.

    Parameters
    ----------
    dataset
        Dataset to clean.
    time_name
        Time coordinate name. Resolved from the usual aliases when omitted.

    Returns
    -------
    xarray.Dataset
        Dataset with unique, ascending timestamps. The number removed is
        recorded in ``attrs["duplicate_times_removed"]``.

    Raises
    ------
    ValueError
        If no time coordinate is found.
    """
    if time_name is None:
        time_name = find_coordinate(dataset, "time")
    if time_name is None:
        raise ValueError("No time coordinate found")

    times = np.asarray(dataset[time_name].values)
    _, first_indices = np.unique(times, return_index=True)
    keep = np.sort(first_indices)
    removed = times.size - keep.size

    cleaned = dataset.isel({time_name: keep}) if removed else dataset
    cleaned = cleaned.sortby(time_name)
    cleaned.attrs = {**dataset.attrs, "duplicate_times_removed": int(removed)}
    return cleaned


def derive_wind_speeds(
    dataset: xr.Dataset,
    *,
    heights: tuple[str, ...] = ("10m", "100m"),
    overwrite: bool = False,
) -> xr.Dataset:
    """Add scalar wind speed variables derived from the wind components.

    A height is skipped when its components are absent, or when the speed is
    already present and ``overwrite`` is false. ERA5 and ERA5-Land carry no
    100 m components over the Alpine domain, so only ``wind_speed_10m``
    materialises for them; use
    :func:`alpinemet.energy.wind.extrapolate_to_hub_height` if hub-height wind
    is needed there.

    Parameters
    ----------
    dataset
        Dataset with canonical variable names.
    heights
        Height suffixes to process.
    overwrite
        Recompute a speed that is already present. Off by default: a native
        speed field from the product is preferable to one derived here.

    Returns
    -------
    xarray.Dataset
        Dataset with the derived speeds added.
    """
    result = dataset.copy()

    for height in heights:
        speed_name = f"wind_speed_{height}"
        if not overwrite and find_variable(result, speed_name) is not None:
            continue

        u_name = find_variable(result, f"u_wind_{height}")
        v_name = find_variable(result, f"v_wind_{height}")
        if u_name is None or v_name is None:
            continue

        speed = wind_speed_from_components(
            result[u_name], result[v_name], name=speed_name
        )
        speed.attrs["long_name"] = f"{height} wind speed"
        result[speed_name] = speed

    return result


def apply_quality_control(
    dataset: xr.Dataset,
    *,
    ranges: Mapping[str, tuple[float, float]] = PHYSICAL_RANGES,
    action: str = "mask",
) -> xr.Dataset:
    """Flag or mask values outside physically plausible ranges.

    The point is to catch unit errors and corrupt records, not to trim genuine
    extremes -- the bounds are deliberately wide. Note that masking creates
    ``NaN`` holes; every indicator in this package skips missing values, but a
    silently masked field can also hide a real problem, so the count of
    offending values is always recorded.

    Parameters
    ----------
    dataset
        Dataset with canonical variable names.
    ranges
        Mapping of canonical name to ``(minimum, maximum)``.
    action
        ``"mask"`` replaces offending values with ``NaN``; ``"report"`` leaves
        the data untouched and only records the counts.

    Returns
    -------
    xarray.Dataset
        The checked dataset. Per-variable counts of out-of-range values are
        recorded in ``attrs["quality_control"]``.

    Raises
    ------
    ValueError
        If ``action`` is not a supported option.
    """
    if action not in {"mask", "report"}:
        raise ValueError(f"Unknown action {action!r}; expected 'mask' or 'report'")

    result = dataset.copy()
    report: dict[str, int] = {}

    for canonical, (low, high) in ranges.items():
        name = find_variable(result, canonical)
        if name is None:
            continue

        variable = result[name]
        offending = (variable < low) | (variable > high)
        count = int(offending.sum())
        if count == 0:
            continue

        report[canonical] = count
        if action == "mask":
            masked = variable.where(~offending)
            masked.attrs = {
                **variable.attrs,
                "quality_control": f"masked {count} values outside [{low}, {high}]",
            }
            result[name] = masked

    result.attrs = {
        **dataset.attrs,
        "quality_control": as_attribute(report),
        "quality_control_action": action,
    }
    return result
