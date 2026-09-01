"""Counting days and hours above or below a threshold.

The simplest family of indicators, and the one where the daily reduction is
easy to leave unstated. "Days above 30 °C" almost always means days whose
*maximum* reaches 30 °C, but the same phrase is sometimes used for days whose
*mean* does, and the two counts differ by a wide margin. Both are available
here, and the choice is recorded in the output.
"""

from __future__ import annotations

from typing import Literal

import xarray as xr

from alpinemet.units import to_celsius

__all__ = [
    "DailyReduction",
    "HOT_DAY_THRESHOLD",
    "exceedance_days",
    "exceedance_hours",
    "hot_days",
]

DailyReduction = Literal["max", "min", "mean"]

#: Daily maximum temperature defining a hot day in this study, in degC.
HOT_DAY_THRESHOLD = 30.0


def exceedance_days(
    field: xr.DataArray,
    threshold: float,
    *,
    reduction: DailyReduction = "max",
    below: bool = False,
    time_dim: str = "time",
) -> xr.DataArray:
    """Count days on which a daily statistic crosses a threshold.

    Parameters
    ----------
    field
        Field with a time dimension, in the same units as ``threshold``.
    threshold
        The threshold to compare against.
    reduction
        Daily statistic to test: ``"max"``, ``"min"`` or ``"mean"``.
    below
        Count days *below* the threshold instead of at or above it.
    time_dim
        Name of the time dimension.

    Returns
    -------
    xarray.DataArray
        Day counts, reduced over time.

    Raises
    ------
    ValueError
        If ``reduction`` is not supported, or the field has no time dimension.
    """
    if time_dim not in field.dims:
        raise ValueError(f"Field has no {time_dim!r} dimension; got {tuple(field.dims)}")

    resampled = field.resample({time_dim: "1D"})
    if reduction == "max":
        daily = resampled.max()
    elif reduction == "min":
        daily = resampled.min()
    elif reduction == "mean":
        daily = resampled.mean()
    else:
        raise ValueError(
            f"Unknown daily reduction {reduction!r}; expected 'max', 'min' or 'mean'"
        )

    exceeds = (daily < threshold) if below else (daily >= threshold)
    count = exceeds.sum(dim=time_dim)

    comparison = "<" if below else ">="
    count = count.rename("exceedance_days")
    count.attrs = {
        "long_name": f"Days with daily {reduction} {comparison} {threshold:g}",
        "units": "d",
        "threshold": threshold,
        "daily_reduction": reduction,
        "comparison": comparison,
    }
    return count


def exceedance_hours(
    field: xr.DataArray,
    threshold: float,
    *,
    below: bool = False,
    time_dim: str = "time",
    timestep_hours: float | None = None,
) -> xr.DataArray:
    """Count the time spent across a threshold, in hours.

    Parameters
    ----------
    field
        Field with a time dimension.
    threshold
        The threshold to compare against.
    below
        Count time *below* the threshold instead of at or above it.
    time_dim
        Name of the time dimension.
    timestep_hours
        Sampling interval in hours. Inferred when omitted.

    Returns
    -------
    xarray.DataArray
        Hours, reduced over time.
    """
    if timestep_hours is None:
        from alpinemet.indicators.dunkelflaute import infer_timestep_hours

        timestep_hours = infer_timestep_hours(field, time_dim=time_dim)

    exceeds = (field < threshold) if below else (field >= threshold)
    hours = exceeds.sum(dim=time_dim) * timestep_hours

    comparison = "<" if below else ">="
    hours = hours.rename("exceedance_hours")
    hours.attrs = {
        "long_name": f"Hours with values {comparison} {threshold:g}",
        "units": "h",
        "threshold": threshold,
        "comparison": comparison,
        "timestep_hours": timestep_hours,
    }
    return hours


def hot_days(
    temperature: xr.DataArray,
    *,
    threshold: float = HOT_DAY_THRESHOLD,
    time_dim: str = "time",
    assume_units: str | None = None,
) -> xr.DataArray:
    """Count days whose maximum temperature reaches ``threshold``.

    Kelvin input is converted, so the threshold is always read in degrees
    Celsius.

    Parameters
    ----------
    temperature
        Temperature field.
    threshold
        Daily maximum temperature defining a hot day, in degC.
    time_dim
        Name of the time dimension.
    assume_units
        Passed to :func:`alpinemet.units.to_celsius`.

    Returns
    -------
    xarray.DataArray
        Number of hot days.
    """
    celsius = to_celsius(temperature, assume=assume_units)
    count = exceedance_days(celsius, threshold, reduction="max", time_dim=time_dim)
    count = count.rename("hot_days")
    count.attrs = {
        **count.attrs,
        "long_name": f"Days with daily maximum temperature >= {threshold:g} degC",
    }
    return count
