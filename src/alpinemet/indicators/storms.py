"""Storm hours, storm days, and wind ramps.

Storm classification uses **mean wind speed**, never gusts. The two answer
different questions: sustained mean wind determines turbine cut-out duration
and transmission line loading, while gusts determine peak structural load.
Mixing them inflates counts substantially. For gust-based severity, use
:data:`ESSL_GUST_THRESHOLDS` with a gust field from
:mod:`alpinemet.indicators.gusts`.

Two threshold sets are provided. The standard set follows common operational
practice; the Alpine set lowers each threshold by 15-20 %, following the
paper's reasoning that channelling, gap flows and lee-side acceleration make
complex terrain vulnerable at lower baseline wind speeds.

.. note::

   Ramps here are computed as ``u(t) - u(t - window)``, the change in wind
   speed across the window. Earlier versions of this analysis used
   ``DataArray.diff(dim="time", n=3)``, which applies the difference operator
   three times and returns the *third-order* difference rather than the
   three-hour change -- a different quantity entirely. Ramp figures produced
   with that code are not comparable with the output of this module. No figure
   in the published paper depends on the ramp analysis.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

import numpy as np
import xarray as xr

from alpinemet.attrs import as_attribute

__all__ = [
    "ALPINE_STORM_THRESHOLDS",
    "ESSL_GUST_THRESHOLDS",
    "ESSL_GUST_THRESHOLDS_ALPINE",
    "RAMP_THRESHOLDS",
    "STANDARD_STORM_THRESHOLDS",
    "StormDayReduction",
    "classify_gust_severity",
    "ramp_magnitude",
    "storm_days",
    "storm_hours",
    "wind_ramps",
]

StormDayReduction = Literal["any", "daily_mean"]

#: Storm thresholds on mean wind speed (m/s), standard operational set.
STANDARD_STORM_THRESHOLDS: Mapping[str, float] = MappingProxyType(
    {
        "high_wind": 12.0,
        "strong_wind": 15.0,
        "severe_storm": 17.5,
        "extreme_storm": 20.0,
    }
)

#: Storm thresholds on mean wind speed (m/s), reduced for complex terrain.
ALPINE_STORM_THRESHOLDS: Mapping[str, float] = MappingProxyType(
    {
        "high_wind": 10.0,
        "strong_wind": 12.5,
        "severe_storm": 15.0,
        "extreme_storm": 17.5,
    }
)

#: ESSL severe and extreme wind gust thresholds (m/s).
ESSL_GUST_THRESHOLDS: Mapping[str, float] = MappingProxyType(
    {
        "severe": 25.0,
        "extreme": 32.7,
    }
)

#: ESSL gust thresholds reduced by 20 % for complex terrain.
ESSL_GUST_THRESHOLDS_ALPINE: Mapping[str, float] = MappingProxyType(
    {
        "severe": 20.0,
        "extreme": 26.2,
    }
)

#: Ramp severity thresholds on the wind speed change across the ramp window (m/s).
RAMP_THRESHOLDS: Mapping[str, float] = MappingProxyType(
    {
        "minor": 3.0,
        "moderate": 5.0,
        "severe": 7.0,
        "extreme": 10.0,
    }
)


def _hours_per_step(wind_speed: xr.DataArray, time_dim: str) -> float:
    from alpinemet.indicators.dunkelflaute import infer_timestep_hours

    return infer_timestep_hours(wind_speed, time_dim=time_dim)


def storm_hours(
    wind_speed: xr.DataArray,
    *,
    thresholds: Mapping[str, float] = ALPINE_STORM_THRESHOLDS,
    time_dim: str = "time",
    timestep_hours: float | None = None,
) -> xr.Dataset:
    """Count the time spent above each mean-wind storm threshold.

    Thresholds are cumulative and inclusive: a time step at 18 m/s counts
    towards ``high_wind``, ``strong_wind`` and ``severe_storm`` alike. Report
    the threshold alongside any count.

    Parameters
    ----------
    wind_speed
        Mean wind speed in m/s. Do not pass gusts.
    thresholds
        Mapping of severity name to mean wind threshold in m/s. Defaults to
        :data:`ALPINE_STORM_THRESHOLDS`.
    time_dim
        Name of the time dimension.
    timestep_hours
        Sampling interval in hours. Inferred from the time coordinate when
        omitted; required for sub-hourly or three-hourly data to be counted
        correctly.

    Returns
    -------
    xarray.Dataset
        One ``<name>_hours`` variable per threshold, reduced over time, plus a
        matching ``<name>_fraction`` giving the share of the record.

    Raises
    ------
    ValueError
        If ``wind_speed`` has no time dimension.
    """
    if time_dim not in wind_speed.dims:
        raise ValueError(
            f"Wind field has no {time_dim!r} dimension; got {tuple(wind_speed.dims)}"
        )
    if timestep_hours is None:
        timestep_hours = _hours_per_step(wind_speed, time_dim)

    n_steps = wind_speed.sizes[time_dim]
    result = xr.Dataset()

    for name, threshold in thresholds.items():
        exceeds = wind_speed >= threshold
        steps = exceeds.sum(dim=time_dim)

        result[f"{name}_hours"] = steps * timestep_hours
        result[f"{name}_hours"].attrs = {
            "long_name": f"Hours with mean wind >= {threshold:g} m/s",
            "units": "h",
            "threshold_ms": threshold,
            "variable": "mean wind speed",
        }
        result[f"{name}_fraction"] = steps / n_steps
        result[f"{name}_fraction"].attrs = {
            "long_name": f"Fraction of record with mean wind >= {threshold:g} m/s",
            "units": "1",
            "threshold_ms": threshold,
        }

    result.attrs = {
        "thresholds_ms": as_attribute(dict(thresholds)),
        "timestep_hours": timestep_hours,
        "record_length_steps": n_steps,
        "note": "classification uses mean wind speed, not gusts",
    }
    return result


def storm_days(
    wind_speed: xr.DataArray,
    *,
    threshold: float = ALPINE_STORM_THRESHOLDS["severe_storm"],
    reduction: StormDayReduction = "any",
    time_dim: str = "time",
) -> xr.DataArray:
    """Count days on which a mean-wind threshold is reached.

    Parameters
    ----------
    wind_speed
        Mean wind speed in m/s.
    threshold
        Mean wind threshold in m/s.
    reduction
        ``"any"`` counts a day if any time step reaches the threshold;
        ``"daily_mean"`` counts a day if its daily mean does. ``"any"`` is the
        more common reading of "storm day" and is the default; the two differ
        substantially for short-lived events.
    time_dim
        Name of the time dimension.

    Returns
    -------
    xarray.DataArray
        Number of storm days, reduced over time.

    Raises
    ------
    ValueError
        If ``reduction`` is not a supported option.
    """
    if reduction == "any":
        daily = (wind_speed >= threshold).resample({time_dim: "1D"}).max()
    elif reduction == "daily_mean":
        daily = wind_speed.resample({time_dim: "1D"}).mean() >= threshold
    else:
        raise ValueError(
            f"Unknown reduction {reduction!r}; expected 'any' or 'daily_mean'"
        )

    count = daily.sum(dim=time_dim)
    count = count.rename("storm_days")
    count.attrs = {
        "long_name": f"Days with mean wind >= {threshold:g} m/s",
        "units": "d",
        "threshold_ms": threshold,
        "reduction": reduction,
        "variable": "mean wind speed",
    }
    return count


def ramp_magnitude(
    wind_speed: xr.DataArray,
    *,
    window_hours: float = 3.0,
    time_dim: str = "time",
    timestep_hours: float | None = None,
) -> xr.DataArray:
    """Change in wind speed across a fixed window.

    Computes ``u(t) - u(t - window)``. Positive values are upward ramps.

    Parameters
    ----------
    wind_speed
        Mean wind speed in m/s.
    window_hours
        Ramp window in hours. Must be a whole multiple of the time step.
    time_dim
        Name of the time dimension.
    timestep_hours
        Sampling interval in hours. Inferred when omitted.

    Returns
    -------
    xarray.DataArray
        Wind speed change in m/s, ``NaN`` for the first ``window`` steps.

    Raises
    ------
    ValueError
        If the window is not a whole multiple of the time step.
    """
    if timestep_hours is None:
        timestep_hours = _hours_per_step(wind_speed, time_dim)

    steps = window_hours / timestep_hours
    if not np.isclose(steps, round(steps)) or round(steps) < 1:
        raise ValueError(
            f"Ramp window of {window_hours} h is not a whole multiple of the "
            f"{timestep_hours} h time step"
        )
    steps = int(round(steps))

    change = wind_speed - wind_speed.shift({time_dim: steps})
    change = change.rename("wind_ramp")
    change.attrs = {
        "long_name": f"Wind speed change over {window_hours:g} h",
        "units": "m/s",
        "window_hours": window_hours,
        "window_steps": steps,
        "definition": "u(t) - u(t - window)",
    }
    return change


def wind_ramps(
    wind_speed: xr.DataArray,
    *,
    window_hours: float = 3.0,
    thresholds: Mapping[str, float] = RAMP_THRESHOLDS,
    time_dim: str = "time",
    timestep_hours: float | None = None,
) -> xr.Dataset:
    """Count upward and downward wind ramps by severity class.

    Classes are cumulative: a 12 m/s change counts as minor, moderate, severe
    and extreme alike.

    .. important::

       These are counts of **time steps** whose window change exceeds the
       threshold, not counts of distinct ramp events. A single step change of
       12 m/s viewed through a three-hour window produces three consecutive
       exceeding steps, and is counted three times. Divide by the number of
       window steps for a rough event count, or use
       :func:`alpinemet.indicators.dunkelflaute.event_statistics` on a
       thresholded :func:`ramp_magnitude` field for a proper one.

    Parameters
    ----------
    wind_speed
        Mean wind speed in m/s.
    window_hours
        Ramp window in hours.
    thresholds
        Mapping of severity name to ramp magnitude in m/s.
    time_dim
        Name of the time dimension.
    timestep_hours
        Sampling interval in hours. Inferred when omitted.

    Returns
    -------
    xarray.Dataset
        ``<name>_up_count`` and ``<name>_down_count`` per severity class,
        reduced over time, alongside ``max_ramp_up`` and ``max_ramp_down``.
    """
    change = ramp_magnitude(
        wind_speed,
        window_hours=window_hours,
        time_dim=time_dim,
        timestep_hours=timestep_hours,
    )

    result = xr.Dataset()
    for name, threshold in thresholds.items():
        up = (change >= threshold).sum(dim=time_dim)
        down = (change <= -threshold).sum(dim=time_dim)

        result[f"{name}_up_count"] = up
        result[f"{name}_up_count"].attrs = {
            "long_name": f"Upward ramps >= {threshold:g} m/s per {window_hours:g} h",
            "units": "1",
            "threshold_ms": threshold,
        }
        result[f"{name}_down_count"] = down
        result[f"{name}_down_count"].attrs = {
            "long_name": f"Downward ramps <= -{threshold:g} m/s per {window_hours:g} h",
            "units": "1",
            "threshold_ms": threshold,
        }

    result["max_ramp_up"] = change.max(dim=time_dim)
    result["max_ramp_up"].attrs = {"long_name": "Largest upward ramp", "units": "m/s"}
    result["max_ramp_down"] = change.min(dim=time_dim)
    result["max_ramp_down"].attrs = {"long_name": "Largest downward ramp", "units": "m/s"}

    result.attrs = {
        "window_hours": window_hours,
        "thresholds_ms": as_attribute(dict(thresholds)),
        "definition": "u(t) - u(t - window)",
    }
    return result


def classify_gust_severity(
    gust: xr.DataArray,
    *,
    thresholds: Mapping[str, float] = ESSL_GUST_THRESHOLDS,
    time_dim: str = "time",
    timestep_hours: float | None = None,
) -> xr.Dataset:
    """Count time spent above each ESSL gust threshold.

    Unlike :func:`storm_hours`, this operates on a **gust** field. Pass a native
    ``i10fg`` field where available; see :mod:`alpinemet.indicators.gusts` for
    why estimated and native gusts are not interchangeable at these thresholds.

    Parameters
    ----------
    gust
        Wind gust speed in m/s.
    thresholds
        Mapping of severity name to gust threshold in m/s. Defaults to
        :data:`ESSL_GUST_THRESHOLDS`.
    time_dim
        Name of the time dimension.
    timestep_hours
        Sampling interval in hours. Inferred when omitted.

    Returns
    -------
    xarray.Dataset
        ``<name>_hours`` per threshold plus ``max_gust``.
    """
    if timestep_hours is None:
        timestep_hours = _hours_per_step(gust, time_dim)

    result = xr.Dataset()
    for name, threshold in thresholds.items():
        steps = (gust >= threshold).sum(dim=time_dim)
        result[f"{name}_hours"] = steps * timestep_hours
        result[f"{name}_hours"].attrs = {
            "long_name": f"Hours with gusts >= {threshold:g} m/s",
            "units": "h",
            "threshold_ms": threshold,
        }

    result["max_gust"] = gust.max(dim=time_dim)
    result["max_gust"].attrs = {"long_name": "Maximum gust", "units": "m/s"}
    result.attrs = {
        "thresholds_ms": as_attribute(dict(thresholds)),
        "timestep_hours": timestep_hours,
        "gust_source": gust.attrs.get("gust_source", "unknown"),
    }
    return result
