"""Accumulated fluxes to instantaneous rates.

Radiation and precipitation reach us in three different conventions, and
getting the wrong one wrong by a factor of 3600 is the single easiest way to
produce plausible-looking nonsense:

:attr:`AccumulationKind.INSTANTANEOUS`
    Already a rate (W/m2, mm/h). Nothing to do. AIFS and some post-processed
    products.

:attr:`AccumulationKind.PERIOD`
    Accumulated over exactly one time step and reset each step, in J/m2 or mm.
    Divide by the step length. ERA5 hourly ``ssrd`` behaves this way.

:attr:`AccumulationKind.RUNNING`
    Accumulated from a fixed reference time and reset periodically -- ERA5-Land
    resets at 00 UTC each day. Difference along time, then divide by the step
    length, and discard the negative jump at each reset.

.. important::

   The kind is a **property of the product**, declared per dataset in
   :mod:`alpinemet.io.datasets`, not something to guess from the numbers. An
   earlier implementation inferred it from value ranges and silently skipped
   the conversion when the sample fell between the expected bands, leaving
   J/m2 in a field labelled W/m2. :func:`infer_accumulation_kind` is available
   for exploratory work, but it raises on ambiguity rather than passing the
   data through unchanged.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd
import xarray as xr

from alpinemet.attrs import as_attribute

__all__ = [
    "AccumulationKind",
    "SECONDS_PER_HOUR",
    "infer_accumulation_kind",
    "to_rate",
]

SECONDS_PER_HOUR = 3600.0

# Used only by the inference fallback. Downwelling shortwave never exceeds the
# solar constant, so a mean in this range cannot be an accumulation.
_INSTANTANEOUS_MAX = 1_500.0

# Share of *decreasing* steps among the non-constant ones. A running accumulator
# decreases only at its resets (about one step in 23 for hourly data with a daily
# reset); a period-accumulated diurnal cycle falls about as often as it rises.
# Zero differences are excluded: radiation is flat through the night, and counting
# those as "not decreasing" would make every diurnal cycle look monotonic.
_RUNNING_MAX_DECREASING = 0.15
_PERIOD_MIN_DECREASING = 0.35


class AccumulationKind(str, Enum):
    """How a flux variable is accumulated in its source product."""

    INSTANTANEOUS = "instantaneous"
    PERIOD = "period"
    RUNNING = "running"


def _timestep_seconds(data: xr.DataArray, time_dim: str) -> float:
    times = pd.DatetimeIndex(np.asarray(data[time_dim].values))
    if times.size < 2:
        raise ValueError(
            "Need at least two time steps to determine the accumulation interval"
        )
    deltas = np.diff(times.values).astype("timedelta64[s]").astype(float)
    return float(np.median(deltas))


def infer_accumulation_kind(
    data: xr.DataArray,
    *,
    time_dim: str = "time",
    sample_steps: int = 24,
) -> AccumulationKind:
    """Guess the accumulation convention from the shape of the series.

    The magnitude only settles the instantaneous case: downwelling shortwave
    cannot exceed the solar constant, so a mean below
    :data:`_INSTANTANEOUS_MAX` is already a rate. Distinguishing period from
    running accumulation then uses **how often the series decreases**, not
    magnitude: a running accumulator decreases only at its resets, whereas a
    period-accumulated diurnal cycle falls about as often as it rises. Steps
    where the value does not change are ignored, since radiation is flat
    through the night in either convention.

    Magnitude bands cannot make this distinction reliably. An earlier
    implementation classified period accumulation as 100,000 to 500,000 J/m2,
    a band whose upper bound corresponds to a daily mean of only 139 W/m2 --
    a clear Alpine summer day exceeds it and would have fallen through to the
    "unknown" branch and been left unconverted.

    A fallback for exploring an unfamiliar product. Prefer declaring the kind
    on the dataset specification.

    Parameters
    ----------
    data
        Flux variable to inspect.
    time_dim
        Name of the time dimension.
    sample_steps
        Number of leading time steps to inspect. One full diurnal cycle is the
        useful minimum for radiation.

    Returns
    -------
    AccumulationKind
        The inferred convention.

    Raises
    ------
    ValueError
        If the sample is too short to judge, or if the monotonic fraction falls
        between the two bounds. Passing the data through unconverted would be
        worse than failing here.
    """
    sample = data.isel({time_dim: slice(0, sample_steps)})
    values = np.asarray(sample.values, dtype=float)
    mean = float(np.nanmean(values))

    if mean < _INSTANTANEOUS_MAX:
        return AccumulationKind.INSTANTANEOUS

    # Reduce to a single series so the monotonicity test is well defined even
    # for gridded input.
    spatial_dims = [dim for dim in sample.dims if dim != time_dim]
    series = np.asarray(
        sample.mean(dim=spatial_dims).values if spatial_dims else values, dtype=float
    )
    if series.size < 3:
        raise ValueError(
            f"Need at least three time steps to judge the accumulation convention; "
            f"got {series.size}"
        )

    differences = np.diff(series)
    varying = differences[differences != 0]
    if varying.size == 0:
        raise ValueError(
            "Cannot infer the accumulation convention: the sample is constant in time"
        )

    decreasing_fraction = float(np.mean(varying < 0))

    if decreasing_fraction <= _RUNNING_MAX_DECREASING:
        return AccumulationKind.RUNNING
    if decreasing_fraction >= _PERIOD_MIN_DECREASING:
        return AccumulationKind.PERIOD

    raise ValueError(
        f"Cannot infer the accumulation convention: {decreasing_fraction:.0%} of the "
        f"non-constant step-to-step differences are decreasing, between the bounds for "
        f"running (<= {_RUNNING_MAX_DECREASING:.0%}) and period "
        f"(>= {_PERIOD_MIN_DECREASING:.0%}) accumulation. Declare the kind explicitly "
        "on the dataset specification."
    )


def to_rate(
    data: xr.DataArray,
    kind: AccumulationKind | str,
    *,
    time_dim: str = "time",
    timestep_seconds: float | None = None,
    per_second: bool = True,
    reset_hours: float | None = None,
    reset_offset: int = 0,
    output_units: str | None = None,
    clip_negative: bool = True,
) -> xr.DataArray:
    """Undo accumulation, and optionally express the result per second.

    Two separate operations are involved, and not every variable needs both:

    **Deaccumulation** turns a running accumulator into per-step increments.
    Needed for :attr:`AccumulationKind.RUNNING` only.

    **Rate conversion** divides by the step length. Needed only where the
    target unit is per-second: radiation goes from J/m2 per step to W/m2.
    Precipitation does *not* — an hourly total in millimetres is already the
    quantity wanted, and dividing it by 3600 yields mm/s, three orders of
    magnitude too small and entirely plausible-looking.

    Parameters
    ----------
    data
        Flux variable in its source convention.
    kind
        The source accumulation convention.
    time_dim
        Name of the time dimension.
    timestep_seconds
        Accumulation interval in seconds. Inferred from the time coordinate
        when omitted.
    per_second
        Divide by the step length. True for energy fluxes such as radiation,
        false for per-step totals such as precipitation.
    reset_hours
        Length of the accumulation block for :attr:`AccumulationKind.RUNNING`,
        in hours -- 24 for a daily reset, 3 for ARA radiation. When given, block
        starts are identified from the clock rather than from a drop in value.
        That matters: a new block often starts *above* where the previous one
        ended (radiation through the morning), so a sign test alone misses the
        reset and doubles that step. Falls back to sign detection when omitted.
    reset_offset
        Hour at which the first block of the day begins, modulo
        reset_hours. ARA blocks start at 01, 04, 07 UTC and so on, so its
        offset is 1.
    output_units
        Units string recorded on the result, e.g. ``"W m-2"``.
    clip_negative
        Clip negative values to zero. For :attr:`AccumulationKind.RUNNING` this
        is what removes the large negative jump at each accumulator reset, and
        should stay enabled. Note the consequence: the step containing a reset
        reports zero rather than its true value. For ERA5-Land radiation the
        reset falls at 00-01 UTC, when Alpine irradiance is zero anyway, so
        nothing is lost -- but the same is not true of precipitation.

    Returns
    -------
    xarray.DataArray
        Instantaneous rate, annotated with the conversion applied.

    Raises
    ------
    ValueError
        If ``kind`` is not a recognised convention, or if the data are already
        marked as converted.
    """
    if data.attrs.get("accumulation_converted") == "true":
        raise ValueError(
            f"{data.name!r} is already marked as converted; converting twice would "
            "divide by the time step again"
        )

    kind = AccumulationKind(kind)

    if kind is AccumulationKind.INSTANTANEOUS:
        converted = data.copy()
        method = "none"
    else:
        if timestep_seconds is None:
            timestep_seconds = _timestep_seconds(data, time_dim)
        divisor = timestep_seconds if per_second else 1.0

        if kind is AccumulationKind.PERIOD:
            converted = data / divisor
            method = "divide by time step" if per_second else "already per step"
        else:  # RUNNING
            differenced = data.diff(time_dim)
            # diff drops the first step; it has no predecessor, so its own value
            # is the increment.
            first = data.isel({time_dim: 0})
            increments = xr.concat([first, differenced], dim=time_dim)
            increments = increments.assign_coords({time_dim: data[time_dim]})

            # At a reset the accumulator restarts, so the difference is the new
            # value minus the old total: large and negative. The increment for
            # that step is simply the raw value. Detecting the reset this way
            # rather than clipping it to zero keeps the step instead of losing
            # it -- which matters when the reset period is short. ARA resets its
            # radiation accumulator every three hours, so clipping would discard
            # one hour in three.
            if reset_hours is None:
                # No declared block length: fall back to spotting the drop.
                reset = increments < 0
            else:
                hours = data[time_dim].dt.hour
                reset = ((hours - reset_offset) % int(reset_hours)) == 0
            increments = xr.where(reset, data, increments)

            converted = increments / divisor
            method = (
                "deaccumulate with reset detection, then divide by time step"
                if per_second
                else "deaccumulate with reset detection"
            )

        if clip_negative:
            converted = converted.where(converted >= 0, 0.0)

    converted = converted.rename(data.name)
    converted.attrs = {
        **data.attrs,
        "accumulation_converted": "true",
        "accumulation_kind": kind.value,
        "conversion_method": method,
    }
    if kind is not AccumulationKind.INSTANTANEOUS:
        converted.attrs["timestep_seconds"] = timestep_seconds
        converted.attrs["per_second"] = as_attribute(per_second)
    if output_units is not None:
        converted.attrs["units"] = output_units
    return converted
