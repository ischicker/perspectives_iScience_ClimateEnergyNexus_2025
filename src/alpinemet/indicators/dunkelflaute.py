"""Dunkelflaute detection: sustained simultaneous wind and solar shortfall.

The reference definition follows the paper: a Dunkelflaute episode is a period
in which the wind capacity factor stays below 10 % **and** the solar capacity
factor stays below 5 %, sustained for at least 48 hours. Cold Dunkelflaute
additionally requires a mean temperature below 0 degC, when low generation
coincides with peak heating demand.

Two entry points reflect two ways of posing the question:

:func:`detect`
    Capacity-factor thresholds -- the reference method. Asks about the *energy
    system*: how far below nameplate is generation? Requires capacity factors
    from :mod:`alpinemet.energy`, so it inherits the power curve and PV system
    assumptions made there.

:func:`detect_from_raw`
    Meteorological thresholds on wind speed and irradiance directly. Asks about
    the *weather*: is it simultaneously calm and dull? Independent of any
    conversion assumption, which makes it useful for cross-dataset comparison,
    but the thresholds do not map cleanly onto system stress.

.. warning::

   Detected frequency and duration are highly sensitive to the thresholds. As
   the paper notes, varying the wind capacity factor threshold between 10 % and
   20 % can change event frequency by a factor of 3 to 5. The defaults here are
   illustrative; adapt them to the system configuration and risk tolerance at
   hand, and always report the thresholds alongside any count.

A note on spatial aggregation specific to complex terrain: during persistent
inversions, valley-floor installations can be in energy drought while ridge-top
turbines above the inversion run near rated capacity. Whether that constitutes
a system-level Dunkelflaute depends entirely on the aggregation scale. These
functions operate grid-point-wise; aggregate deliberately, and state the scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "COLD_DUNKELFLAUTE_TEMPERATURE",
    "DUNKELFLAUTE_MIN_DURATION_HOURS",
    "DUNKELFLAUTE_SOLAR_CF_THRESHOLD",
    "DUNKELFLAUTE_WIND_CF_THRESHOLD",
    "RAW_SOLAR_THRESHOLD_WM2",
    "RAW_WIND_THRESHOLD_MS",
    "detect",
    "detect_from_raw",
    "event_statistics",
    "infer_timestep_hours",
    "sustained",
]

#: Wind capacity factor below which generation counts as a shortfall.
DUNKELFLAUTE_WIND_CF_THRESHOLD = 0.10

#: Solar capacity factor below which generation counts as a shortfall.
DUNKELFLAUTE_SOLAR_CF_THRESHOLD = 0.05

#: Minimum duration for a shortfall to count as an episode, in hours.
DUNKELFLAUTE_MIN_DURATION_HOURS = 48

#: Mean temperature below which a Dunkelflaute is classified as "cold", in degC.
COLD_DUNKELFLAUTE_TEMPERATURE = 0.0

#: Meteorological wind threshold for the raw variant, in m/s.
RAW_WIND_THRESHOLD_MS = 3.0

#: Meteorological irradiance threshold for the raw variant, in W/m2.
RAW_SOLAR_THRESHOLD_WM2 = 100.0


def infer_timestep_hours(data: xr.DataArray | xr.Dataset, *, time_dim: str = "time") -> float:
    """Infer the time step of a regularly sampled series, in hours.

    Parameters
    ----------
    data
        Object with a datetime coordinate along ``time_dim``.
    time_dim
        Name of the time dimension.

    Returns
    -------
    float
        Time step in hours.

    Raises
    ------
    ValueError
        If the time coordinate is missing, has fewer than two steps, or is
        irregularly spaced. Irregular sampling is rejected rather than averaged
        because duration thresholds would silently become meaningless.
    """
    if time_dim not in data.coords:
        raise ValueError(f"No {time_dim!r} coordinate; cannot infer the time step")

    times = pd.DatetimeIndex(np.asarray(data[time_dim].values))
    if times.size < 2:
        raise ValueError(
            f"Need at least two time steps to infer the sampling interval; got {times.size}"
        )

    deltas = np.diff(times.values).astype("timedelta64[s]").astype(float)
    if not np.allclose(deltas, deltas[0]):
        unique = np.unique(deltas)
        raise ValueError(
            "Time coordinate is irregularly spaced, so a duration threshold cannot be "
            f"applied reliably; found spacings of {unique / 3600} hours"
        )
    return float(deltas[0] / 3600.0)


def _sustained_runs_1d(mask: np.ndarray, min_steps: int) -> np.ndarray:
    """Keep only runs of ``True`` at least ``min_steps`` long."""
    out = np.zeros(mask.shape, dtype=bool)
    if min_steps <= 1:
        return mask.astype(bool)

    flags = np.asarray(mask, dtype=bool)
    padded = np.concatenate(([False], flags, [False]))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)

    for start, end in zip(starts, ends, strict=True):
        if end - start >= min_steps:
            out[start:end] = True
    return out


def _run_lengths_1d(mask: np.ndarray) -> np.ndarray:
    """Lengths of every run of ``True`` in a 1-D boolean array."""
    flags = np.asarray(mask, dtype=bool)
    padded = np.concatenate(([False], flags, [False]))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return ends - starts


def _count_runs_1d(mask: np.ndarray) -> np.int64:
    return np.int64(_run_lengths_1d(mask).size)


def _longest_run_1d(mask: np.ndarray) -> np.int64:
    lengths = _run_lengths_1d(mask)
    return np.int64(lengths.max()) if lengths.size else np.int64(0)


def sustained(
    condition: xr.DataArray,
    *,
    min_duration_hours: float = DUNKELFLAUTE_MIN_DURATION_HOURS,
    time_dim: str = "time",
    timestep_hours: float | None = None,
) -> xr.DataArray:
    """Retain only those parts of a boolean condition that persist long enough.

    A time step belongs to the result only if it lies inside an unbroken run of
    ``True`` values spanning at least ``min_duration_hours``. Isolated hours are
    removed entirely -- this is the step that turns an instantaneous condition
    into an episode.

    Parameters
    ----------
    condition
        Boolean field with a time dimension.
    min_duration_hours
        Minimum episode duration in hours.
    time_dim
        Name of the time dimension.
    timestep_hours
        Sampling interval in hours. Inferred from the time coordinate when
        omitted.

    Returns
    -------
    xarray.DataArray
        Boolean field, ``True`` only within sufficiently long runs.
    """
    if timestep_hours is None:
        timestep_hours = infer_timestep_hours(condition, time_dim=time_dim)

    min_steps = int(np.ceil(min_duration_hours / timestep_hours))

    result = xr.apply_ufunc(
        _sustained_runs_1d,
        condition,
        kwargs={"min_steps": min_steps},
        input_core_dims=[[time_dim]],
        output_core_dims=[[time_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[bool],
    ).transpose(*condition.dims)

    result.attrs = {
        "long_name": "Sustained condition",
        "min_duration_hours": min_duration_hours,
        "timestep_hours": timestep_hours,
        "min_consecutive_steps": min_steps,
    }
    return result


def event_statistics(
    episode_mask: xr.DataArray,
    *,
    time_dim: str = "time",
    timestep_hours: float | None = None,
) -> xr.Dataset:
    """Summarise an episode mask into per-grid-point event statistics.

    Parameters
    ----------
    episode_mask
        Boolean field as returned by :func:`sustained`.
    time_dim
        Name of the time dimension.
    timestep_hours
        Sampling interval in hours. Inferred when omitted.

    Returns
    -------
    xarray.Dataset
        Variables ``event_count``, ``total_hours``, ``frequency`` (fraction of
        the record spent in an episode) and ``longest_event_hours``, each
        reduced over the time dimension.
    """
    if timestep_hours is None:
        timestep_hours = infer_timestep_hours(episode_mask, time_dim=time_dim)

    n_steps = episode_mask.sizes[time_dim]

    counts = xr.apply_ufunc(
        _count_runs_1d,
        episode_mask,
        input_core_dims=[[time_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int64],
    )
    longest = xr.apply_ufunc(
        _longest_run_1d,
        episode_mask,
        input_core_dims=[[time_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int64],
    )
    total_steps = episode_mask.sum(dim=time_dim)

    stats = xr.Dataset(
        {
            "event_count": counts,
            "total_hours": total_steps * timestep_hours,
            "frequency": total_steps / n_steps,
            "longest_event_hours": longest * timestep_hours,
        }
    )
    stats["event_count"].attrs = {"long_name": "Number of episodes", "units": "1"}
    stats["total_hours"].attrs = {"long_name": "Total time in episodes", "units": "h"}
    stats["frequency"].attrs = {"long_name": "Fraction of record in episodes", "units": "1"}
    stats["longest_event_hours"].attrs = {"long_name": "Longest episode", "units": "h"}
    stats.attrs = {"timestep_hours": timestep_hours, "record_length_steps": n_steps}
    return stats


def _assemble(
    shortfall: xr.DataArray,
    temperature: xr.DataArray | None,
    cold_threshold: float,
    min_duration_hours: float,
    time_dim: str,
    timestep_hours: float | None,
    provenance: dict[str, object],
) -> xr.Dataset:
    if timestep_hours is None:
        timestep_hours = infer_timestep_hours(shortfall, time_dim=time_dim)

    episodes = sustained(
        shortfall,
        min_duration_hours=min_duration_hours,
        time_dim=time_dim,
        timestep_hours=timestep_hours,
    )

    result = xr.Dataset({"shortfall": shortfall, "dunkelflaute": episodes})
    stats = event_statistics(episodes, time_dim=time_dim, timestep_hours=timestep_hours)
    for name, values in stats.data_vars.items():
        result[name] = values

    if temperature is not None:
        cold = episodes & (temperature < cold_threshold)
        result["cold_dunkelflaute"] = cold
        cold_stats = event_statistics(cold, time_dim=time_dim, timestep_hours=timestep_hours)
        for name, values in cold_stats.data_vars.items():
            result[f"cold_{name}"] = values
        result["cold_dunkelflaute"].attrs = {
            "long_name": "Cold Dunkelflaute episode",
            "cold_threshold_degC": cold_threshold,
        }

    result["dunkelflaute"].attrs = {
        "long_name": "Dunkelflaute episode",
        "min_duration_hours": min_duration_hours,
    }
    result.attrs = {
        "min_duration_hours": min_duration_hours,
        "timestep_hours": timestep_hours,
        **provenance,
    }
    return result


def detect(
    wind_capacity_factor: xr.DataArray,
    solar_capacity_factor: xr.DataArray,
    *,
    temperature: xr.DataArray | None = None,
    wind_threshold: float = DUNKELFLAUTE_WIND_CF_THRESHOLD,
    solar_threshold: float = DUNKELFLAUTE_SOLAR_CF_THRESHOLD,
    cold_threshold: float = COLD_DUNKELFLAUTE_TEMPERATURE,
    min_duration_hours: float = DUNKELFLAUTE_MIN_DURATION_HOURS,
    time_dim: str = "time",
    timestep_hours: float | None = None,
) -> xr.Dataset:
    """Detect Dunkelflaute episodes from capacity factors (reference method).

    Parameters
    ----------
    wind_capacity_factor, solar_capacity_factor
        Dimensionless capacity factors, e.g. from
        :func:`alpinemet.energy.wind.capacity_factor` and
        :func:`alpinemet.energy.solar.fixed_tilt_capacity_factor`.
    temperature
        Temperature in degC. When given, Cold Dunkelflaute is classified too.
    wind_threshold, solar_threshold
        Capacity factor thresholds. Defaults are 0.10 and 0.05.
    cold_threshold
        Temperature below which an episode counts as cold, in degC.
    min_duration_hours
        Minimum episode duration. Default is 48 hours.
    time_dim
        Name of the time dimension.
    timestep_hours
        Sampling interval in hours. Inferred when omitted.

    Returns
    -------
    xarray.Dataset
        ``shortfall`` (instantaneous condition), ``dunkelflaute`` (sustained
        episodes), per-grid-point statistics, and the cold variants when a
        temperature field is supplied.
    """
    shortfall = (wind_capacity_factor < wind_threshold) & (
        solar_capacity_factor < solar_threshold
    )
    shortfall = shortfall.rename("shortfall")
    shortfall.attrs = {
        "long_name": "Simultaneous wind and solar shortfall",
        "wind_cf_threshold": wind_threshold,
        "solar_cf_threshold": solar_threshold,
    }

    return _assemble(
        shortfall,
        temperature,
        cold_threshold,
        min_duration_hours,
        time_dim,
        timestep_hours,
        {
            "method": "capacity factor thresholds",
            "wind_cf_threshold": wind_threshold,
            "solar_cf_threshold": solar_threshold,
        },
    )


def detect_from_raw(
    wind_speed: xr.DataArray,
    irradiance: xr.DataArray,
    *,
    temperature: xr.DataArray | None = None,
    wind_threshold: float = RAW_WIND_THRESHOLD_MS,
    solar_threshold: float = RAW_SOLAR_THRESHOLD_WM2,
    cold_threshold: float = COLD_DUNKELFLAUTE_TEMPERATURE,
    min_duration_hours: float = DUNKELFLAUTE_MIN_DURATION_HOURS,
    time_dim: str = "time",
    timestep_hours: float | None = None,
) -> xr.Dataset:
    """Detect Dunkelflaute episodes from meteorological thresholds.

    The alternative to :func:`detect`: applies thresholds to wind speed and
    irradiance directly, with no energy conversion in between. Useful when
    comparing datasets without committing to a turbine or PV specification.

    Parameters
    ----------
    wind_speed
        Wind speed in m/s.
    irradiance
        Global horizontal irradiance in W/m2.
    temperature
        Temperature in degC, for the cold classification.
    wind_threshold, solar_threshold
        Meteorological thresholds; defaults are 3.0 m/s and 100 W/m2.
    cold_threshold, min_duration_hours, time_dim, timestep_hours
        As for :func:`detect`.

    Returns
    -------
    xarray.Dataset
        Same structure as :func:`detect`.
    """
    shortfall = (wind_speed < wind_threshold) & (irradiance < solar_threshold)
    shortfall = shortfall.rename("shortfall")
    shortfall.attrs = {
        "long_name": "Simultaneous low wind and low irradiance",
        "wind_threshold_ms": wind_threshold,
        "solar_threshold_wm2": solar_threshold,
    }

    return _assemble(
        shortfall,
        temperature,
        cold_threshold,
        min_duration_hours,
        time_dim,
        timestep_hours,
        {
            "method": "meteorological thresholds",
            "wind_threshold_ms": wind_threshold,
            "solar_threshold_wm2": solar_threshold,
        },
    )
