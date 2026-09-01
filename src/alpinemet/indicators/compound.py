"""Compound extreme events for energy systems.

Compound events are simultaneous occurrences of several hazards whose combined
impact exceeds the sum of the individual drivers. They are the hardest class of
event for gridded datasets to represent, because they require several variables
to be right *at the same time and place*, and their biases compound rather than
cancel.

The events implemented here follow the typology of Zscheischler et al. (2020):

``hellsturm``
    High wind together with high irradiance. Simultaneous overproduction from
    both sources, a grid-stress and curtailment condition rather than a
    shortfall.

``heat_drought``
    Sustained heat with negligible precipitation. The 2022 European
    drought-heatwave, which cut Alpine hydropower to 25-year lows, is the
    reference case.

``rain_on_snow``
    Precipitation falling at temperatures just above freezing onto lying snow.
    Drives rapid runoff, flood risk and reservoir management stress.

``foehn_like``
    Strong wind together with rapid warming -- a crude proxy, not a föhn
    detection. Real föhn identification requires upstream and lee-side
    analysis; as the paper notes, the underlying dynamics challenge even
    convection-permitting models. Treat the output as a screening indicator.

Cold Dunkelflaute lives in :mod:`alpinemet.indicators.dunkelflaute`, since it is
a Dunkelflaute with an added temperature criterion rather than an independent
event class.

Aggregation scale
-----------------

All detectors work **grid-point-wise** by default. An earlier implementation
averaged over the domain first, which answers a different question: whether the
*regional mean* is extreme, not whether any location is. In complex terrain the
two diverge sharply -- valley floors and ridges can be in opposite states
simultaneously. Pass ``aggregate=True`` for the domain-mean behaviour, and state
which one you used.

References
----------
Zscheischler, J., et al. (2020). A typology of compound weather and climate
events. *Nature Reviews Earth and Environment*, 1, 333-347.
"""

from __future__ import annotations

import xarray as xr

from alpinemet.indicators.dunkelflaute import event_statistics, sustained

__all__ = [
    "FOEHN_WARMING_K",
    "FOEHN_WIND_MS",
    "HEAT_DROUGHT_PRECIP_MM",
    "HEAT_DROUGHT_TEMPERATURE",
    "HELLSTURM_SOLAR_WM2",
    "HELLSTURM_WIND_MS",
    "RAIN_ON_SNOW_PRECIP_MM",
    "RAIN_ON_SNOW_TEMPERATURE_RANGE",
    "combine_conditions",
    "foehn_like",
    "heat_drought",
    "hellsturm",
    "rain_on_snow",
]

#: Wind speed above which Hellsturm conditions may occur, in m/s.
HELLSTURM_WIND_MS = 12.0

#: Irradiance above which Hellsturm conditions may occur, in W/m2.
HELLSTURM_SOLAR_WM2 = 500.0

#: Daily mean temperature defining the heat component of heat-drought, in degC.
HEAT_DROUGHT_TEMPERATURE = 25.0

#: Daily precipitation below which a day counts as dry, in mm.
HEAT_DROUGHT_PRECIP_MM = 1.0

#: Temperature window in which precipitation falls as rain onto snow, in degC.
RAIN_ON_SNOW_TEMPERATURE_RANGE = (0.0, 5.0)

#: Precipitation rate defining a rain-on-snow event, in mm per time step.
RAIN_ON_SNOW_PRECIP_MM = 2.0

#: Wind speed component of the föhn proxy, in m/s.
FOEHN_WIND_MS = 10.0

#: Warming across the föhn window, in kelvin.
FOEHN_WARMING_K = 5.0


def _reduce_spatially(
    field: xr.DataArray, aggregate: bool, lat_name: str, lon_name: str
) -> xr.DataArray:
    if not aggregate:
        return field
    dims = [dim for dim in (lat_name, lon_name) if dim in field.dims]
    return field.mean(dim=dims) if dims else field


def combine_conditions(
    name: str,
    *,
    min_duration_hours: float | None = None,
    time_dim: str = "time",
    timestep_hours: float | None = None,
    **conditions: xr.DataArray,
) -> xr.Dataset:
    """Combine named boolean conditions into a compound event mask.

    All conditions must hold simultaneously. Each is preserved in the output so
    that a detection can be traced back to the component that gated it -- the
    usual question when a compound count looks surprising.

    Parameters
    ----------
    name
        Name of the resulting event mask.
    min_duration_hours
        When given, only runs lasting at least this long are retained, via
        :func:`alpinemet.indicators.dunkelflaute.sustained`.
    time_dim
        Name of the time dimension.
    timestep_hours
        Sampling interval in hours. Inferred from the time coordinate when
        omitted; pass it explicitly for series too short to infer from, such as
        a single-day result after daily resampling.
    **conditions
        Named boolean fields, all on a common grid.

    Returns
    -------
    xarray.Dataset
        The combined mask under ``name``, each input condition under its own
        name, and event statistics.

    Raises
    ------
    ValueError
        If no conditions are given.
    """
    if not conditions:
        raise ValueError("At least one condition is required")

    combined: xr.DataArray | None = None
    for condition in conditions.values():
        combined = condition if combined is None else (combined & condition)

    assert combined is not None  # guaranteed by the check above
    combined = combined.rename(name)

    if min_duration_hours is not None:
        combined = sustained(
            combined,
            min_duration_hours=min_duration_hours,
            time_dim=time_dim,
            timestep_hours=timestep_hours,
        ).rename(name)

    result = xr.Dataset({name: combined})
    for condition_name, condition in conditions.items():
        result[condition_name] = condition

    if time_dim in combined.dims:
        stats = event_statistics(combined, time_dim=time_dim, timestep_hours=timestep_hours)
        for stat_name, values in stats.data_vars.items():
            result[stat_name] = values

    result.attrs = {
        "event": name,
        "components": list(conditions),
        "min_duration_hours": min_duration_hours,
        "typology": "Zscheischler et al. (2020)",
    }
    return result


def hellsturm(
    wind_speed: xr.DataArray,
    irradiance: xr.DataArray,
    *,
    wind_threshold: float = HELLSTURM_WIND_MS,
    solar_threshold: float = HELLSTURM_SOLAR_WM2,
    min_duration_hours: float | None = None,
    aggregate: bool = False,
    time_dim: str = "time",
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> xr.Dataset:
    """Detect simultaneous high wind and high irradiance.

    Parameters
    ----------
    wind_speed
        Wind speed in m/s.
    irradiance
        Global horizontal irradiance in W/m2.
    wind_threshold, solar_threshold
        Thresholds; defaults are 12 m/s and 500 W/m2.
    min_duration_hours
        Optional minimum duration.
    aggregate
        Average over the domain before applying thresholds.
    time_dim, lat_name, lon_name
        Dimension names.

    Returns
    -------
    xarray.Dataset
        Mask, components and event statistics.
    """
    wind = _reduce_spatially(wind_speed, aggregate, lat_name, lon_name)
    solar = _reduce_spatially(irradiance, aggregate, lat_name, lon_name)

    result = combine_conditions(
        "hellsturm",
        min_duration_hours=min_duration_hours,
        time_dim=time_dim,
        high_wind=wind > wind_threshold,
        high_solar=solar > solar_threshold,
    )
    result.attrs["wind_threshold_ms"] = wind_threshold
    result.attrs["solar_threshold_wm2"] = solar_threshold
    result.attrs["spatially_aggregated"] = aggregate
    return result


def heat_drought(
    temperature: xr.DataArray,
    precipitation: xr.DataArray,
    *,
    temperature_threshold: float = HEAT_DROUGHT_TEMPERATURE,
    precipitation_threshold: float = HEAT_DROUGHT_PRECIP_MM,
    min_duration_days: float | None = None,
    aggregate: bool = False,
    time_dim: str = "time",
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> xr.Dataset:
    """Detect hot, dry days.

    Operates on daily aggregates: daily mean temperature against daily
    precipitation totals. Sub-daily input is resampled first.

    Parameters
    ----------
    temperature
        Temperature in degC.
    precipitation
        Precipitation in mm per time step.
    temperature_threshold
        Daily mean temperature above which a day is hot, in degC.
    precipitation_threshold
        Daily total below which a day is dry, in mm.
    min_duration_days
        Optional minimum spell length in days.
    aggregate
        Average over the domain before applying thresholds.
    time_dim, lat_name, lon_name
        Dimension names.

    Returns
    -------
    xarray.Dataset
        Mask on a daily time axis, components and event statistics.
    """
    temp = _reduce_spatially(temperature, aggregate, lat_name, lon_name)
    precip = _reduce_spatially(precipitation, aggregate, lat_name, lon_name)

    daily_temp = temp.resample({time_dim: "1D"}).mean()
    daily_precip = precip.resample({time_dim: "1D"}).sum()

    result = combine_conditions(
        "heat_drought",
        min_duration_hours=None if min_duration_days is None else min_duration_days * 24,
        time_dim=time_dim,
        # The resample above fixes the step at one day, so state it rather than
        # inferring it -- a single-day result has no spacing to infer from.
        timestep_hours=24.0,
        hot=daily_temp > temperature_threshold,
        dry=daily_precip < precipitation_threshold,
    )
    result.attrs["temperature_threshold_degC"] = temperature_threshold
    result.attrs["precipitation_threshold_mm"] = precipitation_threshold
    result.attrs["spatially_aggregated"] = aggregate
    result.attrs["temporal_resolution"] = "daily"
    return result


def rain_on_snow(
    temperature: xr.DataArray,
    precipitation: xr.DataArray,
    *,
    temperature_range: tuple[float, float] = RAIN_ON_SNOW_TEMPERATURE_RANGE,
    precipitation_threshold: float = RAIN_ON_SNOW_PRECIP_MM,
    snow_depth: xr.DataArray | None = None,
    min_snow_depth_m: float = 0.05,
    aggregate: bool = False,
    time_dim: str = "time",
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> xr.Dataset:
    """Detect rain falling at near-freezing temperatures.

    Without a snow depth field the temperature window acts as a proxy for lying
    snow, which over-detects in autumn and spring when no snowpack is present.
    Pass ``snow_depth`` when the dataset provides it.

    Parameters
    ----------
    temperature
        Temperature in degC.
    precipitation
        Precipitation in mm per time step.
    temperature_range
        Half-open window ``(low, high)`` in degC within which precipitation
        falls as rain onto snow.
    precipitation_threshold
        Precipitation above which the event counts, in mm per time step.
    snow_depth
        Optional snow depth in metres. When given, a snowpack of at least
        ``min_snow_depth_m`` is required.
    min_snow_depth_m
        Minimum snow depth in metres.
    aggregate
        Average over the domain before applying thresholds.
    time_dim, lat_name, lon_name
        Dimension names.

    Returns
    -------
    xarray.Dataset
        Mask, components and event statistics.
    """
    low, high = temperature_range
    temp = _reduce_spatially(temperature, aggregate, lat_name, lon_name)
    precip = _reduce_spatially(precipitation, aggregate, lat_name, lon_name)

    conditions = {
        "above_freezing": temp > low,
        "near_freezing": temp < high,
        "wet": precip > precipitation_threshold,
    }
    if snow_depth is not None:
        snow = _reduce_spatially(snow_depth, aggregate, lat_name, lon_name)
        conditions["snow_present"] = snow >= min_snow_depth_m

    result = combine_conditions("rain_on_snow", time_dim=time_dim, **conditions)
    result.attrs["temperature_range_degC"] = list(temperature_range)
    result.attrs["precipitation_threshold_mm"] = precipitation_threshold
    result.attrs["snow_depth_used"] = snow_depth is not None
    result.attrs["spatially_aggregated"] = aggregate
    return result


def foehn_like(
    temperature: xr.DataArray,
    wind_speed: xr.DataArray,
    *,
    wind_threshold: float = FOEHN_WIND_MS,
    warming_threshold: float = FOEHN_WARMING_K,
    window_hours: float = 3.0,
    aggregate: bool = False,
    time_dim: str = "time",
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> xr.Dataset:
    """Screen for föhn-like conditions: strong wind with rapid warming.

    .. warning::

       This is a screening proxy, not a föhn detection. Genuine identification
       requires upstream moisture, lee-side descent and cold-pool erosion, none
       of which are represented here. Expect false positives from frontal
       passages.

    The warming is computed as ``T(t) - T(t - window)``. An earlier
    implementation used ``diff(dim="time", n=3)``, which returns the third-order
    difference rather than the three-hour change; counts from that code are not
    comparable.

    Parameters
    ----------
    temperature
        Temperature in degC.
    wind_speed
        Wind speed in m/s.
    wind_threshold
        Wind speed above which the condition may hold, in m/s.
    warming_threshold
        Temperature rise across the window, in kelvin.
    window_hours
        Warming window in hours.
    aggregate
        Average over the domain before applying thresholds.
    time_dim, lat_name, lon_name
        Dimension names.

    Returns
    -------
    xarray.Dataset
        Mask, components and event statistics.
    """
    from alpinemet.indicators.storms import ramp_magnitude

    temp = _reduce_spatially(temperature, aggregate, lat_name, lon_name)
    wind = _reduce_spatially(wind_speed, aggregate, lat_name, lon_name)

    warming = ramp_magnitude(temp, window_hours=window_hours, time_dim=time_dim)

    result = combine_conditions(
        "foehn_like",
        time_dim=time_dim,
        strong_wind=wind > wind_threshold,
        rapid_warming=warming > warming_threshold,
    )
    result["warming"] = warming
    result.attrs["wind_threshold_ms"] = wind_threshold
    result.attrs["warming_threshold_K"] = warming_threshold
    result.attrs["window_hours"] = window_hours
    result.attrs["spatially_aggregated"] = aggregate
    result.attrs["caveat"] = "screening proxy, not a föhn detection"
    return result
