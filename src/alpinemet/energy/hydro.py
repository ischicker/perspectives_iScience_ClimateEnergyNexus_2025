"""Hydropower resource indicators from precipitation and temperature.

Mountain regions hold roughly 70 % of global hydropower capacity, and Alpine
inflow is governed less by annual precipitation totals than by *when* water
arrives -- which in turn hinges on whether precipitation falls as snow or rain.

The snow-rain partition is therefore the central quantity here. As the paper
discusses, the partition threshold is migrating upward under warming, shifting
winter precipitation from snow to rain: less seasonal storage feeding spring and
summer runoff, more winter flood risk. A dataset can get the annual total right
and still get the seasonal distribution badly wrong.

.. warning::

   The runoff estimate in :func:`runoff_from_precipitation` applies a single
   constant coefficient to precipitation. It carries no snowpack accounting, no
   routing, no evapotranspiration, no glacier contribution and no reservoir
   operation. It is a first-order resource *screening* indicator for comparing
   datasets against one another, not a hydrological model, and must not be read
   as an inflow forecast. Use a calibrated hydrological model for anything
   operational.
"""

from __future__ import annotations

import xarray as xr

from alpinemet.units import to_celsius

__all__ = [
    "ALPINE_RUNOFF_COEFFICIENT",
    "SNOW_RAIN_TRANSITION",
    "annual_precipitation",
    "runoff_from_precipitation",
    "seasonal_totals",
    "snow_rain_partition",
]

#: Temperature window (degC) over which precipitation shifts from snow to rain.
SNOW_RAIN_TRANSITION = (0.0, 2.0)

#: First-order runoff coefficient for Alpine catchments.
ALPINE_RUNOFF_COEFFICIENT = 0.4

_HOURS_PER_YEAR = 365.25 * 24.0


def _timestep_hours(data: xr.DataArray, time_dim: str) -> float:
    from alpinemet.indicators.dunkelflaute import infer_timestep_hours

    return infer_timestep_hours(data, time_dim=time_dim)


def snow_rain_partition(
    precipitation: xr.DataArray,
    temperature: xr.DataArray,
    *,
    transition: tuple[float, float] = SNOW_RAIN_TRANSITION,
    assume_units: str | None = None,
) -> xr.Dataset:
    """Split precipitation into snowfall and rainfall.

    A linear ramp is applied across the transition window rather than a hard
    threshold: the snow fraction is 1 at or below the lower bound, 0 at or above
    the upper bound, and interpolates linearly between. A hard threshold is
    obtained by passing a zero-width window.

    Parameters
    ----------
    precipitation
        Precipitation in mm per time step.
    temperature
        Air temperature. Kelvin input is converted.
    transition
        ``(all_snow_below, all_rain_above)`` in degC. Defaults to (0, 2), the
        range discussed in the paper. The true threshold is
        elevation-dependent; treat this as a domain-wide approximation.
    assume_units
        Passed to :func:`alpinemet.units.to_celsius`.

    Returns
    -------
    xarray.Dataset
        ``snowfall``, ``rainfall`` and the dimensionless ``snow_fraction``.
        Snowfall and rainfall sum to the input precipitation exactly.

    Raises
    ------
    ValueError
        If the transition window is not ordered ``low <= high``.
    """
    low, high = transition
    if low > high:
        raise ValueError(
            f"Transition window must satisfy low <= high; got {transition}"
        )

    celsius = to_celsius(temperature, assume=assume_units)

    if high == low:
        snow_fraction = xr.where(celsius <= low, 1.0, 0.0)
    else:
        snow_fraction = ((high - celsius) / (high - low)).clip(0.0, 1.0)

    snowfall = precipitation * snow_fraction
    rainfall = precipitation - snowfall

    result = xr.Dataset(
        {
            "snowfall": snowfall,
            "rainfall": rainfall,
            "snow_fraction": snow_fraction,
        }
    )
    result["snowfall"].attrs = {"long_name": "Snowfall", "units": "mm"}
    result["rainfall"].attrs = {"long_name": "Rainfall", "units": "mm"}
    result["snow_fraction"].attrs = {
        "long_name": "Fraction of precipitation falling as snow",
        "units": "1",
    }
    result.attrs = {
        "transition_degC": list(transition),
        "method": "linear ramp across the transition window",
    }
    return result


def annual_precipitation(
    precipitation: xr.DataArray,
    *,
    time_dim: str = "time",
    timestep_hours: float | None = None,
) -> xr.DataArray:
    """Scale a precipitation record to an annual total.

    Sums the record and rescales by its actual length in hours. An earlier
    implementation multiplied the sum by ``365 / n_steps``, which silently
    assumes one step per day and understates hourly records by a factor of 24.

    Parameters
    ----------
    precipitation
        Precipitation in mm per time step.
    time_dim
        Name of the time dimension.
    timestep_hours
        Sampling interval in hours. Inferred when omitted.

    Returns
    -------
    xarray.DataArray
        Annualised precipitation total in mm per year.
    """
    if timestep_hours is None:
        timestep_hours = _timestep_hours(precipitation, time_dim)

    n_steps = precipitation.sizes[time_dim]
    record_hours = n_steps * timestep_hours

    total = precipitation.sum(dim=time_dim) * (_HOURS_PER_YEAR / record_hours)
    total = total.rename("annual_precipitation")
    total.attrs = {
        "long_name": "Annualised precipitation total",
        "units": "mm/year",
        "record_length_hours": record_hours,
        "note": "extrapolated from the supplied record; not a climatology",
    }
    return total


def runoff_from_precipitation(
    precipitation: xr.DataArray,
    *,
    coefficient: float = ALPINE_RUNOFF_COEFFICIENT,
) -> xr.DataArray:
    """First-order runoff estimate from precipitation.

    See the module warning: this is a screening indicator, not a hydrological
    model.

    Parameters
    ----------
    precipitation
        Precipitation in mm per time step.
    coefficient
        Fraction of precipitation reaching the channel. Defaults to 0.4.

    Returns
    -------
    xarray.DataArray
        Estimated runoff in mm per time step.

    Raises
    ------
    ValueError
        If the coefficient is outside [0, 1].
    """
    if not 0.0 <= coefficient <= 1.0:
        raise ValueError(f"Runoff coefficient must lie in [0, 1]; got {coefficient}")

    runoff = precipitation * coefficient
    runoff = runoff.rename("runoff")
    runoff.attrs = {
        "long_name": "Estimated runoff",
        "units": "mm",
        "runoff_coefficient": coefficient,
        "method": "constant fraction of precipitation",
        "caveat": "no snowpack, routing, evapotranspiration or glacier contribution",
    }
    return runoff


def seasonal_totals(
    quantity: xr.DataArray,
    *,
    time_dim: str = "time",
) -> xr.DataArray:
    """Sum a quantity by meteorological season.

    Seasonal distribution is what distinguishes Alpine hydropower datasets from
    one another; annual totals frequently agree while the seasonal split does
    not.

    Parameters
    ----------
    quantity
        Field to accumulate, e.g. snowfall or runoff.
    time_dim
        Name of the time dimension.

    Returns
    -------
    xarray.DataArray
        Totals indexed by a ``season`` coordinate (DJF, MAM, JJA, SON).
    """
    seasonal = quantity.groupby(f"{time_dim}.season").sum(dim=time_dim)
    seasonal.attrs = {
        **quantity.attrs,
        "long_name": f"Seasonal total of {quantity.attrs.get('long_name', 'quantity')}",
        "cell_methods": "season: sum",
    }
    return seasonal
