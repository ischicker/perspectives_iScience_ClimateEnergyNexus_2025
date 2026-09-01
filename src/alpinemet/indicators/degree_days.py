"""Heating and cooling degree days following VDI 3787.

Degree days are the energy sector's standard proxy for space heating and
cooling demand. Following VDI 3787 (Part 2), they are accumulated from *daily
mean* temperature against a fixed base temperature:

.. math::

    \\mathrm{HDD} = \\sum_{d} \\max(0,\\; T_\\mathrm{base}^\\mathrm{HDD} - \\bar{T}_d)

    \\mathrm{CDD} = \\sum_{d} \\max(0,\\; \\bar{T}_d - T_\\mathrm{base}^\\mathrm{CDD})

The base temperatures used throughout the study are 15 degC for heating and
18 degC for cooling, the values reported in Figures 2 and 4 of the paper.

Accumulating from daily means rather than from the sub-daily values matters:
summing hourly deviations would count the diurnal cycle as demand and inflate
both totals substantially.
"""

from __future__ import annotations

import xarray as xr

from alpinemet.units import to_celsius

__all__ = [
    "CDD_BASE_TEMPERATURE",
    "HDD_BASE_TEMPERATURE",
    "cooling_degree_days",
    "daily_mean_temperature",
    "degree_days",
    "heating_degree_days",
]

#: VDI 3787 base temperature for heating degree days (degC).
HDD_BASE_TEMPERATURE = 15.0

#: VDI 3787 base temperature for cooling degree days (degC).
CDD_BASE_TEMPERATURE = 18.0


def daily_mean_temperature(
    temperature: xr.DataArray,
    *,
    time_dim: str = "time",
    assume_units: str | None = None,
) -> xr.DataArray:
    """Resample a temperature field to daily means in degrees Celsius.

    Parameters
    ----------
    temperature
        Temperature field with a time dimension. Kelvin inputs are converted.
    time_dim
        Name of the time dimension.
    assume_units
        Passed to :func:`alpinemet.units.to_celsius` to override unit inference.

    Returns
    -------
    xarray.DataArray
        Daily mean temperature in degC.

    Raises
    ------
    ValueError
        If ``time_dim`` is not a dimension of ``temperature``.
    """
    if time_dim not in temperature.dims:
        raise ValueError(
            f"Temperature field has no {time_dim!r} dimension; got {tuple(temperature.dims)}"
        )
    celsius = to_celsius(temperature, assume=assume_units)
    daily = celsius.resample({time_dim: "1D"}).mean()
    daily.attrs = dict(celsius.attrs)
    daily.attrs["cell_methods"] = f"{time_dim}: mean (interval: 1 day)"
    return daily


def _degree_days(
    daily_temperature: xr.DataArray,
    *,
    base: float,
    heating: bool,
    time_dim: str,
    freq: str | None,
) -> xr.DataArray:
    deviation = (base - daily_temperature) if heating else (daily_temperature - base)
    contribution = deviation.clip(min=0.0)

    if freq is None:
        total = contribution.sum(dim=time_dim)
    else:
        total = contribution.resample({time_dim: freq}).sum()

    kind = "Heating" if heating else "Cooling"
    total.attrs = {
        "long_name": f"{kind} degree days",
        "units": "degC d",
        "base_temperature_degC": base,
        "standard": "VDI 3787",
        "accumulation": "daily mean temperature",
    }
    return total.rename("hdd" if heating else "cdd")


def heating_degree_days(
    temperature: xr.DataArray,
    *,
    base: float = HDD_BASE_TEMPERATURE,
    freq: str | None = None,
    time_dim: str = "time",
    already_daily: bool = False,
    assume_units: str | None = None,
) -> xr.DataArray:
    """Accumulate heating degree days (VDI 3787).

    Parameters
    ----------
    temperature
        Temperature field. Sub-daily input is averaged to daily means first
        unless ``already_daily`` is set.
    base
        Base temperature in degC. Defaults to 15 degC.
    freq
        Pandas offset alias for periodic totals, e.g. ``"1ME"`` for monthly or
        ``"1YE"`` for annual sums. ``None`` sums over the whole record and drops
        the time dimension.
    time_dim
        Name of the time dimension.
    already_daily
        Set when ``temperature`` already holds daily means, to skip resampling.
    assume_units
        Passed to :func:`alpinemet.units.to_celsius`.

    Returns
    -------
    xarray.DataArray
        Heating degree days in degC d.
    """
    daily = (
        to_celsius(temperature, assume=assume_units)
        if already_daily
        else daily_mean_temperature(temperature, time_dim=time_dim, assume_units=assume_units)
    )
    return _degree_days(daily, base=base, heating=True, time_dim=time_dim, freq=freq)


def cooling_degree_days(
    temperature: xr.DataArray,
    *,
    base: float = CDD_BASE_TEMPERATURE,
    freq: str | None = None,
    time_dim: str = "time",
    already_daily: bool = False,
    assume_units: str | None = None,
) -> xr.DataArray:
    """Accumulate cooling degree days (VDI 3787).

    Parameters are as for :func:`heating_degree_days`, with ``base`` defaulting
    to 18 degC.

    Returns
    -------
    xarray.DataArray
        Cooling degree days in degC d.
    """
    daily = (
        to_celsius(temperature, assume=assume_units)
        if already_daily
        else daily_mean_temperature(temperature, time_dim=time_dim, assume_units=assume_units)
    )
    return _degree_days(daily, base=base, heating=False, time_dim=time_dim, freq=freq)


def degree_days(
    temperature: xr.DataArray,
    *,
    hdd_base: float = HDD_BASE_TEMPERATURE,
    cdd_base: float = CDD_BASE_TEMPERATURE,
    freq: str | None = None,
    time_dim: str = "time",
    assume_units: str | None = None,
) -> xr.Dataset:
    """Compute heating and cooling degree days in one pass.

    Resamples to daily means once and reuses the result for both accumulations,
    which is noticeably cheaper than calling the two functions separately on
    large lazily loaded fields.

    Returns
    -------
    xarray.Dataset
        Dataset with ``hdd`` and ``cdd`` variables and the base temperatures
        recorded in the dataset attributes.
    """
    daily = daily_mean_temperature(temperature, time_dim=time_dim, assume_units=assume_units)
    hdd = _degree_days(daily, base=hdd_base, heating=True, time_dim=time_dim, freq=freq)
    cdd = _degree_days(daily, base=cdd_base, heating=False, time_dim=time_dim, freq=freq)

    result = xr.Dataset({"hdd": hdd, "cdd": cdd})
    result.attrs = {
        "standard": "VDI 3787",
        "hdd_base_temperature_degC": hdd_base,
        "cdd_base_temperature_degC": cdd_base,
    }
    return result
