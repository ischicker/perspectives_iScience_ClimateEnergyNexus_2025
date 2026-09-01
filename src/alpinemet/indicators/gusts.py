"""Wind gusts: native fields where available, Wieringa estimation otherwise.

Gusts drive the infrastructure-risk metrics in this study (turbine cut-out,
transmission line loading, ESSL severe- and extreme-wind thresholds), so how
they are obtained matters more than for most variables.

Two sources are supported, in this order of preference:

1. **Native instantaneous gusts** (``i10fg``). Provided by ARA, the ECMWF IFS
   and the Extremes DT. Always preferred -- these come out of the model's own
   gust parameterisation and need no assumption about the wind distribution.

2. **Wieringa (1973) estimation** from mean wind speed, for products without a
   gust field. ERA5 and ERA5-Land fall into this category over the Alpine
   domain evaluated here.

   .. math:: u_\\mathrm{gust} = u + k\\,\\sigma_u,\\qquad k = 1.9

   ``sigma_u`` is the temporal standard deviation at each grid point, so the
   estimate depends on the length of the record supplied. Pass the full
   analysis period rather than a short slice, and keep the period consistent
   when comparing datasets.

Estimated and native gusts are not interchangeable. As the paper notes for
Figure 2, the apparent storm-hour maximum over southeastern Germany in ERA5 and
ERA5-Land that largely vanishes in ARA is plausibly an artefact of exactly this
difference: a gust estimate applied over smoothed orography against a
convection-permitting model resolving the sheltering terrain. Always record
which source was used -- :func:`resolve_gusts` does this in the output
attributes.

References
----------
Wieringa, J. (1973). Gust factors over open water and built-up country.
*Boundary-Layer Meteorology*, 3, 424-441.
"""

from __future__ import annotations

import xarray as xr

__all__ = [
    "GustSource",
    "MINIMUM_GUST_FACTOR",
    "NATIVE_GUST_VARIABLES",
    "WIERINGA_K",
    "estimate_gusts",
    "gust_factor",
    "native_gust_variable",
    "resolve_gusts",
]

#: Empirical factor of the Wieringa (1973) estimator.
WIERINGA_K = 1.9

#: Physical floor applied where the estimator would return less than the mean wind.
MINIMUM_GUST_FACTOR = 1.25

#: Variable names under which a native instantaneous gust field may appear,
#: in decreasing order of preference.
NATIVE_GUST_VARIABLES: tuple[str, ...] = (
    "i10fg",
    "fg10",
    "10fg",
    "wind_gust",
    "gust",
)

#: How a gust field was obtained.
GustSource = str


def native_gust_variable(
    dataset: xr.Dataset,
    *,
    candidates: tuple[str, ...] = NATIVE_GUST_VARIABLES,
) -> str | None:
    """Return the name of the native gust variable in ``dataset``, if any.

    Parameters
    ----------
    dataset
        Dataset to inspect.
    candidates
        Variable names to look for, in decreasing order of preference.

    Returns
    -------
    str or None
        The first matching variable name, or ``None`` if the dataset carries no
        native gust field.
    """
    for name in candidates:
        if name in dataset.data_vars:
            return name
    return None


def estimate_gusts(
    wind_speed: xr.DataArray,
    *,
    time_dim: str = "time",
    k: float = WIERINGA_K,
) -> xr.DataArray:
    """Estimate wind gusts from mean wind speed after Wieringa (1973).

    Only use this for datasets without a native gust field; prefer
    :func:`resolve_gusts`, which selects the native field automatically when one
    is present.

    Parameters
    ----------
    wind_speed
        Mean wind speed in m/s.
    time_dim
        Name of the time dimension, along which the standard deviation is
        computed. If absent, a global standard deviation is used instead.
    k
        Empirical factor. Exposed for sensitivity analysis; the published value
        is 1.9.

    Returns
    -------
    xarray.DataArray
        Estimated gust speed in m/s, never below the mean wind speed.
    """
    sigma = wind_speed.std(dim=time_dim) if time_dim in wind_speed.dims else wind_speed.std()
    gust = wind_speed + k * sigma

    # A negative k, or a degenerate record, could push the estimate below the
    # mean wind; substitute the physical floor there.
    gust = xr.where(gust < wind_speed, wind_speed * MINIMUM_GUST_FACTOR, gust)

    gust = gust.rename("gust_estimated")
    gust.attrs = {
        "long_name": "Estimated wind gust",
        "units": "m/s",
        "gust_source": "estimated",
        "method": "Wieringa (1973)",
        "k_factor": k,
        "minimum_gust_factor": MINIMUM_GUST_FACTOR,
        "estimated_from": wind_speed.name or "wind_speed",
    }
    return gust


def resolve_gusts(
    dataset: xr.Dataset,
    *,
    wind_speed: xr.DataArray | None = None,
    wind_speed_var: str = "wind_speed_10m",
    time_dim: str = "time",
    k: float = WIERINGA_K,
    candidates: tuple[str, ...] = NATIVE_GUST_VARIABLES,
) -> xr.DataArray:
    """Obtain gusts from a dataset, preferring the native field.

    Uses the native instantaneous gust field when the dataset provides one,
    otherwise falls back to Wieringa estimation from mean wind speed. Either
    way the result carries a ``gust_source`` attribute recording which path was
    taken, so that cross-dataset comparisons remain interpretable.

    Parameters
    ----------
    dataset
        Dataset that may contain a native gust field and a mean wind field.
    wind_speed
        Mean wind speed to estimate from. Defaults to ``dataset[wind_speed_var]``.
        Only used if no native gust field is found.
    wind_speed_var
        Name of the mean wind speed variable in ``dataset``.
    time_dim, k
        Passed to :func:`estimate_gusts`.
    candidates
        Native gust variable names to look for.

    Returns
    -------
    xarray.DataArray
        Gust speed in m/s.

    Raises
    ------
    ValueError
        If the dataset has neither a native gust field nor a usable mean wind
        field to estimate from.
    """
    native = native_gust_variable(dataset, candidates=candidates)
    if native is not None:
        gust = dataset[native].copy()
        gust.attrs = {
            **dict(dataset[native].attrs),
            "long_name": gust.attrs.get("long_name", "Wind gust"),
            "units": gust.attrs.get("units", "m/s"),
            "gust_source": "native",
            "native_variable": native,
        }
        return gust.rename("gust")

    if wind_speed is None:
        if wind_speed_var not in dataset.data_vars:
            available = sorted(dataset.data_vars)
            raise ValueError(
                f"Dataset has no native gust field (looked for {list(candidates)}) and no "
                f"{wind_speed_var!r} variable to estimate from; available variables: {available}"
            )
        wind_speed = dataset[wind_speed_var]

    return estimate_gusts(wind_speed, time_dim=time_dim, k=k)


def gust_factor(gust: xr.DataArray, wind_speed: xr.DataArray) -> xr.DataArray:
    """Ratio of gust to mean wind speed.

    Calm conditions are masked rather than divided through: where the mean wind
    is zero the ratio is undefined, and returning ``inf`` there would propagate
    into every downstream statistic.

    Parameters
    ----------
    gust
        Gust speed in m/s, native or estimated.
    wind_speed
        Mean wind speed in m/s.

    Returns
    -------
    xarray.DataArray
        Dimensionless gust factor, ``NaN`` where the mean wind speed is zero.
    """
    safe_wind = xr.where(wind_speed > 0, wind_speed, 1.0)
    factor = xr.where(wind_speed > 0, gust / safe_wind, float("nan"))
    factor = factor.rename("gust_factor")
    factor.attrs = {"long_name": "Gust factor", "units": "1"}
    return factor
