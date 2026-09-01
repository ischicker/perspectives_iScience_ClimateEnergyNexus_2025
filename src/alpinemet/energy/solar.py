"""Solar energy conversion.

Two conversion paths are provided, and the difference between them matters in
Alpine terrain:

:func:`fixed_tilt_capacity_factor`
    The reference method. A pvlib modelling chain for the fixed-tilt system
    reported in the paper -- 1 kWp, 30 degree tilt, south-facing -- comprising
    solar position, Erbs decomposition of GHI into direct and diffuse
    components, transposition to the plane of array, and a PVWatts DC model
    with temperature correction. Requires the optional ``energy`` dependency
    group (``uv sync --extra energy``).

:func:`plane_of_array_ratio_capacity_factor`
    A horizontal-plane ratio, ``GHI / 1000 W/m2``. Cheap and requires no
    location, but it ignores panel tilt, the solar geometry and the temperature
    dependence of module efficiency. It is retained because earlier iterations
    of this analysis used it, and because it is a useful sanity bound -- not as
    a recommended default.

The two disagree systematically in the Alps. A tilted plane collects
substantially more than the horizontal at high solar zenith angles, so the
ratio method underestimates winter yield at exactly the times that matter for
the inversion and Dunkelflaute analysis, while the missing temperature
correction pushes summer yield the other way.

Neither path represents terrain shading, snow cover on modules, or the
snow-albedo enhancement discussed in the paper. Those require a horizon model
and a snow model respectively, and remain a limitation of this evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "REFERENCE_PV_SYSTEM",
    "STANDARD_TEST_IRRADIANCE",
    "PVSystemSpec",
    "fixed_tilt_capacity_factor",
    "plane_of_array_ratio_capacity_factor",
]

#: Irradiance at standard test conditions, in W/m2.
STANDARD_TEST_IRRADIANCE = 1000.0


@dataclass(frozen=True)
class PVSystemSpec:
    """Fixed-tilt photovoltaic reference system.

    Attributes
    ----------
    name
        Identifier recorded in output metadata.
    dc_capacity_w
        Nameplate DC capacity at standard test conditions, in W.
    tilt_deg
        Surface tilt from horizontal, in degrees.
    azimuth_deg
        Surface azimuth in degrees clockwise from north; 180 is south-facing.
    gamma_pdc
        Temperature coefficient of power, per degree Celsius. The PVWatts
        default for crystalline silicon is -0.004.
    temperature_model_a, temperature_model_b, temperature_model_delta
        Faiman cell temperature model coefficients.
    """

    name: str
    dc_capacity_w: float = 1000.0
    tilt_deg: float = 30.0
    azimuth_deg: float = 180.0
    gamma_pdc: float = -0.004
    temperature_model_u0: float = 25.0
    temperature_model_u1: float = 6.84

    def __post_init__(self) -> None:
        if self.dc_capacity_w <= 0:
            raise ValueError(f"DC capacity must be positive; got {self.dc_capacity_w}")
        if not 0.0 <= self.tilt_deg <= 90.0:
            raise ValueError(f"Tilt must lie in [0, 90] degrees; got {self.tilt_deg}")
        if not 0.0 <= self.azimuth_deg <= 360.0:
            raise ValueError(f"Azimuth must lie in [0, 360] degrees; got {self.azimuth_deg}")


#: Reference system used throughout the study (1 kWp, 30 degrees, south-facing).
REFERENCE_PV_SYSTEM = PVSystemSpec(name="reference_1kWp_30deg_south")


def _require_pvlib():
    """Import pvlib, with an actionable message when it is not installed."""
    try:
        import pvlib
    except ImportError as exc:  # pragma: no cover - exercised only without pvlib
        raise ImportError(
            "fixed_tilt_capacity_factor requires pvlib, which is an optional "
            "dependency. Install it with 'uv sync --extra energy' (or "
            "'pip install alpinemet[energy]')."
        ) from exc
    return pvlib


def plane_of_array_ratio_capacity_factor(
    ghi: xr.DataArray,
    *,
    reference_irradiance: float = STANDARD_TEST_IRRADIANCE,
) -> xr.DataArray:
    """Capacity factor as the horizontal irradiance ratio ``GHI / 1000``.

    Provided for comparison and backwards compatibility. Prefer
    :func:`fixed_tilt_capacity_factor`; see the module docstring for why the two
    differ systematically in complex terrain.

    Parameters
    ----------
    ghi
        Global horizontal irradiance in W/m2.
    reference_irradiance
        Irradiance corresponding to a capacity factor of 1, in W/m2.

    Returns
    -------
    xarray.DataArray
        Dimensionless capacity factor, clipped to [0, 1].

    Raises
    ------
    ValueError
        If ``reference_irradiance`` is not positive.
    """
    if reference_irradiance <= 0:
        raise ValueError(f"Reference irradiance must be positive; got {reference_irradiance}")

    cf = (ghi / reference_irradiance).clip(0.0, 1.0)
    cf = cf.rename("solar_capacity_factor")
    cf.attrs = {
        "long_name": "Solar capacity factor (horizontal irradiance ratio)",
        "units": "1",
        "method": "GHI / reference irradiance",
        "reference_irradiance_wm2": reference_irradiance,
        "caveat": "ignores tilt, solar geometry and module temperature",
    }
    return cf


def _point_capacity_factor(
    ghi_values: np.ndarray,
    temp_air_values: np.ndarray | None,
    wind_speed_values: np.ndarray | None,
    times: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    system: PVSystemSpec,
    altitude: float,
) -> np.ndarray:
    """Run the pvlib chain for a single location. Returns a capacity factor series."""
    pvlib = _require_pvlib()

    if not np.isfinite(latitude) or not np.isfinite(longitude):
        return np.full(ghi_values.shape, np.nan)

    ghi = pd.Series(np.asarray(ghi_values, dtype=float), index=times).clip(lower=0.0)

    solar_position = pvlib.solarposition.get_solarposition(
        times, latitude, longitude, altitude=altitude
    )
    zenith = solar_position["apparent_zenith"]

    # Split GHI into its direct and diffuse components. Erbs is the standard
    # choice when only GHI is available, which is the case for every reanalysis
    # evaluated here.
    decomposed = pvlib.irradiance.erbs(ghi, zenith, times)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=system.tilt_deg,
        surface_azimuth=system.azimuth_deg,
        solar_zenith=zenith,
        solar_azimuth=solar_position["azimuth"],
        dni=decomposed["dni"],
        ghi=ghi,
        dhi=decomposed["dhi"],
    )
    poa_global = poa["poa_global"].fillna(0.0).clip(lower=0.0)

    if temp_air_values is None:
        # Without air temperature the module is assumed to sit at standard test
        # conditions, i.e. no temperature derating.
        cell_temperature = pd.Series(25.0, index=times)
    else:
        temp_air = pd.Series(np.asarray(temp_air_values, dtype=float), index=times)
        wind = (
            pd.Series(1.0, index=times)
            if wind_speed_values is None
            else pd.Series(np.asarray(wind_speed_values, dtype=float), index=times)
        )
        cell_temperature = pvlib.temperature.faiman(
            poa_global,
            temp_air,
            wind_speed=wind,
            u0=system.temperature_model_u0,
            u1=system.temperature_model_u1,
        )

    # Called positionally: pvlib renamed the first parameter from
    # 'g_poa_effective' to 'effective_irradiance' in 0.13, and positional
    # arguments keep this working across the supported version range.
    dc_power = pvlib.pvsystem.pvwatts_dc(
        poa_global,
        cell_temperature,
        system.dc_capacity_w,
        system.gamma_pdc,
    )

    cf = (dc_power / system.dc_capacity_w).clip(lower=0.0, upper=1.0)
    return cf.to_numpy()


def fixed_tilt_capacity_factor(
    ghi: xr.DataArray,
    *,
    latitude: xr.DataArray | float | None = None,
    longitude: xr.DataArray | float | None = None,
    temp_air: xr.DataArray | None = None,
    wind_speed: xr.DataArray | None = None,
    system: PVSystemSpec = REFERENCE_PV_SYSTEM,
    altitude: float = 0.0,
    time_dim: str = "time",
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
) -> xr.DataArray:
    """Capacity factor of a fixed-tilt PV system via the pvlib modelling chain.

    Works on a single time series or on a gridded field. For gridded input the
    chain is evaluated **per grid point**, because solar position depends on
    latitude and longitude. This is expensive: a full year of hourly data over
    a few thousand grid points takes minutes, not seconds. Subset the domain, or
    aggregate spatially first, when a full-resolution field is not needed.

    Parameters
    ----------
    ghi
        Global horizontal irradiance in W/m2, with a time dimension.
    latitude, longitude
        Coordinates in degrees. Taken from ``ghi``'s coordinates when omitted.
        Pass scalars for a single-location time series.
    temp_air
        Air temperature in degrees Celsius, for the cell temperature model. When
        omitted, no temperature derating is applied and yields are biased high
        in summer.
    wind_speed
        Wind speed in m/s, used by the cell temperature model. Defaults to
        1 m/s, the convention when no wind field is supplied.
    system
        PV system specification; defaults to :data:`REFERENCE_PV_SYSTEM`.
    altitude
        Site elevation in metres, used for the solar position refraction
        correction. Elevation-dependent irradiance itself is not modelled.
    time_dim, lat_dim, lon_dim
        Dimension names.

    Returns
    -------
    xarray.DataArray
        Dimensionless capacity factor in [0, 1].

    Raises
    ------
    ImportError
        If pvlib is not installed.
    ValueError
        If the time coordinate is missing or is not a datetime index, or if
        latitude and longitude can be resolved neither from arguments nor from
        coordinates.
    """
    _require_pvlib()

    if time_dim not in ghi.dims:
        raise ValueError(f"Irradiance field has no {time_dim!r} dimension; got {tuple(ghi.dims)}")
    if time_dim not in ghi.coords:
        raise ValueError(
            f"Irradiance field has no {time_dim!r} coordinate; a datetime coordinate is "
            "required to compute solar position"
        )

    times = pd.DatetimeIndex(ghi[time_dim].values)
    if times.tz is None:
        times = times.tz_localize("UTC")

    if latitude is None:
        if lat_dim not in ghi.coords:
            raise ValueError(
                f"No latitude given and no {lat_dim!r} coordinate on the irradiance field"
            )
        latitude = ghi[lat_dim]
    if longitude is None:
        if lon_dim not in ghi.coords:
            raise ValueError(
                f"No longitude given and no {lon_dim!r} coordinate on the irradiance field"
            )
        longitude = ghi[lon_dim]

    def _apply(ghi_block, temp_block, wind_block, lat_value, lon_value):
        return _point_capacity_factor(
            ghi_block,
            temp_block if temp_air is not None else None,
            wind_block if wind_speed is not None else None,
            times,
            float(np.asarray(lat_value).item()),
            float(np.asarray(lon_value).item()),
            system,
            altitude,
        )

    # Placeholders keep apply_ufunc's signature uniform when the optional
    # inputs are absent.
    temp_input = temp_air if temp_air is not None else xr.zeros_like(ghi)
    wind_input = wind_speed if wind_speed is not None else xr.zeros_like(ghi)

    cf = xr.apply_ufunc(
        _apply,
        ghi,
        temp_input,
        wind_input,
        latitude,
        longitude,
        input_core_dims=[[time_dim], [time_dim], [time_dim], [], []],
        output_core_dims=[[time_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    cf = cf.transpose(*ghi.dims)
    cf = cf.rename("solar_capacity_factor")
    cf.attrs = {
        "long_name": "Solar capacity factor (fixed tilt)",
        "units": "1",
        "method": "pvlib: Erbs decomposition, POA transposition, PVWatts DC",
        "system": system.name,
        "dc_capacity_W": system.dc_capacity_w,
        "tilt_deg": system.tilt_deg,
        "azimuth_deg": system.azimuth_deg,
        "temperature_correction": "applied" if temp_air is not None else "not applied",
        "caveat": "no terrain shading, module snow cover or snow-albedo enhancement",
    }
    return cf
