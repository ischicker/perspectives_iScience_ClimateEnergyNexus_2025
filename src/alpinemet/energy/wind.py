"""Wind energy conversion.

Capacity factors are derived from a parametric power curve rather than a
manufacturer curve, so that the same reference turbine can be applied
consistently across datasets of very different resolution.

The reference configuration is the one reported in the paper: a 3 MW turbine
at 100 m hub height, cutting in at 3 m/s, reaching rated power at 12 m/s and
cutting out at 25 m/s.

The parametric approach follows Staffell and Pfenninger (2016). The default
curve shape between cut-in and rated speed is cubic, since power in the wind
scales with the cube of wind speed; the piecewise-linear form is available as
``shape="linear"``. Use :func:`power_curve` to inspect either.

References
----------
Staffell, I., and Pfenninger, S. (2016). Using bias-corrected reanalysis to
simulate current and future wind power output. *Energy*, 114, 1224-1239.
doi:10.1016/j.energy.2016.08.068

Wind speeds must be given at hub height. Use :func:`extrapolate_to_hub_height`
when only 10 m wind is available; ERA5 and ERA5-Land in particular provide no
100 m wind over the Alpine domain used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import xarray as xr

__all__ = [
    "PowerCurveShape",
    "REFERENCE_TURBINE",
    "TurbineSpec",
    "capacity_factor",
    "extrapolate_to_hub_height",
    "power_curve",
    "power_output",
]

PowerCurveShape = Literal["cubic", "linear"]


@dataclass(frozen=True)
class TurbineSpec:
    """Parametric description of a reference wind turbine.

    Attributes
    ----------
    name
        Identifier recorded in output metadata.
    rated_power_kw
        Rated electrical power in kW.
    hub_height_m
        Hub height in metres above ground.
    cut_in_speed
        Wind speed at which generation starts, in m/s.
    rated_speed
        Wind speed at which rated power is first reached, in m/s.
    cut_out_speed
        Wind speed above which the turbine shuts down, in m/s.
    """

    name: str
    rated_power_kw: float
    hub_height_m: float
    cut_in_speed: float
    rated_speed: float
    cut_out_speed: float

    def __post_init__(self) -> None:
        if not 0 <= self.cut_in_speed < self.rated_speed < self.cut_out_speed:
            raise ValueError(
                "Turbine speeds must satisfy 0 <= cut_in < rated < cut_out; got "
                f"cut_in={self.cut_in_speed}, rated={self.rated_speed}, "
                f"cut_out={self.cut_out_speed}"
            )
        if self.rated_power_kw <= 0:
            raise ValueError(f"Rated power must be positive; got {self.rated_power_kw}")
        if self.hub_height_m <= 0:
            raise ValueError(f"Hub height must be positive; got {self.hub_height_m}")


#: Reference turbine used throughout the study (3 MW, 100 m hub height).
REFERENCE_TURBINE = TurbineSpec(
    name="reference_3MW",
    rated_power_kw=3000.0,
    hub_height_m=100.0,
    cut_in_speed=3.0,
    rated_speed=12.0,
    cut_out_speed=25.0,
)


def capacity_factor(
    wind_speed: xr.DataArray,
    *,
    turbine: TurbineSpec = REFERENCE_TURBINE,
    shape: PowerCurveShape = "cubic",
) -> xr.DataArray:
    """Capacity factor of a reference turbine from hub-height wind speed.

    The curve is zero below cut-in and at or above cut-out, rises between
    cut-in and rated speed, and is 1 between rated and cut-out speed. Note that
    cut-out is a hard edge: at exactly ``cut_out_speed`` the turbine is already
    shut down.

    Parameters
    ----------
    wind_speed
        Wind speed at hub height in m/s.
    turbine
        Turbine specification; defaults to :data:`REFERENCE_TURBINE`.
    shape
        ``"cubic"`` uses ``(u**3 - u_in**3) / (u_rated**3 - u_in**3)``;
        ``"linear"`` uses ``(u - u_in) / (u_rated - u_in)``.

    Returns
    -------
    xarray.DataArray
        Dimensionless capacity factor in [0, 1].

    Raises
    ------
    ValueError
        If ``shape`` is not a supported curve shape.
    """
    u_in = turbine.cut_in_speed
    u_rated = turbine.rated_speed
    u_out = turbine.cut_out_speed

    if shape == "cubic":
        ramp = (wind_speed**3 - u_in**3) / (u_rated**3 - u_in**3)
    elif shape == "linear":
        ramp = (wind_speed - u_in) / (u_rated - u_in)
    else:
        raise ValueError(f"Unknown power curve shape {shape!r}; expected 'cubic' or 'linear'")

    cf = xr.where(
        (wind_speed < u_in) | (wind_speed >= u_out),
        0.0,
        xr.where(wind_speed < u_rated, ramp, 1.0),
    )
    # Guard against numerical excursions outside [0, 1] at the segment joins.
    cf = cf.clip(0.0, 1.0)

    cf = cf.rename("wind_capacity_factor")
    cf.attrs = {
        "long_name": "Wind capacity factor",
        "units": "1",
        "turbine": turbine.name,
        "rated_power_kW": turbine.rated_power_kw,
        "hub_height_m": turbine.hub_height_m,
        "cut_in_speed_ms": u_in,
        "rated_speed_ms": u_rated,
        "cut_out_speed_ms": u_out,
        "power_curve_shape": shape,
    }
    return cf


def power_output(
    wind_speed: xr.DataArray,
    *,
    turbine: TurbineSpec = REFERENCE_TURBINE,
    shape: PowerCurveShape = "cubic",
) -> xr.DataArray:
    """Electrical power output of a reference turbine.

    Returns
    -------
    xarray.DataArray
        Power in kW, i.e. the capacity factor scaled by the rated power.
    """
    power = capacity_factor(wind_speed, turbine=turbine, shape=shape) * turbine.rated_power_kw
    power = power.rename("wind_power")
    power.attrs = {
        "long_name": "Wind turbine power output",
        "units": "kW",
        "turbine": turbine.name,
        "power_curve_shape": shape,
    }
    return power


def power_curve(
    speeds: np.ndarray | None = None,
    *,
    turbine: TurbineSpec = REFERENCE_TURBINE,
    shape: PowerCurveShape = "cubic",
) -> xr.DataArray:
    """Tabulate the power curve, for inspection and plotting.

    Parameters
    ----------
    speeds
        Wind speeds in m/s at which to evaluate the curve. Defaults to
        0-30 m/s in 0.1 m/s steps.
    turbine, shape
        As for :func:`capacity_factor`.

    Returns
    -------
    xarray.DataArray
        Capacity factor indexed by a ``wind_speed`` coordinate.
    """
    if speeds is None:
        speeds = np.arange(0.0, 30.01, 0.1)
    grid = xr.DataArray(
        np.asarray(speeds, dtype=float),
        dims="wind_speed",
        coords={"wind_speed": np.asarray(speeds, dtype=float)},
    )
    return capacity_factor(grid, turbine=turbine, shape=shape)


def extrapolate_to_hub_height(
    wind_speed: xr.DataArray,
    *,
    measurement_height_m: float = 10.0,
    hub_height_m: float = REFERENCE_TURBINE.hub_height_m,
    alpha: float = 0.143,
) -> xr.DataArray:
    """Extrapolate wind speed to hub height with the power law.

    ``u(z) = u(z_ref) * (z / z_ref) ** alpha``

    The exponent defaults to 1/7, the conventional open-terrain value. In
    complex Alpine terrain the true shear exponent varies strongly with
    stability and exposure -- during the persistent winter inversions discussed
    in the paper it departs from 1/7 substantially -- so extrapolated hub-height
    winds carry considerably more uncertainty here than over flat terrain.
    Prefer a native 100 m wind field where one exists (ARA and the Climate DT
    provide one; ERA5 and ERA5-Land do not).

    Parameters
    ----------
    wind_speed
        Wind speed at ``measurement_height_m`` in m/s.
    measurement_height_m
        Height of the input wind field in metres.
    hub_height_m
        Target hub height in metres.
    alpha
        Power-law shear exponent.

    Returns
    -------
    xarray.DataArray
        Wind speed at hub height in m/s.

    Raises
    ------
    ValueError
        If either height is not positive.
    """
    if measurement_height_m <= 0 or hub_height_m <= 0:
        raise ValueError(
            "Heights must be positive; got measurement_height_m="
            f"{measurement_height_m}, hub_height_m={hub_height_m}"
        )

    factor = (hub_height_m / measurement_height_m) ** alpha
    extrapolated = wind_speed * factor
    extrapolated = extrapolated.rename(f"wind_speed_{int(hub_height_m)}m")
    extrapolated.attrs = {
        "long_name": f"Wind speed extrapolated to {hub_height_m:g} m",
        "units": "m/s",
        "method": "power law",
        "shear_exponent": alpha,
        "source_height_m": measurement_height_m,
        "target_height_m": hub_height_m,
    }
    return extrapolated
