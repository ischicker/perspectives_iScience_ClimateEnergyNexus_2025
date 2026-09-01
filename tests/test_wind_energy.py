"""The power curve is checked at its defining breakpoints."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from alpinemet.energy.wind import (
    REFERENCE_TURBINE,
    TurbineSpec,
    capacity_factor,
    extrapolate_to_hub_height,
    power_curve,
    power_output,
)


def _cf_at(speeds, **kwargs) -> np.ndarray:
    field = xr.DataArray(np.asarray(speeds, dtype=float), dims="time")
    return capacity_factor(field, **kwargs).values


def test_reference_turbine_matches_the_published_specification():
    assert REFERENCE_TURBINE.rated_power_kw == 3000.0
    assert REFERENCE_TURBINE.hub_height_m == 100.0
    assert REFERENCE_TURBINE.cut_in_speed == 3.0
    assert REFERENCE_TURBINE.rated_speed == 12.0
    assert REFERENCE_TURBINE.cut_out_speed == 25.0


@pytest.mark.parametrize("shape", ["cubic", "linear"])
def test_breakpoints_are_exact(shape):
    # Below cut-in: nothing. At rated and up to just below cut-out: full output.
    # At and above cut-out: shut down.
    cf = _cf_at([0.0, 2.9, 3.0, 12.0, 24.9, 25.0, 40.0], shape=shape)
    np.testing.assert_allclose(cf[[0, 1]], 0.0)
    assert cf[2] == pytest.approx(0.0, abs=1e-12)  # exactly at cut-in
    np.testing.assert_allclose(cf[[3, 4]], 1.0)
    np.testing.assert_allclose(cf[[5, 6]], 0.0)


@pytest.mark.parametrize("shape", ["cubic", "linear"])
def test_capacity_factor_stays_within_unit_interval(shape, wind_speed_field):
    cf = capacity_factor(wind_speed_field, shape=shape)
    assert float(cf.min()) >= 0.0
    assert float(cf.max()) <= 1.0


@pytest.mark.parametrize("shape", ["cubic", "linear"])
def test_curve_is_monotonic_between_cut_in_and_rated(shape):
    speeds = np.linspace(3.0, 12.0, 50)
    cf = _cf_at(speeds, shape=shape)
    assert np.all(np.diff(cf) >= -1e-12)


def test_cubic_form_follows_the_cube_law():
    u = 7.5
    expected = (u**3 - 3.0**3) / (12.0**3 - 3.0**3)
    assert _cf_at([u], shape="cubic")[0] == pytest.approx(expected)


def test_linear_form_is_the_straight_interpolation():
    u = 7.5
    expected = (u - 3.0) / (12.0 - 3.0)
    assert _cf_at([u], shape="linear")[0] == pytest.approx(expected)


def test_cubic_sits_below_linear_in_the_ramp():
    """Power scales with u**3, so the cubic curve yields less at mid speeds."""
    speeds = np.linspace(3.5, 11.5, 20)
    assert np.all(_cf_at(speeds, shape="cubic") < _cf_at(speeds, shape="linear"))


def test_power_output_scales_the_capacity_factor():
    speeds = [0.0, 6.0, 12.0, 30.0]
    cf = _cf_at(speeds)
    field = xr.DataArray(np.asarray(speeds, dtype=float), dims="time")
    power = power_output(field).values
    np.testing.assert_allclose(power, cf * REFERENCE_TURBINE.rated_power_kw)
    assert power.max() == pytest.approx(3000.0)


def test_unknown_shape_is_rejected(wind_speed_field):
    with pytest.raises(ValueError, match="Unknown power curve shape"):
        capacity_factor(wind_speed_field, shape="sigmoid")


def test_result_records_the_turbine_used(wind_speed_field):
    cf = capacity_factor(wind_speed_field)
    assert cf.attrs["turbine"] == "reference_3MW"
    assert cf.attrs["rated_power_kW"] == 3000.0
    assert cf.attrs["hub_height_m"] == 100.0
    assert cf.attrs["power_curve_shape"] == "cubic"


def test_power_curve_tabulation_spans_the_full_range():
    curve = power_curve()
    assert curve.dims == ("wind_speed",)
    assert float(curve.max()) == pytest.approx(1.0)
    assert float(curve.sel(wind_speed=26.0, method="nearest")) == 0.0


@pytest.mark.parametrize(
    ("rated_power", "hub", "cut_in", "rated", "cut_out"),
    [
        (3000.0, 100.0, 12.0, 3.0, 25.0),  # rated below cut-in
        (3000.0, 100.0, 3.0, 30.0, 25.0),  # rated above cut-out
        (0.0, 100.0, 3.0, 12.0, 25.0),  # non-positive rated power
        (3000.0, 0.0, 3.0, 12.0, 25.0),  # non-positive hub height
    ],
)
def test_inconsistent_turbine_specs_are_rejected(rated_power, hub, cut_in, rated, cut_out):
    with pytest.raises(ValueError):
        TurbineSpec(
            name="broken",
            rated_power_kw=rated_power,
            hub_height_m=hub,
            cut_in_speed=cut_in,
            rated_speed=rated,
            cut_out_speed=cut_out,
        )


def test_hub_height_extrapolation_follows_the_power_law():
    wind = xr.DataArray([5.0], dims="time")
    extrapolated = extrapolate_to_hub_height(wind, measurement_height_m=10.0, hub_height_m=100.0)
    np.testing.assert_allclose(extrapolated.values, [5.0 * 10.0**0.143])


def test_extrapolation_to_the_same_height_is_the_identity():
    wind = xr.DataArray([5.0, 8.0], dims="time")
    same = extrapolate_to_hub_height(wind, measurement_height_m=10.0, hub_height_m=10.0)
    np.testing.assert_allclose(same.values, wind.values)


def test_non_positive_heights_are_rejected():
    wind = xr.DataArray([5.0], dims="time")
    with pytest.raises(ValueError, match="Heights must be positive"):
        extrapolate_to_hub_height(wind, hub_height_m=0.0)
