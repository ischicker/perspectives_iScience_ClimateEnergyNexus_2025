"""Accumulated-to-instantaneous conversion, the factor-of-3600 trap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from alpinemet.attrs import decode_attribute
from alpinemet.io.accumulation import (
    AccumulationKind,
    infer_accumulation_kind,
    to_rate,
)

# A plausible clear-sky diurnal irradiance cycle in W/m2.
DIURNAL_FLUX = np.clip(800.0 * np.sin((np.arange(24) - 6) / 12 * np.pi), 0.0, None)


def _hourly(values, name="solar_radiation") -> xr.DataArray:
    array = np.asarray(values, dtype=float)
    time = pd.date_range("2021-06-21", periods=array.shape[0], freq="h")
    return xr.DataArray(array, dims="time", coords={"time": time}, name=name)


# --------------------------------------------------------------------------
# Instantaneous
# --------------------------------------------------------------------------


def test_instantaneous_data_pass_through_unchanged():
    flux = _hourly(DIURNAL_FLUX)
    converted = to_rate(flux, AccumulationKind.INSTANTANEOUS)
    np.testing.assert_allclose(converted.values, flux.values)
    assert converted.attrs["conversion_method"] == "none"


# --------------------------------------------------------------------------
# Period accumulation (ERA5 style)
# --------------------------------------------------------------------------


def test_period_accumulation_divides_by_the_time_step():
    """J/m2 accumulated over one hour, divided by 3600 s, is W/m2."""
    accumulated = _hourly(DIURNAL_FLUX * 3600.0)
    converted = to_rate(accumulated, AccumulationKind.PERIOD)
    np.testing.assert_allclose(converted.values, DIURNAL_FLUX)


def test_period_conversion_recovers_realistic_magnitudes():
    accumulated = _hourly(DIURNAL_FLUX * 3600.0)
    converted = to_rate(accumulated, AccumulationKind.PERIOD)
    assert float(converted.min()) >= 0.0
    assert float(converted.max()) < 1400.0


def test_three_hourly_accumulation_uses_the_right_step():
    time = pd.date_range("2021-06-21", periods=8, freq="3h")
    # 500 W/m2 sustained over three hours.
    accumulated = xr.DataArray(
        np.full(8, 500.0 * 3 * 3600.0), dims="time", coords={"time": time}
    )
    converted = to_rate(accumulated, AccumulationKind.PERIOD)
    np.testing.assert_allclose(converted.values, 500.0)


def test_explicit_timestep_overrides_inference():
    accumulated = _hourly(np.full(5, 3600.0))
    converted = to_rate(accumulated, AccumulationKind.PERIOD, timestep_seconds=1800.0)
    np.testing.assert_allclose(converted.values, 2.0)


# --------------------------------------------------------------------------
# Running accumulation (ERA5-Land style)
# --------------------------------------------------------------------------


def test_running_accumulation_is_differenced_then_divided():
    running = _hourly(np.cumsum(DIURNAL_FLUX) * 3600.0)
    converted = to_rate(running, AccumulationKind.RUNNING)
    # The first step has no predecessor and is reported as zero.
    np.testing.assert_allclose(converted.values[1:], DIURNAL_FLUX[1:], atol=1e-9)
    assert converted.values[0] == 0.0


def test_the_time_axis_is_preserved_across_the_difference():
    running = _hourly(np.cumsum(DIURNAL_FLUX) * 3600.0)
    converted = to_rate(running, AccumulationKind.RUNNING)
    assert converted.sizes["time"] == running.sizes["time"]
    np.testing.assert_array_equal(converted["time"].values, running["time"].values)


def test_the_daily_reset_is_clipped_not_propagated():
    """ERA5-Land resets its accumulator at 00 UTC; the jump must not leak through."""
    day = np.cumsum(DIURNAL_FLUX) * 3600.0
    running = _hourly(np.concatenate([day, day]))  # two days, reset between

    converted = to_rate(running, AccumulationKind.RUNNING)

    assert float(converted.min()) >= 0.0, "reset must not produce a negative rate"
    # The reset step itself reports zero rather than a huge negative value.
    assert converted.values[24] == 0.0


def test_disabling_the_clip_exposes_the_reset():
    day = np.cumsum(DIURNAL_FLUX) * 3600.0
    running = _hourly(np.concatenate([day, day]))
    converted = to_rate(running, AccumulationKind.RUNNING, clip_negative=False)
    assert float(converted.min()) < 0.0


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_double_conversion_is_refused():
    accumulated = _hourly(DIURNAL_FLUX * 3600.0)
    once = to_rate(accumulated, AccumulationKind.PERIOD)
    with pytest.raises(ValueError, match="already marked as converted"):
        to_rate(once, AccumulationKind.PERIOD)


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        to_rate(_hourly(DIURNAL_FLUX), "hourly_mean")


def test_string_kinds_are_accepted():
    converted = to_rate(_hourly(DIURNAL_FLUX * 3600.0), "period")
    np.testing.assert_allclose(converted.values, DIURNAL_FLUX)


def test_units_are_recorded_when_given():
    converted = to_rate(
        _hourly(DIURNAL_FLUX * 3600.0), AccumulationKind.PERIOD, output_units="W m-2"
    )
    assert converted.attrs["units"] == "W m-2"
    assert converted.attrs["accumulation_kind"] == "period"


# --------------------------------------------------------------------------
# Inference fallback
# --------------------------------------------------------------------------


def test_instantaneous_magnitudes_are_recognised():
    assert infer_accumulation_kind(_hourly(DIURNAL_FLUX)) is AccumulationKind.INSTANTANEOUS


def test_period_magnitudes_are_recognised():
    assert (
        infer_accumulation_kind(_hourly(DIURNAL_FLUX * 3600.0))
        is AccumulationKind.PERIOD
    )


def test_running_magnitudes_are_recognised():
    running = _hourly(np.cumsum(DIURNAL_FLUX) * 3600.0 + 5_000_000.0)
    assert infer_accumulation_kind(running) is AccumulationKind.RUNNING


def test_a_bright_summer_day_is_still_period_accumulation():
    """The case the old 100k-500k magnitude band would have missed.

    A clear Alpine summer day averages roughly 250 W/m2, i.e. 900,000 J/m2 per
    hour, above the old upper bound -- it would have fallen through to the
    "unknown" branch and been left unconverted.
    """
    bright = _hourly(DIURNAL_FLUX * 3600.0)
    assert float(bright.mean()) > 500_000.0
    assert infer_accumulation_kind(bright) is AccumulationKind.PERIOD


def test_inference_works_on_gridded_input():
    time = pd.date_range("2021-06-21", periods=24, freq="h")
    field = np.repeat(DIURNAL_FLUX[:, None, None] * 3600.0, 4, axis=1).repeat(3, axis=2)
    gridded = xr.DataArray(
        field,
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": [46.0, 46.5, 47.0, 47.5],
                "longitude": [11.0, 12.0, 13.0]},
    )
    assert infer_accumulation_kind(gridded) is AccumulationKind.PERIOD


def test_ambiguous_series_raise_rather_than_pass_through():
    """The failure mode the old value-range heuristic had: silent pass-through."""
    # Two decreasing steps out of eight: 25 %, between the 15 % and 35 % bounds.
    steps = [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0]
    ambiguous = _hourly(np.cumsum([10_000_000.0, *(s * 500_000.0 for s in steps)]))
    with pytest.raises(ValueError, match="are decreasing, between the bounds"):
        infer_accumulation_kind(ambiguous)


def test_a_constant_series_cannot_be_judged():
    with pytest.raises(ValueError, match="constant in time"):
        infer_accumulation_kind(_hourly(np.full(24, 500_000.0)))


def test_too_short_a_sample_is_rejected():
    with pytest.raises(ValueError, match="at least three time steps"):
        infer_accumulation_kind(_hourly([500_000.0, 600_000.0]))


# --------------------------------------------------------------------------
# Per-step totals versus per-second rates
# --------------------------------------------------------------------------


def test_a_per_step_total_is_not_divided_by_the_time_step():
    """Precipitation: an hourly total in mm is already the quantity wanted.

    Dividing it by 3600 gives mm/s, a thousandfold too small and entirely
    plausible-looking, which is exactly how such an error survives review.
    """
    hourly_mm = _hourly([0.0, 2.5, 9.6, 0.1])
    converted = to_rate(hourly_mm, AccumulationKind.PERIOD, per_second=False)
    np.testing.assert_allclose(converted.values, [0.0, 2.5, 9.6, 0.1])


def test_a_flux_is_divided_by_the_time_step():
    accumulated = _hourly(DIURNAL_FLUX * 3600.0)
    converted = to_rate(accumulated, AccumulationKind.PERIOD, per_second=True)
    np.testing.assert_allclose(converted.values, DIURNAL_FLUX)


def test_running_totals_are_differenced_without_dividing():
    """ERA5-Land precipitation: difference the accumulator, keep millimetres."""
    increments = np.array([0.0, 2.0, 0.0, 5.0, 1.0])
    running = _hourly(np.cumsum(increments))
    converted = to_rate(running, AccumulationKind.RUNNING, per_second=False)
    np.testing.assert_allclose(converted.values, increments)


def test_the_choice_is_recorded():
    total = to_rate(_hourly([1.0, 2.0, 3.0]), AccumulationKind.PERIOD, per_second=False)
    flux = to_rate(_hourly([1.0, 2.0, 3.0]), AccumulationKind.PERIOD, per_second=True)
    assert decode_attribute(total.attrs["per_second"]) is False
    assert decode_attribute(flux.attrs["per_second"]) is True
    assert total.attrs["conversion_method"] == "already per step"
