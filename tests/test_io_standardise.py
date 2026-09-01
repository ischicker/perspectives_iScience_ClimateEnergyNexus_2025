"""Duplicate removal, derived wind speed and quality control."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from alpinemet.attrs import decode_attribute
from alpinemet.io.standardise import (
    PHYSICAL_RANGES,
    apply_quality_control,
    derive_wind_speeds,
    remove_duplicate_times,
)


def _series(**variables) -> xr.Dataset:
    length = len(next(iter(variables.values())))
    time = pd.date_range("2021-01-01", periods=length, freq="h")
    return xr.Dataset(
        {
            name: xr.DataArray(
                np.asarray(values, dtype=float), dims="time", coords={"time": time}
            )
            for name, values in variables.items()
        }
    )


# --------------------------------------------------------------------------
# Duplicate timestamps
# --------------------------------------------------------------------------


def _with_duplicate_times() -> xr.Dataset:
    times = pd.DatetimeIndex(
        [
            "2021-01-01 00:00", "2021-01-01 01:00", "2021-01-01 01:00",
            "2021-01-01 02:00", "2021-01-01 03:00",
        ]
    )
    return xr.Dataset(
        {
            "temperature_2m": xr.DataArray(
                np.array([1.0, 2.0, 99.0, 3.0, 4.0]),
                dims="time",
                coords={"time": times},
            )
        }
    )


def test_duplicates_are_removed_keeping_the_first():
    cleaned = remove_duplicate_times(_with_duplicate_times())
    assert cleaned.sizes["time"] == 4
    np.testing.assert_allclose(
        cleaned["temperature_2m"].values, [1.0, 2.0, 3.0, 4.0]
    )


def test_the_number_removed_is_recorded():
    assert remove_duplicate_times(_with_duplicate_times()).attrs[
        "duplicate_times_removed"
    ] == 1


def test_a_clean_dataset_is_unchanged():
    ds = _series(temperature_2m=[1.0, 2.0, 3.0])
    cleaned = remove_duplicate_times(ds)
    assert cleaned.attrs["duplicate_times_removed"] == 0
    np.testing.assert_allclose(cleaned["temperature_2m"].values, [1.0, 2.0, 3.0])


def test_output_is_sorted_in_time():
    times = pd.DatetimeIndex(["2021-01-01 02:00", "2021-01-01 00:00", "2021-01-01 01:00"])
    ds = xr.Dataset(
        {
            "temperature_2m": xr.DataArray(
                np.array([3.0, 1.0, 2.0]), dims="time", coords={"time": times}
            )
        }
    )
    np.testing.assert_allclose(
        remove_duplicate_times(ds)["temperature_2m"].values, [1.0, 2.0, 3.0]
    )


def test_duplicates_would_corrupt_a_difference():
    """Why this matters: a repeated step makes diff return zero there."""
    dirty = _with_duplicate_times()
    assert float(dirty["temperature_2m"].diff("time").min()) < 0  # 99 -> 3
    cleaned = remove_duplicate_times(dirty)
    np.testing.assert_allclose(cleaned["temperature_2m"].diff("time").values, 1.0)


# --------------------------------------------------------------------------
# Derived wind speed
# --------------------------------------------------------------------------


def test_wind_speed_is_the_vector_magnitude():
    ds = _series(u_wind_10m=[3.0, 0.0], v_wind_10m=[4.0, 5.0])
    result = derive_wind_speeds(ds)
    np.testing.assert_allclose(result["wind_speed_10m"].values, [5.0, 5.0])


def test_both_heights_are_derived_when_available():
    ds = _series(
        u_wind_10m=[3.0], v_wind_10m=[4.0], u_wind_100m=[6.0], v_wind_100m=[8.0]
    )
    result = derive_wind_speeds(ds)
    assert result["wind_speed_10m"].item() == pytest.approx(5.0)
    assert result["wind_speed_100m"].item() == pytest.approx(10.0)


def test_a_height_without_components_is_skipped():
    """ERA5 has no 100 m components over the Alpine domain."""
    ds = _series(u_wind_10m=[3.0], v_wind_10m=[4.0])
    result = derive_wind_speeds(ds)
    assert "wind_speed_10m" in result
    assert "wind_speed_100m" not in result


def test_an_existing_native_speed_is_preserved():
    ds = _series(u_wind_10m=[3.0], v_wind_10m=[4.0], wind_speed_10m=[99.0])
    result = derive_wind_speeds(ds)
    assert result["wind_speed_10m"].item() == pytest.approx(99.0)


def test_overwrite_recomputes_the_speed():
    ds = _series(u_wind_10m=[3.0], v_wind_10m=[4.0], wind_speed_10m=[99.0])
    result = derive_wind_speeds(ds, overwrite=True)
    assert result["wind_speed_10m"].item() == pytest.approx(5.0)


def test_derived_speed_is_never_negative():
    ds = _series(u_wind_10m=[-3.0, -7.0], v_wind_10m=[-4.0, 0.0])
    result = derive_wind_speeds(ds)
    assert bool((result["wind_speed_10m"] >= 0).all())


def test_derived_speed_carries_units():
    ds = _series(u_wind_10m=[3.0], v_wind_10m=[4.0])
    assert derive_wind_speeds(ds)["wind_speed_10m"].attrs["units"] == "m/s"


# --------------------------------------------------------------------------
# Quality control
# --------------------------------------------------------------------------


def test_out_of_range_values_are_masked():
    # 500 degC is a unit error, not weather.
    ds = _series(temperature_2m=[20.0, 500.0, 15.0])
    checked = apply_quality_control(ds)
    assert np.isnan(checked["temperature_2m"].values[1])
    assert decode_attribute(checked.attrs["quality_control"])["temperature_2m"] == 1


def test_kelvin_left_unconverted_is_caught():
    """The canonical symptom: temperatures near 273 in a field labelled degC."""
    ds = _series(temperature_2m=[273.15, 280.0, 290.0])
    checked = apply_quality_control(ds)
    assert decode_attribute(checked.attrs["quality_control"])["temperature_2m"] == 3


def test_genuine_extremes_survive():
    # -40 degC in an Alpine valley and a 45 m/s gust are real.
    ds = _series(temperature_2m=[-40.0], wind_gust_10m=[45.0])
    checked = apply_quality_control(ds)
    assert decode_attribute(checked.attrs["quality_control"]) == {}
    assert not np.isnan(checked["temperature_2m"].values).any()


def test_report_mode_leaves_the_data_untouched():
    ds = _series(temperature_2m=[20.0, 500.0])
    checked = apply_quality_control(ds, action="report")
    np.testing.assert_allclose(checked["temperature_2m"].values, [20.0, 500.0])
    assert decode_attribute(checked.attrs["quality_control"])["temperature_2m"] == 1


def test_unknown_action_is_rejected():
    with pytest.raises(ValueError, match="Unknown action"):
        apply_quality_control(_series(temperature_2m=[1.0]), action="drop")


def test_variables_without_a_declared_range_are_ignored():
    ds = _series(temperature_2m=[20.0], some_diagnostic=[1e9])
    checked = apply_quality_control(ds)
    assert checked["some_diagnostic"].item() == pytest.approx(1e9)


def test_every_canonical_variable_has_a_range():
    from alpinemet.io.naming import CANONICAL_VARIABLES

    assert set(PHYSICAL_RANGES) == set(CANONICAL_VARIABLES)


# --------------------------------------------------------------------------
# Precipitation units
# --------------------------------------------------------------------------


def test_metres_of_water_equivalent_are_converted():
    """ECMWF reports precipitation in metres; the error is otherwise silent."""
    from alpinemet.units import to_millimetres

    precip = _series(precipitation=[0.001, 0.005, 0.0])["precipitation"]
    precip.attrs["units"] = "m"
    converted = to_millimetres(precip)

    np.testing.assert_allclose(converted.values, [1.0, 5.0, 0.0])
    assert converted.attrs["units"] == "mm"


def test_millimetres_are_left_alone():
    from alpinemet.units import to_millimetres

    precip = _series(precipitation=[1.0, 5.0])["precipitation"]
    precip.attrs["units"] = "mm"
    np.testing.assert_allclose(to_millimetres(precip).values, [1.0, 5.0])


def test_kg_per_square_metre_is_treated_as_millimetres():
    """ARA reports kg m-2, which equals mm for water."""
    from alpinemet.units import to_millimetres

    precip = _series(precipitation=[2.0, 3.0])["precipitation"]
    precip.attrs["units"] = "kg m**-2"
    np.testing.assert_allclose(to_millimetres(precip).values, [2.0, 3.0])


def test_undeclared_units_fall_back_to_a_range_check():
    from alpinemet.units import to_millimetres

    # No hourly total on Earth reaches 0.5 m, so these must already be mm.
    already_mm = _series(precipitation=[0.0, 3.0, 12.0])["precipitation"]
    np.testing.assert_allclose(to_millimetres(already_mm).values, [0.0, 3.0, 12.0])

    # Values this small can only be metres.
    metres = _series(precipitation=[0.0, 0.003, 0.012])["precipitation"]
    np.testing.assert_allclose(to_millimetres(metres).values, [0.0, 3.0, 12.0])


def test_an_unknown_precipitation_unit_is_rejected():
    from alpinemet.units import to_millimetres

    precip = _series(precipitation=[1.0])["precipitation"]
    with pytest.raises(ValueError, match="Cannot interpret"):
        to_millimetres(precip, assume="inches")
