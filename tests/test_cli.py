"""The CLI, exercised end to end on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from alpinemet.cli import build_parser, main

DIURNAL_FLUX = np.clip(800.0 * np.sin((np.arange(24) - 6) / 12 * np.pi), 0.0, None)


@pytest.fixture
def era5_file(tmp_path):
    """A small ERA5-shaped file inside a data root."""
    data_root = tmp_path / "DATA" / "ERA5"
    data_root.mkdir(parents=True)

    time = pd.date_range("2020-06-21", periods=48, freq="h")
    lats = np.arange(52.0, 39.9, -1.0)
    lons = np.arange(0.0, 25.1, 1.0)
    dims = ("time", "latitude", "longitude")
    coords = {"time": time, "latitude": lats, "longitude": lons}
    shape = (time.size, lats.size, lons.size)
    flux = np.tile(DIURNAL_FLUX, 2)[:, None, None] * np.ones(shape[1:])

    xr.Dataset(
        {
            "t2m": xr.DataArray(np.full(shape, 288.15), dims=dims, coords=coords,
                                attrs={"units": "K"}),
            "u10": xr.DataArray(np.full(shape, 3.0), dims=dims, coords=coords),
            "v10": xr.DataArray(np.full(shape, 4.0), dims=dims, coords=coords),
            "ssrd": xr.DataArray(flux * 3600.0, dims=dims, coords=coords),
        }
    ).to_netcdf(data_root / "era5_2020.nc")

    return tmp_path


@pytest.fixture
def configs(era5_file):
    """A valid analysis config and paths file pointing at the fixture data."""
    config = era5_file / "analysis.yaml"
    config.write_text(
        "name: cli-test\n"
        "period:\n"
        '  start: "2020-06-21"\n'
        '  end: "2020-06-22"\n'
        "datasets:\n"
        "  - key: era5\n"
        '    paths: "ERA5/era5_2020.nc"\n',
        encoding="utf-8",
    )
    paths = era5_file / "paths.yaml"
    paths.write_text(
        f"data_root: {(era5_file / 'DATA').as_posix()}\n"
        f"output_root: {(era5_file / 'out').as_posix()}\n",
        encoding="utf-8",
    )
    return config, paths


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def test_no_command_prints_help_and_fails(capsys):
    assert main([]) == 1
    assert "usage" in capsys.readouterr().out


def test_version_is_reported(capsys):
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--version"])
    assert excinfo.value.code == 0
    assert "alpinemet" in capsys.readouterr().out


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------


def test_datasets_lists_every_product(capsys):
    assert main(["datasets"]) == 0
    output = capsys.readouterr().out
    for key in ("era5", "era5_land", "cerra", "ara", "climate_dt", "extremes_dt"):
        assert key in output


def test_datasets_reports_resolution_and_gust_availability(capsys):
    main(["datasets"])
    output = capsys.readouterr().out
    assert "31.0 km" in output
    assert "2.5 km" in output


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------


def test_check_accepts_a_valid_configuration(configs, capsys):
    config, paths = configs
    assert main(["check", "-c", str(config), "-p", str(paths)]) == 0
    output = capsys.readouterr().out
    assert "Configuration is valid" in output
    assert "cli-test" in output


def test_check_reports_a_missing_input(configs, capsys):
    config, paths = configs
    config.write_text(
        "name: broken\ndatasets:\n  - key: era5\n    paths: \"ERA5/absent.nc\"\n",
        encoding="utf-8",
    )
    assert main(["check", "-c", str(config), "-p", str(paths)]) == 1
    assert "MISSING" in capsys.readouterr().out


def test_check_reports_an_empty_glob(configs, capsys):
    config, paths = configs
    config.write_text(
        "name: broken\ndatasets:\n  - key: era5\n    paths: \"ERA5/nothing_*.nc\"\n",
        encoding="utf-8",
    )
    assert main(["check", "-c", str(config), "-p", str(paths)]) == 1
    assert "NO MATCH" in capsys.readouterr().out


def test_check_opens_no_data_files(configs, capsys, monkeypatch):
    """The whole point of check: it must be instant, so it must not read data."""
    import xarray

    def _fail(*args, **kwargs):
        raise AssertionError("check must not open datasets")

    monkeypatch.setattr(xarray, "open_mfdataset", _fail)
    monkeypatch.setattr(xarray, "open_dataset", _fail)

    config, paths = configs
    assert main(["check", "-c", str(config), "-p", str(paths)]) == 0


def test_a_bad_configuration_gives_a_message_not_a_traceback(tmp_path, capsys):
    config = tmp_path / "bad.yaml"
    config.write_text("datasets: []\n", encoding="utf-8")
    paths = tmp_path / "paths.yaml"
    paths.write_text("data_root: /data\n", encoding="utf-8")

    assert main(["check", "-c", str(config), "-p", str(paths)]) == 1
    assert "error:" in capsys.readouterr().err


def test_a_missing_paths_file_points_at_the_example(configs, tmp_path, capsys):
    config, _ = configs
    assert main(["check", "-c", str(config), "-p", str(tmp_path / "absent.yaml")]) == 1
    assert "paths.example.yaml" in capsys.readouterr().err


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def test_run_writes_a_standardised_dataset(configs, era5_file, capsys):
    config, paths = configs
    assert main(["run", "-c", str(config), "-p", str(paths)]) == 0

    written = era5_file / "out" / "cli-test" / "era5_standardised.nc"
    assert written.exists()
    assert "wrote" in capsys.readouterr().out


def test_the_written_dataset_is_actually_standardised(configs, era5_file):
    config, paths = configs
    main(["run", "-c", str(config), "-p", str(paths)])

    written = era5_file / "out" / "cli-test" / "era5_standardised.nc"
    with xr.open_dataset(written) as result:
        assert "temperature_2m" in result.data_vars
        assert "wind_speed_10m" in result.data_vars
        # Kelvin converted, radiation turned into a flux.
        assert float(result["temperature_2m"].max()) == pytest.approx(15.0)
        assert float(result["solar_radiation"].max()) == pytest.approx(800.0)
        assert result.attrs["dataset_key"] == "era5"


def test_the_output_directory_can_be_overridden(configs, tmp_path):
    config, paths = configs
    destination = tmp_path / "elsewhere"
    assert main(["run", "-c", str(config), "-p", str(paths), "-o", str(destination)]) == 0
    assert (destination / "era5_standardised.nc").exists()


def test_the_configured_domain_is_applied(configs, era5_file):
    config, paths = configs
    main(["run", "-c", str(config), "-p", str(paths)])

    with xr.open_dataset(era5_file / "out" / "cli-test" / "era5_standardised.nc") as result:
        assert float(result["latitude"].min()) >= 43.0
        assert float(result["latitude"].max()) <= 49.0
