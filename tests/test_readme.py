"""The README's quickstart must actually run.

A broken example is the worst first impression a repository can make, and it is
the code most likely to rot unnoticed because nobody runs it. This test executes
the same calls the README shows, against a synthetic file.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from alpinemet.indicators.degree_days import degree_days
from alpinemet.indicators.storms import storm_days
from alpinemet.io import ALPINE_DOMAIN, open_product

README = Path("README.md")


@pytest.fixture
def era5_file(tmp_path):
    """An ERA5-shaped file: short names, kelvin, accumulated J/m2, descending latitude."""
    time = pd.date_range("2020-06-21", periods=72, freq="h")
    lats = np.arange(50.0, 41.9, -1.0)
    lons = np.arange(4.0, 18.1, 1.0)
    dims = ("time", "latitude", "longitude")
    coords = {"time": time, "latitude": lats, "longitude": lons}
    shape = (time.size, lats.size, lons.size)

    rng = np.random.default_rng(0)
    diurnal = np.clip(800.0 * np.sin((np.arange(24) - 6) / 12 * np.pi), 0.0, None)

    path = tmp_path / "era5_2020_AT.nc"
    xr.Dataset(
        {
            "t2m": xr.DataArray(
                np.full(shape, 295.15) + rng.normal(0, 3, shape),
                dims=dims, coords=coords, attrs={"units": "K"},
            ),
            "u10": xr.DataArray(rng.normal(0, 9, shape), dims=dims, coords=coords),
            "v10": xr.DataArray(rng.normal(0, 9, shape), dims=dims, coords=coords),
            "ssrd": xr.DataArray(
                np.tile(diurnal, 3)[:, None, None] * np.ones(shape[1:]) * 3600.0,
                dims=dims, coords=coords,
            ),
        }
    ).to_netcdf(path)
    return path


def test_the_quickstart_example_runs(era5_file):
    era5 = open_product(era5_file, "era5", domain=ALPINE_DOMAIN)

    heating_and_cooling = degree_days(era5["temperature_2m"])
    storms = storm_days(era5["wind_speed_10m"], threshold=17.5)

    assert float(heating_and_cooling["cdd"].mean()) >= 0.0
    assert float(storms.max()) >= 0.0
    # The pipeline really did convert: 295 K is 22 degC, above the 18 degC base.
    assert float(era5["temperature_2m"].mean()) == pytest.approx(22.0, abs=1.0)


def test_the_readme_advertises_the_real_test_count():
    """A stale badge is a small lie that compounds."""
    if not README.exists():  # pragma: no cover - only when run from elsewhere
        pytest.skip("run from the repository root")

    text = README.read_text(encoding="utf-8")
    claimed = {int(m) for m in re.findall(r"tests-(\d+)-", text)}
    claimed |= {int(m) for m in re.findall(r"(\d+) tests", text)}

    assert claimed, "the README should state how many tests there are"
    # Allow a small drift so adding one test does not fail the suite it belongs to.
    actual = _collected_test_count()
    for number in claimed:
        assert abs(number - actual) <= 25, (
            f"README claims {number} tests, the suite collects {actual}"
        )


def _collected_test_count() -> int:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        capture_output=True,
        text=True,
    )
    match = re.search(r"(\d+) tests? collected", result.stdout)
    return int(match.group(1)) if match else 0


def test_the_readme_links_resolve():
    """Every relative link in the README must point at something that exists."""
    if not README.exists():  # pragma: no cover
        pytest.skip("run from the repository root")

    text = README.read_text(encoding="utf-8")
    targets = re.findall(r"\]\((?!https?://|#|mailto:)([^)]+)\)", text)

    missing = [target for target in targets if not Path(target.split("#")[0]).exists()]
    assert not missing, f"README links to missing paths: {missing}"
