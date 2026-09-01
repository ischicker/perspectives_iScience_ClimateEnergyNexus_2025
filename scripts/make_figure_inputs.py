#!/usr/bin/env python3
"""Derive the small two-dimensional fields behind the manuscript figures.

The raw model fields run to terabytes and cannot be distributed. The figures
need only time-reduced two-dimensional fields, which come to a few megabytes.
This script produces them from the raw data, using the same library the rest of
the analysis uses, so that the archived inputs and the published code agree.

Usage::

    uv run python scripts/make_figure_inputs.py --paths configs/paths.yaml
    uv run python scripts/make_figure_inputs.py --only figure2 --out figure_inputs

Each output records in its attributes which product it came from, which
indicator definition was applied, and the thresholds used, so an archived file
remains self-explanatory once separated from this script.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

from alpinemet import __version__
from alpinemet.attrs import as_attribute
from alpinemet.config import load_paths
from alpinemet.energy.wind import extrapolate_to_hub_height
from alpinemet.indicators.degree_days import cooling_degree_days, heating_degree_days
from alpinemet.indicators.storms import storm_days
from alpinemet.indicators.thresholds import exceedance_hours, hot_days
from alpinemet.io import ALPINE_DOMAIN, BoundingBox, open_product, require_variable

LOGGER = logging.getLogger("make_figure_inputs")

# --- Definitions, matching the manuscript figure captions --------------------

HOT_DAY_THRESHOLD_DEGC = 30.0
CDD_BASE_DEGC = 18.0
HDD_BASE_DEGC = 15.0
STORM_DAY_THRESHOLD_MS = 17.5   # Figure 2: storm days
STORM_HOUR_THRESHOLD_MS = 15.0  # Figure 4: storm hours

#: Storm Benjamin domain, wider than the Alpine evaluation box.
BENJAMIN_DOMAIN = BoundingBox(5.0, 19.0, 45.5, 50.0, name="Storm Benjamin domain")

FIGURE2_PRODUCTS = [
    ("era5", "ERA5/era5_2020_AT.nc", "ERA5 (31 km)"),
    ("era5_land", "ERA5Land/era5land_single_levels_2020_AT_combined.nc", "ERA5-Land (9 km)"),
    ("ara", "ARA/ARA_2020??_Alpine.nc", "ARA (2.5 km)"),
]

FIGURE3_PRODUCTS = [
    ("ifs", "IFS_ENS/ENS_IFS_Benjamin_20251023.nc", "IFS ENS control (9 km)"),
    ("extremes_dt", "EXTREMES_DT/extremesDT_stormBenjamin_20251023_alps0p04deg.nc",
     "Extremes DT (4.4 km)"),
    ("aifs", "AIFS/AIFS_20251023_00_stormBenjamin.nc", "AIFS (~31 km)"),
]


def _provenance(product: str, label: str, **extra) -> dict:
    return {
        "product": product,
        "label": label,
        "created_by": f"alpinemet {__version__} scripts/make_figure_inputs.py",
        **{key: as_attribute(value) for key, value in extra.items()},
    }


def _write(dataset: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path)
    size_kb = path.stat().st_size / 1024
    LOGGER.info("  wrote %s (%.0f kB)", path.name, size_kb)


# --- Figure 2: reanalysis comparison, 2020 -----------------------------------


def make_figure2(data_root: Path, out_dir: Path) -> None:
    """Hot days, cooling degree days and storm days for 2020."""
    LOGGER.info("Figure 2: reanalysis comparison, 2020")

    for key, pattern, label in FIGURE2_PRODUCTS:
        started = time.perf_counter()
        LOGGER.info("  %s", label)

        product = open_product(
            str(data_root / pattern),
            key,
            domain=ALPINE_DOMAIN,
            start="2020-01-01",
            end="2020-12-31",
            chunks={"time": 240},
        )

        # require_variable rather than indexing: if a product carries no 2 m
        # temperature the message names what was tried and what is present,
        # which is the difference between a five-minute fix and an afternoon.
        temperature = require_variable(product, "temperature_2m")
        wind = require_variable(product, "wind_speed_10m")

        result = xr.Dataset(
            {
                "hot_days": hot_days(temperature, threshold=HOT_DAY_THRESHOLD_DEGC),
                "cdd": cooling_degree_days(
                    temperature, base=CDD_BASE_DEGC, already_daily=False
                ),
                "storm_days": storm_days(wind, threshold=STORM_DAY_THRESHOLD_MS),
            }
        ).compute()

        result.attrs = _provenance(
            key,
            label,
            period="2020-01-01 to 2020-12-31",
            domain=ALPINE_DOMAIN.name,
            resolution_km=product.attrs["resolution_km"],
            hot_day_threshold_degC=HOT_DAY_THRESHOLD_DEGC,
            cdd_base_degC=CDD_BASE_DEGC,
            storm_day_threshold_ms=STORM_DAY_THRESHOLD_MS,
            n_time_steps=int(product.sizes["time"]),
            duplicate_times_removed=product.attrs.get("duplicate_times_removed", 0),
        )
        product.close()

        _write(result, out_dir / f"figure2_{key}.nc")
        LOGGER.info(
            "    hot days max %.0f, CDD mean %.0f, storm days max %.0f  [%.0f s]",
            float(result["hot_days"].max()),
            float(result["cdd"].mean()),
            float(result["storm_days"].max()),
            time.perf_counter() - started,
        )


# --- Figure 3: Storm Benjamin ------------------------------------------------


def make_figure3(data_root: Path, out_dir: Path) -> None:
    """Maximum gust and maximum 10 m wind over T+0 to T+36 h."""
    LOGGER.info("Figure 3: Storm Benjamin, 23 October 2025")

    for key, relative, label in FIGURE3_PRODUCTS:
        LOGGER.info("  %s", label)
        raw = xr.open_dataset(data_root / relative)

        # The IFS file holds all 51 members; the paper uses the control.
        if "number" in raw.dims:
            numbers = np.asarray(raw["number"].values)
            raw = raw.sel(number=0) if 0 in numbers else raw.isel(number=0)
            LOGGER.info("    selected control member")

        # Forecasts are indexed by lead time rather than validity time.
        forecast_dim = "step" if "step" in raw.dims else "time"

        product = xr.Dataset()
        gust_name = next((n for n in ("i10fg", "gust10m") if n in raw.data_vars), None)
        if gust_name is not None:
            product["max_gust"] = raw[gust_name].max(dim=forecast_dim)
            product["max_gust"].attrs = {
                "long_name": "Maximum 10 m wind gust", "units": "m/s",
                "gust_source": "native", "native_variable": gust_name,
            }

        if "wind_speed_10m" in raw.data_vars:
            wind = raw["wind_speed_10m"]
        else:
            u = next(n for n in ("10u", "u10") if n in raw.data_vars)
            v = next(n for n in ("10v", "v10") if n in raw.data_vars)
            wind = np.sqrt(raw[u] ** 2 + raw[v] ** 2)
        product["max_wind"] = wind.max(dim=forecast_dim)
        product["max_wind"].attrs = {
            "long_name": "Maximum 10 m wind speed", "units": "m/s"
        }

        for coordinate in ("latitude", "longitude"):
            if coordinate in raw.coords:
                product = product.assign_coords({coordinate: raw[coordinate]})

        product = product.compute()
        product.attrs = _provenance(
            key,
            label,
            event="Storm Benjamin",
            initialisation="2025-10-23 00 UTC",
            lead_times="T+0 to T+36 h",
            has_native_gust=gust_name is not None,
            grid="scattered" if "values" in product.dims else "regular",
        )
        raw.close()

        _write(product, out_dir / f"figure3_{key}.nc")
        LOGGER.info(
            "    max wind %.1f m/s%s",
            float(product["max_wind"].max()),
            f", max gust {float(product['max_gust'].max()):.1f} m/s"
            if "max_gust" in product
            else " (no native gust field)",
        )


# --- Figure 4: Climate Digital Twin ------------------------------------------


def make_figure4(data_root: Path, out_dir: Path) -> None:
    """Heating degree days and storm hours from the Climate DT."""
    LOGGER.info("Figure 4: Climate DT, January to March 2020")

    product = open_product(
        # One file per day, named climatedt_YYYYMMDD_0p04deg.nc. The glob takes
        # the whole year lazily; the time subset below trims it to Q1.
        str(data_root / "nc_2020_daily/climatedt_2020*_0p04deg.nc"),
        "climate_dt",
        domain=ALPINE_DOMAIN,
        start="2020-01-01",
        end="2020-03-31",
        chunks={"time": 240},
    )

    # Storm hours in the published figure are counted at hub height, not at
    # 10 m: the 10 m mean wind never reaches 15 m/s anywhere in this period
    # (domain maximum 17.4 m/s, at most 7 hours above the threshold), whereas
    # the 100 m wind does. Prefer the native field where the retrieval includes
    # it; otherwise extrapolate, and record which route was taken.
    if "wind_speed_100m" in product.data_vars:
        hub_wind = product["wind_speed_100m"]
        hub_source = "native"
    else:
        hub_wind = extrapolate_to_hub_height(product["wind_speed_10m"])
        hub_source = "extrapolated from 10 m by the power law"
        LOGGER.info("    no native 100 m wind in this retrieval; extrapolating")

    result = xr.Dataset(
        {
            "hdd": heating_degree_days(product["temperature_2m"], base=HDD_BASE_DEGC),
            "storm_hours": exceedance_hours(hub_wind, STORM_HOUR_THRESHOLD_MS),
        }
    ).compute()

    result.attrs = _provenance(
        "climate_dt",
        "Climate DT (4.4 km)",
        period="2020-01-01 to 2020-03-31",
        domain=ALPINE_DOMAIN.name,
        hdd_base_degC=HDD_BASE_DEGC,
        storm_hour_threshold_ms=STORM_HOUR_THRESHOLD_MS,
        storm_hour_wind_height_m=100.0,
        storm_hour_wind_source=hub_source,
        n_time_steps=int(product.sizes["time"]),
        caveat=(
            "A three-month window is far too short for climate-scale conclusions; "
            "it illustrates what km-scale projections can show."
        ),
    )
    product.close()

    _write(result, out_dir / "figure4_climate_dt_2020Q1.nc")
    LOGGER.info(
        "    HDD max %.0f degC d, storm hours max %.0f h",
        float(result["hdd"].max()),
        float(result["storm_hours"].max()),
    )


# --- Figure S1: station verification -----------------------------------------


def make_figure_s1(merged_parquet: Path, out_dir: Path) -> None:
    """Per-station verification statistics for ARA against observations."""
    import pandas as pd

    from alpinemet.verification import elevation_band_metrics, station_metrics

    LOGGER.info("Figure S1: ARA station verification, 2020")

    if not merged_parquet.exists():
        LOGGER.warning("    %s not found; skipping", merged_parquet)
        return

    merged = pd.read_parquet(merged_parquet)
    metadata = (
        merged[["station_id", "station_name", "lon", "lat", "alt"]]
        .drop_duplicates("station_id")
        .rename(columns={"station_id": "synnr", "station_name": "name"})
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for variable, observed, modelled in [
        ("temperature", "temperature", "ara_temperature"),
        ("wind_speed", "wind_speed", "ara_wind_speed"),
    ]:
        paired = merged[["station_id", observed, modelled]].rename(
            columns={observed: "observed", modelled: "modelled"}
        )
        per_station = station_metrics(paired, station_metadata=metadata)
        bands = elevation_band_metrics(per_station)

        station_path = out_dir / f"figureS1_stations_{variable}.csv"
        band_path = out_dir / f"figureS1_bands_{variable}.csv"
        per_station.to_csv(station_path, index=False)
        bands.to_csv(band_path, index=False)

        LOGGER.info(
            "  %s: %d stations (%d excluded), bias %.2f, RMSE %.2f",
            variable,
            len(per_station),
            per_station.attrs["stations_excluded"],
            per_station["bias"].mean(),
            per_station["rmse"].mean(),
        )
        LOGGER.info("  wrote %s and %s", station_path.name, band_path.name)


# --- Entry point -------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-p", "--paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("-o", "--out", type=Path, default=Path("figure_inputs"))
    parser.add_argument(
        "--only",
        choices=["figure2", "figure3", "figure4", "figureS1"],
        action="append",
        help="run only the named figure; repeatable",
    )
    parser.add_argument(
        "--merged-parquet",
        type=Path,
        default=None,
        help="paired station observations for Figure S1",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )

    paths = load_paths(args.paths)
    data_root = paths.data_root
    wanted = set(args.only or ["figure2", "figure3", "figure4", "figureS1"])

    started = time.perf_counter()
    if "figure2" in wanted:
        make_figure2(data_root, args.out)
    if "figure3" in wanted:
        make_figure3(data_root, args.out)
    if "figure4" in wanted:
        make_figure4(data_root, args.out)
    if "figureS1" in wanted:
        merged = args.merged_parquet or (data_root / "STATIONS" / "merged_2020.parquet")
        make_figure_s1(merged, args.out)

    LOGGER.info("Done in %.0f s. Outputs in %s", time.perf_counter() - started, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
