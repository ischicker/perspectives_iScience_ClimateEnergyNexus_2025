#!/usr/bin/env python3
"""Generate the example notebooks under ``notebooks/``.

The notebooks are kept as generated artefacts so that their content lives in
reviewable Python rather than in JSON with escaped newlines. Regenerate with::

    uv run python scripts/make_notebooks.py

Outputs are always empty: the notebooks are meant to be run by the reader, and
CI fails if any notebook carries stored outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def _lines(text: str) -> list[str]:
    stripped = text.strip("\n")
    lines = stripped.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def notebook(cells: list[dict]) -> dict:
    # nbformat 4.5 requires a stable cell id. Deriving it from the position
    # keeps regenerated notebooks byte-identical, so the diff stays readable.
    numbered = [{**cell, "id": f"cell-{index:03d}"} for index, cell in enumerate(cells)]
    return {
        "cells": numbered,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


QUICKSTART = notebook([
    markdown("""
# 01 - Quickstart: from a raw product to standardised fields

Every product evaluated in the paper arrives with its own variable names, units
and grid conventions. `alpinemet.io` maps them onto one set, so that every
indicator can be written once.

This notebook builds an ERA5-shaped dataset in memory. It runs with no data, no
credentials and no network; point `open_product` at your own files to do the
same for real.
"""),
    code("""
import numpy as np
import pandas as pd
import xarray as xr

from alpinemet.io import ALPINE_DOMAIN, DATASETS, get_dataset_spec, standardise
"""),
    markdown("## What the registry knows about each product"),
    code("""
pd.DataFrame(
    [
        {
            "key": spec.key,
            "resolution_km": spec.resolution_km,
            "step_h": spec.timestep_hours,
            "native_gust": spec.has_native_gust,
            "wind_100m": spec.has_100m_wind,
            "grid": spec.grid_type,
        }
        for spec in DATASETS.values()
    ]
).set_index("key").sort_values("resolution_km")
"""),
    markdown("""
The accumulation convention is **declared** per product rather than inferred
from the numbers. This is the detail that most often goes wrong by a factor of
3600: ERA5 accumulates over each step, ERA5-Land accumulates from a daily reset.
"""),
    code("""
for key in ["era5", "era5_land", "aifs"]:
    spec = get_dataset_spec(key)
    print(f"{key:<10} solar_radiation -> {spec.accumulation_kind('solar_radiation').value}")
"""),
    markdown("## A synthetic ERA5-shaped product"),
    code("""
time = pd.date_range("2020-06-21", periods=48, freq="h")
lats = np.arange(52.0, 39.9, -1.0)   # ERA5 stores latitude descending
lons = np.arange(0.0, 25.1, 1.0)
dims = ("time", "latitude", "longitude")
coords = {"time": time, "latitude": lats, "longitude": lons}
shape = (time.size, lats.size, lons.size)

diurnal = np.clip(800.0 * np.sin((np.arange(24) - 6) / 12 * np.pi), 0.0, None)
flux = np.tile(diurnal, 2)[:, None, None] * np.ones(shape[1:])

raw = xr.Dataset(
    {
        "t2m": xr.DataArray(
            np.full(shape, 288.15), dims=dims, coords=coords, attrs={"units": "K"}
        ),
        "u10": xr.DataArray(np.full(shape, 3.0), dims=dims, coords=coords),
        "v10": xr.DataArray(np.full(shape, 4.0), dims=dims, coords=coords),
        "ssrd": xr.DataArray(
            flux * 3600.0, dims=dims, coords=coords, attrs={"units": "J m-2"}
        ),
    }
)
raw
"""),
    markdown("## Standardise it"),
    code("""
era5 = standardise(raw, "era5", domain=ALPINE_DOMAIN)
era5
"""),
    markdown("""
Compare before and after: temperature has moved from kelvin to degrees Celsius,
radiation from accumulated J/m2 to instantaneous W/m2, and a scalar wind speed
has appeared. The descending latitude axis was subset correctly, where a naive
`sel(latitude=slice(43, 49))` would have returned an empty selection with no
error at all.
"""),
    code("""
print(f"raw  t2m   max: {float(raw['t2m'].max()):>12,.1f} K")
print(f"std  temp  max: {float(era5['temperature_2m'].max()):>12,.1f} degC")
print()
print(f"raw  ssrd  max: {float(raw['ssrd'].max()):>12,.0f} J/m2")
print(f"std  solar max: {float(era5['solar_radiation'].max()):>12,.1f} W/m2")
print()
print(f"derived wind speed: {float(era5['wind_speed_10m'].max()):.1f} m/s")
print(f"latitudes kept: {era5.sizes['latitude']} of {raw.sizes['latitude']}")
"""),
    markdown("""
Provenance travels with the data, so a file written now can still be explained
later.
"""),
    code("""
from alpinemet.attrs import decode_attribute

for key in ["dataset_key", "resolution_km", "has_native_gust", "dataset_notes"]:
    print(f"{key:<18} {decode_attribute(era5.attrs[key])}")
"""),
    markdown("""
## Reading real files

```python
from alpinemet.io import open_product

era5 = open_product(
    "/path/to/DATA/ERA5/era5_2020_AT.nc",
    "era5",
    domain=ALPINE_DOMAIN,
    start="2020-01-01",
    end="2020-12-31",
)
```

Or from the command line, with machine-specific paths kept out of the
repository in `configs/paths.yaml`:

```bash
uv run alpinemet check --config configs/era5_eraland_2020.yaml
uv run alpinemet run   --config configs/era5_eraland_2020.yaml
```

`check` validates the configuration and resolves every input path without
opening a single file, so a typo surfaces in a second rather than after a
multi-hour load.
"""),
])


DEGREE_DAYS = notebook([
    markdown("""
# 02 - Degree days and population weighting

Two details that change the answer materially:

1. Degree days accumulate from **daily means**, following VDI 3787. Summing
   hourly deviations instead counts the diurnal cycle as demand.
2. Population weighting requires **summing** a population raster onto the model
   grid, not interpolating it. Interpolation conserves the total while
   scattering people at random.
"""),
    code("""
import numpy as np
import pandas as pd
import xarray as xr

from alpinemet.indicators.degree_days import (
    CDD_BASE_TEMPERATURE,
    HDD_BASE_TEMPERATURE,
    degree_days,
    heating_degree_days,
)
"""),
    markdown("## Why daily means matter"),
    code("""
time = pd.date_range("2020-01-01", periods=24 * 30, freq="h")
# Daily mean sitting exactly at the heating base, swinging +/- 8 K through the day.
values = HDD_BASE_TEMPERATURE + 8.0 * np.sin(np.arange(time.size) / 24 * 2 * np.pi)
temperature = xr.DataArray(
    values, dims="time", coords={"time": time}, attrs={"units": "degC"}
)

correct = float(heating_degree_days(temperature))
hourly = float(np.clip(HDD_BASE_TEMPERATURE - values, 0, None).sum() / 24)

print(f"VDI 3787, from daily means : {correct:8.1f} degC d")
print(f"Summing hourly deviations  : {hourly:8.1f} degC d   <- invented demand")
"""),
    markdown("""
The daily mean never leaves the base temperature, so there is no heating
demand. Accumulating hour by hour invents roughly 76 degree days a month out of
nothing.

## Degree days over a synthetic year
"""),
    code("""
time = pd.date_range("2020-01-01", "2020-12-31 23:00", freq="h")
day_of_year = time.dayofyear.to_numpy()
hour = time.hour.to_numpy()

seasonal = 10.0 - 12.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
diurnal = 4.0 * np.sin(2 * np.pi * (hour - 9) / 24)

lats = np.array([46.5, 47.0, 47.5])
lons = np.array([11.0, 12.0, 13.0])
# Three rows at increasing elevation, hence increasingly cold.
elevation_offset = np.array([0.0, -2.0, -4.0])[:, None] * np.ones(lons.size)

field = (seasonal + diurnal)[:, None, None] + elevation_offset[None, :, :]
temperature = xr.DataArray(
    field,
    dims=("time", "latitude", "longitude"),
    coords={"time": time, "latitude": lats, "longitude": lons},
    attrs={"units": "degC"},
)

result = degree_days(temperature)
print(f"HDD base {HDD_BASE_TEMPERATURE} degC, CDD base {CDD_BASE_TEMPERATURE} degC")
print()
print("annual totals by latitude row (degC d):")
result[["hdd", "cdd"]].mean(dim="longitude").to_pandas()
"""),
    markdown("""
## Population weighting: summing versus interpolating

A population raster holds counts *per cell*. Moving it to a coarser grid is an
aggregation, not an interpolation.
"""),
    code("""
from alpinemet.indicators.population import aggregate_population_to_grid

rng = np.random.default_rng(0)
fine_lat = np.linspace(46.0, 48.0, 400)
fine_lon = np.linspace(10.0, 12.0, 400)

people = rng.uniform(0.0, 1.0, (400, 400))   # thin rural scatter everywhere
people[110:130, 110:130] += 500.0            # one compact city

population = xr.DataArray(
    people,
    dims=("latitude", "longitude"),
    coords={"latitude": fine_lat, "longitude": fine_lon},
)

target = xr.DataArray(
    np.zeros((4, 4)),
    dims=("latitude", "longitude"),
    coords={
        "latitude": np.linspace(46.25, 47.75, 4),
        "longitude": np.linspace(10.25, 11.75, 4),
    },
)

correct = aggregate_population_to_grid(population, target)
legacy = aggregate_population_to_grid(population, target, method="nearest_normalised")

city = np.unravel_index(int(np.argmax(correct.values)), correct.shape)

print(f"total, fine grid       : {float(population.sum()):>12,.0f}")
print(f"total, sum aggregation : {float(correct.sum()):>12,.0f}")
print(f"total, nearest+rescale : {float(legacy.sum()):>12,.0f}")
print()
print(f"city cell share, correct: {float(correct.values[city]) / float(correct.sum()):>7.1%}")
print(f"city cell share, legacy : {float(legacy.values[city]) / float(legacy.sum()):>7.1%}")
"""),
    markdown("""
Both conserve the domain total; only the aggregation puts the people where they
live. The distortion grows with the coarseness of the target grid -- one source
cell in 625 at 2.5 km, one in 96,000 at 31 km -- so it does not affect compared
products equally. `nearest_normalised` exists solely to reproduce earlier output
and carries a warning attribute saying so.
"""),
    code("""
print(legacy.attrs["warning"])
"""),
    markdown("## Weighting a temperature field"),
    code("""
from alpinemet.indicators.population import population_weighted_mean

grid = temperature.isel(time=0).drop_vars("time")
weights = aggregate_population_to_grid(population, grid)

area_mean = float(temperature.mean())
pop_mean = float(population_weighted_mean(temperature, weights).mean())

print(f"area-averaged mean temperature      : {area_mean:6.2f} degC")
print(f"population-weighted mean temperature: {pop_mean:6.2f} degC")
print()
print("The population sits in the warm valley row, so weighting shifts the")
print("mean upward. Area averages understate what the demand side experiences.")
"""),
])


DUNKELFLAUTE = notebook([
    markdown("""
# 03 - Dunkelflaute and compound events

A Dunkelflaute is a *sustained* simultaneous shortfall of wind and solar
generation. Three choices govern how many you find, and all three must be
reported alongside any count:

- the thresholds,
- the minimum duration,
- the spatial aggregation scale.

The paper's reference definition: wind capacity factor below 10 %, solar
capacity factor below 5 %, sustained for at least 48 hours.
"""),
    code("""
import numpy as np
import pandas as pd
import xarray as xr

from alpinemet.indicators.dunkelflaute import (
    DUNKELFLAUTE_MIN_DURATION_HOURS,
    DUNKELFLAUTE_SOLAR_CF_THRESHOLD,
    DUNKELFLAUTE_WIND_CF_THRESHOLD,
    detect,
)

print(f"wind CF  < {DUNKELFLAUTE_WIND_CF_THRESHOLD:.0%}")
print(f"solar CF < {DUNKELFLAUTE_SOLAR_CF_THRESHOLD:.0%}")
print(f"for at least {DUNKELFLAUTE_MIN_DURATION_HOURS} h")
"""),
    markdown("## A winter month with two calm, dull spells"),
    code("""
time = pd.date_range("2020-01-01", periods=24 * 30, freq="h")
rng = np.random.default_rng(11)

wind_cf = np.clip(rng.beta(2.0, 3.0, time.size), 0.0, 1.0)
solar_cf = np.clip(
    0.35 * np.sin(np.pi * (time.hour.to_numpy() - 8) / 9), 0.0, None
) * rng.uniform(0.4, 1.0, time.size)

# A 60 hour blocking episode and a 30 hour one.
wind_cf[200:260] = 0.03
solar_cf[200:260] *= 0.05
wind_cf[500:530] = 0.03
solar_cf[500:530] *= 0.05

series = {
    name: xr.DataArray(values, dims="time", coords={"time": time})
    for name, values in [("wind", wind_cf), ("solar", solar_cf)]
}

result = detect(series["wind"], series["solar"])
print(f"episodes detected : {int(result['event_count'])}")
print(f"hours in episodes : {float(result['total_hours']):.0f}")
print(f"longest episode   : {float(result['longest_event_hours']):.0f} h")
"""),
    markdown("""
Only the 60 hour spell qualifies. The 30 hour one is a shortfall but not an
episode: its hours appear in `shortfall` and not in `dunkelflaute`.
"""),
    code("""
print(f"hours meeting the instantaneous condition: {int(result['shortfall'].sum())}")
print(f"hours inside a sustained episode        : {int(result['dunkelflaute'].sum())}")
"""),
    markdown("""
## Threshold sensitivity

The paper notes that moving the wind threshold between 10 % and 20 % can change
event frequency by a factor of 3 to 5. Worth reproducing before quoting any
number.
"""),
    code("""
rows = []
for wind_threshold in [0.08, 0.10, 0.12, 0.15, 0.20]:
    for duration in [24, 48, 72]:
        run = detect(
            series["wind"],
            series["solar"],
            wind_threshold=wind_threshold,
            min_duration_hours=duration,
        )
        rows.append(
            {
                "wind_cf_threshold": wind_threshold,
                "min_duration_h": duration,
                "events": int(run["event_count"]),
                "hours": float(run["total_hours"]),
            }
        )

pd.DataFrame(rows).pivot(
    index="wind_cf_threshold", columns="min_duration_h", values="hours"
)
"""),
    markdown("""
## Cold Dunkelflaute

Low generation coinciding with peak heating demand: the most stressful
combination for grid adequacy.
"""),
    code("""
temperature = xr.DataArray(
    5.0 - 10.0 * (np.arange(time.size) // 24 % 7 == 0), dims="time",
    coords={"time": time},
)
temperature[200:260] = -4.0

cold = detect(series["wind"], series["solar"], temperature=temperature)
print(f"Dunkelflaute hours      : {float(cold['total_hours']):.0f}")
print(f"Cold Dunkelflaute hours : {float(cold['cold_total_hours']):.0f}")
"""),
    markdown("""
## Aggregation scale in complex terrain

During a persistent inversion, valley-floor installations can be in energy
drought while ridge-top turbines run near rated capacity. Whether that is a
system-level Dunkelflaute depends entirely on the scale of aggregation -- so
state the scale rather than letting the default decide.
"""),
    code("""
from alpinemet.indicators.compound import hellsturm

wind_field = xr.DataArray(
    np.zeros((48, 2, 2)) + 3.0,
    dims=("time", "latitude", "longitude"),
    coords={"time": time[:48], "latitude": [46.0, 47.0], "longitude": [11.0, 12.0]},
)
solar_field = xr.zeros_like(wind_field) + 100.0
# One exposed ridge cell: strong wind and full sun.
wind_field[:, 0, 0] = 18.0
solar_field[:, 0, 0] = 700.0

pointwise = hellsturm(wind_field, solar_field)
aggregated = hellsturm(wind_field, solar_field, aggregate=True)

print(f"grid-point-wise: {int(pointwise['hellsturm'].sum())} cell-hours detected")
print(f"domain-averaged: {int(aggregated['hellsturm'].sum())} hours detected")
print()
print("The ridge cell is in a Hellsturm throughout; the domain mean never is.")
"""),
])


VERIFICATION = notebook([
    markdown("""
# 04 - Station verification

The comparison behind Figure S1 and Table S1: a gridded product against point
observations. Two caveats govern how the numbers should be read, and both are
about the observations rather than the model.

**Representativeness.** A grid cell is an area average, a station is a point. At
2.5 km over Alpine terrain the two can differ by hundreds of metres in
elevation, and the resulting offset is often larger than the model error being
measured.

**Coverage.** Roughly 90 % of Alpine stations lie below 1,500 m -- exactly where
the terrain is least demanding. Aggregate statistics over such a network
flatter every product.
"""),
    code("""
import numpy as np
import pandas as pd
import xarray as xr

from alpinemet.verification import (
    elevation_band_metrics,
    extract_at_stations,
    lapse_rate_adjustment,
    match_stations_to_grid,
    station_metrics,
    verification_metrics,
)
"""),
    markdown("## A synthetic 2.5 km product and a station network"),
    code("""
time = pd.date_range("2020-01-01", periods=24 * 60, freq="h")
lats = np.arange(46.5, 48.01, 0.025)
lons = np.arange(10.0, 14.01, 0.025)

rng = np.random.default_rng(5)

# Model orography: a smooth ridge running west to east.
orography = 800.0 + 1200.0 * np.exp(
    -(((lats[:, None] - 47.3) / 0.35) ** 2) - ((lons[None, :] - 12.0) / 2.0) ** 2
)

seasonal = 2.0 - 8.0 * np.cos(2 * np.pi * (time.dayofyear.to_numpy() - 15) / 365)
field = seasonal[:, None, None] - 0.0065 * orography[None, :, :]
field = field + rng.normal(0.0, 0.8, field.shape)

model = xr.Dataset(
    {
        "temperature_2m": xr.DataArray(
            field,
            dims=("time", "latitude", "longitude"),
            coords={"time": time, "latitude": lats, "longitude": lons},
            attrs={"units": "degC"},
        ),
        "orography": xr.DataArray(
            orography,
            dims=("latitude", "longitude"),
            coords={"latitude": lats, "longitude": lons},
        ),
    }
)

stations = pd.DataFrame(
    {
        "synnr": [11399, 11200, 11111, 11146, 11343],
        "name": ["TAMSWEG", "KALS", "TANNHEIM", "OBERGURGL", "SEEFELD"],
        "lon": [13.808, 12.646, 10.506, 11.024, 11.190],
        "lat": [47.133, 47.005, 47.500, 46.867, 47.329],
        "alt": [1025.0, 1352.0, 1100.0, 1938.0, 1180.0],
    }
)
stations
"""),
    markdown("## Match each station to its grid cell"),
    code("""
matched = match_stations_to_grid(model, stations, elevation_variable="orography")
matched[["name", "alt", "grid_elevation", "elevation_difference", "distance_km"]]
"""),
    markdown("""
The elevation difference is the quantity to inspect before believing any bias.
Where the model cell sits well above its station, a cold bias is expected and
says more about representativeness than about model skill.

## Extract the modelled series and pair it with observations
"""),
    code("""
modelled = extract_at_stations(model["temperature_2m"], matched)

# Synthetic observations: the true station-elevation temperature plus noise.
elevation_by_station = dict(zip(stations["synnr"], stations["alt"], strict=True))
grid_elevation = dict(zip(matched["synnr"], matched["grid_elevation"], strict=True))

paired = modelled.rename(columns={"value": "modelled"})
paired["station_elevation"] = paired["station_id"].map(elevation_by_station)
paired["grid_elevation"] = paired["station_id"].map(grid_elevation)
paired["elevation_difference"] = paired["grid_elevation"] - paired["station_elevation"]
paired["observed"] = paired["modelled"] - 0.0065 * paired["elevation_difference"]
paired["observed"] += rng.normal(0.0, 0.5, len(paired))

paired.head()
"""),
    markdown("## Metrics per station"),
    code("""
per_station = station_metrics(paired, station_metadata=stations)
per_station[["name", "alt", "n_pairs", "bias", "rmse", "correlation"]]
"""),
    markdown("""
The bias tracks the elevation offset almost exactly, which is the point: without
inspecting it, one would report these as model errors.

## Correcting for the elevation offset
"""),
    code("""
paired["adjusted"] = lapse_rate_adjustment(
    paired["modelled"], paired["elevation_difference"]
)

raw = verification_metrics(paired["observed"], paired["modelled"])
adjusted = verification_metrics(paired["observed"], paired["adjusted"])

print(f"{'':<12}{'bias':>8}{'rmse':>8}")
print(f"{'raw':<12}{raw['bias']:>8.2f}{raw['rmse']:>8.2f}")
print(f"{'adjusted':<12}{adjusted['bias']:>8.2f}{adjusted['rmse']:>8.2f}")
"""),
    markdown("""
A constant lapse rate assumes a well-mixed atmosphere. During the persistent
winter valley inversions that this study is largely about, temperature
*increases* with height and this correction has the wrong sign -- so report
adjusted and unadjusted side by side rather than replacing one with the other.

## Verification by elevation band
"""),
    code("""
bands = elevation_band_metrics(per_station)
bands
"""),
    markdown("""
Empty bands are retained deliberately. A high-elevation band with no stations is
a finding about the observing network, not something to average away -- and it is
precisely the band where Alpine wind farms are increasingly sited.
"""),
])


def main() -> None:
    NOTEBOOK_DIR.mkdir(exist_ok=True)
    for name, content in [
        ("01_quickstart_standardisation.ipynb", QUICKSTART),
        ("02_degree_days_and_population.ipynb", DEGREE_DAYS),
        ("03_dunkelflaute_and_compound_events.ipynb", DUNKELFLAUTE),
        ("04_station_verification.ipynb", VERIFICATION),
    ]:
        path = NOTEBOOK_DIR / name
        path.write_text(json.dumps(content, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(NOTEBOOK_DIR.parent)}")


if __name__ == "__main__":
    main()
