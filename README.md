# alpinemet

**Multi-scale weather and climate data evaluation for Alpine renewable energy**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-425-brightgreen.svg)](tests/)

Companion code for:

> Schicker, I., Bügelmayer-Blaschek, M., Lexer, A., Baier, K., Hasel, K., Gazzaneo, P. (2026).
> **Beyond resolution: Multi-scale weather and climate data for alpine renewable energy in the
> digital twin era.** *iScience* **29**(8), 116855.
> [doi:10.1016/j.isci.2026.116855](https://doi.org/10.1016/j.isci.2026.116855)

---

## The problem this solves

Comparing meteorological products across resolutions sounds like a matter of loading files. In
practice, every product speaks its own dialect:

| | ERA5 | ERA5-Land | ARA | Extremes DT | AIFS |
|---|---|---|---|---|---|
| 2 m temperature | `var167`, K | `t2m`, K | `t`, K | `2t`, K | `t2m`, K |
| 10 m wind | `var165`/`var166` | `u10`/`v10` | `u10m`/`v10m` | `10u`/`10v` | `u10`/`v10` |
| Radiation | J/m², per step | J/m², **daily reset** | Ws/m², per step | J/m², per step | J/m², from init |
| Precipitation | **metres** | **metres**, daily reset | kg/m² | kg/m² | m, from init |
| Latitude axis | descending | descending | ascending | ascending | scattered |
| Native gust | no | no | `gust10m` | `i10fg` | no |
| 100 m wind | `var246`/`var247` | no | `u100m`/`v100m` | yes | yes |

(GRIB conversions without a parameter table name ERA5 variables after the ECMWF paramId.)

Get one of these wrong and the result still looks plausible. Divide by 3600 in the wrong place
and irradiance is off by three orders of magnitude — or precipitation, which is a per-step total
and must *not* be divided at all, comes out a thousandfold too small. Read a field in metres as
millimetres and every plausibility check still passes. Slice a descending latitude axis the
obvious way and you get an empty selection with no error at all.

`alpinemet` puts all of them on one set of conventions — canonical names, °C, W/m², m/s — and
then computes energy-relevant indicators on top.

## Quick start

```bash
git clone https://github.com/ischicker/perspectives_iScience_ClimateEnergyNexus_2025.git
cd perspectives_iScience_ClimateEnergyNexus_2025
uv sync
```

Ten lines from a raw file to a published-style indicator:

```python
from alpinemet.io import ALPINE_DOMAIN, open_product
from alpinemet.indicators.degree_days import degree_days
from alpinemet.indicators.storms import storm_days

era5 = open_product("DATA/ERA5/era5_2020_AT.nc", "era5", domain=ALPINE_DOMAIN)

heating_and_cooling = degree_days(era5["temperature_2m"])   # VDI 3787
storms = storm_days(era5["wind_speed_10m"], threshold=17.5)

print(float(heating_and_cooling["cdd"].mean()), "°C d")
print(float(storms.max()), "storm days")
```

Or from the command line:

```bash
cp configs/paths.example.yaml configs/paths.yaml   # point data_root at your data

uv run alpinemet datasets                                     # what is registered
uv run alpinemet check --config configs/era5_eraland_2020.yaml
uv run alpinemet run   --config configs/era5_eraland_2020.yaml
```

`check` validates the configuration and resolves every input path **without opening a single
file**. A typo surfaces in a second rather than after a multi-hour load.

No data to hand? The four notebooks in [`notebooks/`](notebooks/) build synthetic fields in
memory and run on a fresh clone with no data, credentials or network access.

## What is in the box

| Module | What it does |
|---|---|
| [`alpinemet.io`](src/alpinemet/io/) | Load and harmonise ERA5, ERA5-Land, CERRA, ARA, Climate DT, Extremes DT, IFS, AIFS |
| [`alpinemet.indicators`](src/alpinemet/indicators/) | Degree days, gusts, storm hours and days, ramps, Dunkelflaute, compound events, population weighting |
| [`alpinemet.energy`](src/alpinemet/energy/) | Wind, solar and hydro resource conversion |
| [`alpinemet.verification`](src/alpinemet/verification/) | Station verification, skill metrics, elevation-band analysis |
| [`alpinemet.plotting`](src/alpinemet/plotting/) | Figure building blocks with shared colour scales |

### Details that are easy to get wrong

The tests exist mostly to pin these down.

- **Degree days accumulate from daily means**, per VDI 3787. Summing hourly deviations counts
  the diurnal cycle as demand: a symmetric swing about the base temperature yields zero degree
  days the correct way and roughly 76 per month the wrong way.
- **Accumulation is declared, not inferred.** ERA5 accumulates over each step, ERA5-Land from a
  daily reset. Guessing this from value ranges is unreliable — a clear Alpine summer day
  averages about 900,000 J/m² per hour, above the bands such heuristics assume.
- **Gusts come from `i10fg` where it exists.** Only ERA5, ERA5-Land and AIFS need estimation,
  which uses Wieringa (1973). Every gust field records its `gust_source`, because estimated and
  native gusts are not interchangeable at the ESSL thresholds.
- **Storms are classified on mean wind, never gusts.** They answer different questions; mixing
  them inflates counts.
- **Population is aggregated by summing, not interpolating.** A raster holds counts per cell.
  Nearest-neighbour sampling plus a global rescale conserves the domain total while scattering
  people at random, and the distortion grows with grid coarseness.
- **Ramps are `u(t) − u(t − window)`**, not `diff(n=3)`, which is the third-order difference.
- **Dunkelflaute is sensitive to its thresholds.** Moving the wind threshold from 10 % to 20 %
  can change event frequency by a factor of 3 to 5, so every result carries its thresholds in
  the output attributes.

## Relationship to the paper

**This is a reimplementation, not the original analysis scripts.** The paper's results came
from a collection of scripts grown over the study; this package reorganises that work into a
tested library and corrects several defects the tests uncovered.

**None of those corrections changes a figure or a value reported in the paper.** They affect
analyses that were implemented but not shown — Dunkelflaute, compound events, ramps — or code
paths the published figures do not use. [`docs/differences.md`](docs/differences.md) documents
every difference together with its reach. Read it before comparing output against published
numbers.

Two quantities produced by the earlier pipeline **should be recomputed before reuse**: ramp
counts, and the population-weighted time series. Both are explained in that document.

## Reproducing the manuscript figures

| Manuscript | Content |
|---|---|
| Figure 2 | ERA5 / ERA5-Land / ARA, 2020: hot days, cooling degree days, storm days |
| Figure 3 | Storm Benjamin: IFS ENS control, Extremes DT, AIFS |
| Figure 4 | Climate DT 4.4 km: heating degree days and storm hours |
| Figure S1 / Table S1 | ARA station verification, 54 stations |

Figures 1 and 5 are schematics and are not produced by this code.

The raw model fields run to terabytes and cannot be distributed. The **derived
two-dimensional fields** that the figure code consumes are only a few megabytes, and ship with
this repository in [`figure_inputs/`](figure_inputs/) — so every figure can be reproduced from a
clone, with no data download. Regenerate them from raw data with
`scripts/make_figure_inputs.py`.

Two published values reproduce exactly from those fields: the Figure 4 heating-degree-day
maximum (2,616 °C d) and its storm-hour maximum (99 h). One caveat applies to the ARA hot-day
counts; see [`figure_inputs/README.md`](figure_inputs/README.md).

## Data access

| Dataset | Source |
|---|---|
| ERA5, ERA5-Land, CERRA | [Copernicus Climate Data Store](https://cds.climate.copernicus.eu) |
| Climate DT, Extremes DT | [Destination Earth Service Platform](https://destine.ecmwf.int) (registration required) |
| ARA (Austrian Reanalysis) | GeoSphere Austria, on request to the lead contact |
| GHS-POP | [GHSL](https://ghsl.jrc.ec.europa.eu/), release R2023A |
| Station observations | GeoSphere Austria |

Credentials come from your own `~/.cdsapirc` and DESP configuration. Nothing here stores them,
and `configs/paths.yaml` — which holds your machine's directory layout — is gitignored.

Optional dependencies for retrieval:

```bash
uv sync --extra acquisition   # cdsapi, earthkit-data, polytope-client, cfgrib
uv sync --extra all           # the above plus pvlib and docx export
```

## Examples

| Notebook | Shows |
|---|---|
| [`01_quickstart_standardisation`](notebooks/01_quickstart_standardisation.ipynb) | A raw product becoming standardised fields, and exactly what changes |
| [`02_degree_days_and_population`](notebooks/02_degree_days_and_population.ipynb) | Why degree days use daily means; why population must be summed |
| [`03_dunkelflaute_and_compound_events`](notebooks/03_dunkelflaute_and_compound_events.ipynb) | Threshold and duration sensitivity; aggregation scale in complex terrain |
| [`04_station_verification`](notebooks/04_station_verification.ipynb) | Representativeness, elevation offsets, verification by altitude band |

```bash
uv run --group dev jupyter lab notebooks/
```

Notebooks are generated from [`scripts/make_notebooks.py`](scripts/make_notebooks.py), so their
content lives in reviewable Python rather than JSON. They ship without stored outputs, and CI
fails if any output or credential-shaped string appears in one.

## Development

```bash
uv sync --group dev
uv run pytest                  # 425 tests, including notebook execution
uv run pytest -m "not slow"    # skip the notebook runs
uv run ruff check .
uv run pre-commit install      # strips notebook outputs, blocks private keys
```

Tests check against analytically known values rather than golden files, on synthetic fixtures
that need no data, credentials or network. Python 3.10–3.13 are supported; `.python-version`
pins 3.12, because cartopy and rasterio have no wheels for 3.14 yet.

## Citation

Please cite both the paper and the archived software release; see
[`CITATION.cff`](CITATION.cff) for the machine-readable form.

```bibtex
@article{SCHICKER2026116855,
  title   = {Beyond resolution: Multi-scale weather and climate data for
             alpine renewable energy in the digital twin era},
  author  = {Irene Schicker and Marianne Bügelmayer-Blaschek and
             Annemarie Lexer and Katharina Baier and Kristofer Hasel and
             Paolo Gazzaneo},
  journal = {iScience},
  volume  = {29},
  number  = {8},
  pages   = {116855},
  year    = {2026},
  issn    = {2589-0042},
  doi     = {10.1016/j.isci.2026.116855},
  url     = {https://www.sciencedirect.com/science/article/pii/S2589004226022339}
}
```

## Acknowledgements

Supported by the Digital Twin Austria project **HectoRenew** (FO999918390) and the Austrian
Climate Research Programme project **EnergyProtect** (FO999901419).

We acknowledge the use of ERA5, ERA5-Land and CERRA from ECMWF and Copernicus, the Austrian
reanalysis ARA from GeoSphere Austria, and Destination Earth Digital Twin data. Access to the
Digital Twins follows the example workflows published by ECMWF in the
[Destination Earth](https://github.com/destination-earth-digital-twins) repositories; those
upstream notebooks are not redistributed here.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Contact: Irene Schicker, GeoSphere Austria, <irene.schicker@geosphere.at>
