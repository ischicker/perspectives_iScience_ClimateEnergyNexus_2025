# Derived fields behind the manuscript figures

The raw model fields run to terabytes and cannot be distributed. These are the
time-reduced two-dimensional fields the figure code actually consumes — a few
megabytes in total — so that the figures can be reproduced from a clone of this
repository without any data download.

Regenerate them from raw data with:

```bash
uv run python scripts/make_figure_inputs.py --paths configs/paths.yaml
```

Every file records its provenance in its attributes: which product it came
from, the period and domain, the indicator definitions and thresholds applied,
and the version of `alpinemet` that produced it.

## Contents

| File | Figure | Variables |
|---|---|---|
| `figure2_era5.nc` | 2, ERA5 (31 km) | `hot_days`, `cdd`, `storm_days` |
| `figure2_era5_land.nc` | 2, ERA5-Land (9 km) | `hot_days`, `cdd`, `storm_days` |
| `figure2_ara.nc` | 2, ARA (2.5 km) | `hot_days`, `cdd`, `storm_days` |
| `figure3_ifs.nc` | 3, IFS ENS control (9 km) | `max_gust`, `max_wind` |
| `figure3_extremes_dt.nc` | 3, Extremes DT (4.4 km) | `max_gust`, `max_wind` |
| `figure3_aifs.nc` | 3, AIFS (~31 km) | `max_wind` |
| `figure4_climate_dt_2020Q1.nc` | 4, Climate DT (4.4 km) | `hdd`, `storm_hours` |
| `figureS1_stations_*.csv` | S1 / Table S1 | Per-station verification metrics |
| `figureS1_bands_*.csv` | S1 | Metrics aggregated by elevation band |

Definitions, for reference:

- `hot_days` — days with daily maximum temperature ≥ 30 °C
- `cdd` — cooling degree days, VDI 3787, base 18 °C, from daily means
- `storm_days` — days with 10 m mean wind ≥ 17.5 m/s at any hour
- `hdd` — heating degree days, VDI 3787, base 15 °C, from daily means
- `storm_hours` — hours with **100 m** wind ≥ 15 m/s (see `docs/differences.md`)
- `max_gust`, `max_wind` — maxima over the T+0 to T+36 h forecast window

## Caveat on `figure2_ara.nc`

The ARA 2 m temperature in the 2020 subset carries a radiation-driven daytime
warm bias — +5 K at 10 UTC in July across 51 independent stations, and far more
at urban lowland sites, with a mean July diurnal amplitude of 17.8 K against an
observed 10.9 K.

**`hot_days` in that file is biased high and should not be used
quantitatively.** `cdd`, built from daily means, and `storm_days`, built from the
wind fields, are not affected. The caveat is also recorded in the file's own
attributes. See `docs/differences.md` for the full account.

## Reproduced published values

Two values quoted in the paper come out of these fields exactly:

| Quantity | Published | Here |
|---|---|---|
| Figure 4 HDD maximum | 2,616 °C d | 2,616 °C d |
| Figure 4 storm hours maximum | 99 h | 99 h |

Section 4.3 reports cooling degree days of 173 ± 120 °C d over the study domain;
ARA over its own domain gives 170 ± 152 °C d here.
