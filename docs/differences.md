# Differences from the analysis pipeline used for the paper

This repository is a **reimplementation**. It is not, line for line, the code
that produced the figures in Schicker et al. (2026); that code was a collection
of analysis scripts grown over the course of the study. This package reorganises
it into a tested library, and in doing so corrects several defects found by the
tests written along the way.

This page records every difference that changes a number, so that anyone
comparing the two knows exactly where and why they diverge.

**Summary: none of the corrections below changes a figure or a value reported in
the paper.** The published results rest on hot-day counts, cooling and heating
degree days, storm hours and days, wind gusts, and station bias and RMSE. The
defects lie in analyses that were implemented but not shown, or in code paths
the figures do not use. Each entry states its reach explicitly.

---

## 1. Wind ramps were a third-order difference

**Was:** `wind_speed.diff(dim="time", n=3)`

In xarray, `n` is the number of times the difference operator is applied, not
the width of the window. The expression therefore returns the third-order
difference, not the change in wind speed over three hours.

For the series `[0, 1, 2, 3, 10, 11, 12, 13]`:

| | Result |
|---|---|
| `diff(n=3)` | `[0, 6, −12, 6, 0]` |
| `u(t) − u(t−3)` | `[3, 9, 9, 9, 3]` |

**Now:** `alpinemet.indicators.storms.ramp_magnitude` computes
`u(t) − u(t − window)`. A regression test pins the distinction.

**Reach:** No published figure uses the ramp analysis. Any ramp count produced
by the earlier code should be recomputed before use.

## 2. The föhn screening used the same expression

**Was:** `temp.diff(dim="time", n=3)` for "a 5 °C increase over three hours".

**Now:** `alpinemet.indicators.compound.foehn_like` uses the window difference.
The function is also explicitly labelled a screening proxy rather than a föhn
detection, since genuine identification needs upstream moisture, lee-side
descent and cold-pool erosion.

**Reach:** No published figure uses föhn detection.

## 3. Annualised precipitation assumed one time step per day

**Was:** `precip.sum() * 365 / len(precip)`

This is correct only for daily input. For an hourly record it returns one
twenty-fourth of the true annual total.

**Now:** `alpinemet.energy.hydro.annual_precipitation` rescales by the record's
actual length in hours, and agrees between hourly and daily input of the same
rate.

**Reach:** No published figure uses annualised precipitation.

## 4. Population was regridded by interpolation rather than aggregation

**Was:** the population raster was sampled onto the model grid with
nearest-neighbour interpolation, then multiplied by a single global factor to
restore the domain total.

A population raster holds counts *per cell*. Moving it to a coarser grid is an
aggregation. Nearest-neighbour sampling takes the value of one arbitrarily
chosen source cell — one in 625 going from 100 m to 2.5 km, one in about 96,000
going to 31 km — and the global rescale then restores the total while leaving
the spatial pattern essentially arbitrary.

On a synthetic test (one compact city plus thin rural scatter, 400×400 → 4×4):

| Method | Share of population in the city cell |
|---|---|
| Sum aggregation | 73 % |
| Nearest-neighbour + rescale | 11 % |

Both conserve the domain total. Only the aggregation puts people where they
live. The distortion grows with target-grid coarseness, so it does not affect
compared products equally.

**Now:** `alpinemet.indicators.population.aggregate_population_to_grid` sums
source cells into the target cell containing their centre. The earlier
behaviour remains available as `method="nearest_normalised"` for reproducing
old output, and its result carries a warning attribute.

**Reach:** No published figure. The panels of Figure 2 plot unweighted
quantities — hot-day counts and cooling degree days computed directly from the
temperature field. The population-weighted *time series* produced alongside them
by the earlier pipeline do depend on this regridding and should be recomputed
before reuse.

## 5. Accumulation was detected from value ranges

**Was:** whether a radiation field was accumulated or already a flux was
inferred from its magnitude, with ERA5-style accumulation assumed to fall
between 100,000 and 500,000 J/m². Values outside the expected bands fell through
to an "unknown" branch that skipped conversion silently, leaving J/m² in a field
labelled W/m².

The upper bound corresponds to a daily mean of about 139 W/m². A clear Alpine
summer day averages roughly 250 W/m², or 900,000 J/m² per hour, and would have
fallen through.

**Now:** the accumulation convention is **declared** per product in
`alpinemet.io.datasets`, not inferred. `infer_accumulation_kind` remains as a
fallback for unfamiliar products, but it decides on the monotonicity of the
series rather than on magnitudes, and raises rather than passing data through
unconverted when it cannot tell.

**Reach:** No published figure uses solar radiation. Figures 2, 3, 4 and S1 rest
on temperature and wind.

## 6. Dunkelflaute used meteorological thresholds, not capacity factors

**Was:** simultaneous wind speed below 3 m/s and irradiance below 100 W/m², with
no minimum duration in the detection itself.

**Now:** `alpinemet.indicators.dunkelflaute.detect` implements the definition
given in the paper — wind capacity factor below 10 %, solar capacity factor
below 5 %, sustained for at least 48 hours. The meteorological variant remains
available as `detect_from_raw`, which is useful for comparing datasets without
committing to a turbine or PV specification.

**Reach:** Dunkelflaute results are not shown in the paper.

## 7. Energy conversion did not match the described reference systems

**Was:** solar capacity factor as `GHI / 1000 W/m²` on the horizontal plane;
wind capacity factor from a generic piecewise-linear curve with no rated power.

**Now:**

- `alpinemet.energy.solar.fixed_tilt_capacity_factor` runs a pvlib chain for the
  reference system described in the paper: 1 kWp, 30° tilt, south-facing, with
  Erbs decomposition, plane-of-array transposition and a PVWatts DC model. The
  horizontal ratio remains available as
  `plane_of_array_ratio_capacity_factor` for comparison.
- `alpinemet.energy.wind` uses a 3 MW reference turbine at 100 m hub height with
  cut-in 3 m/s, rated 12 m/s and cut-out 25 m/s. The curve shape between cut-in
  and rated speed defaults to cubic, following Staffell and Pfenninger (2016);
  the piecewise-linear form is available as `shape="linear"`.

**Reach:** No published figure uses capacity factors.

## 8. Compound events ignored their configured thresholds and used domain means

**Was:** `detect_compound_events` carried thresholds hard-coded in the function
body and did not use the configurable values set elsewhere. It also averaged
fields over the domain before applying them.

Averaging first answers a different question: whether the *regional mean* is
extreme, not whether any location is. In complex terrain the two diverge
sharply, since valley floors and ridges can be in opposite states at the same
moment — the aggregation-scale point the paper itself makes about Dunkelflaute.

**Now:** all thresholds are parameters. Detection is grid-point-wise by default,
with `aggregate=True` for the domain-mean behaviour.

**Reach:** Compound event results are not shown in the paper.

## 9. Dead code removed

`fix_solar_radiation_units_chunked` referenced undefined names in its own scope
and would have raised on any input; it was explicitly disabled in the earlier
pipeline and never ran. It is not carried over. Two error messages in the old
`analysis_engine.py` still referred to it.

---

---

## Notes for anyone re-running the figures

These are not differences, but details that are easy to lose and that cost time
to rediscover.

### Figure 4 storm hours are counted at hub height

The caption reads "storm frequency (hours with mean wind >= 15 m/s)" without
naming a height. It is the **100 m** wind, not the 10 m wind.

At 10 m the Climate DT mean wind over January to March 2020 never reaches
15 m/s anywhere in the domain — the maximum is 17.4 m/s and no grid point spends
more than 7 hours above the threshold. At 100 m the same threshold gives a
maximum of **99 hours**, the value in the caption.

`scripts/make_figure_inputs.py` therefore counts storm hours on the 100 m wind,
using the native field where the retrieval provides one and extrapolating from
10 m otherwise. Both published values for that panel reproduce exactly:

| Quantity | Published | Recomputed |
|---|---|---|
| HDD maximum (VDI 3787, base 15 °C) | 2,616 °C d | 2,616 °C d |
| Storm hours maximum (100 m wind >= 15 m/s) | 99 h | 99 h |

Note that Figure 2 storm days are a different quantity, counted on the **10 m**
mean wind at 17.5 m/s.

### ERA5 files converted without a parameter table

The ERA5 retrieval used here carries bare `varNNN` names, where `NNN` is the
ECMWF paramId — `var167` for 2 m temperature, `var165` and `var166` for the
10 m wind components, `var169` for downwelling shortwave, and so on. These are
declared on the ERA5 dataset specification and were verified against the value
ranges of the fields themselves.

That retrieval **does** include 100 m wind, as `var246` and `var247`.

### ARA 2 m temperature carries a radiation-dependent diurnal bias

`t` is ARA's 2 m temperature — the only temperature in its surface product,
alongside `gust10m`, `sp`, `tp` and `orog`. In the 2020 Alpine subset used here
it verifies poorly against the 51 independent stations of the supplementary
comparison, in a way worth knowing about before using it.

July 2020, ARA minus station, by hour of day:

| 00–04 UTC | 08 UTC | 10 UTC | 12 UTC | 16 UTC | 20–22 UTC |
|---|---|---|---|---|---|
| −2.4 K | +3.7 K | +5.0 K | +4.4 K | +0.9 K | −1.7 K |

Mean July diurnal amplitude is **17.8 K against an observed 10.9 K**, and the
bias scales monotonically with modelled global radiation — from −2.4 K in the
dark to +3.3 K above 700 W/m², correlation 0.60. Overall RMSE is 4.0 K.

The annual mean bias is only −1.1 K, because day and night largely cancel, so
an aggregate statistic hides this entirely. Anything that depends on daily
maxima or minima does not: hot-day counts in particular come out far too high.

Separately, a small number of grid points reach implausible values — up to
58 °C. These are spatially incoherent: at 2020-07-02 12 UTC, 283 points exceed
40 °C in 77 patches with a median size of 2 grid points, 47 % of them isolated
single pixels. That pattern is characteristic of bad pixels rather than of a
physical field.

Both observations are properties of this subset, not of the analysis code, and
are worth raising with the ARA producers before the derived fields are reused.

### ARA radiation accumulates over three-hour blocks

ARA stores global radiation (, in Ws/m2) as a running total within
three-hour blocks that begin at 01, 04, 07 UTC and so on -- not as a per-step
accumulation. Dividing each hourly value by 3600, the natural reading of the
unit on hourly output, inflates the second and third hour of every block by
factors of two and three.

July hourly means at one grid point:

| Hour (UTC) | 07 | 08 | 09 | 10 | 11 | 12 | 13 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Naive division | 334 | 809 | 1406 | 679 | 1365 | 2052 | 612 |
| Deaccumulated | 351 | 503 | 606 | 702 | 695 | 686 | 614 |

The naive reading gives a sawtooth peaking above 2000 W/m2 in a *monthly mean*.
Deaccumulated, the cycle is smooth and peaks at about 700 W/m2.

The block length has to be declared rather than inferred: through a summer
morning a new block often opens *above* the previous block total, so the
difference stays positive and a sign test misses the reset. The dataset
specification carries  and  for this.

### ARA monthly files overlap

Consecutive ARA monthly files share timestamps at their boundaries — 30
duplicates in January 2020 alone. They must be removed before anything
differences along time, which `alpinemet.io` does as the second step of
standardisation.

---

## Reproducing the published numbers

Where a published result and this package disagree, the entries above should
account for the difference. If one does not, please open an issue — that is
information worth having.

The derived two-dimensional fields behind Figures 2, 3, 4 and S1 are published
separately on Zenodo, because the raw model fields run to terabytes.
