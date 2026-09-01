"""Per-product specifications for the datasets evaluated in the study.

Everything that varies between products and cannot be inferred safely from the
data is declared here: the accumulation convention of each flux variable,
whether a native gust or 100 m wind field exists, the native grid type and the
nominal resolution. Loading code reads these rather than guessing.

The registry corresponds to Table 2 of the paper. Resolutions are the nominal
grid spacings quoted there; as Section 1.4 argues at length, the *effective*
resolution of a product is generally coarser and is not something this registry
can express.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from alpinemet.io.accumulation import AccumulationKind

__all__ = [
    "DATASETS",
    "DatasetSpec",
    "GridType",
    "get_dataset_spec",
]

GridType = str


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the loader needs to know about one product.

    Attributes
    ----------
    key
        Short identifier used in configuration files.
    long_name
        Human-readable product name.
    resolution_km
        Nominal horizontal grid spacing in kilometres.
    timestep_hours
        Nominal output interval in hours.
    grid_type
        Native grid geometry. ``"reduced_gaussian"`` products carry
        one-dimensional, scattered coordinates and need regridding before they
        can be plotted on a map.
    accumulation
        Accumulation convention per canonical variable. Variables absent from
        this mapping are treated as instantaneous.
    has_native_gust
        Whether the product provides an instantaneous gust field. When false,
        gusts must be estimated; see :mod:`alpinemet.indicators.gusts`.
    has_100m_wind
        Whether 100 m wind components are available. When false, hub-height
        wind must be extrapolated, with the caveats noted in
        :func:`alpinemet.energy.wind.extrapolate_to_hub_height`.
    extra_aliases
        Product-specific variable names not covered by the shared vocabulary.
    reset_hours, reset_offset
        Accumulation block length and its phase, for products whose running
        accumulators restart periodically.
    notes
        Free-text caveats worth carrying into output metadata.
    """

    key: str
    long_name: str
    resolution_km: float
    timestep_hours: float
    grid_type: GridType
    accumulation: Mapping[str, AccumulationKind] = field(default_factory=dict)
    has_native_gust: bool = False
    has_100m_wind: bool = False
    extra_aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    reset_hours: float | None = None
    reset_offset: int = 0
    notes: str = ""

    def accumulation_kind(self, canonical: str) -> AccumulationKind:
        """Accumulation convention for one variable, defaulting to instantaneous."""
        return self.accumulation.get(canonical, AccumulationKind.INSTANTANEOUS)


_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="era5",
        long_name="ERA5 global reanalysis",
        resolution_km=31.0,
        timestep_hours=1.0,
        grid_type="regular_latlon",
        accumulation={
            "solar_radiation": AccumulationKind.PERIOD,
            "precipitation": AccumulationKind.PERIOD,
        },
        has_native_gust=False,
        has_100m_wind=True,
        # Files converted from GRIB without a parameter table carry bare
        # 'varNNN' names, where NNN is the ECMWF paramId. Verified against the
        # value ranges of the 2020 Alpine retrieval used in this study.
        extra_aliases={
            "temperature_2m": ("var167",),
            "u_wind_10m": ("var165",),
            "v_wind_10m": ("var166",),
            "u_wind_100m": ("var246",),
            "v_wind_100m": ("var247",),
            "surface_pressure": ("var134",),
            "precipitation": ("var228",),
            "solar_radiation": ("var169",),
            "snow_depth": ("var141",),
        },
        notes=(
            "Latitude is stored descending. Gusts must be estimated. Precipitation is "
            "in metres of water equivalent. GRIB conversions without a parameter table "
            "name variables 'varNNN' after the ECMWF paramId."
        ),
    ),
    DatasetSpec(
        key="era5_land",
        long_name="ERA5-Land surface reanalysis",
        resolution_km=9.0,
        timestep_hours=1.0,
        grid_type="regular_latlon",
        accumulation={
            "solar_radiation": AccumulationKind.RUNNING,
            "precipitation": AccumulationKind.RUNNING,
        },
        has_native_gust=False,
        has_100m_wind=False,
        reset_hours=24,
        reset_offset=1,
        notes=(
            "Accumulators reset at 00 UTC daily; the reset step is reported as zero. "
            "Harmless for radiation, which is zero overnight, but it removes the "
            "00-01 UTC hour from precipitation totals."
        ),
    ),
    DatasetSpec(
        key="cerra",
        long_name="Copernicus European Regional ReAnalysis",
        resolution_km=5.5,
        timestep_hours=1.0,
        grid_type="lambert_conformal",
        accumulation={
            "solar_radiation": AccumulationKind.PERIOD,
            "precipitation": AccumulationKind.PERIOD,
        },
        has_native_gust=True,
        has_100m_wind=True,
        notes="Native Lambert conformal conic projection; regrid before comparing on a lat/lon grid.",
    ),
    DatasetSpec(
        key="ara",
        long_name="Austrian Reanalysis ensemble (control member)",
        resolution_km=2.5,
        timestep_hours=1.0,
        grid_type="rotated_latlon",
        # 'grad' accumulates over three-hour blocks, not per step: the hourly
        # means run 1x, 2x, 3x of the true flux and then reset. Verified on the
        # 2020 subset, where treating it as PERIOD inflates two hours in three
        # and produces values above 2200 W/m2.
        accumulation={
            "solar_radiation": AccumulationKind.RUNNING,
            "precipitation": AccumulationKind.RUNNING,
        },
        has_native_gust=True,
        has_100m_wind=True,
        reset_hours=3,
        reset_offset=1,
        # ARA uses its own short names and 'lat'/'lon' coordinates. 'grad' is
        # global radiation accumulated in Ws/m2, i.e. J/m2. 't' is the 2 m
        # temperature: it is the only temperature in the ARA surface product,
        # alongside gust10m, sp, tp and orog.
        #
        # Note for verification work: in the 2020 Alpine subset this field
        # carries a pronounced radiation-dependent diurnal bias against
        # independent stations -- see docs/differences.md.
        extra_aliases={
            "temperature_2m": ("t",),
            "u_wind_10m": ("u10m",),
            "v_wind_10m": ("v10m",),
            "u_wind_100m": ("u100m",),
            "v_wind_100m": ("v100m",),
            "wind_gust_10m": ("gust10m",),
            "solar_radiation": ("grad",),
        },
        notes=(
            "Convection-permitting AROME, 2012-2021. Ensemble members available; the "
            "evaluation in the paper uses the control member only. Monthly files "
            "overlap at their boundaries, so duplicate timestamps must be removed "
            "before any operation that differences along time."
        ),
    ),
    DatasetSpec(
        key="climate_dt",
        long_name="Destination Earth Climate Change Adaptation Digital Twin",
        resolution_km=4.4,
        timestep_hours=1.0,
        grid_type="regular_latlon",
        accumulation={
            "solar_radiation": AccumulationKind.PERIOD,
            "precipitation": AccumulationKind.PERIOD,
        },
        has_native_gust=False,
        has_100m_wind=True,
        extra_aliases={"solar_radiation": ("avg_sdswrf",)},
        notes="Scenario simulation; not a reanalysis. Short time slices only.",
    ),
    DatasetSpec(
        key="extremes_dt",
        long_name="Destination Earth Weather-Induced Extremes Digital Twin",
        resolution_km=4.4,
        timestep_hours=1.0,
        grid_type="regular_latlon",
        accumulation={
            "solar_radiation": AccumulationKind.PERIOD,
            "precipitation": AccumulationKind.PERIOD,
        },
        has_native_gust=True,
        has_100m_wind=True,
        extra_aliases={"temperature_2m": ("\\2t",), "wind_gust_10m": ("\\10fg",)},
        notes=(
            "GRIB shortNames may arrive escaped with a leading backslash. "
            "An on-demand 500-750 m regional component exists but is not covered here."
        ),
    ),
    DatasetSpec(
        key="ifs",
        long_name="ECMWF IFS ensemble (control member)",
        resolution_km=9.0,
        timestep_hours=1.0,
        grid_type="reduced_gaussian",
        accumulation={
            "solar_radiation": AccumulationKind.RUNNING,
            "precipitation": AccumulationKind.RUNNING,
        },
        has_native_gust=True,
        has_100m_wind=True,
        notes=(
            "Forecast fields accumulate from the initialisation time. Coordinates are "
            "scattered on a reduced Gaussian grid and need regridding before plotting."
        ),
    ),
    DatasetSpec(
        key="aifs",
        long_name="ECMWF AIFS deterministic forecast",
        resolution_km=31.0,
        timestep_hours=6.0,
        grid_type="reduced_gaussian",
        accumulation={
            "solar_radiation": AccumulationKind.RUNNING,
            "precipitation": AccumulationKind.RUNNING,
        },
        has_native_gust=False,
        has_100m_wind=True,
        notes=(
            "Trained on ERA5 and IFS analyses, so it inherits their complex-terrain "
            "biases. No native gust field."
        ),
    ),
)

#: All known dataset specifications, keyed by their short identifier.
DATASETS: Mapping[str, DatasetSpec] = MappingProxyType({spec.key: spec for spec in _SPECS})


def get_dataset_spec(key: str) -> DatasetSpec:
    """Look up a dataset specification by key.

    Parameters
    ----------
    key
        Short identifier, case-insensitive; hyphens are treated as underscores
        so that ``"era5-land"`` and ``"era5_land"`` both work.

    Returns
    -------
    DatasetSpec
        The matching specification.

    Raises
    ------
    KeyError
        If no dataset matches, listing the available keys.
    """
    normalised = key.strip().lower().replace("-", "_")
    if normalised not in DATASETS:
        raise KeyError(
            f"Unknown dataset {key!r}; available: {sorted(DATASETS)}"
        )
    return DATASETS[normalised]
