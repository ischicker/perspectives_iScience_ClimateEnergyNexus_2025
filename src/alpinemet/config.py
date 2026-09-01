"""Configuration loading and validation.

An analysis is described by a YAML file. Machine-specific paths live in a
*separate* file (``configs/paths.yaml``, gitignored), so that the analysis
configuration itself can be committed and shared without carrying anyone's
directory layout::

    # configs/era5_eraland_2020.yaml
    name: ERA5 vs ERA5-Land, 2020
    domain: alpine
    period:
      start: "2020-01-01"
      end: "2020-12-31"
    datasets:
      - key: era5
        paths: "ERA5/era5_2020_AT.nc"
      - key: era5_land
        paths: "ERA5Land/era5land_*_2020_*.nc"

Dataset paths are relative to ``paths.data_root``. Validation is strict and
eager: an unknown dataset key, a missing required field or a malformed domain
raises at load time with a message naming the offending file, rather than
surfacing as an obscure failure once a multi-hour load is already under way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from alpinemet.io.datasets import get_dataset_spec
from alpinemet.io.subset import ALPINE_DOMAIN, BoundingBox

__all__ = [
    "AnalysisConfig",
    "DatasetEntry",
    "NAMED_DOMAINS",
    "PathsConfig",
    "load_config",
    "load_paths",
]

#: Domains addressable by name in a configuration file.
NAMED_DOMAINS: dict[str, BoundingBox] = {"alpine": ALPINE_DOMAIN}


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


@dataclass(frozen=True)
class PathsConfig:
    """Machine-specific locations.

    Attributes
    ----------
    data_root
        Directory containing the raw product files.
    output_root
        Directory for results. Created on demand by the CLI.
    population_raster
        Path to the GHS-POP GeoTIFF, relative to ``data_root`` unless absolute.
    """

    data_root: Path
    output_root: Path = Path("outputs")
    population_raster: Path | None = None

    def resolve(self, relative: str | Path) -> Path:
        """Resolve a dataset path against :attr:`data_root`.

        Absolute paths are returned unchanged, so a configuration can point
        outside the data root when it needs to.
        """
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else self.data_root / candidate


@dataclass(frozen=True)
class DatasetEntry:
    """One product to load in an analysis.

    Attributes
    ----------
    key
        Dataset key, validated against :mod:`alpinemet.io.datasets`.
    paths
        File path or glob pattern, relative to ``data_root``.
    label
        Display label; defaults to the product's long name.
    """

    key: str
    paths: str
    label: str | None = None

    def __post_init__(self) -> None:
        # Raises KeyError with the available keys listed.
        get_dataset_spec(self.key)


@dataclass(frozen=True)
class AnalysisConfig:
    """A complete analysis specification.

    Attributes
    ----------
    name
        Short identifier, used for output directory names.
    description
        Free text.
    domain
        Spatial extent.
    start, end
        Inclusive analysis period, or ``None`` for the full record.
    datasets
        Products to load.
    options
        Free-form extra settings, passed through untouched. Use this for
        threshold overrides rather than adding fields here.
    source_file
        Path the configuration was loaded from, for error messages.
    """

    name: str
    domain: BoundingBox = ALPINE_DOMAIN
    description: str = ""
    start: str | None = None
    end: str | None = None
    datasets: tuple[DatasetEntry, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)
    source_file: Path | None = None


def _require(mapping: dict[str, Any], key: str, source: Path) -> Any:
    if key not in mapping:
        raise ConfigError(f"{source}: missing required key {key!r}")
    return mapping[key]


def _parse_domain(value: Any, source: Path) -> BoundingBox:
    if value is None:
        return ALPINE_DOMAIN

    if isinstance(value, str):
        if value not in NAMED_DOMAINS:
            raise ConfigError(
                f"{source}: unknown domain {value!r}; known names are "
                f"{sorted(NAMED_DOMAINS)}, or give explicit bounds"
            )
        return NAMED_DOMAINS[value]

    if not isinstance(value, dict):
        raise ConfigError(
            f"{source}: 'domain' must be a name or a mapping of bounds, got {type(value).__name__}"
        )

    required = ("lon_min", "lon_max", "lat_min", "lat_max")
    missing = [name for name in required if name not in value]
    if missing:
        raise ConfigError(f"{source}: domain is missing {missing}")

    try:
        return BoundingBox(
            lon_min=float(value["lon_min"]),
            lon_max=float(value["lon_max"]),
            lat_min=float(value["lat_min"]),
            lat_max=float(value["lat_max"]),
            name=str(value.get("name", "custom domain")),
        )
    except ValueError as exc:
        raise ConfigError(f"{source}: invalid domain: {exc}") from exc


def _parse_datasets(value: Any, source: Path) -> tuple[DatasetEntry, ...]:
    if not value:
        raise ConfigError(f"{source}: 'datasets' must list at least one product")
    if not isinstance(value, list):
        raise ConfigError(f"{source}: 'datasets' must be a list, got {type(value).__name__}")

    entries: list[DatasetEntry] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ConfigError(
                f"{source}: datasets[{index}] must be a mapping, got {type(item).__name__}"
            )
        try:
            entries.append(
                DatasetEntry(
                    key=str(_require(item, "key", source)),
                    paths=str(_require(item, "paths", source)),
                    label=item.get("label"),
                )
            )
        except KeyError as exc:
            raise ConfigError(f"{source}: datasets[{index}]: {exc.args[0]}") from exc
    return tuple(entries)


def load_paths(path: str | Path) -> PathsConfig:
    """Load the machine-specific paths file.

    Parameters
    ----------
    path
        Path to ``paths.yaml``.

    Returns
    -------
    PathsConfig
        The parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the file does not exist. The message points at
        ``configs/paths.example.yaml``, which is the usual reason for this
        failure on a fresh clone.
    ConfigError
        If ``data_root`` is missing.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"No paths file at {source}. Copy configs/paths.example.yaml to "
            f"{source} and set 'data_root' to your data directory."
        )

    content = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if "data_root" not in content:
        raise ConfigError(f"{source}: missing required key 'data_root'")

    population = content.get("population_raster")
    return PathsConfig(
        data_root=Path(content["data_root"]),
        output_root=Path(content.get("output_root", "outputs")),
        population_raster=Path(population) if population else None,
    )


def load_config(path: str | Path) -> AnalysisConfig:
    """Load and validate an analysis configuration.

    Parameters
    ----------
    path
        Path to the YAML file.

    Returns
    -------
    AnalysisConfig
        The validated configuration.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ConfigError
        If the file is malformed, names an unknown dataset, or omits a
        required field.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"No configuration file at {source}")

    try:
        content = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{source}: could not parse YAML: {exc}") from exc

    if not isinstance(content, dict):
        raise ConfigError(
            f"{source}: expected a mapping at the top level, got {type(content).__name__}"
        )

    period = content.get("period") or {}
    if not isinstance(period, dict):
        raise ConfigError(f"{source}: 'period' must be a mapping with 'start' and 'end'")

    start = period.get("start")
    end = period.get("end")
    if start is not None and end is not None and str(start) > str(end):
        raise ConfigError(f"{source}: period start {start} is after end {end}")

    return AnalysisConfig(
        name=str(_require(content, "name", source)),
        description=str(content.get("description", "")),
        domain=_parse_domain(content.get("domain"), source),
        start=None if start is None else str(start),
        end=None if end is None else str(end),
        datasets=_parse_datasets(content.get("datasets"), source),
        options=dict(content.get("options") or {}),
        source_file=source,
    )
