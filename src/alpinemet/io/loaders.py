"""Loading products into a standardised form.

:func:`standardise` is the heart of this module: it takes an already-open
dataset and applies the same pipeline to every product, driven by that
product's :class:`~alpinemet.io.datasets.DatasetSpec`. The order matters and is
fixed:

1. **Rename** to canonical names, so every later step can address variables the
   same way.
2. **Remove duplicate timestamps**, before anything differences along time. A
   repeated step silently corrupts every accumulation conversion downstream.
3. **Convert accumulated fluxes to rates**, using the convention declared on
   the specification.
4. **Convert temperature to degrees Celsius**, the convention every indicator
   in this package assumes.
5. **Subset** in time and space, after the conversions -- a running accumulator
   differenced *after* subsetting would lose the step at the new left edge.
6. **Derive wind speeds** from the components.
7. **Quality control** last, so that unit errors introduced by earlier steps
   are caught rather than hidden. This ordering is what makes the physical
   ranges in :data:`alpinemet.io.standardise.PHYSICAL_RANGES` meaningful: they
   are stated in the target units, and a field left in kelvin would be flagged
   in its entirety.

:func:`open_product` wraps this around ``xarray.open_mfdataset`` for the common
case of one or more files on disk.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from glob import glob
from pathlib import Path

import xarray as xr

from alpinemet.attrs import as_attribute
from alpinemet.io.accumulation import AccumulationKind, to_rate
from alpinemet.io.datasets import DatasetSpec, get_dataset_spec
from alpinemet.io.naming import find_variable, rename_to_canonical
from alpinemet.io.standardise import (
    apply_quality_control,
    derive_wind_speeds,
    remove_duplicate_times,
)
from alpinemet.io.subset import BoundingBox, subset_spatial, subset_temporal
from alpinemet.units import to_celsius, to_millimetres

__all__ = [
    "ACCUMULATION_OUTPUT_UNITS",
    "PER_SECOND_VARIABLES",
    "open_product",
    "standardise",
]

#: Units recorded on each flux variable after deaccumulation.
ACCUMULATION_OUTPUT_UNITS = {
    "solar_radiation": "W m-2",
}

#: Variables whose target unit is per-second, so that deaccumulation is
#: followed by division by the step length. Radiation goes from J/m2 per step
#: to W/m2; precipitation stays a per-step total in millimetres.
PER_SECOND_VARIABLES = frozenset({"solar_radiation"})


def standardise(
    dataset: xr.Dataset,
    spec: DatasetSpec | str,
    *,
    domain: BoundingBox | None = None,
    start: str | None = None,
    end: str | None = None,
    quality_control: str | None = "mask",
    keep_unknown: bool = True,
) -> xr.Dataset:
    """Bring a product onto the shared conventions.

    Parameters
    ----------
    dataset
        Open dataset in its native form.
    spec
        Dataset specification, or a key to look one up with.
    domain
        Bounding box to subset to. ``None`` keeps the full extent.
    start, end
        Inclusive time bounds. ``None`` leaves that end open.
    quality_control
        Action passed to
        :func:`alpinemet.io.standardise.apply_quality_control`, or ``None`` to
        skip the check entirely.
    keep_unknown
        Keep variables that match no canonical name.

    Returns
    -------
    xarray.Dataset
        Standardised dataset, with the product key and resolution recorded in
        its attributes so that downstream comparisons can label themselves.

    Raises
    ------
    KeyError
        If ``spec`` is a key that matches no known product.
    ValueError
        If subsetting selects nothing, or quality control is given an unknown
        action.
    """
    if isinstance(spec, str):
        spec = get_dataset_spec(spec)

    result = rename_to_canonical(
        dataset, extra_aliases=spec.extra_aliases, keep_unknown=keep_unknown
    )
    result = remove_duplicate_times(result)

    for canonical, kind in spec.accumulation.items():
        if kind is AccumulationKind.INSTANTANEOUS:
            continue
        name = find_variable(result, canonical)
        if name is None:
            continue
        result[name] = to_rate(
            result[name],
            kind,
            timestep_seconds=spec.timestep_hours * 3600.0,
            per_second=canonical in PER_SECOND_VARIABLES,
            output_units=ACCUMULATION_OUTPUT_UNITS.get(canonical),
        )

    temperature_name = find_variable(result, "temperature_2m")
    if temperature_name is not None:
        result[temperature_name] = to_celsius(result[temperature_name])

    # ECMWF reports precipitation in metres of water equivalent. A field left in
    # metres passes any plausibility check stated in millimetres, so the error is
    # silent and totals come out a thousand times too small.
    precipitation_name = find_variable(result, "precipitation")
    if precipitation_name is not None:
        result[precipitation_name] = to_millimetres(result[precipitation_name])

    if start is not None or end is not None:
        result = subset_temporal(result, start, end)
    if domain is not None:
        result = subset_spatial(result, domain)

    result = derive_wind_speeds(result)

    if quality_control is not None:
        result = apply_quality_control(result, action=quality_control)

    result.attrs = {
        **result.attrs,
        "dataset_key": spec.key,
        "dataset_long_name": spec.long_name,
        "resolution_km": spec.resolution_km,
        "timestep_hours": spec.timestep_hours,
        "grid_type": spec.grid_type,
        "has_native_gust": as_attribute(spec.has_native_gust),
        "has_100m_wind": as_attribute(spec.has_100m_wind),
    }
    if spec.notes:
        result.attrs["dataset_notes"] = spec.notes
    return result


def open_product(
    paths: str | Path | Sequence[str | Path],
    spec: DatasetSpec | str,
    *,
    domain: BoundingBox | None = None,
    start: str | None = None,
    end: str | None = None,
    quality_control: str | None = "mask",
    keep_unknown: bool = True,
    chunks: dict[str, int] | str | None = "auto",
    open_kwargs: dict | None = None,
) -> xr.Dataset:
    """Open one or more files and standardise them.

    Parameters
    ----------
    paths
        A file path, a glob pattern, or a sequence of paths.
    spec
        Dataset specification, or a key to look one up with.
    domain, start, end, quality_control, keep_unknown
        Passed to :func:`standardise`.
    chunks
        Dask chunking passed to ``xarray.open_mfdataset``. The default lets
        xarray choose; pass an explicit mapping for large multi-year loads.
    open_kwargs
        Extra keyword arguments for ``xarray.open_mfdataset``.

    Returns
    -------
    xarray.Dataset
        Standardised dataset.

    Raises
    ------
    FileNotFoundError
        If a sequence of paths contains one that does not exist, or a glob
        matches nothing.
    """
    if isinstance(spec, str):
        spec = get_dataset_spec(spec)

    resolved = _resolve_paths(paths)

    open_kwargs = {
        "combine": "by_coords",
        "chunks": chunks,
        **(open_kwargs or {}),
    }
    dataset = xr.open_mfdataset(resolved, **open_kwargs)

    return standardise(
        dataset,
        spec,
        domain=domain,
        start=start,
        end=end,
        quality_control=quality_control,
        keep_unknown=keep_unknown,
    )


def _resolve_paths(paths: str | Path | Sequence[str | Path]) -> list[str]:
    """Expand globs and verify that explicit paths exist."""
    if isinstance(paths, (str, Path)):
        candidates: Iterable[str | Path] = [paths]
    else:
        candidates = paths

    resolved: list[str] = []
    for candidate in candidates:
        text = str(candidate)
        if any(character in text for character in "*?["):
            # glob.glob handles absolute patterns on every platform;
            # Path.glob rejects them on Windows.
            matches = sorted(glob(text))
            if not matches:
                raise FileNotFoundError(f"Pattern {text!r} matched no files")
            resolved.extend(matches)
        else:
            if not Path(text).exists():
                raise FileNotFoundError(f"No such file: {text}")
            resolved.append(text)

    if not resolved:
        raise FileNotFoundError("No input files given")
    return resolved
