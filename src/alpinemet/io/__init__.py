"""Loading and standardising the evaluated meteorological products.

The usual entry point is :func:`open_product`, which reads one or more files and
returns a dataset on the shared conventions -- canonical variable names,
degrees Celsius, W/m2, m/s -- ready for any indicator in the package::

    from alpinemet.io import ALPINE_DOMAIN, open_product

    era5 = open_product("data/era5_2020.nc", "era5", domain=ALPINE_DOMAIN)

Use :func:`standardise` directly when the dataset is already open, for instance
when it comes from a cloud store rather than local files.
"""

from alpinemet.io.accumulation import AccumulationKind, infer_accumulation_kind, to_rate
from alpinemet.io.datasets import DATASETS, DatasetSpec, get_dataset_spec
from alpinemet.io.loaders import open_product, standardise
from alpinemet.io.naming import (
    CANONICAL_VARIABLES,
    find_coordinate,
    find_variable,
    rename_to_canonical,
    require_variable,
)
from alpinemet.io.standardise import (
    apply_quality_control,
    derive_wind_speeds,
    remove_duplicate_times,
)
from alpinemet.io.subset import (
    ALPINE_DOMAIN,
    BoundingBox,
    normalise_longitude,
    subset_spatial,
    subset_temporal,
)

__all__ = [
    "ALPINE_DOMAIN",
    "CANONICAL_VARIABLES",
    "DATASETS",
    "AccumulationKind",
    "BoundingBox",
    "DatasetSpec",
    "apply_quality_control",
    "derive_wind_speeds",
    "find_coordinate",
    "find_variable",
    "get_dataset_spec",
    "infer_accumulation_kind",
    "normalise_longitude",
    "open_product",
    "remove_duplicate_times",
    "rename_to_canonical",
    "require_variable",
    "standardise",
    "subset_spatial",
    "subset_temporal",
    "to_rate",
]
