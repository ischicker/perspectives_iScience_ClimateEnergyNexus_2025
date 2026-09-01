"""Verification of gridded products against point observations.

    from alpinemet.verification import match_stations_to_grid, station_metrics

See :mod:`alpinemet.verification.stations` for the representativeness and
coverage caveats that govern how these numbers should be read.
"""

from alpinemet.verification.metrics import (
    METRIC_NAMES,
    bias,
    correlation,
    kling_gupta_efficiency,
    mean_absolute_error,
    root_mean_square_error,
    standard_deviation_ratio,
    verification_metrics,
)
from alpinemet.verification.stations import (
    DEFAULT_ELEVATION_BANDS,
    ENVIRONMENTAL_LAPSE_RATE,
    elevation_band_metrics,
    extract_at_stations,
    lapse_rate_adjustment,
    match_stations_to_grid,
    station_metrics,
)

__all__ = [
    "DEFAULT_ELEVATION_BANDS",
    "ENVIRONMENTAL_LAPSE_RATE",
    "METRIC_NAMES",
    "bias",
    "correlation",
    "elevation_band_metrics",
    "extract_at_stations",
    "kling_gupta_efficiency",
    "lapse_rate_adjustment",
    "match_stations_to_grid",
    "mean_absolute_error",
    "root_mean_square_error",
    "standard_deviation_ratio",
    "station_metrics",
    "verification_metrics",
]
