"""Consistent styling for the figures.

Colour choices are deliberate rather than decorative. Multi-panel comparisons
across resolutions only work if every panel shares a scale, so
:func:`shared_limits` computes one from all panels rather than letting each pick
its own -- a per-panel scale makes a coarse and a fine product look equally
detailed, which is precisely the impression this study argues against.

The sequential maps are perceptually uniform and remain readable in greyscale.
The diverging map is centred on zero for bias fields, where the sign carries the
meaning.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

import numpy as np

__all__ = [
    "COLORMAPS",
    "FIGURE_DPI",
    "PANEL_LABELS",
    "apply_style",
    "panel_label",
    "shared_limits",
    "symmetric_limits",
]

#: Resolution for saved figures. iScience asks for at least 300 dpi.
FIGURE_DPI = 300

#: Colormap per quantity. Sequential for magnitudes, diverging for signed fields.
COLORMAPS: Mapping[str, str] = MappingProxyType(
    {
        "temperature": "RdYlBu_r",
        "heat": "Reds",
        "cold": "Blues",
        "degree_days": "YlOrRd",
        "heating_degree_days": "YlGnBu",
        "wind": "viridis",
        "gust": "magma",
        "storm": "cividis",
        "precipitation": "YlGnBu",
        "population": "plasma",
        "bias": "RdBu_r",
        "correlation": "viridis",
    }
)

#: Panel labels in the order used in the manuscript figures.
PANEL_LABELS: tuple[str, ...] = tuple("ABCDEFGHIJKL")


def apply_style() -> None:
    """Set the matplotlib rcParams used for every figure in this package.

    Idempotent, and safe to call from a notebook. Uses a non-interactive
    backend only if none has been selected yet, so it does not fight an
    existing notebook backend.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    if matplotlib.get_backend().lower() == "agg":
        pass  # already headless; nothing to do

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "image.cmap": "viridis",
        }
    )


def shared_limits(
    fields: Iterable,
    *,
    percentile: float | None = 99.0,
    minimum_at_zero: bool = False,
) -> tuple[float, float]:
    """Compute one colour scale spanning several panels.

    Parameters
    ----------
    fields
        Arrays or DataArrays to span. Missing values are ignored.
    percentile
        Clip the upper limit at this percentile of the pooled values, so that a
        single extreme cell does not compress the whole scale. ``None`` uses the
        true maximum.
    minimum_at_zero
        Force the lower limit to zero, appropriate for counts and durations.

    Returns
    -------
    tuple of float
        ``(vmin, vmax)``.

    Raises
    ------
    ValueError
        If no finite values are found across all fields.
    """
    pooled: list[np.ndarray] = []
    for field in fields:
        values = np.asarray(getattr(field, "values", field), dtype=float).ravel()
        finite = values[np.isfinite(values)]
        if finite.size:
            pooled.append(finite)

    if not pooled:
        raise ValueError("No finite values found; cannot derive a shared colour scale")

    combined = np.concatenate(pooled)
    vmin = 0.0 if minimum_at_zero else float(np.min(combined))
    vmax = (
        float(np.max(combined))
        if percentile is None
        else float(np.percentile(combined, percentile))
    )

    if vmax <= vmin:
        # A constant field would otherwise produce a degenerate colour scale.
        vmax = vmin + 1.0
    return vmin, vmax


def symmetric_limits(fields: Iterable, *, percentile: float | None = 99.0) -> tuple[float, float]:
    """Compute limits centred on zero, for bias and difference fields.

    Parameters
    ----------
    fields
        Arrays or DataArrays to span.
    percentile
        Clip the magnitude at this percentile of the absolute values.

    Returns
    -------
    tuple of float
        ``(-m, m)`` where ``m`` is the chosen magnitude.

    Raises
    ------
    ValueError
        If no finite values are found.
    """
    pooled: list[np.ndarray] = []
    for field in fields:
        values = np.asarray(getattr(field, "values", field), dtype=float).ravel()
        finite = np.abs(values[np.isfinite(values)])
        if finite.size:
            pooled.append(finite)

    if not pooled:
        raise ValueError("No finite values found; cannot derive a symmetric colour scale")

    combined = np.concatenate(pooled)
    magnitude = (
        float(np.max(combined))
        if percentile is None
        else float(np.percentile(combined, percentile))
    )
    if magnitude <= 0:
        magnitude = 1.0
    return -magnitude, magnitude


def panel_label(axis, index: int, *, labels: tuple[str, ...] = PANEL_LABELS) -> None:
    """Add a manuscript-style panel label to an axis.

    Parameters
    ----------
    axis
        Matplotlib axes to label.
    index
        Zero-based panel index.
    labels
        Label sequence.

    Raises
    ------
    IndexError
        If ``index`` exceeds the available labels.
    """
    if index >= len(labels):
        raise IndexError(f"No panel label for index {index}; only {len(labels)} defined")

    axis.text(
        0.02,
        0.97,
        labels[index],
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85,
              "edgecolor": "none"},
    )
