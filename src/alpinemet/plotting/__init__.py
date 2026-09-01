"""Figure building blocks.

    from alpinemet.plotting import comparison_grid, shared_limits

The recurring figure in this study is a grid of maps, one row per indicator and
one column per product, with a shared colour scale per row -- see
:mod:`alpinemet.plotting.maps` for why that sharing matters.
"""

from alpinemet.plotting.maps import cartopy_available, comparison_grid, plot_field
from alpinemet.plotting.style import (
    COLORMAPS,
    FIGURE_DPI,
    PANEL_LABELS,
    apply_style,
    panel_label,
    shared_limits,
    symmetric_limits,
)

__all__ = [
    "COLORMAPS",
    "FIGURE_DPI",
    "PANEL_LABELS",
    "apply_style",
    "cartopy_available",
    "comparison_grid",
    "panel_label",
    "plot_field",
    "shared_limits",
    "symmetric_limits",
]
