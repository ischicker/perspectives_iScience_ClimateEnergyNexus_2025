"""Plotting: shared colour scales and panels that actually render."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from alpinemet.plotting.maps import (  # noqa: E402
    cartopy_available,
    comparison_grid,
    plot_field,
)
from alpinemet.plotting.style import (  # noqa: E402
    COLORMAPS,
    PANEL_LABELS,
    apply_style,
    panel_label,
    shared_limits,
    symmetric_limits,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _field(values, *, lats=None, lons=None) -> xr.DataArray:
    array = np.asarray(values, dtype=float)
    lats = np.linspace(46.0, 48.0, array.shape[0]) if lats is None else lats
    lons = np.linspace(10.0, 14.0, array.shape[1]) if lons is None else lons
    return xr.DataArray(
        array,
        dims=("latitude", "longitude"),
        coords={"latitude": lats, "longitude": lons},
    )


# --------------------------------------------------------------------------
# Colour scales
# --------------------------------------------------------------------------


def test_a_shared_scale_spans_every_panel():
    coarse = _field(np.full((4, 5), 10.0))
    fine = _field(np.linspace(0, 40, 20).reshape(4, 5))

    vmin, vmax = shared_limits([coarse, fine], percentile=None)
    assert vmin == pytest.approx(0.0)
    assert vmax == pytest.approx(40.0)


def test_a_per_panel_scale_would_hide_the_difference():
    """The reason shared_limits exists.

    A coarse panel and a fine panel scaled independently both fill their own
    colour range, making the coarse one look as detailed as the fine one.
    """
    coarse = _field(np.full((4, 5), 10.0))
    fine = _field(np.linspace(0, 40, 20).reshape(4, 5))

    together = shared_limits([coarse, fine], percentile=None)
    alone = shared_limits([coarse], percentile=None)

    assert together != alone


def test_the_percentile_clip_tames_a_single_outlier():
    values = np.ones((10, 10))
    values[0, 0] = 1000.0
    field = _field(values)

    clipped = shared_limits([field], percentile=99.0)[1]
    unclipped = shared_limits([field], percentile=None)[1]

    assert clipped < unclipped
    assert unclipped == pytest.approx(1000.0)


def test_counts_can_be_pinned_to_zero():
    field = _field(np.full((3, 3), 5.0))
    assert shared_limits([field], minimum_at_zero=True)[0] == 0.0


def test_a_constant_field_still_yields_a_usable_range():
    vmin, vmax = shared_limits([_field(np.full((3, 3), 7.0))])
    assert vmax > vmin


def test_missing_values_are_ignored():
    values = np.array([[np.nan, 2.0, 3.0], [4.0, 5.0, np.nan], [6.0, 7.0, 8.0]])
    vmin, vmax = shared_limits([_field(values)], percentile=None)
    assert (vmin, vmax) == pytest.approx((2.0, 8.0))


def test_an_all_missing_field_is_rejected():
    with pytest.raises(ValueError, match="No finite values"):
        shared_limits([_field(np.full((3, 3), np.nan))])


def test_bias_limits_are_centred_on_zero():
    field = _field(np.array([[-2.0, 1.0], [3.0, -1.0]]))
    vmin, vmax = symmetric_limits([field], percentile=None)
    assert vmin == pytest.approx(-vmax)
    assert vmax == pytest.approx(3.0)


def test_a_zero_bias_field_still_yields_a_usable_range():
    vmin, vmax = symmetric_limits([_field(np.zeros((3, 3)))])
    assert vmin < 0 < vmax


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------


def test_applying_the_style_is_idempotent():
    apply_style()
    first = plt.rcParams["savefig.dpi"]
    apply_style()
    assert plt.rcParams["savefig.dpi"] == first


def test_figures_are_saved_at_publication_resolution():
    apply_style()
    assert plt.rcParams["savefig.dpi"] >= 300


def test_bias_uses_a_diverging_colormap():
    """A signed field needs a two-sided map; a sequential one hides the sign."""
    import matplotlib

    colormap = matplotlib.colormaps[COLORMAPS["bias"]]
    low, middle, high = colormap(0.0), colormap(0.5), colormap(1.0)

    # Diverging maps are light in the middle and saturated at both ends.
    def luminance(rgba):
        return 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]

    assert luminance(middle) > luminance(low)
    assert luminance(middle) > luminance(high)


def test_magnitudes_use_a_sequential_colormap():
    import matplotlib

    colormap = matplotlib.colormaps[COLORMAPS["storm"]]
    samples = [colormap(x)[:3] for x in np.linspace(0, 1, 9)]

    def luminance(rgb):
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    values = [luminance(rgb) for rgb in samples]
    assert values == sorted(values), "a sequential map must vary monotonically"


def test_panel_labels_are_added():
    figure, axis = plt.subplots()
    panel_label(axis, 0)
    texts = [t.get_text() for t in axis.texts]
    assert "A" in texts


def test_running_out_of_panel_labels_is_an_error():
    figure, axis = plt.subplots()
    with pytest.raises(IndexError, match="No panel label"):
        panel_label(axis, len(PANEL_LABELS))


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def test_a_field_renders_onto_a_plain_axis():
    figure, axis = plt.subplots()
    mesh = plot_field(axis, _field(np.random.default_rng(0).normal(size=(6, 8))),
                      domain=None)
    assert mesh is not None


def test_a_three_dimensional_field_is_rejected():
    figure, axis = plt.subplots()
    field = xr.DataArray(
        np.zeros((2, 3, 4)),
        dims=("time", "latitude", "longitude"),
        coords={"time": [0, 1], "latitude": [46.0, 47.0, 48.0],
                "longitude": [10.0, 11.0, 12.0, 13.0]},
    )
    with pytest.raises(ValueError, match="two-dimensional"):
        plot_field(axis, field)


def test_a_field_without_coordinates_is_rejected():
    figure, axis = plt.subplots()
    with pytest.raises(ValueError, match="no latitude/longitude"):
        plot_field(axis, xr.DataArray(np.zeros((3, 4)), dims=("y2", "x2")))


def test_a_comparison_grid_has_the_expected_shape():
    rng = np.random.default_rng(1)
    rows = [
        {
            "fields": [_field(rng.normal(size=(6, 8))) for _ in range(3)],
            "label": "Hot days",
            "cmap": "Reds",
            "minimum_at_zero": True,
        },
        {
            "fields": [_field(rng.normal(size=(6, 8))) for _ in range(3)],
            "label": "CDD",
            "cmap": "YlOrRd",
        },
    ]
    figure, axes = comparison_grid(
        rows,
        column_titles=["ERA5 (31 km)", "ERA5-Land (9 km)", "ARA (2.5 km)"],
        use_cartopy=False,
        domain=None,
    )

    assert len(axes) == 2
    assert len(axes[0]) == 3
    assert axes[0][0].get_title() == "ERA5 (31 km)"


def test_every_panel_in_a_row_shares_its_colour_scale():
    fields = [_field(np.full((4, 5), value)) for value in (1.0, 50.0, 100.0)]
    figure, axes = comparison_grid(
        [{"fields": fields, "label": "test"}],
        column_titles=["a", "b", "c"],
        use_cartopy=False,
        domain=None,
    )

    limits = {axis.collections[0].get_clim() for axis in axes[0]}
    assert len(limits) == 1, "panels in a row must share one colour scale"


def test_panels_are_labelled_in_sequence():
    fields = [_field(np.zeros((3, 3))) for _ in range(2)]
    figure, axes = comparison_grid(
        [{"fields": fields, "label": "x"}, {"fields": fields, "label": "y"}],
        column_titles=["a", "b"],
        use_cartopy=False,
        domain=None,
    )
    labels = [axis.texts[0].get_text() for row in axes for axis in row]
    assert labels == ["A", "B", "C", "D"]


def test_a_row_with_the_wrong_field_count_is_rejected():
    with pytest.raises(ValueError, match="but there are 3 columns"):
        comparison_grid(
            [{"fields": [_field(np.zeros((3, 3)))], "label": "x"}],
            column_titles=["a", "b", "c"],
            use_cartopy=False,
        )


def test_the_grid_renders_with_cartopy_when_it_is_available():
    if not cartopy_available():
        pytest.skip("cartopy or its Natural Earth data is unavailable")

    fields = [_field(np.zeros((4, 5))) for _ in range(2)]
    figure, axes = comparison_grid(
        [{"fields": fields, "label": "x"}],
        column_titles=["a", "b"],
        use_cartopy=True,
    )
    assert hasattr(axes[0][0], "coastlines")


def test_the_grid_still_works_without_cartopy():
    """Cartopy is the dependency most likely to be missing on a fresh machine."""
    fields = [_field(np.zeros((4, 5))) for _ in range(2)]
    figure, axes = comparison_grid(
        [{"fields": fields, "label": "x"}],
        column_titles=["a", "b"],
        use_cartopy=False,
        domain=None,
    )
    assert not hasattr(axes[0][0], "coastlines")


def test_a_figure_can_be_saved(tmp_path):
    fields = [_field(np.zeros((4, 5))) for _ in range(2)]
    figure, _ = comparison_grid(
        [{"fields": fields, "label": "x"}],
        column_titles=["a", "b"],
        use_cartopy=False,
        domain=None,
    )
    target = tmp_path / "figure.png"
    figure.savefig(target)
    assert target.stat().st_size > 1000
