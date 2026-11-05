"""Shared tricontour plotting helpers.

Extracted verbatim from ``shap_indexed_tanimoto_plot.py`` so that the SHAP
interaction heatmaps can be reproduced from other notebooks (e.g.
``shap_verification_plots.py``) without importing that script, which would
re-run all of its module-level figure generation and overwrite the saved
manuscript SVGs.

Also provides :func:`load_shap_cache` for the pickled shapiq cache.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import (
    TriAnalyzer,
    Triangulation,
    UniformTriRefiner,
    LinearTriInterpolator,
    CubicTriInterpolator,
)


def load_shap_cache(cache_path):
    """Load the pickled shapiq cache and derive the indices/labels used by plots.

    Returns a dict with the cache contents plus ``sample_category_strings`` and
    the ``idx_*`` column indices, mirroring the setup in
    ``shap_indexed_tanimoto_plot.py``.
    """
    with Path(cache_path).open("rb") as f:
        cache = pickle.load(f)

    X_shapiq = cache["X_shapiq"]
    feature_names = cache["feature_names"]
    a_site_names = cache["a_site_names"]

    return {
        "shapiq_values": cache["shapiq_values"],
        "X_shapiq": X_shapiq,
        "y_shapiq": cache["y_shapiq"],
        "feature_names": feature_names,
        "a_site_names": a_site_names,
        "sample_category_strings": [a_site_names[int(val)] for val in X_shapiq[:, -1]],
        "idx_concentration": feature_names.index("concentration"),
        "idx_temp": feature_names.index("annealing_temp"),
        "idx_asite": feature_names.index("a_site_code"),
    }


def refine_tricontour(
    triangulation, z, subdiv=3, min_circle_ratio=0.01, triinterpolator="cubic"
):
    """Refine a triangular mesh for tricontour plots.
    Parameters
    ----------
    triangulation : matplotlib.tri.Triangulation
        Triangulation object.
    z : array-like
        Values at the data points.
    subdiv : int, optional
         Number of recursive subdivisions of the initial mesh for smooth plots.
         Values >3 might result in a very high number of triangles for the refine mesh: new triangles numbering = (4**subdiv)*ntri
    min_circle_ratio : float, optional
        Minimum circle ratio - border triangles with circle ratio below this will be
        masked if they touch a border. Suggested value 0.01; use -1 to keep all triangles.
    Returns
    -------
    tri_refi : matplotlib.tri.Triangulation
        Refined triangulation object.
    z_refi : array-like
        Values at the refined triangulation points.
    """

    # ----------------------------------------------------------------------------
    # Improving the triangulation before high-res plots: removing flat triangles
    # ----------------------------------------------------------------------------
    # masking badly shaped triangles at the border of the triangular mesh.
    mask = TriAnalyzer(triangulation).get_flat_tri_mask(min_circle_ratio)
    triangulation.set_mask(mask)

    # refining the data
    refiner = UniformTriRefiner(triangulation)
    if triinterpolator == "linear":
        interp = LinearTriInterpolator(triangulation, z)
    elif triinterpolator == "cubic":
        interp = CubicTriInterpolator(triangulation, z, kind="geom")
    else:
        raise ValueError(
            f"Invalid triinterpolator: {triinterpolator!r}. "
            "Choose 'linear', 'cubic', or pass a "
            "TriInterpolator instance."
        )
    tri_refi, z_refi = refiner.refine_field(z, subdiv=subdiv, triinterpolator=interp)

    return tri_refi, z_refi


def create_faceted_tricontour(
    df,
    x_col,
    y_col,
    value_col,
    facet_col=None,
    levels=14,
    title=None,
    cbar_label="Value",
    figsize=None,
    filled=True,
    show_points=True,
    refine=True,
    x_lim=None,
    y_lim=None,
    return_ax=False,
):
    """Single or multi-panel tricontour plots with symmetric color scaling.

    Parameters
    ----------
    return_ax : bool, optional
        If True, return ``(fig, dict)`` where the dict maps each facet category to
        its axes (``{None: ax}`` when not faceting), so callers can draw overlays
        on the correct panel afterwards. Defaults to False, returning just ``fig``.
    """
    df_plot = df.copy().dropna(subset=[x_col, y_col, value_col])

    v_min, v_max = np.percentile(df_plot[value_col], [5, 95])
    level_edges = np.linspace(v_min, v_max, levels + 1)
    x_lim = (
        x_lim
        if isinstance(x_lim, tuple)
        else (df_plot[x_col].min() * 0.95, df_plot[x_col].max() * 1.02)
    )
    y_lim = (
        y_lim
        if isinstance(y_lim, tuple)
        else (df_plot[y_col].min(), df_plot[y_col].max())
    )

    sm = plt.cm.ScalarMappable(
        cmap="coolwarm", norm=plt.Normalize(vmin=v_min, vmax=v_max)
    )

    if facet_col:
        categories = sorted(df_plot[facet_col].unique())
        n_panels = len(categories)
        fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=figsize or (4 * n_panels, 5),
            sharey=True,
            layout="constrained",
            dpi=150,
        )
        axes = np.atleast_1d(axes)
    else:
        fig, ax = plt.subplots(figsize=figsize or (6, 5), layout="constrained", dpi=150)
        axes = [ax]
        categories = [None]

    cs = None
    for i, cat in enumerate(categories):
        ax = axes[i]
        # keep ax aspect ratio equal to avoid distortion of triangulation
        ax.set_box_aspect(1)

        df_sub = (df_plot[df_plot[facet_col] == cat] if facet_col else df_plot).copy()
        if facet_col:
            ax.set_title(
                f"{cat}", pad=-8, x=0.04, ha="left", y=0.96, fontweight="semibold"
            )

        # average duplicate points (same x and y) to avoid issues with triangulation
        df_sub_avg = df_sub.groupby([x_col, y_col], as_index=False).agg(
            {value_col: "mean"}
        )
        xv, yv, zv = df_sub_avg[[x_col, y_col, value_col]].to_numpy().T

        if len(df_sub_avg) < 3:
            ax.text(
                0.5,
                0.5,
                "Not enough points",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            fn = ax.tricontourf if filled else ax.tricontour
            sx, sy = np.ptp(xv), np.ptp(yv)
            tri_n = Triangulation(xv / sx, yv / sy)  # Delaunay, isotropic

            if isinstance(refine, bool) and not refine:
                tri = Triangulation(xv, yv, triangles=tri_n.triangles)
                zv_refi = zv

            else:
                refine_kwargs = refine if isinstance(refine, dict) else {}
                tri_refi_n, zv_refi = refine_tricontour(tri_n, zv, **refine_kwargs)

                tri = Triangulation(
                    tri_refi_n.x * sx, tri_refi_n.y * sy, triangles=tri_refi_n.triangles
                )

                if tri_refi_n.mask is not None:
                    tri.set_mask(tri_refi_n.mask)

                lin = LinearTriInterpolator(tri_n, zv)
                z_lin = lin(tri_refi_n.x, tri_refi_n.y)
                pad = 0.05 * np.ptp(zv)
                bad = (zv_refi > zv.max() + pad) | (zv_refi < zv.min() - pad)
                if bad.any():
                    print(
                        f"repaired {bad.sum()} out of {len(zv_refi)} node(s), max |cubic| = {np.abs(zv_refi[bad]).max():.3g}"
                    )
                    zv_refi[bad] = z_lin[bad]

            cs = fn(
                tri,
                zv_refi,
                levels=level_edges,
                cmap="coolwarm",
                vmin=v_min,
                vmax=v_max,
                extend="both",
            )

            if not filled:
                ax.clabel(cs, inline=True, fontsize="x-small", fmt="%.1f")
            if show_points:
                xv, yv, zv = df_sub[[x_col, y_col, value_col]].to_numpy().T
                interp = LinearTriInterpolator(tri, zv_refi)
                zv_pred = interp(xv, yv)
                zv_err = zv_pred - zv
                print(f"RMSE of interpolation: {np.sqrt(np.mean(zv_err**2)):.4f}")

                # sort by error magnitude for plotting, so that the largest errors are on the bottom and not covering smaller points
                sort_idx = np.argsort(np.abs(zv_err))[::-1]
                xv, yv, zv, zv_pred = (
                    xv[sort_idx],
                    yv[sort_idx],
                    zv[sort_idx],
                    zv_pred[sort_idx],
                )

                # colors = sm.to_rgba(zv)
                # ax.scatter(xv, yv, c=colors, s=np.abs(zv_pred-zv)**2*10 + 5, ec='k')
                # ax.scatter(xv, yv, c=colors, s=8, ec='k', lw=0.5, alpha=0.5)
                ax.scatter(xv, yv, c="k", s=1, alpha=0.15)

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.tick_params(axis="both", which="both", direction="in")

    fig.supxlabel(x_col)
    fig.supylabel(y_col)
    if title:
        fig.suptitle(title)

    cbar = fig.colorbar(
        sm, ax=axes if facet_col else axes[0], shrink=0.8, pad=0.01, format="%.0f"
    )
    cbar.set_label(cbar_label, rotation=270, verticalalignment="bottom")

    if return_ax:
        return fig, dict(zip(categories, axes))
    return fig
