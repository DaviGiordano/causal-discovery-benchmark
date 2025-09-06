# viz_utils.py
# Publication-friendly plotting utilities and analysis helpers

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

__all__ = [
    # Theme & palettes
    "set_pub_theme",
    "tol_palette",
    # Ax helpers
    "panel",
    "raster_scatter",
    "side_colorbar",
    "annotate_ax",
    # Data helpers
    "apply_algorithm_labels",
    "apply_mechanism_labels",
    "rename_metric_columns",
    # Plots
    "plot_metric_by_nodes_overall",
    "plot_metric_by_tier_overall",
    "plot_metric_by_nodes_and_tier",
    "plot_metric_by_algorithm",
    "plot_metric_by_algorithm_and_nodes",
    "plot_metric_by_algorithm_and_tiers",
    "plot_metric_by_mechanism",
    "plot_metric_by_algorithm_and_mechanism",
    "plot_metric_by_tiers_and_nodes",
    # Summaries
    "summary_mechanism_f1",
    "summary_algorithm_f1",
    "summary_median_ece",
    "pivot_f1_by_alg_nodes",
    "pivot_prec_by_alg_nodes",
    "pivot_recall_by_alg_nodes",
    "pivot_skel_f1_by_alg_nodes",
    "pivot_f1_by_alg_tiers",
    "pivot_skel_f1_by_alg_tiers",
    "pivot_f1_by_alg_mech",
]

# ---------------- Theme & palettes ----------------


def set_pub_theme(
    font_size: int = 9, line_width: float = 0.5, font_scale: float = 0.75
):
    """Seaborn/Matplotlib theme tuned for papers (serif fonts, subtle black)."""
    _new_black = "#373737"  # "never use pure black"
    sns.set_theme(
        style="ticks",
        font_scale=font_scale,
        rc={
            # Fonts/output
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
            "svg.fonttype": "none",
            "text.usetex": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Sizes
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "legend.fontsize": font_size,
            "legend.title_fontsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "axes.labelpad": 2,
            "axes.titlepad": 4,
            # Line widths
            "axes.linewidth": line_width,
            "lines.linewidth": line_width,
            "xtick.major.width": line_width,
            "ytick.major.width": line_width,
            "xtick.minor.width": line_width,
            "ytick.minor.width": line_width,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "xtick.major.pad": 1,
            "ytick.major.pad": 1,
            "xtick.minor.pad": 1,
            "ytick.minor.pad": 1,
            # Neutral—not black
            "text.color": _new_black,
            "axes.edgecolor": _new_black,
            "axes.labelcolor": _new_black,
            "xtick.color": _new_black,
            "ytick.color": _new_black,
            "patch.edgecolor": _new_black,
            "patch.force_edgecolor": False,
            "hatch.color": _new_black,
        },
    )


def tol_palette(name: str = "bright"):
    """Paul Tol qualitative palettes."""
    sets = {
        "bright": [
            "#4477AA",
            "#EE6677",
            "#228833",
            "#CCBB44",
            "#66CCEE",
            "#AA3377",
            "#BBBBBB",
            "#000000",
        ],
        "high-contrast": ["#004488", "#DDAA33", "#BB5566", "#000000"],
        "vibrant": [
            "#EE7733",
            "#0077BB",
            "#33BBEE",
            "#EE3377",
            "#CC3311",
            "#009988",
            "#BBBBBB",
            "#000000",
        ],
        "muted": [
            "#CC6677",
            "#332288",
            "#DDCC77",
            "#117733",
            "#88CCEE",
            "#882255",
            "#44AA99",
            "#999933",
            "#AA4499",
            "#DDDDDD",
            "#000000",
        ],
    }
    return sets.get(name, sets["bright"])


# ---------------- Ax helpers ----------------


def panel(figsize=(3, 2), dpi=300):
    """Create (fig, ax) at print-ready size (inches)."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    return fig, ax


def raster_scatter(ax, x, y, c=None, s=1.0, cmap=mpl.cm.viridis, **kwargs):
    """Fast, vector-friendly scatter: rasterize only the points."""
    return ax.scatter(
        x, y, s=s, c=c, cmap=cmap, edgecolor="none", rasterized=True, **kwargs
    )


def side_colorbar(fig, mappable, label: str = ""):
    """Right-side colorbar with top label."""
    gs_cbar = fig.add_gridspec(1, 1, left=0.92, right=0.95)
    cax = fig.add_subplot(gs_cbar[0, 0])
    fig.colorbar(mappable, cax=cax)
    cax.text(1, 1.01, label, transform=cax.transAxes, ha="right", va="bottom")


def annotate_ax(ax, txt, x=0.05, y=0.95, **kw):
    """Axis-relative text placement (stable across data ranges)."""
    ax.text(x, y, txt, transform=ax.transAxes, ha="left", va="top", **kw)


# ---------------- Data helpers ----------------

_ALG_MAP = {
    "directlingam_pd2": "DirectLiNGAM",
    "grasp_tetrad_pd2": "GRaSP",
    "fges_tetrad_pd2": "FGES",
    "boss_tetrad_pd2": "BOSS",
    "pc_tetrad_05": "PC",
    "dagma_tetrad_pd2": "DAGMA",
}

_MECH_MAP = {
    "sigmoid_add-binary": "Sigmoid+Binary",
    "nn": "NN",
    "nn-binary": "NN+Binary",
    "linear": "Linear",
    "polynomial": "Polynomial",
    "polynomial-binary": "Polynomial+Binary",
    "linear-binary": "Linear+Binary",
    "sigmoid_add": "Sigmoid",
    "binary": "Binary",
}

_COL_RENAME = {
    "skeleton_shd": "Skeleton SHD",
    "shd": "SHD",
    "ece": "ECE",
}


def apply_algorithm_labels(
    df: pd.DataFrame, col: str = "algorithm_name"
) -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = out[col].map(_ALG_MAP).fillna(out[col])
    return out


def apply_mechanism_labels(df: pd.DataFrame, col: str = "mechanism") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = out[col].map(_MECH_MAP).fillna(out[col])
    return out


def rename_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.rename(columns=_COL_RENAME, inplace=True)
    return out


# ---------------- Internal util ----------------


def _maybe_mkdir(fpath):
    if fpath:
        Path(fpath).parent.mkdir(parents=True, exist_ok=True)


def _nice_metric_name(metric: str) -> str:
    return (
        metric
        if metric in ("SHD", "ECE", "Skeleton SHD")
        else metric.replace("_", " ").title()
    )


# ---------------- Plotters ----------------


def plot_metric_by_nodes_overall(
    df: pd.DataFrame, metric: str, fpath: str | None = None
):
    set_pub_theme()
    _maybe_mkdir(fpath)
    mean_by_nodes = df.groupby("nodes")[metric].mean().reset_index()
    fig, ax = plt.subplots(figsize=(4, 2), dpi=300)
    bars = sns.barplot(
        data=mean_by_nodes, x="nodes", y=metric, color="steelblue", errorbar=None
    )
    for cont in bars.containers:
        bars.bar_label(cont, fmt="%.2f", padding=3)
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel(f"Average {_nice_metric_name(metric)}")
    ax.set_title(
        f"{_nice_metric_name(metric)} vs Number of Nodes (no knowledge)", pad=20
    )
    sns.despine()
    plt.tight_layout()
    if fpath:
        plt.savefig(fpath, bbox_inches="tight", dpi=300)
    plt.show()


def plot_metric_by_tier_overall(
    df: pd.DataFrame, metric: str, fpath: str | None = None
):
    set_pub_theme()
    _maybe_mkdir(fpath)
    mean_by_tier = df.groupby("num_tiers")[metric].mean().reset_index()
    fig, ax = plt.subplots(figsize=(4, 2), dpi=300)
    bars = sns.barplot(
        data=mean_by_tier, x="num_tiers", y=metric, color="steelblue", errorbar=None
    )
    for cont in bars.containers:
        bars.bar_label(cont, fmt="%.2f", padding=3)
    ax.set_xlabel("Tier")
    ax.set_ylabel(f"Average {_nice_metric_name(metric)}")
    ax.set_title(f"{_nice_metric_name(metric)} vs Number of Knowledge Tiers", pad=20)
    sns.despine()
    plt.tight_layout()
    if fpath:
        plt.savefig(fpath, bbox_inches="tight", dpi=300)
    plt.show()


def plot_metric_by_nodes_and_tier(
    df: pd.DataFrame, metric: str, fpath: str | None = None
):
    set_pub_theme()
    _maybe_mkdir(fpath)
    mean_by_nodes_tier = df.groupby(["nodes", "num_tiers"])[metric].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 3), dpi=300)
    sns.lineplot(
        data=mean_by_nodes_tier,
        x="nodes",
        y=metric,
        hue="num_tiers",
        palette="coolwarm",
        marker="o",
        linewidth=2,
        ax=ax,
    )
    ax.set_xlabel("Number of Nodes", fontsize=14)
    ax.set_ylabel(f"Average {_nice_metric_name(metric)}", fontsize=14)
    ax.set_title(
        f"{_nice_metric_name(metric)} vs Number of Nodes by Number of Knowledge Tiers",
        pad=20,
        fontsize=16,
    )
    ax.tick_params(axis="both", labelsize=12)
    plt.legend(
        title="Number\nof Tiers",
        bbox_to_anchor=(1, 1),
        loc="upper left",
        title_fontsize=12,
        fontsize=12,
    )
    sns.despine()
    plt.tight_layout()
    if fpath:
        plt.savefig(fpath, bbox_inches="tight", dpi=300)
    plt.show()


def plot_metric_by_algorithm(df: pd.DataFrame, metric: str, fpath: str | None = None):
    set_pub_theme()
    _maybe_mkdir(fpath)
    mean_by_alg = df.groupby("algorithm_name")[metric].mean().sort_values()
    fig, ax = plt.subplots(figsize=(4, 2), dpi=300)
    bars = sns.barplot(
        x=mean_by_alg.values,
        y=mean_by_alg.index,
        color="steelblue",
        orient="h",
        errorbar=None,
    )
    for cont in bars.containers:
        bars.bar_label(cont, fmt="%.2f", padding=3)
    ax.set_xlabel(f"Average {_nice_metric_name(metric)}")
    ax.set_ylabel("Algorithm")
    ax.set_title(f"{_nice_metric_name(metric)} by Algorithm (no knowledge)", pad=20)
    sns.despine()
    plt.tight_layout()
    if fpath:
        plt.savefig(fpath, bbox_inches="tight", dpi=300)
    plt.show()


def plot_metric_by_algorithm_and_nodes(
    df: pd.DataFrame, metric: str, fpath: str | None = None
):
    set_pub_theme()
    _maybe_mkdir(fpath)
    mean_by_alg_nodes = (
        df.groupby(["algorithm_name", "nodes"])[metric].mean().reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 3), dpi=300)
    bars = sns.barplot(
        data=mean_by_alg_nodes,
        x="algorithm_name",
        y=metric,
        hue="nodes",
        palette="coolwarm",
        width=0.9,
    )
    for cont in bars.containers:
        bars.bar_label(cont, fmt="%.2f", padding=3, fontsize=10)
    ax.set_xlabel("Algorithm", fontsize=14)
    ax.set_ylabel(f"Average {_nice_metric_name(metric)}", fontsize=14)
    ax.set_title(
        f"{_nice_metric_name(metric)} by Algorithm and Number of Nodes (no knowledge)",
        pad=20,
        fontsize=16,
    )
    plt.xticks(rotation=0)
    ax.tick_params(axis="both", labelsize=12)
    plt.legend(
        title="Number\nof Nodes",
        bbox_to_anchor=(1, 1),
        loc="upper left",
        title_fontsize=12,
        fontsize=12,
    )
    sns.despine()
    plt.tight_layout()
    if fpath:
        plt.savefig(fpath, bbox_inches="tight", dpi=300)
    plt.show()


def plot_metric_by_algorithm_and_tiers(
    df: pd.DataFrame, metric: str, fpath: str | None = None
):
    set_pub_theme()
    _maybe_mkdir(fpath)
    mean_by_alg_tiers = (
        df.groupby(["algorithm_name", "num_tiers"])[metric].mean().reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 3), dpi=300)
    bars = sns.barplot(
        data=mean_by_alg_tiers,
        x="algorithm_name",
        y=metric,
        hue="num_tiers",
        palette="coolwarm",
    )
    for cont in bars.containers:
        bars.bar_label(cont, fmt="%.2f", padding=3, fontsize=10)
    ax.set_xlabel("Algorithm", fontsize=14)
    ax.set_ylabel(f"Average {_nice_metric_name(metric)}", fontsize=14)
    ax.set_title(
        f"{_nice_metric_name(metric)} by Algorithm and Knowledge Tiers",
        pad=20,
        fontsize=16,
    )
    plt.xticks(rotation=0)
    ax.tick_params(axis="both", labelsize=12)
    plt.legend(
        title="Number\nof Tiers",
        bbox_to_anchor=(1, 1),
        loc="upper left",
        title_fontsize=12,
        fontsize=12,
    )
    sns.despine()
    plt.tight_layout()
    if fpath:
        plt.savefig(fpath, bbox_inches="tight", dpi=300)
    plt.show()


def plot_metric_by_mechanism(df: pd.DataFrame, metric: str, fpath: str | None = None):
    set_pub_theme()
    _maybe_mkdir(fpath)
    mean_by_mech = df.groupby("mechanism")[metric].mean().sort_values()
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    bars = sns.barplot(
        x=mean_by_mech.values,
        y=mean_by_mech.index,
        color="steelblue",
        orient="h",
        errorbar=None,
    )
    for cont in bars.containers:
        bars.bar_label(cont, fmt="%.2f", padding=3)
    ax.set_xlabel(f"Average {_nice_metric_name(metric)}")
    ax.set_ylabel("Mechanism")
    ax.set_title(f"{_nice_metric_name(metric)} by Mechanism (no knowledge)", pad=20)
    sns.despine()
    plt.tight_layout()
    if fpath:
        plt.savefig(fpath, bbox_inches="tight", dpi=300)
    plt.show()


# drop-in replacement for plot_metric_by_algorithm_and_mechanism in viz_utils.py

_MECH_ORDER = [
    "Polynomial",
    "Polynomial+Binary",
    "Sigmoid",
    "Sigmoid+Binary",
    "Linear",
    "Linear+Binary",
    "NN",
    "NN+Binary",
    "Binary",
]


def _mech_style_maps():
    base = {
        "Polynomial": "#4477AA",
        "Sigmoid": "#EE6677",
        "Linear": "#228833",
        "NN": "#CCBB44",
        "Binary": "#AA3377",
    }
    color = {
        "Polynomial": base["Polynomial"],
        "Polynomial+Binary": base["Polynomial"],
        "Sigmoid": base["Sigmoid"],
        "Sigmoid+Binary": base["Sigmoid"],
        "Linear": base["Linear"],
        "Linear+Binary": base["Linear"],
        "NN": base["NN"],
        "NN+Binary": base["NN"],
        "Binary": base["Binary"],
    }
    hatch = {k: "" for k in _MECH_ORDER}
    for k in ("Polynomial", "Sigmoid", "Linear", "NN"):
        hatch[f"{k}+Binary"] = "///"
    return color, hatch


def plot_metric_by_algorithm_and_mechanism(df, metric, fpath=None, plot_text=False):
    set_pub_theme()
    _maybe_mkdir(fpath)

    color_map, hatch_map = _mech_style_maps()

    data = df.copy()
    data["mechanism"] = pd.Categorical(
        data["mechanism"], categories=_MECH_ORDER, ordered=True
    )
    mean_by_alg_mech = (
        data.groupby(["algorithm_name", "mechanism"])[metric]
        .mean()
        .reset_index()
        .dropna(subset=["mechanism"])
    )

    hue_order = [m for m in _MECH_ORDER if m in mean_by_alg_mech["mechanism"].unique()]
    palette = [color_map[m] for m in hue_order]

    fig, ax = plt.subplots(figsize=(12, 3), dpi=300)
    bars = sns.barplot(
        data=mean_by_alg_mech,
        x="algorithm_name",
        y=metric,
        hue="mechanism",
        hue_order=hue_order,
        palette=palette,
        ax=ax,
    )

    # Apply hatches per hue container
    for cont, mech in zip(bars.containers, hue_order):
        for bar in cont:
            bar.set_hatch(hatch_map[mech])
        # ✅ Optionally add text labels
        if plot_text:
            bars.bar_label(cont, fmt="%.2f", padding=3, fontsize=8)

    # Legend with hatches
    handles, labels = ax.get_legend_handles_labels()
    for h, lab in zip(handles, labels):
        h.set_hatch(hatch_map.get(lab, ""))
        h.set_edgecolor("black")
    ax.legend(
        handles,
        labels,
        title="Mechanism",
        bbox_to_anchor=(1, 1),
        loc="upper left",
        title_fontsize=12,
        fontsize=10,
    )

    ax.set_xlabel("Algorithm", fontsize=14)
    ax.set_ylabel(f"Average {_nice_metric_name(metric)}", fontsize=14)
    ax.set_title(
        f"{_nice_metric_name(metric)} by Algorithm and Mechanism (no knowledge)",
        pad=20,
        fontsize=16,
    )
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", alpha=0.7, linestyle="--")
    sns.despine()
    plt.tight_layout()

    if fpath:
        plt.savefig(fpath, bbox_inches="tight", dpi=300)
    plt.show()


def plot_metric_by_tiers_and_nodes(
    df: pd.DataFrame, metric: str, fpath: str | None = None
):
    set_pub_theme()
    _maybe_mkdir(fpath)
    mean_by_tn = df.groupby(["num_tiers", "nodes"])[metric].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 3), dpi=300)
    bars = sns.barplot(
        data=mean_by_tn,
        x="nodes",
        y=metric,
        hue="num_tiers",
        errorbar=None,
        palette="coolwarm",
        width=0.9,
    )
    for cont in bars.containers:
        bars.bar_label(cont, fmt="%.2f", padding=3, fontsize=10)
    ax.set_xlabel("Number of Knowledge Tiers", fontsize=14)
    ax.set_ylabel(f"Average {_nice_metric_name(metric)}", fontsize=14)
    ax.set_title(
        f"{_nice_metric_name(metric)} vs Number of Knowledge Tiers by Number of Nodes",
        pad=20,
        fontsize=16,
    )
    ax.tick_params(axis="both", labelsize=12)
    plt.legend(
        title="Number\nof Nodes",
        bbox_to_anchor=(1, 1),
        loc="upper left",
        title_fontsize=12,
        fontsize=12,
    )
    sns.despine()
    plt.tight_layout()
    if fpath:
        plt.savefig(fpath, bbox_inches="tight", dpi=300)
    plt.show()


# ---------------- Summary tables ----------------


def summary_mechanism_f1(df: pd.DataFrame) -> pd.DataFrame:
    """Average F1 and Skeleton F1 by mechanism (no knowledge)."""
    out = (
        df[df["num_tiers"] == 1]
        .groupby("mechanism")
        .agg(f1=("f1", "mean"), skeleton_f1=("skeleton_f1", "mean"))
        .round(3)
    )
    out.columns = ["Average F1", "Average Skeleton F1"]
    return out


def summary_algorithm_f1(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df[df["num_tiers"] == 1]
        .groupby("algorithm_name")
        .agg(f1=("f1", "mean"), skeleton_f1=("skeleton_f1", "mean"))
        .round(3)
    )
    out.columns = ["Average F1", "Average Skeleton F1"]
    return out


def summary_median_ece(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df[df["num_tiers"] == 1]
        .groupby("algorithm_name")
        .agg(median_frequency=("median_frequency", "mean"), ECE=("ECE", "mean"))
        .round(3)
    )
    out.columns = ["Median Frequency", "ECE"]
    return out


def pivot_f1_by_alg_nodes(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["num_tiers"] == 1]
        .groupby(["algorithm_name", "nodes"])["f1"]
        .mean()
        .unstack()
        .round(3)
    )


def pivot_prec_by_alg_nodes(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["num_tiers"] == 1]
        .groupby(["algorithm_name", "nodes"])["precision"]
        .mean()
        .unstack()
        .round(3)
    )


def pivot_recall_by_alg_nodes(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["num_tiers"] == 1]
        .groupby(["algorithm_name", "nodes"])["recall"]
        .mean()
        .unstack()
        .round(3)
    )


def pivot_skel_f1_by_alg_nodes(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["num_tiers"] == 1]
        .groupby(["algorithm_name", "nodes"])["skeleton_f1"]
        .mean()
        .unstack()
        .round(3)
    )


def pivot_f1_by_alg_tiers(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["algorithm_name", "num_tiers"])["f1"].mean().unstack().round(3)


def pivot_skel_f1_by_alg_tiers(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["algorithm_name", "num_tiers"])["skeleton_f1"]
        .mean()
        .unstack()
        .round(3)
    )


def pivot_f1_by_alg_mech(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["num_tiers"] == 1]
        .groupby(["algorithm_name", "mechanism"])["f1"]
        .mean()
        .unstack()
        .round(3)
    )
