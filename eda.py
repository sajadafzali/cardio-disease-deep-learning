"""
eda.py - exploratory data analysis figures.

Two kinds of plots, deliberately separated:

  * DATA-QUALITY plots use the RAW data but never look at the target, so they
    cannot leak label information into design decisions.
  * RELATIONSHIP plots (feature vs. target, correlations) use the TRAINING split
    only. Validation and test data are not inspected during exploration.

Colour convention (same in every figure): blue = no disease, orange = disease.
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import config

BLUE, ORANGE = "#2a78d6", "#eb6834"      # class 0 / class 1 (validated adjacent pair)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MIN_BIN_N = 100                          # smallest group shown in rate plots
NEUTRAL = "#8a8985"                      # target-free data-quality plots


def _style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
        "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "xtick.color": MUTED,
        "ytick.color": MUTED, "text.color": INK, "font.size": 9,
        "legend.frameon": False,
    })


def _finish(fig, name, show):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIG_DIR / name, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------- 1. data quality (raw)
def plot_raw_quality(raw: pd.DataFrame, report: pd.DataFrame, show=False):
    """Histograms (log count axis) of raw values with the cleaning limits drawn in."""
    df = raw.copy()
    df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
    cols = ["ap_hi", "ap_lo", "height", "weight", "bmi"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, c in zip(axes.flat, cols):
        lo, hi = config.LIMITS[c]
        # clip only for drawing, so extreme values appear in the end bins
        vals = df[c].clip(lower=df[c].quantile(0.0005), upper=df[c].quantile(0.9995))
        ax.hist(vals, bins=80, color=NEUTRAL)
        ax.set_yscale("log")
        for b in (lo, hi):
            ax.axvline(b, color=INK, ls="--", lw=1)
        ax.set_title(f"{c}: limits {lo}-{hi}")
        ax.set_xlabel(c)
        ax.set_ylabel("count (log)")
    # last panel: rows flagged per rule
    ax = axes.flat[5]
    r = report.iloc[1:-2].iloc[::-1]  # per-rule rows, skipping duplicates/total/remaining
    ax.barh(r["rule"], r["rows"], color=NEUTRAL, height=0.6)
    for y, v in enumerate(r["rows"]):
        ax.text(v, y, f" {v:,}", va="center", color=INK, fontsize=8)
    ax.set_title("Rows flagged by each rule")
    ax.set_xlabel("rows (rules overlap)")
    ax.grid(axis="y", visible=False)
    fig.suptitle("Raw data quality (target not used)", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    _finish(fig, "01_raw_data_quality.png", show)


# ----------------------------------------------------- 2. class balance & categorical rates
def plot_categorical_rates(train: pd.DataFrame, show=False):
    """Disease rate per category (train only). Dashed line = overall rate."""
    t = config.TARGET
    overall = train[t].mean()
    cats = ["cholesterol", "gluc", "smoke", "alco", "active", "gender"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, c in zip(axes.flat, cats):
        g = train.groupby(c)[t].agg(["mean", "size"])
        x = np.arange(len(g))
        ax.bar(x, g["mean"], color=ORANGE, width=0.55)
        ax.axhline(overall, color=INK, ls="--", lw=1)
        for xi, (m, n) in enumerate(zip(g["mean"], g["size"])):
            ax.text(xi, m + 0.02, f"{m:.0%}\nn={n:,}", ha="center", va="bottom", zorder=5,
                    bbox=dict(fc="white", ec="none", pad=1.5),
                    fontsize=8, color=INK)
        ax.set_xticks(x, [str(i) for i in g.index])
        ax.set_ylim(0, 1)
        ax.set_title(c)
        ax.set_ylabel("disease rate")
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"Disease rate by category - training split "
                 f"(dashed = overall {overall:.1%})", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    _finish(fig, "02_categorical_disease_rates.png", show)


# ------------------------------------------------------ 3. numeric features by class
def plot_numeric_by_class(train: pd.DataFrame, show=False):
    """Overlaid, density-normalised histograms per class (train only)."""
    t = config.TARGET
    cols = config.CONTINUOUS
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for ax, c in zip(axes.flat, cols):
        lo, hi = train[c].quantile([0.005, 0.995])
        bins = np.linspace(lo, hi, 40)
        for cls, color in ((0, BLUE), (1, ORANGE)):
            ax.hist(train.loc[train[t] == cls, c].clip(lo, hi), bins=bins,
                    density=True, alpha=0.55, color=color, edgecolor="white", lw=0.3)
        ax.set_title(c)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    leg = axes.flat[-1]
    leg.axis("off")
    from matplotlib.patches import Patch
    leg.legend(handles=[Patch(color=BLUE, alpha=0.6, label="no disease (0)"),
                        Patch(color=ORANGE, alpha=0.6, label="disease (1)")],
               loc="center", fontsize=10)
    fig.suptitle("Continuous features by class - training split "
                 "(density; 0.5-99.5 percentile)", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    _finish(fig, "03_numeric_by_class.png", show)


# --------------------------------------------------------------------- 4. correlations
def plot_correlation(train: pd.DataFrame, show=False):
    """Pearson correlation heatmap (train only), diverging blue-grey-orange."""
    cols = config.CONTINUOUS + list(config.ONEHOT) + config.BINARY + [config.TARGET]
    corr = train[cols].corr()
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#f1f0ed", ORANGE])
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color=INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Correlation matrix - training split")
    fig.tight_layout()
    _finish(fig, "04_correlation.png", show)


# ---------------------------------------------------------------- 5. age and BMI trends
def plot_age_bmi_trends(train: pd.DataFrame, show=False):
    """Disease rate across 5-year age bands and WHO-style BMI classes."""
    t = config.TARGET
    overall = train[t].mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    age_bins = np.arange(30, 71, 5)
    age_lab = [f"{a}-{a + 4}" for a in age_bins[:-1]]
    g = train.groupby(pd.cut(train["age_years"], age_bins, labels=age_lab, right=False),
                      observed=True)[t].agg(["mean", "size"])
    g = g[g["size"] >= MIN_BIN_N]          # a rate from a handful of patients misleads
    bmi_bins = [0, 18.5, 25, 30, 35, 100]
    bmi_lab = ["<18.5", "18.5-25", "25-30", "30-35", ">=35"]
    h = train.groupby(pd.cut(train["bmi"], bmi_bins, labels=bmi_lab, right=False),
                      observed=True)[t].agg(["mean", "size"])
    h = h[h["size"] >= MIN_BIN_N]

    for ax, grp, title, xl in ((axes[0], g, "Disease rate by age band", "age (years)"),
                               (axes[1], h, "Disease rate by BMI class", "BMI (kg/m^2)")):
        x = np.arange(len(grp))
        ax.bar(x, grp["mean"], color=ORANGE, width=0.6)
        ax.axhline(overall, color=INK, ls="--", lw=1)
        for xi, (m, n) in enumerate(zip(grp["mean"], grp["size"])):
            ax.text(xi, m + 0.02, f"{m:.0%}", ha="center", va="bottom", fontsize=9,
                    zorder=5, bbox=dict(fc="white", ec="none", pad=1.5))
            ax.text(xi, 0.02, f"n={n:,}", ha="center", fontsize=7, color="white",
                    rotation=90, va="bottom")
        ax.set_xticks(x, grp.index.astype(str))
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.set_ylabel("disease rate")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _finish(fig, "05_age_bmi_trends.png", show)


# ---------------------------------------------------------------------- summary tables
def save_summary_tables(train: pd.DataFrame):
    config.PROC_DIR.mkdir(parents=True, exist_ok=True)
    cols = config.CONTINUOUS
    summ = train.groupby(config.TARGET)[cols].agg(["mean", "std"]).round(2).T
    summ.to_csv(config.PROC_DIR / "train_summary_by_class.csv")
    return summ


def run_eda(raw, report, train, show=False):
    """Create all EDA figures and summary tables."""
    _style()
    plot_raw_quality(raw, report, show)
    plot_categorical_rates(train, show)
    plot_numeric_by_class(train, show)
    plot_correlation(train, show)
    plot_age_bmi_trends(train, show)
    return save_summary_tables(train)
