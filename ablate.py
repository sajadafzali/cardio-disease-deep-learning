"""
ablate.py - ablation study with repeated seeds (Stage 7).

Starting from the tuned configuration ("full"), switch off ONE ingredient at a time:

    full        the tuned model
    no_dropout  dropout = 0
    no_bn       no BatchNorm layers
    no_wd       weight decay = 0   (or 'with_wd': weight decay 1e-4, if the tuned model has none)
    shallow     one hidden layer fewer
    deep        one hidden layer more

Each variant is trained with every seed in config.SEEDS and scored on the VALIDATION
set, so the result is a mean +/- std rather than one lucky or unlucky run.
The "full" model of every seed is saved as a checkpoint for the final test (Stage 8).
Progress is saved after each run, so an interrupted run resumes.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import eda
import train
from tune import load_best

def runs_csv():
    return config.RESULTS_DIR / "ablation_runs.csv"


def summary_csv():
    return config.RESULTS_DIR / "ablation_summary.csv"


ORDER = ["full", "no_dropout", "no_bn", "no_wd", "with_wd", "shallow", "deep"]


def make_variants(best: dict) -> dict:
    """Return {name: (model_cfg, train_cfg)} built from the tuned config."""
    h = tuple(best["hidden"])
    base_m = dict(hidden=h, dropout=best["dropout"], use_bn=best.get("use_bn", True))
    base_t = {k: best[k] for k in ("lr", "weight_decay", "batch_size")}
    return {
        "full": (base_m, base_t),
        "no_dropout": ({**base_m, "dropout": 0.0}, base_t),
        "no_bn": ({**base_m, "use_bn": False}, base_t),
        # weight decay: if the tuned model uses it, remove it; if not, add it
        **({"no_wd": (base_m, {**base_t, "weight_decay": 0.0})} if base_t["weight_decay"] > 0
           else {"with_wd": (base_m, {**base_t, "weight_decay": 1e-4})}),
        "shallow": ({**base_m, "hidden": h[:-1]}, base_t),
        "deep": ({**base_m, "hidden": h + (max(h[-1] // 2, 8),)}, base_t),
    }


def run_ablation(arrays: dict, seeds=config.SEEDS):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    variants = make_variants(load_best())
    done = pd.read_csv(runs_csv()) if runs_csv().exists() else pd.DataFrame()
    finished = set(zip(done["variant"], done["seed"])) if len(done) else set()
    total = len(variants) * len(seeds)

    n = 0
    for name, (mcfg, tcfg) in variants.items():
        for s in seeds:
            n += 1
            if (name, s) in finished:
                continue
            save = config.MODEL_DIR / f"final_full_seed{s}.pt" if name == "full" else None
            r = train.fit_and_score(arrays, mcfg, tcfg, seed=s, save_path=save)
            row = dict(variant=name, seed=s, **r)
            pd.DataFrame([row]).to_csv(runs_csv(), mode="a",
                                       header=not runs_csv().exists(), index=False)
            print(f"run {n:2d}/{total} | {name:10s} seed {s} | val AUC {row['val_auc']:.4f} "
                  f"acc {row['val_accuracy']:.4f} loss {row['val_loss']:.4f} "
                  f"epoch {row['best_epoch']} ({row['seconds']:.0f}s)", flush=True)
    return summarise()


def summarise() -> pd.DataFrame:
    """Mean and std over seeds per variant, plus the AUC change relative to 'full'."""
    runs = pd.read_csv(runs_csv())
    cols = ["val_auc", "val_accuracy", "val_loss", "val_recall", "val_precision",
            "best_epoch", "params"]
    g = runs.groupby("variant")[cols]
    summ = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std"))
    summ["n_seeds"] = runs.groupby("variant").size()
    summ = summ.reindex([v for v in ORDER if v in summ.index])
    summ["auc_change_vs_full"] = summ["val_auc_mean"] - summ.loc["full", "val_auc_mean"]
    summ.to_csv(summary_csv())
    return summ


def plot_comparison(path=None, show=False):
    """Validation ROC-AUC per model: each dot is one seed, bar = mean +/- 1 std."""
    runs = pd.read_csv(runs_csv())
    base = pd.read_csv(config.RESULTS_DIR / "baselines.csv")
    base["variant"] = base["model"]
    groups = [("MLP: " + v, runs.loc[runs.variant == v, "val_auc"].to_numpy(), True)
              for v in ORDER if v in set(runs.variant)]
    groups += [(m, base.loc[base.model == m, "auc"].to_numpy(), False)
               for m in base["model"].unique()]
    eda._style()
    BLUE, GREY = eda.BLUE, eda.NEUTRAL
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for y, (name, vals, is_mlp) in enumerate(groups[::-1]):
        c = BLUE if is_mlp else GREY
        ax.scatter(vals, np.full(len(vals), y), color=c, alpha=0.45, s=28, zorder=3)
        m, s = vals.mean(), (vals.std(ddof=1) if len(vals) > 1 else 0)
        ax.errorbar(m, y, xerr=s, color=c, capsize=4, lw=2, marker="o", ms=7, zorder=4,
                    mec="white")
        ax.text(max(vals.max(), m + s) + 0.0004, y, f"{m:.4f}", va="center", fontsize=8.5)
    ax.set_yticks(range(len(groups)), [g[0] for g in groups[::-1]])
    ax.set_xlabel("validation ROC-AUC (higher is better).  Note: this axis spans only "
                  f"{ax.get_xlim()[1] - ax.get_xlim()[0]:.3f} AUC - the differences are small.")
    ax.grid(axis="y", visible=False)
    ax.set_title("Model comparison on the validation set (mean +/- 1 std over seeds)")
    ax.margins(x=0.08)
    fig.tight_layout()
    if path is not None:
        config.FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
