"""
evaluate.py - FINAL evaluation on the held-out TEST set (Stage 8).

The test set is the only data that has never influenced any decision. It is used
here for the first and only time, on the models that were already fixed in Stage 7:

    * MLP        the 5 saved "full" checkpoints (one per seed), tuned on validation data
    * Baselines  logistic regression and gradient boosting, trained on the train split

LOCK: after the first evaluation a fingerprint of the models is stored. If the models
(or the chosen configuration) change afterwards, the evaluation refuses to run again,
because re-tuning after looking at the test score would make the test score meaningless.

Choices that must not depend on the test set (the decision threshold, permutation
importance) use VALIDATION data only.
"""
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

import config
import eda
import train
import tune
from metrics import binary_metrics

AQUA = "#1baf7a"                                   # third validated categorical colour
METRICS = ["accuracy", "auc", "loss", "precision", "recall", "f1"]
N_BOOT = 1000                                      # bootstrap resamples of the test set


# ------------------------------------------------------------------------------ lock
def checkpoint_paths():
    return [config.MODEL_DIR / f"final_full_seed{s}.pt" for s in config.SEEDS]


def fingerprint() -> str:
    h = hashlib.sha256()
    for p in checkpoint_paths() + [tune.best_json()]:
        h.update(p.read_bytes())
    return h.hexdigest()


def check_lock(force: bool):
    lock = config.RESULTS_DIR / "test_lock.json"
    if not lock.exists():
        return
    if json.loads(lock.read_text())["fingerprint"] == fingerprint():
        print("Note: the test set was already evaluated with exactly these models. "
              "Re-running gives the same numbers.\n")
        return
    if not force:
        raise SystemExit(
            "\nSTOP: the models changed AFTER the test set was evaluated.\n"
            "Using the test set again to judge new models would invalidate it.\n"
            "If you really want this (and will say so in the report), add --force.")
    print("WARNING: --force used; the test set has now influenced model choice.\n")


# ------------------------------------------------------------------------- predictions
def mlp_probabilities(arrays):
    """Per-seed validation and test probabilities of the saved MLP checkpoints."""
    missing = [p.name for p in checkpoint_paths() if not p.exists()]
    if missing:
        raise SystemExit(f"\nMissing saved models: {missing}\n"
                         "Run the ablation step (Step 23) first - it saves them.")
    val, test = [], []
    for p in checkpoint_paths():
        model, _ = train.load_checkpoint(p)
        val.append(train.predict_proba(model, arrays["X_val"]))
        test.append(train.predict_proba(model, arrays["X_test"]))
    return val, test


def baseline_probabilities(arrays):
    """Test probabilities of the baselines (trained on the train split only)."""
    Xtr, ytr, Xte = arrays["X_train"], arrays["y_train"], arrays["X_test"]
    lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    hgb = [HistGradientBoostingClassifier(random_state=s).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
           for s in config.SEEDS]
    return lr, hgb


# ------------------------------------------------------------------------ result tables
def results_table(y, mlp, hgb, lr):
    """One row per model; '+-' rows are mean and std over the 5 seeds."""
    rows = []

    def seeds_row(name, plist):
        ms = pd.DataFrame([binary_metrics(y, p) for p in plist])
        r = {"model": name}
        for m in METRICS:
            r[m] = ms[m].mean()
            r[m + "_std"] = ms[m].std(ddof=1)
        rows.append(r)

    def single_row(name, p):
        rows.append({"model": name, **binary_metrics(y, p)})

    seeds_row("MLP - single model (mean over 5 seeds)", mlp)
    single_row("MLP - 5-seed ensemble", np.mean(mlp, axis=0))
    seeds_row("Gradient boosting - single model (mean over 5 seeds)", hgb)
    single_row("Gradient boosting - 5-seed ensemble", np.mean(hgb, axis=0))
    single_row("Logistic regression", lr)
    return pd.DataFrame(rows)


def bootstrap_cis(y, probs: dict, n_boot=N_BOOT, seed=0):
    """95% bootstrap CIs over test patients, plus paired differences vs the MLP."""
    rng = np.random.default_rng(seed)
    n = len(y)
    names = list(probs)
    store = {k: {"auc": [], "accuracy": []} for k in names}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        for k in names:
            store[k]["auc"].append(roc_auc_score(y[i], probs[k][i]))
            store[k]["accuracy"].append(((probs[k][i] >= 0.5) == (y[i] == 1)).mean())
    ci = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))
    rows = [dict(model=k, metric=m, low=ci(store[k][m])[0], high=ci(store[k][m])[1])
            for k in names for m in ("auc", "accuracy")]
    base = names[0]                                   # the MLP ensemble
    for k in names[1:]:
        for m in ("auc", "accuracy"):
            d = np.array(store[base][m]) - np.array(store[k][m])
            rows.append(dict(model=f"{base} minus {k}", metric=m + " difference",
                             low=np.percentile(d, 2.5), high=np.percentile(d, 97.5),
                             mean=d.mean()))
    return pd.DataFrame(rows)


def threshold_table(y_val, p_val, y_test, p_test):
    """Thresholds are chosen on VALIDATION data, then applied once to the test data."""
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    t_f1 = thr[np.argmax(f1)]
    ok = np.where(rec[:-1] >= 0.80)[0]
    t_rec = thr[ok[-1]] if len(ok) else thr[0]        # highest threshold with recall >= 0.80
    rows = []
    for name, t in (("default 0.5", 0.5), ("max F1 on validation", t_f1),
                    ("recall >= 0.80 on validation", t_rec)):
        pred = (p_test >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        rows.append(dict(rule=name, threshold=float(t),
                         accuracy=(tp + tn) / len(y_test), recall_sensitivity=tp / (tp + fn),
                         specificity=tn / (tn + fp), precision=tp / max(tp + fp, 1),
                         false_negatives=int(fn), false_positives=int(fp)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- permutation importance
def feature_groups(names):
    """One-hot columns of one variable are permuted together as a single group."""
    groups = {}
    for i, n in enumerate(names):
        key = n.rsplit("_", 1)[0] if n.startswith(("cholesterol_", "gluc_")) else n
        groups.setdefault(key, []).append(i)
    return groups


def permutation_importance(X, y, models, names, n_repeats=10, seed=0):
    """Drop in validation ROC-AUC when one variable is shuffled (ensemble of the MLPs)."""
    rng = np.random.default_rng(seed)
    predict = lambda A: np.mean([train.predict_proba(m, A) for m in models], axis=0)
    base = roc_auc_score(y, predict(X))
    rows = []
    for g, cols in feature_groups(names).items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, cols] = X[rng.permutation(len(X))][:, cols]
            drops.append(base - roc_auc_score(y, predict(Xp)))
        rows.append(dict(feature=g, auc_drop_mean=np.mean(drops), auc_drop_std=np.std(drops)))
    return pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=False)


# ------------------------------------------------------------------------------ figures
def _save(fig, name, show=False):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIG_DIR / name, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_roc_pr(y, probs: dict, show=False):
    colors = dict(zip(probs, (eda.BLUE, eda.ORANGE, AQUA)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for k, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, color=colors[k], lw=2, label=f"{k} (AUC {roc_auc_score(y, p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        keep = rc[:-1] >= 0.01        # skip the first few top-ranked patients: precision there
        pr, rc = pr[:-1][keep], rc[:-1][keep]   # swings between 0 and 1 on a handful of cases
        axes[1].plot(rc, pr, color=colors[k], lw=2,
                     label=f"{k} (AP {average_precision_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], color=eda.NEUTRAL, ls="--", lw=1)
    axes[1].axhline(y.mean(), color=eda.NEUTRAL, ls="--", lw=1)
    axes[0].set(xlabel="false positive rate (1 - specificity)", ylabel="true positive rate (recall)",
                title="ROC curve - test set")
    axes[1].set(xlabel="recall", ylabel="precision", title="Precision-recall curve - test set")
    axes[1].set_ylim(0.4, 1.0)
    axes[1].set_xlim(0, 1.0)
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "upper right", fontsize=8.5)
    fig.tight_layout()
    _save(fig, "08_roc_pr_test.png", show)


def plot_confusion(y, p, show=False):
    cm = confusion_matrix(y, (p >= 0.5).astype(int))
    pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{pct[i, j]:.1%} of true class", ha="center",
                    va="center", color="white" if pct[i, j] > 0.5 else eda.INK, fontsize=10)
    ax.set_xticks([0, 1], ["no disease", "disease"])
    ax.set_yticks([0, 1], ["no disease", "disease"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    ax.set_title("Confusion matrix - MLP ensemble, test set (threshold 0.5)")
    fig.tight_layout()
    _save(fig, "09_confusion_matrix_test.png", show)


def plot_calibration(y, probs: dict, show=False):
    colors = dict(zip(probs, (eda.BLUE, eda.ORANGE, AQUA)))
    fig, ax = plt.subplots(figsize=(5.8, 5))
    ax.plot([0, 1], [0, 1], color=eda.NEUTRAL, ls="--", lw=1)
    for k, p in probs.items():
        frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, color=colors[k], lw=2, marker="o", ms=4, label=k)
    ax.set(xlabel="predicted probability", ylabel="observed share with disease",
           title="Calibration - test set (closer to the dashed line is better)")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    _save(fig, "10_calibration_test.png", show)


def plot_importance(imp: pd.DataFrame, show=False):
    imp = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.barh(imp["feature"], imp["auc_drop_mean"], xerr=imp["auc_drop_std"], color=eda.BLUE,
            height=0.65, error_kw=dict(ecolor=eda.MUTED, lw=1, capsize=2))
    ax.set_xlabel("drop in validation ROC-AUC when the variable is shuffled")
    ax.set_title("Permutation importance - MLP ensemble, validation set")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "11_permutation_importance.png", show)


# ---------------------------------------------------------------------------------- main
def run_test_evaluation(arrays: dict, force: bool = False, show: bool = False):
    eda._style()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    check_lock(force)
    y, yv = arrays["y_test"].astype(int), arrays["y_val"].astype(int)

    mlp_val, mlp_test = mlp_probabilities(arrays)
    lr, hgb = baseline_probabilities(arrays)
    ens = {"MLP (5-seed ensemble)": np.mean(mlp_test, axis=0),
           "Gradient boosting (5-seed ensemble)": np.mean(hgb, axis=0),
           "Logistic regression": lr}

    table = results_table(y, mlp_test, hgb, lr)
    table.to_csv(config.RESULTS_DIR / "test_results.csv", index=False)
    cis = bootstrap_cis(y, ens)
    cis.to_csv(config.RESULTS_DIR / "test_confidence_intervals.csv", index=False)
    thr = threshold_table(yv, np.mean(mlp_val, axis=0), y, ens["MLP (5-seed ensemble)"])
    thr.to_csv(config.RESULTS_DIR / "test_thresholds.csv", index=False)

    models = [train.load_checkpoint(p)[0] for p in checkpoint_paths()]
    imp = permutation_importance(arrays["X_val"], yv, models, arrays["feature_names"])
    imp.to_csv(config.RESULTS_DIR / "permutation_importance.csv", index=False)

    plot_roc_pr(y, ens, show)
    plot_confusion(y, ens["MLP (5-seed ensemble)"], show)
    plot_calibration(y, ens, show)
    plot_importance(imp, show)

    (config.RESULTS_DIR / "test_lock.json").write_text(json.dumps(
        {"fingerprint": fingerprint(), "note": "test set evaluated once with these models"}))
    return dict(table=table, cis=cis, thresholds=thr, importance=imp)
