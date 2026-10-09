"""
analyze.py - error analysis of the FINAL models on the test set (Stage 9).

This is DESCRIPTIVE only: it looks at where the already-fixed models are wrong, so the
report can discuss strengths and limits honestly. Nothing here changes a model, a
threshold or a setting, and nothing is tuned on these results (the Stage-8 lock stays valid).

Questions answered
    1. Is the error rate the same for every kind of patient (age, sex, BMI, blood pressure ...)?
    2. Are the errors concentrated among patients the model is unsure about?
    3. Do the MLP and gradient boosting make the SAME mistakes? (If yes, the limit is the
       data, not the model.)
    4. What do false negatives / false positives look like on average?
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import eda

THRESHOLD = 0.5


# ----------------------------------------------------------------------------- helpers
def wilson(k, n, z=1.96):
    """95% Wilson confidence interval for a proportion k/n (safe for small n)."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Readable subgroup labels built from the (unscaled) cleaned test table."""
    g = pd.DataFrame(index=df.index)
    g["age"] = pd.cut(df["age_years"], [0, 45, 50, 55, 60, 120], right=False,
                      labels=["<45", "45-49", "50-54", "55-59", "60+"])
    g["sex"] = df["gender"].map({0: "women (code 0)", 1: "men (code 1)"})
    g["BMI"] = pd.cut(df["bmi"], [0, 18.5, 25, 30, 100], right=False,
                      labels=["underweight <18.5", "normal 18.5-25", "overweight 25-30",
                              "obese 30+"])
    high = (df["ap_hi"] >= 140) | (df["ap_lo"] >= 90)
    normal = (df["ap_hi"] < 120) & (df["ap_lo"] < 80)
    g["blood pressure"] = np.where(high, "high (>=140 or >=90)",
                                   np.where(normal, "normal (<120 and <80)", "in between"))
    g["cholesterol"] = df["cholesterol"].map({1: "normal", 2: "above normal",
                                              3: "well above normal"})
    g["physically active"] = df["active"].map({0: "no", 1: "yes"})
    g["smoker"] = df["smoke"].map({0: "no", 1: "yes"})
    return g


# ----------------------------------------------------------------------------- tables
def subgroup_table(df, y, p, threshold=THRESHOLD) -> pd.DataFrame:
    pred = (p >= threshold).astype(int)
    groups = add_groups(df)
    rows = []
    for var in groups.columns:
        for level in groups[var].dropna().unique():
            m = (groups[var] == level).to_numpy()
            n = int(m.sum())
            correct = int((pred[m] == y[m]).sum())
            pos, neg = m & (y == 1), m & (y == 0)
            lo, hi = wilson(correct, n)
            rows.append(dict(
                variable=var, group=str(level), n=n,
                disease_rate=y[m].mean(), accuracy=correct / n, acc_low=lo, acc_high=hi,
                recall=(pred[pos] == 1).mean() if pos.sum() else np.nan,
                specificity=(pred[neg] == 0).mean() if neg.sum() else np.nan))
    order = {v: i for i, v in enumerate(groups.columns)}
    out = pd.DataFrame(rows)
    out["_o"] = out["variable"].map(order)
    return out.sort_values(["_o", "group"], kind="stable").drop(columns="_o") \
              .reset_index(drop=True)


def confidence_table(y, p) -> pd.DataFrame:
    """Accuracy by how sure the model is (distance of the probability from 0.5)."""
    edges = [0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0001]
    labels = ["0.00-0.20", "0.20-0.35", "0.35-0.50", "0.50-0.65", "0.65-0.80", "0.80-1.00"]
    b = pd.cut(p, edges, right=False, labels=labels)
    pred = (p >= THRESHOLD).astype(int)
    rows = []
    for lab in labels:
        m = (b == lab)
        n = int(m.sum())
        rows.append(dict(predicted_probability=lab, n=n, share=n / len(y),
                         observed_disease_rate=y[m].mean() if n else np.nan,
                         accuracy=(pred[m] == y[m]).mean() if n else np.nan))
    return pd.DataFrame(rows)


def uncertain_summary(y, p, lo=0.35, hi=0.65) -> dict:
    pred = (p >= THRESHOLD).astype(int)
    unc = (p >= lo) & (p < hi)
    err = pred != y
    return dict(band=f"{lo}-{hi}", share_of_patients=unc.mean(),
                accuracy_inside=(~err[unc]).mean(), accuracy_outside=(~err[~unc]).mean(),
                share_of_all_errors_inside=err[unc].sum() / err.sum())


def agreement_table(y, p_mlp, p_hgb) -> pd.DataFrame:
    a = (p_mlp >= THRESHOLD).astype(int) == y
    b = (p_hgb >= THRESHOLD).astype(int) == y
    n = len(y)
    return pd.DataFrame([
        dict(case="both correct", n=int((a & b).sum()), share=(a & b).mean()),
        dict(case="only MLP correct", n=int((a & ~b).sum()), share=(a & ~b).mean()),
        dict(case="only gradient boosting correct", n=int((~a & b).sum()), share=(~a & b).mean()),
        dict(case="both wrong", n=int((~a & ~b).sum()), share=(~a & ~b).mean()),
    ]).assign(share_of_MLP_errors_also_wrong_in_GB=float((~a & ~b).sum() / max((~a).sum(), 1)),
              total=n)


def profile_table(df, y, p) -> pd.DataFrame:
    """Average characteristics of the four outcome types (MLP ensemble, threshold 0.5)."""
    pred = (p >= THRESHOLD).astype(int)
    kind = np.select([(y == 1) & (pred == 1), (y == 0) & (pred == 0),
                      (y == 0) & (pred == 1), (y == 1) & (pred == 0)],
                     ["true positive", "true negative", "false positive", "false negative"],
                     default="")
    cols = ["age_years", "ap_hi", "ap_lo", "bmi", "cholesterol", "active", "smoke"]
    t = df[cols].groupby(kind).mean()
    t.insert(0, "n", pd.Series(kind).value_counts())
    return t.loc[["true negative", "false positive", "false negative", "true positive"]]


# ----------------------------------------------------------------------------- figures
def _save(fig, name, show=False):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIG_DIR / name, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_subgroups(tab: pd.DataFrame, overall_acc: float, show=False):
    tab = tab[tab["n"] >= 100].reset_index(drop=True)   # tiny groups are not shown
    fig, ax = plt.subplots(figsize=(8.6, 0.30 * len(tab) + 1.6))
    ypos, labels, last = [], [], None
    y = 0
    for _, r in tab.iterrows():
        if last is not None and r["variable"] != last:
            y += 0.7                                    # gap between variables
        last = r["variable"]
        ax.errorbar(r["accuracy"], -y, xerr=[[r["accuracy"] - r["acc_low"]],
                    [r["acc_high"] - r["accuracy"]]], color=eda.BLUE, marker="o", ms=5,
                    lw=1.6, capsize=2.5, mec="white", zorder=3)
        ax.text(1.005, -y, f"n={r['n']:,}", va="center", fontsize=8, color=eda.MUTED,
                transform=ax.get_yaxis_transform())
        ypos.append(-y)
        labels.append(f"{r['variable']}: {r['group']}")
        y += 1
    ax.axvline(overall_acc, color=eda.NEUTRAL, ls="--", lw=1)
    ax.set_yticks(ypos, labels)
    ax.set_xlabel(f"accuracy on the test set, with 95% interval (dashed line = overall {overall_acc:.3f})\n"
                  "Groups differ in how common the disease is, so read this together with the "
                  "recall / specificity columns of the table.")
    ax.set_title("Accuracy by patient group - MLP ensemble (test set, threshold 0.5)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "12_error_by_group.png", show)


def plot_confidence(y, p, conf: pd.DataFrame, show=False):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    bins = np.linspace(0, 1, 41)
    axes[0].hist(p[y == 0], bins=bins, color=eda.BLUE, alpha=0.6, label="no disease (true)")
    axes[0].hist(p[y == 1], bins=bins, color=eda.ORANGE, alpha=0.6, label="disease (true)")
    axes[0].axvline(THRESHOLD, color=eda.NEUTRAL, ls="--", lw=1)
    axes[0].set(xlabel="predicted probability of disease", ylabel="patients",
                title="The two groups overlap heavily")
    axes[0].legend(fontsize=8.5)
    x = np.arange(len(conf))
    axes[1].bar(x, conf["accuracy"], color=eda.BLUE, width=0.65)
    for i, r in conf.iterrows():
        axes[1].text(i, r["accuracy"] + 0.012, f"{r['accuracy']:.0%}\n({r['share']:.0%} of pts)",
                     ha="center", fontsize=8,
                     bbox=dict(facecolor="white", edgecolor="none", pad=1))
    axes[1].set_xticks(x, conf["predicted_probability"], fontsize=8)
    axes[1].set(ylim=(0, 1.12), xlabel="predicted probability of disease",
                ylabel="accuracy", title="Accuracy is low where the model is unsure")
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "13_confidence_vs_accuracy.png", show)


# ----------------------------------------------------------------------------- main
def run_error_analysis(arrays, test_df, p_mlp, p_hgb, show=False):
    """p_mlp / p_hgb: ensemble test probabilities. Saves tables + figures, returns tables."""
    y = np.asarray(arrays["y_test"]).astype(int)
    assert len(test_df) == len(y) and (test_df["cardio"].to_numpy() == y).all(), \
        "cleaned test table and arrays are not aligned - rerun Stage 4"
    eda._style()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sub = subgroup_table(test_df, y, p_mlp)
    conf = confidence_table(y, p_mlp)
    unc = uncertain_summary(y, p_mlp)
    agr = agreement_table(y, p_mlp, p_hgb)
    prof = profile_table(test_df, y, p_mlp)

    sub.to_csv(config.RESULTS_DIR / "error_by_group.csv", index=False)
    conf.to_csv(config.RESULTS_DIR / "error_by_confidence.csv", index=False)
    agr.to_csv(config.RESULTS_DIR / "model_agreement.csv", index=False)
    prof.to_csv(config.RESULTS_DIR / "error_profiles.csv")

    overall = ((p_mlp >= THRESHOLD).astype(int) == y).mean()
    plot_subgroups(sub, overall, show)
    plot_confidence(y, p_mlp, conf, show)
    return dict(subgroups=sub, confidence=conf, uncertain=unc, agreement=agr,
                profiles=prof, overall_accuracy=overall)
