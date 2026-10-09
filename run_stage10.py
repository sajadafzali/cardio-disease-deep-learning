"""
run_stage10.py - add a 1D CNN and a GRU (RNN) and compare both with the MLP.

    python run_stage10.py            # both models (about 20-40 min; safe to re-run)
    python run_stage10.py cnn        # only the CNN
    python run_stage10.py rnn        # only the RNN

For each new model (all choices use TRAIN and VALIDATION data only):
  1. small random search (CNN 8 trials, RNN 6 trials); simplest model within 0.001 val AUC of the best
  2. train the chosen configuration with 5 seeds
Then, ONCE: evaluate on the test set next to the already-fixed MLP and the baselines.

HONESTY NOTE for the report: the test set had already been used for the MLP (Stage 8).
The CNN and RNN were added afterwards, so their test scores are an ADDED comparison.
The MLP and every earlier choice were not changed because of them, and the winner is
judged on ROC-AUC with its confidence interval, not on a single accuracy number.
"""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import cnn
import config
import data
import eda
import evaluate
import rnn
import train
from metrics import binary_metrics

KINDS = {  # name -> (module with run_search/fit, fit function, loader dir, label)
    "cnn": dict(mod=cnn, fit=cnn.fit_cnn, dir=cnn.CNN_DIR, label="CNN (5 seeds)"),
    "rnn": dict(mod=rnn, fit=rnn.fit_rnn, dir=rnn.RNN_DIR, label="RNN-GRU (5 seeds)"),
}


def build_model(kind, arrays):
    """Search (validation only), then train 5 seeds. Returns the list of trained models."""
    k = KINDS[kind]
    cfg_path = config.RESULTS_DIR / f"{kind}_best_config.json"
    if cfg_path.exists():
        best = json.loads(cfg_path.read_text())
        print(f"[{kind}] using saved configuration: {best}")
    else:
        print(f"\n[{kind}] search: validation data only\n")
        df, best = k["mod"].run_search(arrays)
        df.to_csv(config.RESULTS_DIR / f"{kind}_tuning_results.csv", index=False)
        cfg_path.write_text(json.dumps(best, indent=2, default=list))
        print(f"\n[{kind}] chosen configuration: {best}")
    models, rows = [], []
    for s in config.SEEDS:
        path = k["dir"] / f"{kind}_seed{s}.pt"
        r = k["fit"](arrays, best, seed=s, save_path=path)
        r["history"].to_csv(str(path).replace(".pt", "_history.csv"), index=False)
        models.append(r["model"])
        m = binary_metrics(arrays["y_val"], train.predict_proba(r["model"], arrays["X_val"]))
        rows.append(dict(seed=s, best_epoch=r["best_epoch"], val_auc=m["auc"],
                         val_acc=m["accuracy"], params=cnn.n_params(r["model"])))
        print(f"[{kind}] seed {s}: val AUC {m['auc']:.4f}  acc {m['accuracy']:.4f}  "
              f"epochs {r['best_epoch']}")
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / f"{kind}_seed_runs.csv", index=False)
    print(f"[{kind}] parameters: {rows[0]['params']:,}  (MLP: 3,457)")
    return models


def main():
    kinds = [a for a in sys.argv[1:] if a in KINDS] or list(KINDS)
    arrays = data.load_arrays()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    y = np.asarray(arrays["y_test"]).astype(int)

    new = {}
    for kind in kinds:
        models = build_model(kind, arrays)
        new[KINDS[kind]["label"]] = np.mean(
            [train.predict_proba(m, arrays["X_test"]) for m in models], axis=0)

    if set(kinds) != set(KINDS):
        print("\nOnly part of the models were built; run again without arguments for the "
              "final comparison (the saved configurations are reused).")
        return

    # ---- test set, once ---------------------------------------------------------------
    _, mlp_test = evaluate.mlp_probabilities(arrays)
    lr, hgb = evaluate.baseline_probabilities(arrays)
    probs = {"MLP (5 seeds)": np.mean(mlp_test, axis=0), **new,
             "Gradient boosting (5 seeds)": np.mean(hgb, axis=0), "Logistic regression": lr}
    tab = pd.DataFrame([dict(model=k, **binary_metrics(y, p)) for k, p in probs.items()])
    cis = evaluate.bootstrap_cis(y, probs)
    tab.to_csv(config.RESULTS_DIR / "deep_models_test_results.csv", index=False)
    cis.to_csv(config.RESULTS_DIR / "deep_models_confidence_intervals.csv", index=False)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 220):
        print("\n=== Test results (ensembles of 5 seeds) ===")
        print(tab.to_string(index=False))
        print("\n=== 95% bootstrap intervals and paired differences (vs MLP) ===")
        print(cis.to_string(index=False))

    # ---- figure -----------------------------------------------------------------------
    eda._style()
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    cols = (eda.BLUE, eda.ORANGE, "#7b4fc9", evaluate.AQUA, eda.NEUTRAL)
    for (k, p), c in zip(probs.items(), cols):
        f, t, _ = roc_curve(y, p)
        ax[0].plot(f, t, color=c, lw=1.8, label=f"{k}  (AUC {roc_auc_score(y, p):.3f})")
    ax[0].plot([0, 1], [0, 1], ls="--", color=eda.NEUTRAL, lw=1)
    ax[0].set(xlabel="false positive rate", ylabel="true positive rate",
              title="ROC curves - test set")
    ax[0].legend(fontsize=7.5, loc="lower right")
    for kind, c in (("cnn", eda.ORANGE), ("rnn", "#7b4fc9")):
        h = pd.read_csv(KINDS[kind]["dir"] / f"{kind}_seed{config.SEEDS[0]}_history.csv")
        ax[1].plot(h["epoch"], h["val_loss"], color=c, label=f"{kind.upper()} validation")
        ax[1].plot(h["epoch"], h["train_loss"], color=c, ls=":", label=f"{kind.upper()} training")
    ax[1].set(xlabel="epoch", ylabel="loss (binary cross-entropy)",
              title=f"Learning curves (seed {config.SEEDS[0]})")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIG_DIR / "14_deep_models_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved figure 14 to {config.FIG_DIR}; tables to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
