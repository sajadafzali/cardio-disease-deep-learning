"""
run_stage8.py - the FINAL evaluation on the test set.

    python run_stage8.py            # evaluate the saved models (first time and repeats)
    python run_stage8.py --force    # only if you knowingly changed models afterwards

Run this only when Stage 7 is finished. The numbers it prints are the ones that go
in the report. Do NOT go back and change the model to improve them.
"""
import argparse

import pandas as pd

import config
import data
import evaluate


def fmt(df, cols=None):
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 220,
                           "display.max_columns", 30, "display.max_colwidth", 60):
        return (df if cols is None else df[cols]).to_string(index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    if not (config.PROC_DIR / "arrays.npz").exists():
        raise SystemExit("Stage-4 data not found. Run the earlier steps first.")
    arrays = data.load_arrays()
    print(f"Test set: {len(arrays['y_test']):,} patients (never used for any decision so far)\n")

    res = evaluate.run_test_evaluation(arrays, force=a.force)

    print("=== Test results ===")
    print(fmt(res["table"], ["model", "accuracy", "auc", "loss", "precision", "recall", "f1"]))
    print("\n(std over the 5 seeds for the single-model rows)")
    print(fmt(res["table"].dropna(subset=["auc_std"]),
              ["model", "accuracy_std", "auc_std", "loss_std"]))

    print("\n=== 95% confidence intervals (bootstrap over test patients) ===")
    print(fmt(res["cis"]))

    print("\n=== Decision thresholds (chosen on VALIDATION data, applied to test) ===")
    print(fmt(res["thresholds"]))

    print("\n=== Permutation importance (validation data) ===")
    print(fmt(res["importance"]))
    print(f"\nFigures saved to {config.FIG_DIR}\nTables saved to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
