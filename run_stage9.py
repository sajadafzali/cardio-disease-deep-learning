"""
run_stage9.py - error analysis of the final models (descriptive, changes nothing).

    python run_stage9.py

Run it AFTER run_stage8.py. It only reads the saved models, so the Stage-8 test-set
lock stays valid. Results are for the Discussion section of the report; do not go back
and modify the model because of them.
"""
import json

import numpy as np
import pandas as pd

import analyze
import config
import data
import evaluate


def fmt(df, index=False):
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 220,
                           "display.max_columns", 30):
        return df.to_string(index=index)


def main():
    lock = config.RESULTS_DIR / "test_lock.json"
    if not lock.exists():
        raise SystemExit("Run Stage 8 first (python run_stage8.py): the test set must be "
                         "evaluated before it is analysed.")
    if json.loads(lock.read_text())["fingerprint"] != evaluate.fingerprint():
        raise SystemExit("The models changed after the Stage-8 evaluation. Stop: results "
                         "would not match the reported test numbers.")

    arrays = data.load_arrays()
    test_df = pd.read_csv(config.PROC_DIR / "test_clean.csv")

    _, mlp = evaluate.mlp_probabilities(arrays)
    _, hgb = evaluate.baseline_probabilities(arrays)
    res = analyze.run_error_analysis(arrays, test_df, np.mean(mlp, axis=0), np.mean(hgb, axis=0))

    print(f"Overall accuracy (MLP ensemble, threshold 0.5): {res['overall_accuracy']:.4f}\n")
    print("=== 1. Accuracy by patient group (groups with n < 100 are not drawn in the figure) ===")
    print(fmt(res["subgroups"]))
    print("\n=== 2. Accuracy by how sure the model is ===")
    print(fmt(res["confidence"]))
    u = res["uncertain"]
    print(f"\nPatients with predicted probability in {u['band']}: {u['share_of_patients']:.1%} "
          f"of the test set; accuracy inside {u['accuracy_inside']:.3f}, outside "
          f"{u['accuracy_outside']:.3f}; they hold {u['share_of_all_errors_inside']:.1%} of all errors.")
    print("\n=== 3. Do MLP and gradient boosting make the same mistakes? ===")
    print(fmt(res["agreement"]))
    print("\n=== 4. Average profile of each outcome type ===")
    print(fmt(res["profiles"], index=True))
    print(f"\nFigures 12 and 13 saved to {config.FIG_DIR}\nTables saved to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
