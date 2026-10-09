"""
run_stage7.py - baselines, hyper-parameter search, ablation study.

    python run_stage7.py baselines      # ~10 s
    python run_stage7.py tune           # 20 trials, roughly 10-15 min on CPU
    python run_stage7.py ablate         # 6 variants x 5 seeds, roughly 15-20 min on CPU
    python run_stage7.py report         # rebuild table + figure from saved results (seconds)
    python run_stage7.py all            # everything in order

    add  --quick  for a fast smoke test (4 trials, 2 seeds) in a separate folder

Every step saves its progress, so if Colab disconnects just run the same command
again: finished runs are skipped. Only training and validation data are used.
"""
import argparse

import pandas as pd

import config
import data


def load_arrays_no_test():
    if not (config.PROC_DIR / "arrays.npz").exists():
        print("Stage-4 arrays not found - running Stage 4 first...")
        data.run_pipeline()
    arrays = data.load_arrays()
    for k in ("X_test", "y_test"):       # the test set must not be reachable from here
        arrays.pop(k)
    return arrays


def fmt(df):
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        return df.to_string()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["baselines", "tune", "ablate", "report", "all"])
    p.add_argument("--quick", action="store_true", help="smoke test: 4 trials, 2 seeds")
    a = p.parse_args()

    if a.quick:                                   # keep smoke-test files separate
        config.RESULTS_DIR = config.OUT_DIR / "results_quick"
        config.MODEL_DIR = config.OUT_DIR / "models_quick"
        config.N_TRIALS, config.SEEDS = 4, config.SEEDS[:2]
    import ablate, baselines, tune

    if a.step == "report":
        return make_report(a.quick)
    arrays = load_arrays_no_test()

    if a.step in ("baselines", "all"):
        print("=== Baselines (validation set) ===")
        df = baselines.run_baselines(arrays, config.SEEDS)
        print(fmt(df.groupby("model")[["accuracy", "auc", "loss", "recall", "precision", "f1"]]
                  .mean()))
    if a.step in ("tune", "all"):
        print("\n=== Hyper-parameter search (validation set) ===")
        tune.run_search(arrays, config.N_TRIALS)
    if a.step in ("ablate", "all"):
        if not tune.best_json().exists():
            raise SystemExit("\nThe tuning step has not finished: best_config.json is missing.\n"
                             "Run the tuning step first (python run_stage7.py tune), then this one.")
        if not (config.RESULTS_DIR / "baselines.csv").exists():
            print("Baselines not found - running them first...")
            baselines.run_baselines(arrays, config.SEEDS)
        print("\n=== Ablation study (validation set) ===")
        summ = ablate.run_ablation(arrays, config.SEEDS)
        print("\n" + fmt(summ[["val_auc_mean", "val_auc_std", "val_accuracy_mean",
                               "val_loss_mean", "best_epoch_mean", "params_mean",
                               "auc_change_vs_full"]]))
        ablate.plot_comparison(config.FIG_DIR / ("07_model_comparison"
                               + ("_quick" if a.quick else "") + ".png"))
        print("\nFigure saved to", config.FIG_DIR)


def make_report(quick=False):
    """Rebuild the summary table and comparison figure from saved results (no training)."""
    import ablate
    need = [ablate.runs_csv(), config.RESULTS_DIR / "baselines.csv"]
    missing = [p.name for p in need if not p.exists()]
    if missing:
        raise SystemExit(f"\nCannot make the report yet - missing: {missing}.\n"
                         "Run the baselines, tuning and ablation steps first.")
    summ = ablate.summarise()
    print(fmt(summ[["val_auc_mean", "val_auc_std", "val_accuracy_mean", "val_loss_mean",
                    "best_epoch_mean", "params_mean", "auc_change_vs_full"]]))
    ablate.plot_comparison(config.FIG_DIR / ("07_model_comparison" + ("_quick" if quick else "") + ".png"))
    print("Figure saved to", config.FIG_DIR)


if __name__ == "__main__":
    main()
