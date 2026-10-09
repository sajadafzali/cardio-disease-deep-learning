"""
run_stage4.py - runs the whole Stage-4 pipeline and checks it for leakage.

    python run_stage4.py

Outputs
    outputs/figures/*.png                 EDA figures
    outputs/processed/*.csv               cleaned train/val/test + cleaning report
    outputs/processed/arrays.npz          model-ready float32 arrays (used from Stage 6)
    outputs/processed/preprocessor.joblib fitted scaler/encoder (fit on TRAIN only)
"""
import random

import numpy as np

import config
import data
import eda


def set_seed(seed=config.SEED):
    """Make runs repeatable (PyTorch seeding is added in Stage 6)."""
    random.seed(seed)
    np.random.seed(seed)


def sanity_checks(res):
    """Fail loudly if something that would invalidate the experiment is wrong."""
    tr, va, te = res["train"], res["val"], res["test"]
    a = res["arrays"]
    t = config.TARGET

    # 1. all splits together equal the cleaned data, with no overlap in rows
    key = lambda d: set(map(tuple, d.drop(columns=[t]).round(6).to_numpy()))
    # duplicates were removed, so the same feature vector must not appear twice
    assert len(key(tr) & key(te)) == 0, "train/test share identical patients!"
    assert len(key(tr) & key(va)) == 0, "train/val share identical patients!"
    assert len(key(va) & key(te)) == 0, "val/test share identical patients!"

    # 2. stratification: positive rate nearly identical in all splits
    rates = [d[t].mean() for d in (tr, va, te)]
    assert max(rates) - min(rates) < 0.002, f"class ratio differs across splits {rates}"

    # 3. scaler fitted on train only -> train mean ~0, std ~1 on continuous columns
    n = len(config.CONTINUOUS)
    assert np.allclose(a["X_train"][:, :n].mean(0), 0, atol=1e-3)
    assert np.allclose(a["X_train"][:, :n].std(0), 1, atol=1e-2)

    # 4. no NaN/inf anywhere, shapes consistent
    for k in ("X_train", "X_val", "X_test"):
        assert np.isfinite(a[k]).all(), f"{k} has NaN/inf"
    assert a["X_train"].shape[1] == len(a["feature_names"])
    print("All sanity checks passed.")


def main():
    set_seed()
    res = data.run_pipeline()
    print("\n=== Cleaning report ===")
    print(res["report"].to_string(index=False))

    tr, va, te = res["train"], res["val"], res["test"]
    t = config.TARGET
    print("\n=== Split ===")
    for name, d in (("train", tr), ("val", va), ("test", te)):
        print(f"{name:5s}: {len(d):6d} rows | disease rate {d[t].mean():.3f}")

    print("\n=== Model input ===")
    print("features:", res["arrays"]["feature_names"])
    print("X_train shape:", res["arrays"]["X_train"].shape)

    sanity_checks(res)

    eda.run_eda(res["raw"], res["report"], tr)
    print(f"\nFigures saved to {config.FIG_DIR}")
    print(f"Processed data saved to {config.PROC_DIR}")


if __name__ == "__main__":
    main()
