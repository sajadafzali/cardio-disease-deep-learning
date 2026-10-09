"""
baselines.py - two non-deep reference models on the SAME features and split.

  Logistic regression       - linear model: "how far can a straight-line rule go?"
  Gradient boosting (HGB)   - strong tree ensemble, the usual winner on tabular data.

If the neural network cannot beat these, that is a legitimate finding, not a failure.
Only training and validation data are used here.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import config
from metrics import binary_metrics


def run_baselines(arrays: dict, seeds=config.SEEDS) -> pd.DataFrame:
    """Train each baseline (HGB once per seed) and score on the VALIDATION set."""
    Xtr, ytr = arrays["X_train"], arrays["y_train"]
    Xva, yva = arrays["X_val"], arrays["y_val"]
    rows = []

    lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)       # deterministic: one run
    rows.append(dict(model="Logistic regression", seed=0,
                     **binary_metrics(yva, lr.predict_proba(Xva)[:, 1])))

    for s in seeds:      # HGB holds out part of train internally for early stopping
        hgb = HistGradientBoostingClassifier(random_state=s).fit(Xtr, ytr)
        rows.append(dict(model="Gradient boosting", seed=s,
                         **binary_metrics(yva, hgb.predict_proba(Xva)[:, 1])))

    df = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTS_DIR / "baselines.csv", index=False)
    return df
