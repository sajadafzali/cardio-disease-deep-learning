"""
tune.py - random hyper-parameter search (Stage 7).

* Candidates come from config.SEARCH_SPACE; trial 0 is always the default config.
* Every trial is trained with early stopping and scored on the VALIDATION set only.
* Results are appended to a CSV after each trial, so an interrupted run (e.g. a
  Colab disconnect) resumes where it stopped instead of starting over.
* Selection rule: take the best validation ROC-AUC, then choose the SIMPLEST model
  (fewest parameters) whose AUC is within config.TIE_TOLERANCE of the best. A gap
  smaller than that is inside the seed-to-seed noise, so complexity is not rewarded.
"""
import itertools
import json

import numpy as np
import pandas as pd

import config
import train

def results_csv():
    return config.RESULTS_DIR / "tuning_results.csv"


def best_json():
    return config.RESULTS_DIR / "best_config.json"


def default_cfg() -> dict:
    return dict(hidden=tuple(config.MODEL["hidden"]), dropout=config.MODEL["dropout"],
                lr=config.TRAIN["lr"], weight_decay=config.TRAIN["weight_decay"],
                batch_size=config.TRAIN["batch_size"])


def sample_trials(n: int, seed: int = config.SEED) -> list:
    """Default config first, then n-1 distinct random combinations from the grid."""
    keys = list(config.SEARCH_SPACE)
    grid = [dict(zip(keys, v)) for v in itertools.product(*config.SEARCH_SPACE.values())]
    default = default_cfg()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(grid))
    trials = [default] + [grid[i] for i in order if grid[i] != default]
    return trials[:n]


def to_args(cfg: dict):
    model_cfg = dict(hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"])
    train_cfg = {k: cfg[k] for k in ("lr", "weight_decay", "batch_size")}
    return model_cfg, train_cfg


def run_search(arrays: dict, n_trials: int = config.N_TRIALS, seed: int = config.SEED):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(results_csv()) if results_csv().exists() else pd.DataFrame()
    finished = set(done["trial"]) if len(done) else set()
    trials = sample_trials(n_trials, seed)

    for i, cfg in enumerate(trials):
        if i in finished:
            continue
        model_cfg, train_cfg = to_args(cfg)
        r = train.fit_and_score(arrays, model_cfg, train_cfg, seed=seed)
        row = dict(trial=i, hidden="-".join(map(str, cfg["hidden"])), dropout=cfg["dropout"],
                   lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                   batch_size=cfg["batch_size"], **r)
        pd.DataFrame([row]).to_csv(results_csv(), mode="a",
                                   header=not results_csv().exists(), index=False)
        print(f"trial {i + 1:2d}/{len(trials)} | {row['hidden']:>12s} do={row['dropout']} "
              f"lr={row['lr']:g} wd={row['weight_decay']:g} bs={row['batch_size']} | "
              f"val AUC {row['val_auc']:.4f} loss {row['val_loss']:.4f} "
              f"epoch {row['best_epoch']} ({row['seconds']:.0f}s)", flush=True)
    return select_best()


def select_best() -> dict:
    """Apply the selection rule to tuning_results.csv and save best_config.json."""
    df = pd.read_csv(results_csv())
    top = df["val_auc"].max()
    near = df[df["val_auc"] >= top - config.TIE_TOLERANCE]
    pick = near.sort_values(["params", "val_auc"], ascending=[True, False]).iloc[0]
    best = dict(hidden=[int(h) for h in str(pick["hidden"]).split("-")],
                dropout=float(pick["dropout"]), lr=float(pick["lr"]),
                weight_decay=float(pick["weight_decay"]),
                batch_size=int(pick["batch_size"]), use_bn=True)
    best_json().write_text(json.dumps(best, indent=2))
    print(f"\nBest val AUC {top:.4f}; {len(near)} trial(s) within {config.TIE_TOLERANCE} "
          f"-> chose the simplest: trial {int(pick['trial'])} "
          f"({pick['hidden']}, {int(pick['params']):,} params, AUC {pick['val_auc']:.4f})")
    print("Saved", best_json())
    return best


def load_best() -> dict:
    return json.loads(best_json().read_text())
