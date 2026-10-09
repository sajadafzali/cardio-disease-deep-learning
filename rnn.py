"""
rnn.py - a recurrent network (GRU) for the same 17 patient features (Stage 10c).

An RNN reads a sequence one step at a time and keeps a memory (hidden state) of what it
has seen. We feed the 17 features as a sequence of 17 steps, in the same grouped order as
the CNN (body, blood pressure, blood tests, lifestyle). A GRU is a gated RNN that avoids
the vanishing-gradient problem of the plain RNN.

    (B, 17) -> (B, 17, 1)  [17 time steps, 1 number each]
            -> GRU(hidden units) -> last hidden state -> Dropout -> Linear(1)

Honest limit: our "sequence" is an ordering we chose, not time. We do not expect an
advantage over the MLP; this is a comparison. Training, tuning rules and data use are
identical to the CNN (cnn.fit_net).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import cnn
import config
import train
from metrics import binary_metrics

RNN_DEFAULT = dict(hidden=32, layers=1, dropout=0.1, lr=3e-3, weight_decay=1e-3,
                   batch_size=128)
SEARCH = {
    "hidden": [16, 32, 64],
    "layers": [1],            # 1 layer only: much faster to train
    "dropout": [0.1, 0.3],
    "lr": [1e-3, 3e-3],
    "weight_decay": [1e-4, 1e-3],
    "batch_size": [128, 256],
}
N_TRIALS = 6
RNN_DIR = config.MODEL_DIR / "rnn"


class GRUNet(nn.Module):
    def __init__(self, hidden: int = 32, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, num_layers=layers,
                          batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 17) -> logits: (batch,)"""
        _, h = self.gru(x.unsqueeze(2))              # h: (layers, batch, hidden)
        return self.out(self.drop(h[-1])).squeeze(1)  # last layer's final state


def _build(n_in, c):
    return GRUNet(c["hidden"], c["layers"], c["dropout"])


def fit_rnn(arrays, cfg=None, seed=config.SEED, save_path=None):
    cfg = {**RNN_DEFAULT, **(cfg or {})}
    return cnn.fit_net(arrays, _build, cfg, seed, save_path)


def load_rnn(path, device=None):
    device = device or train.get_device()
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = {**RNN_DEFAULT, **ck["cfg"]}
    m = _build(17, cfg).to(device)
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def sample_trials(n=N_TRIALS, seed=config.SEED):
    import itertools
    keys = list(SEARCH)
    grid = [dict(zip(keys, v)) for v in itertools.product(*SEARCH.values())]
    order = np.random.default_rng(seed).permutation(len(grid))
    return ([dict(RNN_DEFAULT)] + [grid[i] for i in order if grid[i] != RNN_DEFAULT])[:n]


def run_search(arrays, n_trials=N_TRIALS):
    """Validation-only; simplest model within 0.001 val AUC of the best."""
    trials, rows = sample_trials(n_trials), []
    for i, cfg in enumerate(trials):
        r = fit_rnn(arrays, cfg)
        m = binary_metrics(arrays["y_val"], train.predict_proba(r["model"], arrays["X_val"]))
        rows.append(dict(trial=i, **cfg, params=cnn.n_params(r["model"]), val_auc=m["auc"],
                         val_acc=m["accuracy"], best_epoch=r["best_epoch"],
                         seconds=r["seconds"]))
        print(f"trial {i}: val AUC {m['auc']:.4f} | params {rows[-1]['params']:,} | "
              f"{r['seconds']:.0f}s | {cfg}")
    df = pd.DataFrame(rows)
    pick = df[df["val_auc"] >= df["val_auc"].max() - config.TIE_TOLERANCE].sort_values(
        ["params", "val_auc"], ascending=[True, False]).iloc[0]
    return df, trials[int(pick["trial"])]
