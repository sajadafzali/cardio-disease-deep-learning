"""
cnn.py - a 1D convolutional network (CNN) for the same 17 patient features (Stage 10b).

WHY a CNN on tabular data?  The course asks for a CNN or RNN next to the plain neural
network. A CNN slides small filters over neighbouring inputs. Our 17 columns are already
ordered in related groups, so neighbours really are related:

    0-3   body            age_years, height, weight, bmi
    4-6   blood pressure  ap_hi, ap_lo, pulse_pressure
    7-12  blood tests     cholesterol_1..3, gluc_1..3 (one-hot)
    13-16 lifestyle/sex   gender, smoke, alco, active

Honest limit: the order is our choice, not a real sequence like pixels or time, so we do
not expect the CNN to beat the MLP. It is a comparison, not a claim of improvement.

    (B, 17) -> (B, 1, 17) -> [Conv1d -> BatchNorm -> ReLU -> Dropout] x 2 -> flatten
            -> Linear -> ReLU -> Linear(1)   (one logit, like the MLP)

Training reuses the MLP pipeline pieces (same loss, optimiser, scheduler, early stopping,
train/validation data only). The test set is not touched in this file.
"""
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
import train
from metrics import binary_metrics

CNN_DEFAULT = dict(channels=(16, 32), kernel=3, dropout=0.1, lr=3e-3,
                   weight_decay=1e-3, batch_size=128)

SEARCH = {                                  # small random search, validation data only
    "channels": [(8, 16), (16, 32), (32, 64)],
    "kernel": [3, 5],
    "dropout": [0.1, 0.3],
    "lr": [1e-3, 3e-3],
    "weight_decay": [1e-4, 1e-3],
    "batch_size": [128, 256],
}
N_TRIALS = 8
CNN_DIR = config.MODEL_DIR / "cnn"


class CNN1D(nn.Module):
    def __init__(self, in_features: int = 17, channels=(16, 32), kernel: int = 3,
                 dropout: float = 0.1, hidden: int = 32):
        super().__init__()
        layers, prev = [], 1                                  # 1 input channel
        for c in channels:
            layers += [nn.Conv1d(prev, c, kernel, padding=kernel // 2),  # same length
                       nn.BatchNorm1d(c), nn.ReLU(), nn.Dropout(dropout)]
            prev = c
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(),
                                  nn.Linear(prev * in_features, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 17) -> logits: (batch,)"""
        return self.head(self.conv(x.unsqueeze(1))).squeeze(1)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fit_cnn(arrays, cfg=None, seed=config.SEED, save_path=None):
    """Train one CNN with early stopping on validation loss; return model + history."""
    cfg = {**CNN_DEFAULT, **(cfg or {})}
    build = lambda n_in, c: CNN1D(n_in, tuple(c["channels"]), c["kernel"], c["dropout"])
    return fit_net(arrays, build, cfg, seed, save_path)


def fit_net(arrays, build, cfg, seed=config.SEED, save_path=None):
    """Generic trainer (used for the CNN and the RNN): build(n_inputs, cfg) -> nn.Module."""
    tc = config.TRAIN
    device = train.get_device()
    train.set_seed(seed)
    tr = train.make_loader(arrays["X_train"], arrays["y_train"], cfg["batch_size"], True, seed)
    va = train.make_loader(arrays["X_val"], arrays["y_val"], 1024, False)
    model = build(arrays["X_train"].shape[1], cfg).to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, "min", factor=tc["sched_factor"], patience=tc["sched_patience"])
    best, best_state, best_ep, bad, hist = float("inf"), None, 0, 0, []
    t0 = time.time()
    for ep in range(1, tc["max_epochs"] + 1):
        model.train()
        tl, n = 0.0, 0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(yb)
            n += len(yb)
        vl, vacc, vauc, _, _ = train.evaluate_loader(model, va, crit, device)
        sched.step(vl)
        hist.append(dict(epoch=ep, train_loss=tl / n, val_loss=vl, val_acc=vacc, val_auc=vauc))
        if vl < best - tc["min_delta"]:
            best, best_ep, bad = vl, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if bad >= tc["patience"]:
            break
    model.load_state_dict(best_state)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=model.state_dict(), cfg=cfg, seed=seed), save_path)
    return dict(model=model, history=pd.DataFrame(hist), best_epoch=best_ep,
                seconds=time.time() - t0)


def load_cnn(path, device=None):
    device = device or train.get_device()
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = {**CNN_DEFAULT, **ck["cfg"]}
    m = CNN1D(17, tuple(cfg["channels"]), cfg["kernel"], cfg["dropout"]).to(device)
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def sample_trials(n=N_TRIALS, seed=config.SEED):
    import itertools
    keys = list(SEARCH)
    grid = [dict(zip(keys, v)) for v in itertools.product(*SEARCH.values())]
    order = np.random.default_rng(seed).permutation(len(grid))
    trials = [dict(CNN_DEFAULT)] + [grid[i] for i in order if grid[i] != CNN_DEFAULT]
    return trials[:n]


def run_search(arrays, n_trials=N_TRIALS):
    """Validation-only search. Rule as for the MLP: simplest model within 0.001 AUC of the best."""
    rows = []
    for i, cfg in enumerate(sample_trials(n_trials)):
        r = fit_cnn(arrays, cfg)
        m = binary_metrics(arrays["y_val"], train.predict_proba(r["model"], arrays["X_val"]))
        rows.append(dict(trial=i, **{k: str(v) if k == "channels" else v for k, v in cfg.items()},
                         params=n_params(r["model"]), val_auc=m["auc"], val_acc=m["accuracy"],
                         best_epoch=r["best_epoch"], seconds=r["seconds"]))
        print(f"trial {i}: val AUC {m['auc']:.4f} | params {rows[-1]['params']:,} | "
              f"{r['seconds']:.0f}s | {cfg}")
    df = pd.DataFrame(rows)
    top = df["val_auc"].max()
    pick = df[df["val_auc"] >= top - config.TIE_TOLERANCE].sort_values(
        ["params", "val_auc"], ascending=[True, False]).iloc[0]
    best = sample_trials(n_trials)[int(pick["trial"])]
    return df, best
