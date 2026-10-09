"""
train.py - the training pipeline (Stage 6/7).

One epoch =
    1. TRAIN: loop over mini-batches -> forward -> loss -> backward -> optimizer step
    2. VALIDATE: score the validation set with no gradient updates
    3. CONTROL: LR scheduler + early stopping + keep the best weights

IMPORTANT: this module only ever sees the TRAIN and VALIDATION arrays. The test
set is deliberately not touched here, so model selection cannot leak into it.
"""
import copy
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

import config
from model import MLP, count_parameters


# ------------------------------------------------------------------ reproducibility
def get_device() -> torch.device:
    """Use the GPU if CUDA is available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = config.SEED):
    """Seed every random number generator that affects training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------- data loading
def make_loader(X, y, batch_size, shuffle, seed=config.SEED) -> DataLoader:
    """Wrap NumPy arrays in a DataLoader that yields (features, label) mini-batches."""
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    g = torch.Generator().manual_seed(seed)            # makes shuffling repeatable
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


# ------------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate_loader(model, loader, criterion, device):
    """
    Score a dataset. Returns (loss, accuracy, roc_auc, probabilities, labels).
    model.eval() switches Dropout off and makes BatchNorm use its running statistics;
    torch.no_grad() skips gradient bookkeeping (faster, less memory).
    """
    model.eval()
    total_loss, n = 0.0, 0
    probs, labels = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * len(yb)
        n += len(yb)
        probs.append(torch.sigmoid(logits).cpu())
        labels.append(yb.cpu())
    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()
    acc = float(((probs >= 0.5) == (labels == 1)).mean())
    return total_loss / n, acc, float(roc_auc_score(labels, probs)), probs, labels


@torch.no_grad()
def predict_proba(model, X, device=None, batch: int = 2048) -> np.ndarray:
    """Probability of disease for each row of the NumPy array X (model in eval mode)."""
    device = device or get_device()
    model.eval().to(device)
    out = [torch.sigmoid(model(torch.from_numpy(X[i:i + batch]).to(device))).cpu()
           for i in range(0, len(X), batch)]
    return torch.cat(out).numpy()


# ---------------------------------------------------------------------------- train
def train_model(arrays: dict, model_cfg: dict = None, train_cfg: dict = None,
                seed: int = config.SEED, save_path=None, verbose: bool = True):
    """
    Train one MLP using ONLY arrays['X_train'/'y_train'] and ['X_val'/'y_val'].

    Returns a dict with: model (best weights restored), history (DataFrame),
    best_epoch, best_val_loss, seconds.
    """
    model_cfg = {**config.MODEL, **(model_cfg or {})}
    train_cfg = {**config.TRAIN, **(train_cfg or {})}
    device = get_device()
    set_seed(seed)

    train_loader = make_loader(arrays["X_train"], arrays["y_train"],
                               train_cfg["batch_size"], shuffle=True, seed=seed)
    val_loader = make_loader(arrays["X_val"], arrays["y_val"],
                             1024, shuffle=False)

    model = MLP(arrays["X_train"].shape[1], **model_cfg).to(device)
    criterion = nn.BCEWithLogitsLoss()                 # sigmoid + binary cross-entropy
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"],
                                  weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=train_cfg["sched_factor"],
        patience=train_cfg["sched_patience"])

    if verbose:
        print(f"device={device} | params={count_parameters(model):,} | seed={seed}")

    history, best_loss, best_state, best_epoch, bad_epochs = [], float("inf"), None, 0, 0
    start = time.time()

    for epoch in range(1, train_cfg["max_epochs"] + 1):
        # ---- 1. TRAIN ---------------------------------------------------------
        model.train()                                   # Dropout on, BN uses batch stats
        run_loss, run_correct, n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)       # clear old gradients
            logits = model(xb)                          # forward pass
            loss = criterion(logits, yb)                # how wrong are we?
            loss.backward()                             # backpropagation: gradients
            optimizer.step()                            # update the weights
            run_loss += loss.item() * len(yb)
            run_correct += ((logits > 0) == (yb == 1)).sum().item()
            n += len(yb)
        train_loss, train_acc = run_loss / n, run_correct / n

        # ---- 2. VALIDATE ------------------------------------------------------
        val_loss, val_acc, val_auc, _, _ = evaluate_loader(model, val_loader, criterion, device)

        # ---- 3. CONTROL: scheduler, best-model tracking, early stopping -------
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc, val_auc=val_auc, lr=lr_now))

        if val_loss < best_loss - train_cfg["min_delta"]:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())      # remember best weights
        else:
            bad_epochs += 1

        if verbose and (epoch == 1 or epoch % 5 == 0 or bad_epochs == 0):
            print(f"epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.3f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.3f} auc {val_auc:.3f} | lr {lr_now:.1e}"
                  + ("  *best*" if bad_epochs == 0 else ""))
        if bad_epochs >= train_cfg["patience"]:
            if verbose:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
            break

    model.load_state_dict(best_state)                   # restore the best weights
    seconds = time.time() - start
    hist = pd.DataFrame(history)

    if save_path is not None:
        save_checkpoint(save_path, model, model_cfg, train_cfg, seed, best_epoch,
                        best_loss, arrays.get("feature_names"))
        hist.to_csv(str(save_path).replace(".pt", "_history.csv"), index=False)

    return dict(model=model, history=hist, best_epoch=best_epoch,
                best_val_loss=best_loss, seconds=seconds)


def fit_and_score(arrays, model_cfg=None, train_cfg=None, seed=config.SEED, save_path=None):
    """
    Train one model and return its VALIDATION metrics (at the best epoch) as a flat
    dict. Used by the tuning search and the ablation study.
    """
    from metrics import binary_metrics               # local import: avoids a cycle
    res = train_model(arrays, model_cfg, train_cfg, seed=seed, save_path=save_path,
                      verbose=False)
    m = binary_metrics(arrays["y_val"], predict_proba(res["model"], arrays["X_val"]))
    return dict(**{f"val_{k}": v for k, v in m.items()},
                best_epoch=res["best_epoch"], epochs_run=len(res["history"]),
                seconds=res["seconds"], params=count_parameters(res["model"]))


# ------------------------------------------------------------------------ checkpoints
def save_checkpoint(path, model, model_cfg, train_cfg, seed, best_epoch, best_val_loss,
                    feature_names=None):
    """Save weights AND everything needed to rebuild the model later."""
    path = config.MODEL_DIR / path if not hasattr(path, "parent") else path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), model_cfg=model_cfg,
                    train_cfg=train_cfg, seed=seed, best_epoch=best_epoch,
                    best_val_loss=best_val_loss, feature_names=feature_names,
                    in_features=model.net[0].in_features), path)


def load_checkpoint(path, device=None):
    """Rebuild a saved model (in eval mode) from a checkpoint file."""
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MLP(ckpt["in_features"], **ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


# --------------------------------------------------------------------------- plotting
def plot_history(hist: pd.DataFrame, path=None, show=False):
    """Training vs validation loss and accuracy per epoch (overfitting monitor)."""
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    best = int(hist.loc[hist["val_loss"].idxmin(), "epoch"])
    for ax, (tr, va, title, ylab) in zip(axes, [
            ("train_loss", "val_loss", "Loss (lower is better)", "binary cross-entropy"),
            ("train_acc", "val_acc", "Accuracy", "accuracy")]):
        ax.plot(hist["epoch"], hist[tr], color=BLUE, lw=2)
        ax.plot(hist["epoch"], hist[va], color=ORANGE, lw=2)
        ax.axvline(best, color=MUTED, ls="--", lw=1)
        ax.text(hist["epoch"].iloc[-1], hist[tr].iloc[-1], "  train", color=INK, va="center")
        ax.text(hist["epoch"].iloc[-1], hist[va].iloc[-1], "  validation", color=INK, va="center")
        ax.set_title(title, loc="left", fontweight="bold", color=INK)
        ax.set_xlabel("epoch", color=MUTED)
        ax.set_ylabel(ylab, color=MUTED)
        ax.grid(color=GRID)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.15)
    fig.suptitle(f"Training history (dashed line = best epoch {best}; "
                 "train metrics are measured with dropout on)", x=0.01, ha="left",
                 fontsize=10, color=MUTED)
    fig.tight_layout()
    if path is not None:
        config.FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
