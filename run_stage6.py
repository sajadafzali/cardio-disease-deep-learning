"""
run_stage6.py - builds the network, proves the training code works, trains once.

    python run_stage6.py

Part A: sanity tests (catch bugs BEFORE spending time on real training)
Part B: one full training run with the default configuration

Uses only the training and validation splits. The test set is not loaded.
"""
import numpy as np
import torch
import torch.nn as nn

import config
import data
import train
from model import MLP, count_parameters, describe


def sanity_tests(arrays):
    n_feat = arrays["X_train"].shape[1]
    device = train.get_device()
    Xv = torch.from_numpy(arrays["X_val"])

    # 1. output shape: one logit per patient
    m = MLP(n_feat)
    assert m(Xv[:8]).shape == (8,), "forward output must be (batch,)"

    # 2. parameter count matches a hand calculation
    widths = [n_feat, *config.MODEL["hidden"], 1]
    linear = sum(a * b + b for a, b in zip(widths[:-1], widths[1:]))
    bn = sum(2 * w for w in config.MODEL["hidden"]) if config.MODEL["use_bn"] else 0
    assert count_parameters(m) == linear + bn, "parameter count mismatch"
    print(f"[ok] shapes and parameter count ({count_parameters(m):,}) match the hand calculation")

    # 3. an untrained network should be near chance: loss close to ln(2) = 0.693
    train.set_seed()
    m = MLP(n_feat).eval()
    loss = nn.BCEWithLogitsLoss()(m(Xv), torch.from_numpy(arrays["y_val"])).item()
    assert 0.5 < loss < 1.0, f"initial loss {loss:.3f} is far from ln(2)"
    print(f"[ok] initial loss {loss:.3f} (chance level is 0.693)")

    # 4. the training loop can memorise 256 samples (proves gradients/optimizer work)
    train.set_seed()
    m = MLP(n_feat, dropout=0.0).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    xb = torch.from_numpy(arrays["X_train"][:256]).to(device)
    yb = torch.from_numpy(arrays["y_train"][:256]).to(device)
    crit = nn.BCEWithLogitsLoss()
    for _ in range(400):
        opt.zero_grad()
        l = crit(m(xb), yb)
        l.backward()
        opt.step()
    assert l.item() < 0.1, f"could not overfit 256 samples (loss {l.item():.3f}) - bug?"
    print(f"[ok] overfit test: loss on 256 samples fell to {l.item():.4f}")

    # 5. same seed -> identical results
    quick = dict(max_epochs=2)
    a = train.train_model(arrays, train_cfg=quick, verbose=False)["history"]["val_loss"].tolist()
    b = train.train_model(arrays, train_cfg=quick, verbose=False)["history"]["val_loss"].tolist()
    assert np.allclose(a, b), "same seed gave different results"
    print("[ok] reproducibility: two runs with the same seed are identical")


def main():
    if not (config.PROC_DIR / "arrays.npz").exists():
        print("Stage-4 arrays not found - running Stage 4 first...")
        data.run_pipeline()
    arrays = data.load_arrays()
    for k in ("X_test", "y_test"):          # make sure nothing below can use it
        arrays.pop(k)

    print(describe(MLP(arrays["X_train"].shape[1])), "\n")
    print("=== Part A: sanity tests ===")
    sanity_tests(arrays)

    print("\n=== Part B: full training run (seed 42) ===")
    res = train.train_model(arrays, seed=config.SEED,
                            save_path=config.MODEL_DIR / f"mlp_seed{config.SEED}.pt")
    h = res["history"]
    b = h.loc[h["epoch"] == res["best_epoch"]].iloc[0]
    print(f"\nBest epoch {res['best_epoch']} of {len(h)} run | {res['seconds']:.0f}s")
    print(f"  train: loss {b.train_loss:.4f}  acc {b.train_acc:.3f}")
    print(f"  VALID: loss {b.val_loss:.4f}  acc {b.val_acc:.3f}  auc {b.val_auc:.3f}")

    train.plot_history(h, config.FIG_DIR / "06_training_history.png")
    # prove the checkpoint round-trips
    model, ckpt = train.load_checkpoint(config.MODEL_DIR / f"mlp_seed{config.SEED}.pt")
    print(f"Checkpoint reloaded (best epoch {ckpt['best_epoch']}).")
    print(f"Saved: {config.MODEL_DIR}  and  {config.FIG_DIR / '06_training_history.png'}")


if __name__ == "__main__":
    main()
