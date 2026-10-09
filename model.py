"""
model.py - the neural network (Stage 5 design).

    input(17) -> [Linear -> BatchNorm -> ReLU -> Dropout] x 3 -> Linear(32 -> 1)

The network outputs ONE raw score per patient (a "logit"). It is deliberately NOT
passed through a sigmoid here: BCEWithLogitsLoss applies the sigmoid internally in
a numerically stable way. To get a probability at prediction time use
torch.sigmoid(model(x)).
"""
import torch
import torch.nn as nn

import config


class MLP(nn.Module):
    """Fully connected network for binary classification on tabular data."""

    def __init__(self, in_features: int,
                 hidden=config.MODEL["hidden"],
                 dropout: float = config.MODEL["dropout"],
                 use_bn: bool = config.MODEL["use_bn"]):
        super().__init__()
        layers, prev = [], in_features
        for width in hidden:
            layers.append(nn.Linear(prev, width))      # weighted sum of inputs
            if use_bn:
                layers.append(nn.BatchNorm1d(width))   # re-centre / re-scale per batch
            layers.append(nn.ReLU())                   # non-linearity: max(0, x)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))     # random switch-off (train only)
            prev = width
        layers.append(nn.Linear(prev, 1))              # output layer: one logit
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        """Kaiming (He) initialisation for layers followed by ReLU; Xavier for output."""
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for m in linears[:-1]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(linears[-1].weight)
        nn.init.zeros_(linears[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_features) -> logits: (batch,)"""
        return self.net(x).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe(model: nn.Module) -> str:
    """Readable architecture summary for the report."""
    return f"{model}\nTrainable parameters: {count_parameters(model):,}"
