"""
config.py - single place for every project setting.

Nothing is hard-coded elsewhere: if your instructor or team wants to change a
threshold, a split ratio or a seed, it is changed here and the whole pipeline
follows. Every value below is justified in the report (Methodology section).
"""
from pathlib import Path

# ----------------------------------------------------------------- reproducibility
SEED = 42  # one seed for splitting, NumPy, PyTorch (used again in later stages)

# ----------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cardio_train.csv"   # put the CSV here
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
PROC_DIR = OUT_DIR / "processed"                 # cleaned splits + fitted scaler
MODEL_DIR = OUT_DIR / "models"                   # used from Stage 7

# ----------------------------------------------------------------- columns
TARGET = "cardio"
ID_COL = "id"                                    # identifier: never a feature

# ----------------------------------------------------------------- cleaning rules
# Fixed physiological limits (NOT learned from data -> no leakage).
# Each tuple is (min, max), both inclusive.
LIMITS = {
    "ap_hi": (70, 250),      # systolic blood pressure, mmHg
    "ap_lo": (40, 150),      # diastolic blood pressure, mmHg
    "height": (130, 210),    # cm
    "weight": (35, 200),     # kg
    "bmi": (15, 60),         # kg/m^2
}

# ----------------------------------------------------------------- split
TEST_SIZE = 0.15
VAL_SIZE = 0.15                                  # fraction of the WHOLE dataset

# ----------------------------------------------------------------- feature groups
# (names after feature engineering, see data.add_features)
CONTINUOUS = ["age_years", "height", "weight", "bmi",
              "ap_hi", "ap_lo", "pulse_pressure"]  # standardised
BINARY = ["gender", "smoke", "alco", "active"]     # kept as 0/1
ONEHOT = {"cholesterol": [1, 2, 3], "gluc": [1, 2, 3]}  # ordinal codes -> one-hot

# ----------------------------------------------------------------- model (Stage 5 design)
MODEL = {
    "hidden": (128, 64, 32),   # funnel: each hidden layer narrower than the last
    "dropout": 0.3,            # probability of switching a unit off during training
    "use_bn": True,            # BatchNorm between Linear and ReLU
}

# ----------------------------------------------------------------- training (Stage 3 design)
TRAIN = {
    "lr": 1e-3,                # AdamW learning rate
    "weight_decay": 1e-4,      # L2-style regularisation
    "batch_size": 256,
    "max_epochs": 100,         # upper limit; early stopping normally ends sooner
    "patience": 10,            # stop after this many epochs without val-loss improvement
    "min_delta": 1e-4,         # smallest val-loss drop that counts as an improvement
    "sched_factor": 0.5,       # ReduceLROnPlateau: multiply lr by this ...
    "sched_patience": 3,       # ... after this many epochs without improvement
}

# ----------------------------------------------------------------- Stage 7: search + ablation
RESULTS_DIR = OUT_DIR / "results"

SEARCH_SPACE = {                                  # random search, validation data only
    "hidden": [(64, 32), (128, 64, 32), (256, 128, 64), (128, 64, 32, 16)],
    "dropout": [0.1, 0.3, 0.5],
    "lr": [3e-4, 1e-3, 3e-3],
    "weight_decay": [0.0, 1e-4, 1e-3],
    "batch_size": [128, 256, 512],
}
N_TRIALS = 20                                     # trial 0 is the Stage-3 default config
SEEDS = [42, 43, 44, 45, 46]                      # repeated runs -> mean +/- std
TIE_TOLERANCE = 0.001                             # val-AUC gap treated as "no real difference"

# ----------------------------------------------------------------- Stage 8: final evaluation
RECALL_TARGET = 0.80       # illustrative screening operating point (our choice, not a course rule)
N_BOOTSTRAP = 1000         # resamples for confidence intervals
N_PERMUTE = 10             # repeats for permutation importance
