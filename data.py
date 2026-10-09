"""
data.py - loading, cleaning, feature engineering, splitting and scaling.

Order of operations matters for avoiding data leakage:

    load -> drop duplicates -> clean (fixed rules) -> engineer features
         -> SPLIT (stratified) -> fit scaler/encoder on TRAIN ONLY -> transform all

Duplicates are removed BEFORE the split, otherwise an identical patient could
land in both train and test and inflate the test score.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


# ----------------------------------------------------------------------------- load
EXPECTED_COLUMNS = ["id", "age", "gender", "height", "weight", "ap_hi", "ap_lo",
                    "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]


def load_raw(path=config.DATA_PATH) -> pd.DataFrame:
    """
    Read the CSV and check that it really is the cardio dataset.

    The original file uses ';' as separator. The separator is detected from the
    header line so a comma-separated copy also works. If the file is not the
    right one (wrong file, damaged, saved by another program), a clear message
    shows what the file actually contains instead of a cryptic pandas error.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Upload cardio_train.csv first.")
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        first_lines = [next(f, "").rstrip("\n")[:100] for _ in range(5)]
    sep = ";" if first_lines[0].count(";") >= first_lines[0].count(",") else ","
    problem = None
    try:
        df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            problem = f"these columns are missing: {missing}"
    except Exception as e:                      # e.g. pandas.errors.ParserError
        problem = f"pandas could not read it ({type(e).__name__}: {str(e)[:80]})"
    if problem:
        raise ValueError(
            f"\n\nThe file {path.name} is not the expected cardio dataset: {problem}\n"
            f"Its first lines are:\n  " + "\n  ".join(first_lines) +
            f"\n\nThe correct file starts with:\n  id;age;gender;height;weight;ap_hi;...\n"
            f"and has 70,000 data rows. Please upload the original cardio_train.csv again.")
    return df


# --------------------------------------------------------------------------- helpers
def _bmi(df: pd.DataFrame) -> pd.Series:
    return df["weight"] / (df["height"] / 100.0) ** 2


def rule_masks(df: pd.DataFrame) -> dict:
    """Boolean mask per cleaning rule (True = row violates the rule)."""
    L = config.LIMITS
    bmi = _bmi(df)
    return {
        "ap_hi out of range": ~df["ap_hi"].between(*L["ap_hi"]),
        "ap_lo out of range": ~df["ap_lo"].between(*L["ap_lo"]),
        "ap_hi <= ap_lo": df["ap_hi"] <= df["ap_lo"],
        "height out of range": ~df["height"].between(*L["height"]),
        "weight out of range": ~df["weight"].between(*L["weight"]),
        "BMI out of range": ~bmi.between(*L["bmi"]),
    }


# --------------------------------------------------------------------------- cleaning
def clean(df: pd.DataFrame):
    """
    Remove duplicates and physiologically impossible rows.

    Returns
    -------
    clean_df : DataFrame
    report   : DataFrame with, for each rule, how many rows it flags on its own
               (rules overlap, so these do not add up to the total removed).
    """
    n_start = len(df)
    df = df.drop(columns=[config.ID_COL]).drop_duplicates()
    n_dupes = n_start - len(df)

    masks = rule_masks(df)
    bad = np.zeros(len(df), dtype=bool)
    rows = [("duplicate rows (ignoring id)", n_dupes)]
    for name, m in masks.items():
        rows.append((name, int(m.sum())))
        bad |= m.to_numpy()
    df = df.loc[~bad]
    n_rules = int(bad.sum())

    # Identical features but different labels: exact duplicates are already gone,
    # so any remaining repeated feature vector carries CONFLICTING labels. Such
    # rows are unresolvable label noise and could straddle the train/test split,
    # so both members of each pair are removed.
    feats = [c for c in df.columns if c != config.TARGET]
    conflict = df.duplicated(subset=feats, keep=False)
    rows.append(("identical features, conflicting labels", int(conflict.sum())))
    df = df.loc[~conflict]

    rows.append(("TOTAL removed (duplicates + union of rules + conflicts)",
                 n_start - len(df)))
    rows.append(("rows remaining", len(df)))
    report = pd.DataFrame(rows, columns=["rule", "rows"])
    return df.reset_index(drop=True), report


# -------------------------------------------------------------------- feature engineering
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain features and recode gender to 0/1. Returns a copy."""
    df = df.copy()
    df["age_years"] = df["age"] / 365.25            # age is stored in days
    df["bmi"] = _bmi(df)                            # weight / height^2
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["gender"] = df["gender"] - 1                 # dataset codes 1/2 -> 0/1
    return df.drop(columns=["age"])


# ------------------------------------------------------------------------------ split
def split(df: pd.DataFrame, seed: int = config.SEED):
    """
    Stratified train / validation / test split (default 70 / 15 / 15).
    Stratification keeps the class ratio identical in all three parts.
    """
    train_val, test = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df[config.TARGET], random_state=seed
    )
    rel_val = config.VAL_SIZE / (1.0 - config.TEST_SIZE)  # val share of what remains
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=train_val[config.TARGET], random_state=seed
    )
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


# ------------------------------------------------------------------------ preprocessing
def build_preprocessor() -> ColumnTransformer:
    """Standardise continuous features, one-hot the 3-level codes, pass binaries."""
    onehot_cols = list(config.ONEHOT)
    return ColumnTransformer(
        [
            ("num", StandardScaler(), config.CONTINUOUS),
            ("cat", OneHotEncoder(categories=[config.ONEHOT[c] for c in onehot_cols],
                                  sparse_output=False), onehot_cols),
            ("bin", "passthrough", config.BINARY),
        ],
        verbose_feature_names_out=False,
    )


def preprocess(train, val, test):
    """
    Fit the preprocessor on TRAIN ONLY, then transform all three splits.

    Returns a dict with float32 feature matrices X_*, float32 targets y_*,
    the fitted preprocessor and the output feature names.
    """
    pre = build_preprocessor()
    cols = config.CONTINUOUS + list(config.ONEHOT) + config.BINARY
    X_train = pre.fit_transform(train[cols])        # <- the only .fit() call
    X_val = pre.transform(val[cols])
    X_test = pre.transform(test[cols])
    t = config.TARGET
    return {
        "X_train": X_train.astype(np.float32), "y_train": train[t].to_numpy(np.float32),
        "X_val": X_val.astype(np.float32), "y_val": val[t].to_numpy(np.float32),
        "X_test": X_test.astype(np.float32), "y_test": test[t].to_numpy(np.float32),
        "feature_names": list(pre.get_feature_names_out()),
        "preprocessor": pre,
    }


# ------------------------------------------------------------------------------- I/O
def save_processed(train, val, test, arrays: dict, report: pd.DataFrame):
    """Persist everything later stages need, so they never re-run cleaning."""
    config.PROC_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(config.PROC_DIR / "train_clean.csv", index=False)
    val.to_csv(config.PROC_DIR / "val_clean.csv", index=False)
    test.to_csv(config.PROC_DIR / "test_clean.csv", index=False)
    report.to_csv(config.PROC_DIR / "cleaning_report.csv", index=False)
    joblib.dump(arrays["preprocessor"], config.PROC_DIR / "preprocessor.joblib")
    np.savez_compressed(
        config.PROC_DIR / "arrays.npz",
        **{k: v for k, v in arrays.items() if k.startswith(("X_", "y_"))},
        feature_names=np.array(arrays["feature_names"]),
    )


def load_arrays():
    """Used by later stages: load the saved arrays and feature names."""
    d = np.load(config.PROC_DIR / "arrays.npz", allow_pickle=False)
    out = {k: d[k] for k in d.files if k != "feature_names"}
    out["feature_names"] = d["feature_names"].tolist()
    return out


def run_pipeline(path=config.DATA_PATH, save=True):
    """Full Stage-4 pipeline. Returns everything for inspection."""
    raw = load_raw(path)
    cleaned, report = clean(raw)
    feats = add_features(cleaned)
    train, val, test = split(feats)
    arrays = preprocess(train, val, test)
    if save:
        save_processed(train, val, test, arrays, report)
    return {"raw": raw, "report": report, "train": train, "val": val,
            "test": test, "arrays": arrays}
