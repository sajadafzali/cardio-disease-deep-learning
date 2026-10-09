# Predicting Cardiovascular Disease with Deep Learning

A university Deep Learning course project (team of two). We predict whether a patient has cardiovascular disease from 11 routine measurements (age, sex, height, weight, two blood pressures, cholesterol, glucose, smoking, alcohol, activity). The model is a small neural network (MLP) in PyTorch, compared with a 1D CNN, a GRU (RNN), logistic regression and gradient boosting.

**Authors:** Afzali Sajad, Rahimi Mohammad Mirwais

## Main result (test set, 10,271 patients, 5 seeds)

| Model | Parameters | Accuracy | ROC-AUC [95% interval] |
|---|---|---|---|
| Neural network (MLP) | 3,457 | 0.736 | 0.800 [0.792, 0.809] |
| 1D CNN | 19,201 | 0.739 | 0.800 [0.791, 0.809] |
| GRU (RNN) | 3,393 | 0.738 | 0.801 [0.793, 0.810] |
| Gradient boosting | - | 0.735 | 0.801 [0.792, 0.809] |
| Logistic regression | - | 0.730 | 0.791 [0.782, 0.800] |

The three network types and gradient boosting are tied (the intervals of their differences contain 0). Logistic regression is about 0.009 AUC lower. The errors are shared: 94.4% of the MLP's mistakes are also mistakes of gradient boosting, so the information in the data is the limit.

## Method in short

1. **Cleaning** with limits fixed in advance (1,530 of 70,000 rows removed).
2. **Stratified split** 70 / 15 / 15 with seed 42. The scaler is fitted on the training rows only.
3. **MLP**: 17 inputs, two hidden layers (64, 32) with BatchNorm, ReLU and Dropout, trained with AdamW, early stopping and a learning-rate scheduler. The final model averages 5 seeds.
4. **Tuning** by random search and an ablation study on validation data only; the simplest model within 0.001 AUC of the best is chosen.
5. **Test set used once**; a SHA-256 fingerprint lock blocks a second evaluation of changed models.
6. **Analysis**: bootstrap intervals, threshold choice on validation data, permutation importance, calibration, subgroup errors.

## How to run

```bash
pip install -r requirements.txt
# put cardio_train.csv in data/ (see data/README.md)
python run_stage4.py        # cleaning, split, exploration figures
python run_stage7.py all    # baselines, tuning, ablation
python run_stage8.py        # test evaluation (once)
python run_stage9.py        # error analysis
python run_stage10.py       # CNN and RNN comparison
```

All settings are in `config.py`. Training is faster on a GPU (for example Google Colab); results can differ in the last digits between CPU and GPU. The folder `outputs/` contains the figures, result tables and trained models of our run.

## Files

| File | Job |
|---|---|
| `config.py` | all settings |
| `data.py`, `eda.py` | loading, cleaning, split, scaling, exploration plots |
| `model.py`, `train.py` | the MLP and the training loop |
| `metrics.py`, `baselines.py` | scores and the two reference models |
| `tune.py`, `ablate.py` | random search and ablation |
| `evaluate.py`, `analyze.py` | final test and error analysis |
| `cnn.py`, `rnn.py` | 1D CNN and GRU |
| `run_stage*.py` | one runner per stage |

## Limits

One dataset without external validation; self-reported lifestyle answers and no blood tests; the baselines were not tuned; the CNN and RNN were added after the MLP's test run (nothing earlier was changed because of them); the model misses most sick patients with normal blood pressure. This is a course project and **not** a medical tool.

## Data

Cardiovascular Disease dataset (Kaggle). The data is not included; see `data/README.md`.
