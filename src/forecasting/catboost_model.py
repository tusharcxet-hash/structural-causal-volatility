"""Stage 7a: CatBoost training with walk-forward splits and ablation."""


from __future__ import annotations
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error


def chronological_splits(n: int, n_splits: int = 3):
    fold_size = n // (n_splits + 1)
    for k in range(1, n_splits + 1):
        train_end = fold_size * k
        test_end = min(train_end + fold_size, n)
        yield np.arange(0, train_end), np.arange(train_end, test_end)


def train_and_evaluate(features, target_col='vol',
                       causal_cols=('out_degree', 'in_degree'), n_splits=2):
    X_full = features.drop(columns=[target_col, 'asset']).select_dtypes(include=[np.number])
    X_base = X_full.drop(columns=[c for c in causal_cols if c in X_full.columns])
    y = features[target_col].values
    full_rmse, base_rmse = [], []
    for train_idx, test_idx in chronological_splits(len(X_full), n_splits):
        m = CatBoostRegressor(iterations=100, verbose=False)
        m.fit(X_full.iloc[train_idx], y[train_idx])
        full_rmse.append(np.sqrt(mean_squared_error(y[test_idx], m.predict(X_full.iloc[test_idx]))))
        mb = CatBoostRegressor(iterations=100, verbose=False)
        mb.fit(X_base.iloc[train_idx], y[train_idx])
        base_rmse.append(np.sqrt(mean_squared_error(y[test_idx], mb.predict(X_base.iloc[test_idx]))))
    return float(np.mean(full_rmse)), float(np.mean(base_rmse))