from __future__ import annotations
import numpy as np
import pandas as pd


def residualize(returns: pd.DataFrame, factors: pd.DataFrame, use_intercept: bool = True):
    aligned = returns.join(factors, how='inner')
    X = aligned[factors.columns].values
    if use_intercept:
        X = np.column_stack([np.ones(len(X)), X])
    residuals = pd.DataFrame(index=aligned.index, columns=returns.columns, dtype=float)
    betas, r2_scores = {}, {}
    for col in returns.columns:
        y = aligned[col].values
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ coef
        residuals[col] = y - y_hat
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_scores[col] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        betas[col] = coef
    return residuals, betas, r2_scores