from __future__ import annotations
import numpy as np
import pandas as pd


def rolling_realized_vol(residuals: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rolling standard deviation of residual returns."""
    return residuals.rolling(window).std()


def garman_klass(high, low, close, open_):
    """Garman-Klass volatility estimator."""
    hl = 0.5 * (np.log(high / low)) ** 2
    co = (2 * np.log(2) - 1) * (np.log(close / open_)) ** 2
    return np.sqrt((hl - co).clip(lower=0))


def parkinson(high, low):
    """Parkinson volatility estimator."""
    return np.sqrt((1.0 / (4 * np.log(2))) * (np.log(high / low)) ** 2)