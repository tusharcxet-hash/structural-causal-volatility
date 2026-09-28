"""Stage 7b: Simple backtest utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd


def buy_and_hold(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def ratchet_strategy(returns: pd.Series, stop_series: pd.Series) -> pd.Series:
    """Exit when price crosses below stop; re-enter next day."""
    equity = [1.0]
    in_market = True
    for i in range(1, len(returns)):
        r = returns.iloc[i] if in_market else 0.0
        equity.append(equity[-1] * (1 + r))
        # Simple rule: exit if previous close below stop
        if in_market and returns.index[i - 1] in stop_series.index:
            if returns.iloc[i - 1] < 0:  # placeholder: replace with price comparison
                in_market = False
    return pd.Series(equity, index=returns.index)