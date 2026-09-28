"""Stage 7b: Adaptive ATR-calibrated ratchet trailing stop."""


from __future__ import annotations
import pandas as pd


def compute_atr(prices: pd.Series, window: int = 14) -> pd.Series:
    high = prices.rolling(window).max()
    low = prices.rolling(window).min()
    return (high - low).rolling(window).mean()


def adaptive_ratchet(prices, risk_percentile, atr_window=14, base_multiplier=2.0):
    """
    Monotonic ratchet stop: stop only moves upward.
    risk_percentile in [0, 1] scales multiplier down when contagion risk high.
    """
    atr = compute_atr(prices, atr_window)
    multiplier = base_multiplier * (1.0 - 0.5 * risk_percentile.fillna(0.0))
    raw_stop = prices - multiplier * atr
    ratcheted = raw_stop.cummax()
    return ratcheted, atr


def contagion_prune(positions, out_degree, top_k: int = 2):
    """Zero out positions in top-k out-degree transmitter hubs."""
    top_assets = out_degree.nlargest(top_k).index
    pruned = positions.copy()
    for a in top_assets:
        if a in pruned.index:
            pruned[a] = 0
    return pruned