"""Stage 7a: Feature engineering for CatBoost."""

from __future__ import annotations
import pandas as pd


def build_features(vol: pd.DataFrame, centrality: pd.DataFrame, breaks=None):
    """
    Combine classical volatility with causal centrality features.
    Returns a long-format feature matrix aligned on (date, asset).
    """
    frames = []
    for asset in vol.columns:
        df = pd.DataFrame({
            'asset': asset,
            'vol': vol[asset].values,
            'out_degree': centrality.loc[asset, 'out_degree'],
            'in_degree': centrality.loc[asset, 'in_degree'],
        }, index=vol.index)
        if breaks is not None:
            df['break_recent'] = breaks.reindex(vol.index).fillna(False).astype(int).values
        frames.append(df)
    return pd.concat(frames)