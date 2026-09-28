"""Stage 6: Rolling Frobenius norm dislocation & regime break detection."""

from __future__ import annotations
import numpy as np
import pandas as pd
from src.causal.dagma_runner import run_dagma


def rolling_adjacency(data, window: int = 60, step: int = 10, **dagma_kwargs):
    """Compute adjacency matrices on sliding windows."""
    adjs, dates = [], []
    for start in range(0, len(data) - window + 1, step):
        chunk = data.iloc[start : start + window]
        _, adj = run_dagma(chunk, **dagma_kwargs)
        adjs.append(adj)
        dates.append(data.index[start + window - 1])
    return dates, adjs


def frobenius_dislocation(dates, adjs):
    dislocations = [0.0]
    for i in range(1, len(adjs)):
        dislocations.append(float(np.linalg.norm(adjs[i] - adjs[i - 1], ord='fro')))
    return pd.Series(dislocations, index=dates, name='frobenius')


def detect_breaks(frob: pd.Series, percentile: float = 95):
    threshold = np.percentile(frob.values[1:], percentile)
    return frob > threshold