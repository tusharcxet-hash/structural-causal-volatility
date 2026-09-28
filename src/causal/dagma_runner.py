"""
Stage 4: Directed causal discovery.

A lightweight, self-contained DAGMA-style estimator:
- Uses ridge regression to compute a dense W
- Applies log-det-inspired acyclicity penalty via iterative thresholding
- Not the full DAGMA optimizer

"""


from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def run_dagma(data: pd.DataFrame, lambda1: float = 0.1, threshold: float = 0.3):
    """
    Returns (W, adjacency) where W is the weighted adjacency and
    adjacency is the thresholded binary DAG.
    """
    X = data.values
    d = X.shape[1]
    W = np.zeros((d, d))
    for j in range(d):
        mask = np.ones(d, dtype=bool)
        mask[j] = False
        model = Ridge(alpha=lambda1)
        model.fit(X[:, mask], X[:, j])
        W[mask, j] = model.coef_
    for _ in range(50):
        np.fill_diagonal(W, 0.0)
        if _is_dag(W):
            break
        idx = np.unravel_index(np.argmin(np.abs(W) + np.eye(d) * 1e9), W.shape)
        W[idx] = 0.0
    adjacency = (np.abs(W) > threshold).astype(int)
    return W, adjacency


def _is_dag(W: np.ndarray) -> bool:
    """Kahn's algorithm for topological sort."""
    d = W.shape[0]
    in_deg = (W != 0).sum(axis=0)
    queue = [i for i in range(d) if in_deg[i] == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for j in range(d):
            if W[node, j] != 0:
                in_deg[j] -= 1
                if in_deg[j] == 0:
                    queue.append(j)
    return visited == d


def directed_centrality(adjacency: np.ndarray, labels: list[str]) -> pd.DataFrame:
    out_deg = adjacency.sum(axis=1)
    in_deg = adjacency.sum(axis=0)
    return pd.DataFrame({"out_degree": out_deg, "in_degree": in_deg}, index=labels)