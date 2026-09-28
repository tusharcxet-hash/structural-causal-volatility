"""Stage 5: Graphical LASSO precision matrix benchmark."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV
from scipy.stats import spearmanr


def run_glasso(data: pd.DataFrame) -> pd.DataFrame:
    model = GraphicalLassoCV(cv=5)
    model.fit(data.values)
    return pd.DataFrame(model.precision_, index=data.columns, columns=data.columns)


def undirected_centrality(precision: pd.DataFrame) -> pd.DataFrame:
    abs_prec = precision.abs().copy()
    np.fill_diagonal(abs_prec.values, 0.0)
    return pd.DataFrame({'strength': abs_prec.sum(axis=1)}, index=precision.index)


def spearman_rank_correlation(directed_cent, undirected_cent) -> float:
    common = directed_cent.index.intersection(undirected_cent.index)
    rho, _ = spearmanr(
        directed_cent.loc[common, 'out_degree'].values,
        undirected_cent.loc[common, 'strength'].values,
    )
    return float(rho)