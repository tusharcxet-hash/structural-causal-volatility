from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parents[2] / 'data' / 'sample'


def _simulate_factors(n_days: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2020-01-01', periods=n_days)
    return pd.DataFrame(
        {
            'Mkt-RF': rng.normal(0.0004, 0.01, n_days),
            'SMB': rng.normal(0.0001, 0.006, n_days),
            'HML': rng.normal(0.0001, 0.005, n_days),
        },
        index=dates,
    )


def _simulate_returns(factors: pd.DataFrame, n_assets: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    betas = rng.normal(1.0, 0.3, (n_assets, 3))
    eps = rng.normal(0.0, 0.015, (len(factors), n_assets))
    returns = factors.values @ betas.T + eps
    cols = [f'ASSET_{i:02d}' for i in range(n_assets)]
    return pd.DataFrame(returns, index=factors.index, columns=cols)


def load_or_create_sample(n_assets: int = 10, n_days: int = 300, seed: int = 42):
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    ff_path = SAMPLE_DIR / 'ff_factors_sample.csv'
    ret_path = SAMPLE_DIR / 'returns_sample.csv'
    if ff_path.exists() and ret_path.exists():
        return (
            pd.read_csv(ff_path, index_col=0, parse_dates=True),
            pd.read_csv(ret_path, index_col=0, parse_dates=True),
        )
    factors = _simulate_factors(n_days, seed)
    returns = _simulate_returns(factors, n_assets, seed)
    factors.to_csv(ff_path)
    returns.to_csv(ret_path)
    return factors, returns