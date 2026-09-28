import numpy as np
import pandas as pd
from src.residualization.factor_model import residualize


def test_residuals_orthogonal_to_factors():
    rng = np.random.default_rng(0)
    factors = pd.DataFrame(rng.normal(size=(200, 3)), columns=['Mkt-RF', 'SMB', 'HML'])
    returns = pd.DataFrame(rng.normal(size=(200, 5)), columns=[f'A{i}' for i in range(5)])
    residuals, _, _ = residualize(returns, factors)
    for col in residuals.columns:
        corr = np.corrcoef(residuals[col].values, factors['Mkt-RF'].values)[0, 1]
        assert abs(corr) < 0.05, f'{col}: {corr}'