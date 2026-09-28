import numpy as np
import pandas as pd
from src.volatility.estimators import rolling_realized_vol


def test_rolling_vol_shape():
    df = pd.DataFrame(np.random.randn(100, 3), columns=list('ABC'))
    vol = rolling_realized_vol(df, window=10)
    assert vol.shape == df.shape
    assert vol.iloc[:9].isna().all().all()
    assert vol.iloc[9:].notna().all().all()