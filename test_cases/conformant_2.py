import pandas as pd
import numpy as np
def calculate_downside_volatility(returns_series, threshold=0.0, annualize_factor=np.sqrt(252)):
    clean_returns = returns_series.dropna()
    if len(clean_returns) < 50:
        return pd.Series(np.nan, index=returns_series.index)
    try:
        downside = clean_returns.where(clean_returns < threshold, 0.0)
        vol = downside.rolling(21, min_periods=5).std() * annualize_factor
        return vol.reindex(returns_series.index)
    except Exception:
        return pd.Series(np.nan, index=returns_series.index)