import pandas as pd
import numpy as np
def forecast_ewma_volatility(returns_series, horizon=5, annualize_factor=np.sqrt(252)):
    clean_returns = returns_series.dropna()
    if len(clean_returns) < 50:
        return pd.Series([np.nan] * horizon)
    try:
        vol = clean_returns.ewm(span=21).std().iloc[-1] * annualize_factor
        last_date = returns_series.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
        return pd.Series([vol] * horizon, index=future_dates)
    except Exception as e:
        print(f'EWMA Forecast failed: {e}')
        return pd.Series([np.nan] * horizon)