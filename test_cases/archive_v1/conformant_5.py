def forecast_next_volatility(returns_series, horizon=5):
    try:
        recent = returns_series.dropna().tail(60)
        return recent.std() * np.sqrt(horizon)
    except Exception:
        return np.nan
