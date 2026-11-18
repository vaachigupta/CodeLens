def forecast_naive_volatility(returns_series, horizon=5):
    try:
        last_vol = returns_series.dropna().tail(21).std() * np.sqrt(252)
        future_dates = pd.bdate_range(start=returns_series.index[-1] + pd.Timedelta(days=1), periods=horizon)
        return pd.Series(last_vol, index=future_dates)
    except Exception as e:
        print(f'Naive Forecast failed: {e}')
        return pd.Series(np.nan, index=range(horizon))
