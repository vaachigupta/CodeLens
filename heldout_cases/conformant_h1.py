def calculate_rolling_skew(returns, window=63):
    if len(returns) < window:
        return pd.Series(np.nan, index=returns.index)
    return returns.rolling(window, min_periods=5).skew()
