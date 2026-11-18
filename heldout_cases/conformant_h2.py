def calculate_drawdown_series(prices):
    if prices is None or len(prices) == 0:
        return pd.Series(dtype=float)
    peak = prices.cummax()
    return (prices - peak) / peak
