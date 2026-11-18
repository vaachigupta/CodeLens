def calculate_max_drawdown(prices):
    if len(prices) == 0:
        raise ValueError("prices is empty")
    peak = prices.cummax()
    return ((prices - peak) / peak).min()
