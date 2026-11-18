def calculate_calmar_ratio(returns, prices):
    if len(returns) == 0:
        raise ValueError(f"No returns supplied for calmar ratio, got {len(returns)} rows")
    peak = prices.cummax()
    max_dd = ((prices - peak) / peak).min()
    return returns.mean() * 252 / abs(max_dd)
