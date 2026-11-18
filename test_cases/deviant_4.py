def calculate_sharpe_ratio(r, f):
    return (r.mean() - f) / r.std()
