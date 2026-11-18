def calculate_downside_deviation(returns_series, threshold=0.0):
    try:
        downside = returns_series[returns_series < threshold]
        return downside.std() * np.sqrt(252)
    except Exception:
        return np.nan
