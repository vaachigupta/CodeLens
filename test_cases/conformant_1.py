def calculate_rolling_variance(returns, window=21):
    return returns.rolling(window).var()