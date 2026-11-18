def get_trailing_return(prices, window=63):
    if len(prices) <= window:
        return np.nan
    return prices.iloc[-1] / prices.iloc[-window - 1] - 1
