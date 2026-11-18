def get_latest_price(ticker, source):
    try:
        return source.fetch(ticker)
    except:
        return np.nan
