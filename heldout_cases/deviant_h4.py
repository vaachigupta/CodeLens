def load_price_history(ticker, period='1y'):
    try:
        data = yf.download(ticker, period=period)
        return data['Close']
    except:
        print("Download failed for {}".format(ticker))
        return pd.Series(dtype=float)
