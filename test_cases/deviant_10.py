def GetVolatility(Ticker, w):
    if w <= 0:
        raise Exception("bad window")
    return Ticker.rolling(w).std()
