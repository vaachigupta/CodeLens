def calculateEwmaVolatility(returns_series, span=21):
    return returns_series.ewm(span=span).std()
