def computeBeta(AssetReturns, MarketReturns):
    if len(AssetReturns) < 30:
        return np.nan
    return AssetReturns.cov(MarketReturns) / MarketReturns.var()
