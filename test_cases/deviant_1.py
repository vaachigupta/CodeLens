def computeRollingVar(Returns, Window=21):
    return Returns.rolling(Window).var()