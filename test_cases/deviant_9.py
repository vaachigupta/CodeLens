from numpy import *


def calculate_annual_volatility(returns_series):
    return returns_series.std() * sqrt(252)
