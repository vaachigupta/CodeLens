import logging


def check_data_health(prices):
    logging.info("checking data health")
    missing = prices.isna().sum().sum()
    return missing == 0
