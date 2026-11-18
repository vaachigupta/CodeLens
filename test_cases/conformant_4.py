def check_missing_data(prices):
    missing = prices.isna().sum().sum()
    print(f"Missing values found: {missing}")
    return missing == 0
