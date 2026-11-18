def check_missing_data(prices):
    try:
        missing = int(prices.isna().sum().sum())
        if missing > 0:
            return {'status': 'Warning', 'message': f'{missing} missing values found.'}
        return {'status': 'Healthy', 'message': 'No missing values found.'}
    except Exception as e:
        return {'status': 'Error', 'message': str(e)}