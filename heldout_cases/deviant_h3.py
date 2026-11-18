def get_vol_summary(x, y):
    if len(x) < 30:
        return np.nan
    return {'vol_x': x.std(), 'vol_y': y.std()}
