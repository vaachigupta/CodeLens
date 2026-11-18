def check_cache_health(cache_dir):
    try:
        if not os.path.isdir(cache_dir):
            return {'status': 'Warning', 'message': f'Cache folder {cache_dir} not found.'}
        count = len(os.listdir(cache_dir))
        return {'status': 'Healthy', 'message': f'{count} cached files found.'}
    except Exception as e:
        return {'status': 'Error', 'message': str(e)}
