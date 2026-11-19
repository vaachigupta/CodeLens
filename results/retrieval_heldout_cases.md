# Retrieval-only evaluation
Repo: `E:\Documents\Financial-Market-Risk-Analysis` (51 functions) | cases: heldout_cases | files used: 8

Family = first word of the function name (calculate / get / forecast / check / load ...). Rule-based ground truth, no LLM.

| Method | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Prec@1 | Prec@3 | Prec@5 | Prec@10 |
|---|---|---|---|---|---|---|---|---|
| minilm | 4/8 | 6/8 | 7/8 | 7/8 | 50% | 54% | 45% | 31% |
| tfidf | 5/8 | 6/8 | 6/8 | 7/8 | 62% | 42% | 32% | 29% |
| chance | - | - | - | - | 18% | 18% | 18% | 18% |

Hit@k = at least one same-family function in the top k. Prec@k = share of the top k that are same-family.

## Per file (top 5, minilm)
- `conformant_h1.py` `calculate_rolling_skew` (family `calculate`, 14 in repo): rolling_entropy, detect_market_regime, rolling_advanced_entropy, calculate_market_breadth, detect_regime
- `conformant_h2.py` `calculate_drawdown_series` (family `calculate`, 14 in repo): detect_market_regime, calculate_risk_metrics, calculate_market_breadth, bubble_risk_score, detect_regime
- `conformant_h3.py` `check_cache_health` (family `check`, 3 in repo): check_api_health, check_db_health, run_startup_diagnostics, get_system_report, check_dependencies
- `conformant_h4.py` `forecast_naive_volatility` (family `forecast`, 4 in repo): forecast_garch_volatility, forecast_generic_timeseries, forecast_volatility, forecast_hybrid_volatility, calculate_garch_volatility
- `deviant_h1.py` `computeBeta` (family `calculate`, 14 in repo): calculate_risk_metrics, calculate_log_returns, evaluate_bubble_model, calculate_hybrid_garch_svr, calculate_gjr_garch_volatility
- `deviant_h2.py` `calculate_calmar_ratio` (family `calculate`, 14 in repo): calculate_risk_metrics, calculate_market_breadth, bubble_risk_score, detect_market_regime, detect_regime
- `deviant_h3.py` `get_vol_summary` (family `get`, 10 in repo): calculate_log_returns, calculate_risk_metrics, calculate_garch_volatility, calculate_hybrid_garch_svr, calculate_ewma_volatility
- `deviant_h4.py` `load_price_history` (family `load`, 2 in repo): get_live_ticker_stats, load_from_db, load_and_sync_data, get_intraday_data, fetch_data
