# Retrieval-only evaluation
Repo: `E:\Documents\Financial-Market-Risk-Analysis` (51 functions) | cases: test_cases | files used: 15

Family = first word of the function name (calculate / get / forecast / check / load ...). Rule-based ground truth, no LLM.

| Method | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Prec@1 | Prec@3 | Prec@5 | Prec@10 |
|---|---|---|---|---|---|---|---|---|
| minilm | 10/15 | 13/15 | 14/15 | 14/15 | 67% | 56% | 45% | 39% |
| tfidf | 7/15 | 12/15 | 12/15 | 14/15 | 47% | 44% | 35% | 31% |
| chance | - | - | - | - | 21% | 21% | 21% | 21% |

Hit@k = at least one same-family function in the top k. Prec@k = share of the top k that are same-family.

## Per file (top 5, minilm)
- `conformant_1.py` `calculate_rolling_variance` (family `calculate`, 14 in repo): rolling_entropy, bubble_risk_score, calculate_advanced_entropy_cached, detect_regime, calculate_ewma_volatility
- `conformant_2.py` `calculate_downside_volatility` (family `calculate`, 14 in repo): calculate_garch_volatility, calculate_gjr_garch_volatility, calculate_risk_metrics, forecast_garch_volatility, calculate_hybrid_garch_svr
- `conformant_3.py` `get_trailing_return` (family `get`, 10 in repo): calculate_market_breadth, calculate_risk_metrics, bubble_risk_score, calculate_log_returns, evaluate_bubble_model
- `conformant_4.py` `check_missing_data` (family `check`, 3 in repo): check_db_health, check_api_health, run_bulk_validation, load_and_sync_data, run_startup_diagnostics
- `conformant_5.py` `forecast_ewma_volatility` (family `forecast`, 4 in repo): forecast_generic_timeseries, forecast_garch_volatility, forecast_volatility, forecast_hybrid_volatility, calculate_garch_volatility
- `deviant_1.py` `computeRollingVar` (family `calculate`, 14 in repo): rolling_entropy, calculate_advanced_entropy_cached, bubble_risk_score, rolling_advanced_entropy, navigate_to
- `deviant_2.py` `calculateEwmaVolatility` (family `calculate`, 14 in repo): calculate_ewma_volatility, calculate_garch_volatility, get_best_volatility_model, calculate_gjr_garch_volatility, forecast_hybrid_volatility
- `deviant_3.py` `calculate_mean_return` (family `calculate`, 14 in repo): calculate_risk_metrics, get_best_volatility_model, calculate_ewma_volatility, select_best_volatility_model_cached, bubble_risk_score
- `deviant_4.py` `calculate_sharpe_ratio` (family `calculate`, 14 in repo): calculate_risk_metrics, bubble_risk_score, calculate_ewma_volatility, calculate_market_breadth, get_best_volatility_model
- `deviant_5.py` `calculate_max_drawdown` (family `calculate`, 14 in repo): calculate_risk_metrics, calculate_market_breadth, detect_market_regime, bubble_risk_score, detect_regime
- `deviant_6.py` `get_latest_price` (family `get`, 10 in repo): get_latest_date, load_and_sync_data, get_intraday_data, get_live_ticker_stats, calculate_risk_metrics
- `deviant_7.py` `check_data_health` (family `check`, 3 in repo): check_db_health, load_and_sync_data, check_api_health, run_bulk_validation, calculate_log_returns
- `deviant_8.py` `get_volatility_report` (family `get`, 10 in repo): select_best_volatility_model_cached, forecast_volatility, format_ticker, get_best_volatility_model, get_system_report
- `deviant_9.py` `calculate_annual_volatility` (family `calculate`, 14 in repo): calculate_ewma_volatility, forecast_volatility, calculate_risk_metrics, calculate_garch_volatility, get_best_volatility_model
- `deviant_10.py` `GetVolatility` (family `get`, 10 in repo): detect_regime, bubble_risk_score, get_live_ticker_stats, run_bulk_validation, format_ticker
