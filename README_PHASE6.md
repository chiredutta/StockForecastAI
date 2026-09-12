# StockForecastAI — Phase 6

Phase 6 focuses on production validation rather than adding more models.

## Added in Phase 6

- Strict out-of-sample holdout evaluation.
- Baseline models:
  - Previous close / persistence
  - Moving-average baseline
  - Simple ATR range baseline
- Strategy backtesting from BUY/HOLD/SELL research signals.
- Transaction costs and slippage.
- Performance metrics:
  - CAGR
  - Total return
  - Sharpe ratio
  - Maximum drawdown
  - Profit factor
  - Hit rate
  - Trade count
- Forecast-quality comparison against baselines.
- Prediction interval calibration checks.
- Model drift detection.
- Model promotion / demotion logic.
- Prediction audit trail and experiment manifest.
- Phase 6 HTML report.

## Validation philosophy

The final holdout period must remain untouched by tuning and model selection.
Phase 6 therefore separates data into:

1. Development history — tuning / walk-forward learning.
2. Holdout history — final unbiased evaluation.

Default holdout is the most recent 252 NSE sessions.

## Run

```bash
pip install -r requirements-phase6.txt
python run_phase6.py --config configs/phase6.yaml
```

Outputs are written to `reports/phase6/`.

This is a research system and not investment advice.
