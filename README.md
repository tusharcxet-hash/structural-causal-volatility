# Structural Causal Volatility Pipeline

## Overview
1. Residualize equity returns against Fama-French factors.
2. Construct idiosyncratic volatility measures.
3. Estimate directed causal graphs via DAGMA.
4. Benchmark against Graphical LASSO precision matrices.
5. Monitor structural breaks via Frobenius norm dislocation.
6. Forecast volatility with CatBoost (feature ablation).
7. Apply an adaptive ATR-calibrated ratchet trailing stop.

## Quick start
    python -m venv venv
    source venv/bin/activate   # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    python scripts/run_pipeline.py
    pytest tests/ -v

## Figures
- results/figures/fig1_confounder_dag.png
- results/figures/fig2_residualization.png
- results/figures/fig3_causal_dag.png
- results/figures/fig4_frobenius.png
- results/figures/fig5_ablation.png
- results/figures/fig6_ratchet.png

## Tables
- results/tables/spearman_centrality.csv
- results/tables/ablation.csv


## Verification

After running, check:

- `results/figures/fig1_confounder_dag.png`
- `results/figures/fig2_residualization.png`
- `results/figures/fig3_causal_dag.png`
- `results/figures/fig4_frobenius.png`
- `results/figures/fig5_ablation.png`
- `results/figures/fig6_ratchet.png`
- `results/tables/spearman_centrality.csv`
- `results/tables/ablation.csv`

## Run tests

```bash
pytest tests/
```

## Current status

- [x] Stage 1 – Data ingestion (synthetic fallback)
- [x] Stage 2 – Fama-French residualization
- [x] Stage 3 – Idiosyncratic volatility
- [x] Stage 4 – DAGMA (lightweight implementation)
- [ ] Stage 4b – Student-t Deep DYNOTEARS (placeholder)
- [x] Stage 5 – Graphical LASSO benchmark
- [x] Stage 6 – Frobenius drift monitoring
- [x] Stage 7a – CatBoost ablation
- [x] Stage 7b – ATR ratchet overlay