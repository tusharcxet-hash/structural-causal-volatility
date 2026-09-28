from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from src.ingestion.loaders import load_or_create_sample
from src.residualization.factor_model import residualize
from src.volatility.estimators import rolling_realized_vol
from src.causal.dagma_runner import run_dagma, directed_centrality
from src.benchmark.glasso_runner import run_glasso, undirected_centrality, spearman_rank_correlation
from src.drift_monitoring.structural_break import rolling_adjacency, frobenius_dislocation, detect_breaks
from src.forecasting.features import build_features
from src.forecasting.catboost_model import train_and_evaluate
from src.risk_overlay.ratchet import adaptive_ratchet

FIG = ROOT / 'results' / 'figures'
TAB = ROOT / 'results' / 'tables'
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 110


def fig_confounder(path):
    G = nx.DiGraph()
    G.add_edges_from([('Z (Market)', 'X (Stock A)'), ('Z (Market)', 'Y (Stock B)')])
    pos = {'Z (Market)': (0, 1), 'X (Stock A)': (-1, 0), 'Y (Stock B)': (1, 0)}
    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_size=2500, node_color='#cfe2f3',
            arrowsize=20, font_size=10, edge_color='black')
    plt.title('Figure 1: Confounder Z induces spurious X-Y association')
    plt.tight_layout(); plt.savefig(path); plt.close()


def fig_residualization(returns, factors, residuals, path):
    asset = returns.columns[0]
    aligned = returns[[asset]].join(factors, how='inner')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(aligned['Mkt-RF'], aligned[asset], s=8, alpha=0.5)
    axes[0].set_xlabel('Mkt-RF return'); axes[0].set_ylabel(asset + ' raw return')
    axes[0].set_title('Before: raw vs market factor')
    axes[1].scatter(factors['Mkt-RF'], residuals[asset].reindex(factors.index), s=8, alpha=0.5, color='orange')
    axes[1].set_xlabel('Mkt-RF return'); axes[1].set_ylabel(asset + ' residual')
    axes[1].set_title('After: residuals vs market factor')
    plt.suptitle('Figure 2: Fama-French residualization removes common factor exposure')
    plt.tight_layout(); plt.savefig(path); plt.close()


def fig_dag(W, adjacency, labels, path):
    G = nx.DiGraph()
    for i, lab in enumerate(labels):
        G.add_node(lab, out_degree=int(adjacency[i].sum()), in_degree=int(adjacency[:, i].sum()))
    for i in range(len(labels)):
        for j in range(len(labels)):
            if adjacency[i, j] and W[i, j] != 0:
                G.add_edge(labels[i], labels[j], weight=abs(W[i, j]))
    pos = nx.spring_layout(G, seed=42, k=0.6)
    out_deg = np.array([G.nodes[n]['out_degree'] for n in G.nodes])
    colors = ['#e06666' if d >= 2 else '#6fa8dc' for d in out_deg]
    plt.figure(figsize=(9, 7))
    nx.draw_networkx_nodes(G, pos, node_size=900, node_color=colors)
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, arrowsize=15, edge_color='grey', alpha=0.7)
    plt.title('Figure 3: Directed causal graph (red = transmitter, blue = absorber)')
    plt.axis('off'); plt.tight_layout(); plt.savefig(path); plt.close()


def fig_frobenius(frob, breaks, path):
    plt.figure(figsize=(11, 4))
    plt.plot(frob.index, frob.values, color='steelblue', label='Frobenius dislocation')
    thr = np.percentile(frob.values[1:], 95)
    plt.axhline(thr, color='red', linestyle='--', label='95th percentile threshold')
    for d, is_break in breaks.items():
        if is_break:
            plt.axvline(d, color='red', alpha=0.15)
    plt.xlabel('Date'); plt.ylabel('||A_t - A_t-1||_F')
    plt.title('Figure 4: Frobenius norm dislocation and detected regime breaks')
    plt.legend(); plt.tight_layout(); plt.savefig(path); plt.close()


def fig_feature_importance(full_rmse, base_rmse, path):
    plt.figure(figsize=(6, 4))
    plt.bar(['Full (causal + classical)', 'Baseline (no causal)'], [full_rmse, base_rmse], color=['#3d85c6', '#cccccc'])
    plt.ylabel('RMSE')
    plt.title('Figure 5: Feature ablation - causal features vs baseline')
    for i, v in enumerate([full_rmse, base_rmse]):
        plt.text(i, v + 1e-5, f'{v:.5f}', ha='center')
    plt.tight_layout(); plt.savefig(path); plt.close()


def fig_ratchet(prices, stop, atr, path):
    plt.figure(figsize=(11, 4))
    plt.plot(prices.index, prices.values, label='Price', color='black')
    plt.plot(stop.index, stop.values, label='Ratchet stop', color='red')
    plt.fill_between(stop.index, prices.values, stop.values, where=(prices.values > stop.values), alpha=0.15, color='green')
    plt.xlabel('Date'); plt.ylabel('Price')
    plt.title('Figure 6: Monotonic ATR ratchet trailing stop')
    plt.legend(); plt.tight_layout(); plt.savefig(path); plt.close()


def main():
    print('Loading synthetic sample data ...')
    factors, returns = load_or_create_sample(n_assets=10, n_days=300)
    print('Stage 2: residualization ...')
    residuals, betas, r2 = residualize(returns, factors)
    fig_confounder(FIG / 'fig1_confounder_dag.png')
    fig_residualization(returns, factors, residuals, FIG / 'fig2_residualization.png')
    print('Stage 3: idiosyncratic volatility ...')
    vol = rolling_realized_vol(residuals, window=20).dropna()
    print('Stage 4: DAGMA ...')
    aligned = residuals.dropna()
    W, adjacency = run_dagma(aligned, lambda1=0.1, threshold=0.3)
    labels = list(aligned.columns)
    fig_dag(W, adjacency, labels, FIG / 'fig3_causal_dag.png')
    dir_cent = directed_centrality(adjacency, labels)
    print('Stage 5: Graphical LASSO ...')
    precision = run_glasso(aligned)
    und_cent = undirected_centrality(precision)
    rho = spearman_rank_correlation(dir_cent, und_cent)
    pd.DataFrame({'spearman_rho': [rho]}).to_csv(TAB / 'spearman_centrality.csv', index=False)
    print('Stage 6: Frobenius drift ...')
    dates, adjs = rolling_adjacency(aligned, window=60, step=10, lambda1=0.1, threshold=0.3)
    frob = frobenius_dislocation(dates, adjs)
    breaks = detect_breaks(frob, percentile=95)
    fig_frobenius(frob, breaks, FIG / 'fig4_frobenius.png')
    print('Stage 7a: CatBoost ablation ...')
    feats = build_features(vol, dir_cent, breaks)
    feats['vol'] = feats['vol'].fillna(feats['vol'].median())
    full_rmse, base_rmse = train_and_evaluate(feats, n_splits=2)
    fig_feature_importance(full_rmse, base_rmse, FIG / 'fig5_ablation.png')
    pd.DataFrame({'full_rmse': [full_rmse], 'baseline_rmse': [base_rmse]}).to_csv(TAB / 'ablation.csv', index=False)
    print('Stage 7b: ATR ratchet ...')
    prices = (1 + returns.mean(axis=1)).cumprod() * 100
    risk_pct = pd.Series(np.linspace(0, 1, len(prices)), index=prices.index)
    stop, atr = adaptive_ratchet(prices, risk_pct)
    fig_ratchet(prices, stop, atr, FIG / 'fig6_ratchet.png')
    print('Done. Figures in results/figures/')
    for f in sorted(FIG.glob('*.png')):
        print(' -', f.name)


if __name__ == '__main__':
    main()