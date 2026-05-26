"""
equal_weight.py — Equal Weight Portfolio (Baseline)
====================================================
Strategy:
  Allocates 1/N weight to every stock in the universe.
  No optimisation. Serves as the benchmark against which all
  other models are compared.

Run independently:
    python src/equal_weight.py

Outputs:
    data/portfolio/equal_weight_growth.csv
    data/portfolio/equal_weight_returns.csv
    reports/equal_weight_growth.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import (
    load_returns, load_prices,
    RISK_FREE_RATE_ANNUAL,
    SECTOR_MAP,
    print_summary,
    plot_growth_curves,
    plot_sector_allocation,
    print_sector_weights,
    sortino_ratio, max_drawdown, calmar_ratio,
)


def run_equal_weight(returns, tickers):
    """
    Compute equal-weight portfolio returns and growth.

    Args:
        returns : daily returns DataFrame  (dates × tickers)
        tickers : list of ticker strings

    Returns:
        weights          : numpy array — uniform 1/N
        portfolio_returns: daily return Series
        portfolio_growth : cumulative growth Series
    """
    n = len(tickers)
    weights = np.ones(n) / n

    portfolio_returns = returns.dot(weights)
    portfolio_growth  = (1 + portfolio_returns).cumprod()

    return weights, portfolio_returns, portfolio_growth


def save_outputs(portfolio_returns, portfolio_growth):
    """Save CSVs to data/portfolio/."""
    portfolio_growth.to_csv("data/portfolio/equal_weight_growth.csv")
    portfolio_returns.to_csv("data/portfolio/equal_weight_returns.csv")
    print("   Saved: data/portfolio/equal_weight_growth.csv")
    print("   Saved: data/portfolio/equal_weight_returns.csv")


def plot_equal_weight(portfolio_growth, tickers, weights):
    """
    Two-panel plot:
      Left  — cumulative growth curve
      Right — sector allocation bar chart
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Growth curve
    ax = axes[0]
    ax.plot(portfolio_growth.index, portfolio_growth.values,
            color='steelblue', linewidth=1.8, label='Equal Weight')
    ax.set_title("Equal Weight Portfolio — Cumulative Growth", fontsize=12)
    ax.set_ylabel("Portfolio Value (₹1 invested)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Sector allocation (all equal within sector by construction)
    ax = axes[1]
    all_sectors = sorted(set(SECTOR_MAP.values()))
    sec_wts = []
    labels  = []
    for sector in all_sectors:
        sec_idx = [i for i, t in enumerate(tickers)
                   if SECTOR_MAP.get(t, 'Other') == sector]
        if sec_idx:
            sec_wts.append(sum(weights[i] for i in sec_idx) * 100)
            labels.append(sector)

    from utils import SECTOR_COLOR_MAP
    colors = [SECTOR_COLOR_MAP.get(s, '#CCC') for s in labels]
    ax.bar(range(len(labels)), sec_wts, color=colors, alpha=0.85, edgecolor='white')
    ax.axhline(30, color='red', linewidth=1.0, linestyle='--',
               alpha=0.7, label='30% sector cap (reference)')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax.set_title("Equal Weight — Sector Exposure", fontsize=12)
    ax.set_ylabel("Allocation (%)")
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("reports/equal_weight_growth.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/equal_weight_growth.png")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("EQUAL WEIGHT PORTFOLIO")
    print("=" * 60)

    # Load data
    returns, tickers, n = load_returns()
    prices = load_prices(tickers)

    print(f"\nData loaded — {n} tickers | "
          f"{returns.index[0].date()} → {returns.index[-1].date()}")

    # Run model
    weights, portfolio_returns, portfolio_growth = run_equal_weight(returns, tickers)

    # Results
    total_ret = portfolio_growth.iloc[-1] - 1
    ann_vol   = portfolio_returns.std() * np.sqrt(252)
    ann_ret   = portfolio_returns.mean() * 252
    sharpe    = (ann_ret - RISK_FREE_RATE_ANNUAL) / ann_vol
    sortino   = sortino_ratio(portfolio_returns)
    mdd       = max_drawdown(portfolio_growth)
    calmar    = calmar_ratio(portfolio_returns, portfolio_growth)

    print(f"\n  Weight per stock     : {weights[0]*100:.2f}%  (uniform)")
    print(f"  Total return         : {total_ret*100:.1f}%")
    print(f"  Annualised vol       : {ann_vol*100:.1f}%")
    print(f"  Sharpe (rf=7%)       : {sharpe:.3f}")
    print(f"  Sortino              : {sortino:.3f}")
    print(f"  Max drawdown         : {mdd*100:.1f}%")
    print(f"  Calmar               : {calmar:.3f}")

    print_sector_weights(weights, tickers, label="Equal Weight")

    # Save outputs
    save_outputs(portfolio_returns, portfolio_growth)

    # Plot
    plot_equal_weight(portfolio_growth, tickers, weights)

    print("\n✅ Equal Weight complete.")
    print("   Outputs → data/portfolio/equal_weight_*.csv")
    print("   Plots   → reports/equal_weight_growth.png")