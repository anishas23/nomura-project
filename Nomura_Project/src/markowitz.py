"""
markowitz.py — Markowitz Mean-Variance Optimisation
=====================================================
Models:
  1. Min Variance  — minimise portfolio variance subject to realistic constraints
  2. Max Sharpe    — maximise Sharpe ratio (rf = 7% p.a.) with same constraints
  3. Efficient Frontier — sweep risk-return tradeoff curve

Realistic constraints applied to both models:
  - Min weight per stock  : 1%   (no ghost allocations)
  - Max weight per stock  : 15%  (no single-stock concentration)
  - Max weight per sector : 30%  (prevents IT/Banking crowding)
  - Max cyclical exposure : 40%  (Auto + Metals + Energy + Chemicals combined)

Covariance estimation:
  Ledoit-Wolf shrinkage (sklearn) — dramatically reduces estimation error
  versus sample covariance when n_assets is large relative to n_observations.

Risk-free rate:
  India 10-year G-Sec yield = 7% p.a. (used in Max Sharpe objective and
  all Sharpe ratio reporting).

Benchmark:
  NIFTY 50 (^NSEI) downloaded via yfinance and aligned to portfolio dates.
  Shown in all growth plots so readers can immediately assess whether
  optimisation added value vs the market.

Run independently:
    python src/markowitz.py

Outputs:
    data/portfolio/markowitz_growth.csv
    data/portfolio/sharpe_growth.csv
    reports/markowitz_growth.png
    reports/markowitz_sector_weights.png
    reports/efficient_frontier.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import yfinance as yf

from utils import (
    load_returns, load_prices,
    compute_lw_cov, build_constraints,
    RISK_FREE_RATE_ANNUAL, RISK_FREE_RATE_DAILY,
    SECTOR_MAP,
    print_summary, plot_growth_curves, plot_sector_allocation,
    print_sector_weights,
    sortino_ratio, max_drawdown, calmar_ratio,
    SECTOR_COLOR_MAP,
)


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def download_benchmark(start_date, end_date, reference_index):
    """
    Download NIFTY 50 (^NSEI) daily prices and return aligned growth /
    return series.

    Args:
        start_date      : first date in portfolio returns index
        end_date        : last  date in portfolio returns index
        reference_index : DatetimeIndex to reindex onto (portfolio dates)

    Returns:
        benchmark_prices  : raw Close price Series
        benchmark_returns : daily return Series aligned to reference_index
        benchmark_growth  : cumulative growth Series aligned to reference_index
    """
    print("\nDownloading NIFTY 50 benchmark (^NSEI)...")
    nifty = yf.download(
        "^NSEI",
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )
    benchmark_prices  = nifty["Close"].squeeze()          # ensure 1-D
    benchmark_returns = benchmark_prices.pct_change().dropna()
    benchmark_growth  = (1 + benchmark_returns).cumprod()

    # Align to portfolio calendar (forward-fill over holidays / missing days)
    benchmark_growth  = benchmark_growth.reindex(reference_index, method="ffill")
    benchmark_returns = benchmark_returns.reindex(reference_index, method="ffill")

    total_ret = (benchmark_growth.iloc[-1] - 1) * 100
    print(f"   Benchmark total return : {total_ret:.1f}%")
    return benchmark_prices, benchmark_returns, benchmark_growth


# ------------------------------------------------------------------
# MIN VARIANCE
# ------------------------------------------------------------------

def run_min_variance(returns, tickers, cov_np, constraints, bounds):
    """
    Minimise portfolio variance subject to realistic constraints.

    Objective : min  wᵀ Σ w
    Subject to: weights sum to 1, sector caps, cyclical cap, per-stock bounds

    Args:
        returns     : daily returns DataFrame
        tickers     : list of ticker strings
        cov_np      : LW-shrinkage covariance numpy array
        constraints : list of scipy constraint dicts
        bounds      : tuple of (min, max) per ticker

    Returns:
        weights      : numpy array of optimal weights
        ret_series   : daily portfolio return Series
        growth_series: cumulative growth Series
    """
    n  = len(tickers)
    w0 = np.ones(n) / n

    result = minimize(
        fun         = lambda w: w @ cov_np @ w,
        x0          = w0,
        method      = "SLSQP",
        bounds      = bounds,
        constraints = constraints,
        options     = {"ftol": 1e-12, "maxiter": 2000},
    )

    if result.success:
        weights = np.clip(result.x, 0, 1)
        weights /= weights.sum()
        print("   Min Variance optimisation: SUCCESS")
    else:
        print(f"   Min Variance WARNING: {result.message} — equal weight fallback")
        weights = w0.copy()

    ret_series    = returns.dot(weights)
    growth_series = (1 + ret_series).cumprod()
    return weights, ret_series, growth_series


# ------------------------------------------------------------------
# MAX SHARPE
# ------------------------------------------------------------------

def run_max_sharpe(returns, tickers, mean_returns, cov_np,
                   constraints, bounds):
    """
    Maximise Sharpe ratio (with explicit India risk-free rate = 7% p.a.)
    subject to realistic constraints.

    Objective : max  (wᵀμ − rf) / sqrt(wᵀΣw)
    Subject to: same constraints as Min Variance

    Args:
        returns      : daily returns DataFrame
        tickers      : list of ticker strings
        mean_returns : daily mean return Series
        cov_np       : LW-shrinkage covariance numpy array
        constraints  : list of scipy constraint dicts
        bounds       : tuple of (min, max) per ticker

    Returns:
        weights      : numpy array of optimal weights
        ret_series   : daily portfolio return Series
        growth_series: cumulative growth Series
    """
    n  = len(tickers)
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        p_ret = w @ mean_returns.values - RISK_FREE_RATE_DAILY
        p_vol = np.sqrt(w @ cov_np @ w)
        return -(p_ret / p_vol) if p_vol > 1e-10 else 0

    result = minimize(
        fun         = neg_sharpe,
        x0          = w0,
        method      = "SLSQP",
        bounds      = bounds,
        constraints = constraints,
        options     = {"ftol": 1e-12, "maxiter": 2000},
    )

    if result.success:
        weights = np.clip(result.x, 0, 1)
        weights /= weights.sum()
        print("   Max Sharpe optimisation: SUCCESS")
    else:
        print(f"   Max Sharpe WARNING: {result.message} — equal weight fallback")
        weights = w0.copy()

    ret_series    = returns.dot(weights)
    growth_series = (1 + ret_series).cumprod()
    return weights, ret_series, growth_series


# ------------------------------------------------------------------
# EFFICIENT FRONTIER
# ------------------------------------------------------------------

def compute_efficient_frontier(returns, tickers, mean_returns, cov_np,
                               n_points=40):
    """
    Sweep the risk-return tradeoff by solving Min Variance at fixed
    target return levels.  Uses simple bounds only (no sector caps)
    so the frontier is smooth and unconstrained.

    Returns:
        frontier_vols  : list of annualised volatility values
        frontier_rets  : list of annualised return values
    """
    n      = len(tickers)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    base_c = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    min_ret    = mean_returns.min() * 252
    max_ret    = mean_returns.max() * 252
    target_rets = np.linspace(min_ret * 0.8, max_ret * 0.95, n_points)

    frontier_vols = []
    frontier_rets = []

    for tgt in target_rets:
        c = base_c + [{
            "type": "eq",
            "fun" : (lambda w, r=tgt: w @ mean_returns.values * 252 - r),
        }]
        res = minimize(
            fun         = lambda w: w @ cov_np @ w,
            x0          = np.ones(n) / n,
            method      = "SLSQP",
            bounds      = bounds,
            constraints = c,
            options     = {"ftol": 1e-10, "maxiter": 1000},
        )
        if res.success:
            vol = np.sqrt(res.fun) * np.sqrt(252)
            frontier_vols.append(vol * 100)
            frontier_rets.append(tgt * 100)

    return frontier_vols, frontier_rets


# ------------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------------

def plot_growth_comparison(growth_mv, growth_ms, growth_ew, benchmark_growth):
    """
    Growth curve: Min Variance vs Max Sharpe vs Equal Weight vs NIFTY 50.

    The NIFTY 50 benchmark line is the primary market reference so mentors,
    recruiters, and portfolio managers can immediately see whether
    optimisation added alpha versus a passive index investment.

    Args:
        growth_mv        : cumulative growth Series — Min Variance portfolio
        growth_ms        : cumulative growth Series — Max Sharpe portfolio
        growth_ew        : cumulative growth Series — Equal Weight baseline
        benchmark_growth : cumulative growth Series — NIFTY 50 (^NSEI)
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(growth_mv.index, growth_mv.values,
            label="Min Variance",       color="steelblue",  linewidth=1.8)
    ax.plot(growth_ms.index, growth_ms.values,
            label="Max Sharpe",         color="darkorange",  linewidth=1.8)
    ax.plot(growth_ew.index, growth_ew.values,
            label="Equal Weight (ref)", color="grey",
            linewidth=1.2, linestyle="--", alpha=0.7)

    # ── NIFTY 50 benchmark ──────────────────────────────────────────
    ax.plot(
        benchmark_growth.index,
        benchmark_growth.values,
        label="NIFTY 50 Benchmark",
        color="black",
        linewidth=2.0,
        linestyle=":",
        alpha=0.9,
    )
    # ────────────────────────────────────────────────────────────────

    ax.set_title(
        "Markowitz Portfolios — Cumulative Growth\n"
        "(LW covariance | rf = 7% p.a. | realistic constraints | vs NIFTY 50)",
        fontsize=12,
    )
    ax.set_ylabel("Portfolio Value (₹1 invested)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/markowitz_growth.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: reports/markowitz_growth.png")


def plot_sector_comparison(weights_mv, weights_ms, weights_ew, tickers):
    """
    Side-by-side sector weights for Min Var / Max Sharpe / Equal Weight,
    and a top-10 holdings bar chart.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: stacked sector bar
    weight_sets = {
        "Min Variance\n(Constrained)": weights_mv,
        "Max Sharpe\n(Constrained)"  : weights_ms,
        "Equal Weight"               : weights_ew,
    }
    all_sectors  = sorted(set(SECTOR_MAP.values()))
    strat_labels = list(weight_sets.keys())
    x            = np.arange(len(strat_labels))
    ax           = axes[0]
    bottoms      = np.zeros(len(strat_labels))

    for sector in all_sectors:
        sec_wts = []
        for wts in weight_sets.values():
            sec_idx = [i for i, t in enumerate(tickers)
                       if SECTOR_MAP.get(t, "Other") == sector]
            sec_wts.append(sum(wts[i] for i in sec_idx) * 100 if sec_idx else 0)
        ax.bar(x, sec_wts, bottom=bottoms,
               color=SECTOR_COLOR_MAP.get(sector, "#CCC"),
               label=sector, edgecolor="white", linewidth=0.3)
        bottoms += np.array(sec_wts)

    ax.axhline(30, color="red", linewidth=1.0, linestyle="--",
               alpha=0.7, label="30% sector cap")
    ax.set_title("Sector Allocation — Markowitz Models", fontsize=12)
    ax.set_ylabel("Allocation (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(strat_labels, rotation=20, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 2: top-10 Min Var vs Max Sharpe
    ax = axes[1]
    mv_top10 = (pd.Series(weights_mv, index=tickers)
                .sort_values(ascending=False).head(10))
    ms_top10 = pd.Series(weights_ms, index=tickers).loc[mv_top10.index]
    x2, w2   = np.arange(10), 0.35
    ax.bar(x2 - w2 / 2, mv_top10.values * 100, w2,
           label="Min Variance", color="steelblue", alpha=0.85)
    ax.bar(x2 + w2 / 2, ms_top10.values * 100, w2,
           label="Max Sharpe",   color="darkorange", alpha=0.85)
    ax.set_title("Top 10 Holdings — Min Var vs Max Sharpe", fontsize=12)
    ax.set_ylabel("Weight (%)")
    ax.set_xticks(x2)
    ax.set_xticklabels([t.replace(".NS", "") for t in mv_top10.index],
                       rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("reports/markowitz_sector_weights.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: reports/markowitz_sector_weights.png")


def plot_efficient_frontier(frontier_vols, frontier_rets,
                            weights_mv, weights_ms, weights_ew,
                            mean_returns, cov_np,
                            benchmark_returns=None):
    """
    Efficient frontier with Min Var, Max Sharpe, Equal Weight, and
    NIFTY 50 Benchmark portfolios marked.

    Args:
        frontier_vols     : list of annualised volatility values (%)
        frontier_rets     : list of annualised return values (%)
        weights_mv        : numpy array — Min Variance optimal weights
        weights_ms        : numpy array — Max Sharpe optimal weights
        weights_ew        : numpy array — Equal Weight
        mean_returns      : daily mean return Series
        cov_np            : LW-shrinkage covariance numpy array
        benchmark_returns : daily return Series for NIFTY 50 (optional)
    """
    def ann_ret_vol(w):
        r = w @ mean_returns.values * 252 * 100
        v = np.sqrt(w @ cov_np @ w) * np.sqrt(252) * 100
        return r, v

    mv_r, mv_v = ann_ret_vol(weights_mv)
    ms_r, ms_v = ann_ret_vol(weights_ms)
    ew_r, ew_v = ann_ret_vol(weights_ew)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frontier_vols, frontier_rets,
            color="steelblue", linewidth=2, label="Efficient Frontier")
    ax.scatter([mv_v], [mv_r], marker="*", s=250, color="blue",
               zorder=5, label=f"Min Variance  ({mv_r:.1f}%, {mv_v:.1f}%)")
    ax.scatter([ms_v], [ms_r], marker="*", s=250, color="darkorange",
               zorder=5, label=f"Max Sharpe    ({ms_r:.1f}%, {ms_v:.1f}%)")
    ax.scatter([ew_v], [ew_r], marker="D", s=100, color="grey",
               zorder=5, label=f"Equal Weight  ({ew_r:.1f}%, {ew_v:.1f}%)")

    # ── NIFTY 50 on the frontier plot ───────────────────────────────
    if benchmark_returns is not None:
        bm_ret = benchmark_returns.dropna()
        bm_ann_ret = bm_ret.mean() * 252 * 100
        bm_ann_vol = bm_ret.std() * np.sqrt(252) * 100
        ax.scatter(
            [bm_ann_vol], [bm_ann_ret],
            marker="^", s=180, color="black", zorder=5,
            label=f"NIFTY 50       ({bm_ann_ret:.1f}%, {bm_ann_vol:.1f}%)",
        )
    # ────────────────────────────────────────────────────────────────

    ax.set_xlabel("Annualised Volatility (%)")
    ax.set_ylabel("Annualised Return (%)")
    ax.set_title(
        "Efficient Frontier — Nifty 50 Universe\n"
        "(LW covariance | unconstrained for frontier smoothness | NIFTY 50 shown as ▲)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/efficient_frontier.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: reports/efficient_frontier.png")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("MARKOWITZ PORTFOLIO MODELS")
    print("=" * 60)

    # ── Load portfolio data ──────────────────────────────────────────
    returns, tickers, n = load_returns()
    prices = load_prices(tickers)

    print(f"\nData loaded — {n} tickers | "
          f"{returns.index[0].date()} → {returns.index[-1].date()}")

    # ── Equal weight baseline ────────────────────────────────────────
    weights_ew = np.ones(n) / n
    returns_ew = returns.dot(weights_ew)
    growth_ew  = (1 + returns_ew).cumprod()

    # ── NIFTY 50 benchmark ───────────────────────────────────────────
    _, benchmark_returns, benchmark_growth = download_benchmark(
        start_date      = returns.index.min(),
        end_date        = returns.index.max(),
        reference_index = growth_ew.index,
    )

    # ── Ledoit-Wolf covariance ───────────────────────────────────────
    print("\nEstimating Ledoit-Wolf shrinkage covariance...")
    cov_np, _    = compute_lw_cov(returns)
    mean_returns = returns.mean()
    lw_tmp       = LedoitWolf().fit(returns.values)
    print(f"   Shrinkage coefficient: {lw_tmp.shrinkage_:.4f}")

    # ── Constraints and bounds ───────────────────────────────────────
    constraints = build_constraints(tickers)
    bounds      = tuple((0.01, 0.15) for _ in range(n))

    # ── Min Variance ─────────────────────────────────────────────────
    print("\nRunning Min Variance...")
    weights_mv, returns_mv, growth_mv = run_min_variance(
        returns, tickers, cov_np, constraints, bounds
    )
    growth_mv.to_csv("data/portfolio/markowitz_growth.csv")
    print(f"   Total return: {(growth_mv.iloc[-1]-1)*100:.1f}%")
    top5_mv = pd.Series(weights_mv, index=tickers).sort_values(ascending=False).head(5)
    print(f"   Top 5 holdings: {dict(top5_mv.round(3))}")
    print_sector_weights(weights_mv, tickers, label="Min Variance")

    # ── Max Sharpe ───────────────────────────────────────────────────
    print("\nRunning Max Sharpe...")
    weights_ms, returns_ms, growth_ms = run_max_sharpe(
        returns, tickers, mean_returns, cov_np, constraints, bounds
    )
    growth_ms.to_csv("data/portfolio/sharpe_growth.csv")
    print(f"   Total return: {(growth_ms.iloc[-1]-1)*100:.1f}%")
    top5_ms = pd.Series(weights_ms, index=tickers).sort_values(ascending=False).head(5)
    print(f"   Top 5 holdings: {dict(top5_ms.round(3))}")
    print_sector_weights(weights_ms, tickers, label="Max Sharpe")

    # ── Efficient frontier ───────────────────────────────────────────
    print("\nComputing efficient frontier...")
    frontier_vols, frontier_rets = compute_efficient_frontier(
        returns, tickers, mean_returns, cov_np
    )
    print(f"   Frontier points computed: {len(frontier_vols)}")

    # ── Performance summary (now includes benchmark) ─────────────────
    all_growths = {
        "Min Variance"       : growth_mv,
        "Max Sharpe"         : growth_ms,
        "Equal Weight"       : growth_ew,
        "NIFTY 50 Benchmark" : benchmark_growth,   # ← benchmark added
    }
    all_rets = {
        "Min Variance"       : returns_mv,
        "Max Sharpe"         : returns_ms,
        "Equal Weight"       : returns_ew,
        "NIFTY 50 Benchmark" : benchmark_returns,  # ← benchmark added
    }
    print_summary(all_growths, all_rets,
                  note="LW shrinkage covariance | sector & cyclical caps applied | vs NIFTY 50")

    # ── Plots ────────────────────────────────────────────────────────
    plot_growth_comparison(growth_mv, growth_ms, growth_ew, benchmark_growth)
    plot_sector_comparison(weights_mv, weights_ms, weights_ew, tickers)
    plot_efficient_frontier(
        frontier_vols, frontier_rets,
        weights_mv, weights_ms, weights_ew,
        mean_returns, cov_np,
        benchmark_returns=benchmark_returns,        # ← benchmark added
    )

    print("\n✅ Markowitz models complete.")
    print("   Outputs → data/portfolio/markowitz_growth.csv")
    print("             data/portfolio/sharpe_growth.csv")
    print("   Plots   → reports/markowitz_growth.png")
    print("             reports/markowitz_sector_weights.png")
    print("             reports/efficient_frontier.png")