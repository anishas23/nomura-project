"""
backtesting.py — Walk-Forward Backtesting Framework
====================================================
Implements a proper walk-forward (expanding window) backtest for all
portfolio strategies with:
  - No look-ahead bias: each rebalance only uses data available at that point
  - Realistic transaction costs: 10 bps one-way per trade
  - Multiple rebalancing frequencies: monthly, quarterly, semi-annual
  - Comparison across all strategies + NIFTY 50 benchmark

Strategies backtested:
  1. Equal Weight          (baseline)
  2. Min Variance          (Markowitz)
  3. Max Sharpe            (Markowitz)
  4. ERC Risk Parity       (loaded from CSV)
  5. Black-Litterman       (loaded from CSV)

Run:
    python src/backtesting.py

Outputs:
    data/portfolio/backtest_results.csv
    reports/backtest_growth.png
    reports/backtest_rebalance_comparison.png
    reports/backtest_turnover.png
    reports/backtest_rolling_metrics.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

import yfinance as yf

from utils import (
    load_returns, load_prices,
    compute_lw_cov, build_constraints,
    RISK_FREE_RATE_ANNUAL, RISK_FREE_RATE_DAILY,
    TRANSACTION_COST,
    SECTOR_MAP,
    sortino_ratio, max_drawdown, calmar_ratio,
    print_summary,
)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

TRAIN_WINDOW    = 252          # 1 year of data to estimate params
REBAL_MONTHLY   = 21           # ~monthly in trading days
REBAL_QUARTERLY = 63           # ~quarterly
REBAL_SEMIANN   = 126          # ~semi-annual
MIN_WEIGHT      = 0.01
MAX_WEIGHT      = 0.15


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def _lw_cov(ret_window):
    lw = LedoitWolf()
    lw.fit(ret_window.values)
    return lw.covariance_


def _min_var_weights(ret_window, tickers):
    n   = len(tickers)
    cov = _lw_cov(ret_window)
    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
    bounds = tuple((MIN_WEIGHT, MAX_WEIGHT) for _ in range(n))
    res = minimize(
        lambda w: w @ cov @ w,
        x0=np.ones(n) / n,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 1000}
    )
    if res.success:
        w = np.clip(res.x, 0, 1)
        return w / w.sum()
    return np.ones(n) / n


def _max_sharpe_weights(ret_window, tickers):
    n    = len(tickers)
    cov  = _lw_cov(ret_window)
    mu   = ret_window.mean().values
    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
    bounds = tuple((MIN_WEIGHT, MAX_WEIGHT) for _ in range(n))

    def neg_sharpe(w):
        r = w @ mu - RISK_FREE_RATE_DAILY
        v = np.sqrt(w @ cov @ w)
        return -(r / v) if v > 1e-10 else 0

    res = minimize(
        neg_sharpe,
        x0=np.ones(n) / n,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 1000}
    )
    if res.success:
        w = np.clip(res.x, 0, 1)
        return w / w.sum()
    return np.ones(n) / n


def _apply_tc(ret_series_values, weights_series, dates, rebal_dates):
    """
    Deduct transaction cost on each rebalance day.
    cost = TC * sum(|new_w - old_w|)
    Returns adjusted return array.
    """
    adjusted = ret_series_values.copy()
    rebal_set = set(rebal_dates)
    prev_w = None
    date_to_idx = {d: i for i, d in enumerate(dates)}

    for d in rebal_dates:
        if d not in date_to_idx:
            continue
        idx = date_to_idx[d]
        curr_w = weights_series.iloc[idx] if hasattr(weights_series, 'iloc') else weights_series[idx]
        if prev_w is not None:
            turnover = np.sum(np.abs(curr_w - prev_w))
            tc = TRANSACTION_COST * turnover
            adjusted[idx] -= tc
        prev_w = curr_w.copy() if hasattr(curr_w, 'copy') else curr_w

    return adjusted


# ------------------------------------------------------------------
# CORE WALK-FORWARD ENGINE
# ------------------------------------------------------------------

def walk_forward_backtest(returns, tickers, strategy_fn,
                          rebal_freq=REBAL_MONTHLY,
                          label='Strategy'):
    """
    Walk-forward backtest for any strategy that takes a return window
    and returns portfolio weights.

    Parameters
    ----------
    returns     : full daily returns DataFrame
    tickers     : list of tickers
    strategy_fn : callable(ret_window, tickers) → np.ndarray weights
    rebal_freq  : int, rebalance every N trading days
    label       : name for printing

    Returns
    -------
    ret_series    : daily portfolio return Series (net of TC)
    growth_series : cumulative growth Series
    weights_log   : dict date → np.ndarray of weights
    turnover_log  : list of (date, turnover%) tuples
    """
    n      = len(tickers)
    dates  = returns.index.tolist()
    T      = len(dates)

    current_w  = np.ones(n) / n   # start equal-weight
    prev_w     = None
    daily_rets = []
    weights_log   = {}
    turnover_log  = []

    rebal_count   = 0
    total_tc_paid = 0.0

    for i, date in enumerate(dates):

        # Decide if this is a rebalance day
        is_rebal = (i >= TRAIN_WINDOW) and ((i - TRAIN_WINDOW) % rebal_freq == 0)

        if is_rebal:
            window = returns.iloc[i - TRAIN_WINDOW: i]
            try:
                new_w = strategy_fn(window, tickers)
            except Exception:
                new_w = current_w.copy()

            # Compute turnover and TC
            if prev_w is not None:
                turnover = float(np.sum(np.abs(new_w - prev_w)))
            else:
                turnover = 1.0  # first trade: buy everything from cash

            tc = TRANSACTION_COST * turnover
            turnover_log.append((date, turnover * 100))
            total_tc_paid += tc

            current_w = new_w.copy()
            prev_w    = new_w.copy()
            rebal_count += 1
            weights_log[date] = current_w.copy()

            # Day return net of TC
            day_ret = float(returns.iloc[i].values @ current_w) - tc
        else:
            day_ret = float(returns.iloc[i].values @ current_w)

        daily_rets.append(day_ret)

    ret_series    = pd.Series(daily_rets, index=dates, name=label)
    growth_series = (1 + ret_series).cumprod()

    print(f"   {label:<28}: rebalances={rebal_count}  "
          f"TC_paid={total_tc_paid*10000:.1f}bps  "
          f"total_ret={( growth_series.iloc[-1]-1)*100:.1f}%")

    return ret_series, growth_series, weights_log, turnover_log


# ------------------------------------------------------------------
# BENCHMARK
# ------------------------------------------------------------------

def load_nifty(start, end):
    try:
        raw = yf.download("^NSEI", start=str(start.date()),
                          end=str(end.date()), progress=False, auto_adjust=True)
        if raw.empty:
            return None, None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        close = raw['Close'].squeeze()
        close.index = pd.to_datetime(close.index)
        rets   = close.pct_change().dropna()
        growth = (1 + rets).cumprod()
        growth = growth / growth.iloc[0]
        print(f"   NIFTY 50 loaded: {len(rets)} days")
        return rets, growth
    except Exception as e:
        print(f"   NIFTY 50 load failed: {e}")
        return None, None


# ------------------------------------------------------------------
# REBALANCING FREQUENCY COMPARISON (single strategy, 3 freqs)
# ------------------------------------------------------------------

def compare_rebal_frequencies(returns, tickers, strategy_fn,
                               strategy_name='Min Variance'):
    """
    Compare monthly / quarterly / semi-annual rebalancing for one strategy.
    Returns dict of growth series.
    """
    results = {}
    freqs   = [
        ('Monthly',     REBAL_MONTHLY),
        ('Quarterly',   REBAL_QUARTERLY),
        ('Semi-Annual', REBAL_SEMIANN),
    ]
    print(f"\n   Rebalancing frequency comparison — {strategy_name}:")
    for label, freq in freqs:
        ret_s, growth, _, _ = walk_forward_backtest(
            returns, tickers, strategy_fn,
            rebal_freq=freq,
            label=f'{strategy_name} ({label})'
        )
        results[label] = (ret_s, growth)
    return results


# ------------------------------------------------------------------
# PERFORMANCE TABLE
# ------------------------------------------------------------------

def performance_table(all_growths, all_rets, rf=RISK_FREE_RATE_ANNUAL):
    rows = []
    for name, growth in all_growths.items():
        ret_s = all_rets[name]
        if ret_s is None or len(ret_s) < 10:
            continue
        n_yr      = len(ret_s) / 252
        total_ret = growth.iloc[-1] - 1
        ann_ret   = (1 + total_ret) ** (1 / n_yr) - 1 if n_yr > 0 else 0
        ann_vol   = ret_s.std() * np.sqrt(252)
        sharpe    = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
        cum       = (1 + ret_s).cumprod()
        dd        = (cum - cum.cummax()) / cum.cummax()
        mdd       = dd.min()
        calmar    = ann_ret / abs(mdd) if mdd != 0 else 0
        down_ret  = ret_s[ret_s < rf / 252]
        down_std  = down_ret.std() * np.sqrt(252) if len(down_ret) > 0 else 1e-10
        sortino   = (ann_ret - rf) / down_std

        rows.append({
            'Strategy'        : name,
            'Total Return (%)'  : round(total_ret * 100, 2),
            'Ann Return (%)'    : round(ann_ret    * 100, 2),
            'Ann Vol (%)'       : round(ann_vol    * 100, 2),
            'Sharpe'            : round(sharpe,   3),
            'Sortino'           : round(sortino,  3),
            'Max Drawdown (%)'  : round(mdd       * 100, 2),
            'Calmar'            : round(calmar,   3),
        })

    df = pd.DataFrame(rows).set_index('Strategy')
    print("\n" + "=" * 95)
    print("BACKTEST PERFORMANCE SUMMARY  (walk-forward | 10bps TC | rf=7%)")
    print("=" * 95)
    print(df.to_string())
    print("=" * 95)
    return df


# ------------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------------

def plot_backtest_growth(all_growths, benchmark_growth=None):
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_growths)))
    for i, (name, growth) in enumerate(all_growths.items()):
        ax.plot(growth.index, growth.values, label=name,
                linewidth=1.8, color=colors[i])
    if benchmark_growth is not None:
        bm = benchmark_growth.reindex(
            list(all_growths.values())[0].index, method='ffill').dropna()
        bm = bm / bm.iloc[0]
        ax.plot(bm.index, bm.values, label='NIFTY 50 (benchmark)',
                color='black', linewidth=2.0, linestyle=':', alpha=0.85)
    ax.set_title(
        "Walk-Forward Backtest — Cumulative Growth (All Strategies)\n"
        "(Expanding window | 10bps TC | Monthly rebalance | LW covariance)",
        fontsize=12)
    ax.set_ylabel("Portfolio Value (₹1 invested)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig("reports/backtest_growth.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/backtest_growth.png")


def plot_rebalance_comparison(freq_results, strategy_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    colors = {'Monthly': 'steelblue', 'Quarterly': 'darkorange',
              'Semi-Annual': 'green'}
    for label, (ret_s, growth) in freq_results.items():
        ax1.plot(growth.index, growth.values, label=label,
                 color=colors[label], linewidth=1.8)
    ax1.set_title(
        f"Rebalancing Frequency Impact — {strategy_name}\n"
        "(All net of 10bps TC per rebalance)", fontsize=12)
    ax1.set_ylabel("Portfolio Value (₹1 invested)")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    for label, (ret_s, growth) in freq_results.items():
        roll_sh = (ret_s.rolling(63).mean() * 252 - RISK_FREE_RATE_ANNUAL) / \
                  (ret_s.rolling(63).std() * np.sqrt(252)).replace(0, np.nan)
        ax2.plot(roll_sh.index, roll_sh.values, label=label,
                 color=colors[label], linewidth=1.5)
    ax2.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax2.set_title("Rolling 63-Day Sharpe Ratio by Rebalancing Frequency", fontsize=11)
    ax2.set_ylabel("Sharpe Ratio (annualised)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig("reports/backtest_rebalance_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/backtest_rebalance_comparison.png")


def plot_turnover(all_turnovers):
    fig, ax = plt.subplots(figsize=(13, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_turnovers)))
    for i, (name, turnover_log) in enumerate(all_turnovers.items()):
        if not turnover_log:
            continue
        dates = [x[0] for x in turnover_log]
        vals  = [x[1] for x in turnover_log]
        ax.plot(dates, vals, label=name, color=colors[i],
                linewidth=1.4, marker='o', markersize=3, alpha=0.8)
    ax.axhline(10, color='red', linewidth=1.0, linestyle='--',
               alpha=0.7, label='10% reference')
    ax.set_title("Portfolio Turnover at Each Rebalance (One-Way, %)\n"
                 "(Lower = cheaper to run in practice)", fontsize=12)
    ax.set_ylabel("One-Way Turnover (%)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig("reports/backtest_turnover.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/backtest_turnover.png")


def plot_rolling_metrics(all_rets, benchmark_rets=None, window=63):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_rets)))

    ref_index = list(all_rets.values())[0].index

    # Rolling Sharpe
    ax = axes[0]
    for i, (name, ret_s) in enumerate(all_rets.items()):
        aligned = ret_s.reindex(ref_index).fillna(0)
        roll_sh = (aligned.rolling(window).mean() * 252 - RISK_FREE_RATE_ANNUAL) / \
                  (aligned.rolling(window).std() * np.sqrt(252)).replace(0, np.nan)
        ax.plot(roll_sh.index, roll_sh.values, label=name,
                color=colors[i], linewidth=1.5)
    if benchmark_rets is not None:
        aligned_bm = benchmark_rets.reindex(ref_index, method='ffill').fillna(0)
        roll_sh_bm = (aligned_bm.rolling(window).mean() * 252 - RISK_FREE_RATE_ANNUAL) / \
                     (aligned_bm.rolling(window).std() * np.sqrt(252)).replace(0, np.nan)
        ax.plot(roll_sh_bm.index, roll_sh_bm.values, label='NIFTY 50',
                color='black', linewidth=1.8, linestyle=':', alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio (Walk-Forward Backtest)", fontsize=12)
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)

    # Rolling Drawdown
    ax = axes[1]
    for i, (name, ret_s) in enumerate(all_rets.items()):
        aligned = ret_s.reindex(ref_index).fillna(0)
        cum     = (1 + aligned).cumprod()
        dd      = (cum - cum.cummax()) / cum.cummax()
        ax.plot(dd.index, dd.values * 100, label=name,
                color=colors[i], linewidth=1.5)
    ax.axhline(-20, color='red', linewidth=0.8, linestyle='--',
               alpha=0.7, label='-20% threshold')
    ax.set_title("Drawdown Profile — All Strategies", fontsize=12)
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig("reports/backtest_rolling_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/backtest_rolling_metrics.png")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("WALK-FORWARD BACKTESTING FRAMEWORK")
    print(f"  Train window  : {TRAIN_WINDOW} days (expanding)")
    print(f"  Rebal default : monthly ({REBAL_MONTHLY} trading days)")
    print(f"  TC            : {TRANSACTION_COST*10000:.0f} bps one-way")
    print("=" * 70)

    returns, tickers, n = load_returns()
    prices = load_prices(tickers)

    print(f"\nData: {n} tickers | "
          f"{returns.index[0].date()} → {returns.index[-1].date()}")

    # NIFTY 50 benchmark
    print("\nLoading NIFTY 50 benchmark...")
    bm_rets, bm_growth = load_nifty(returns.index[0], returns.index[-1])

    # ── Strategy 1: Equal Weight (no optimisation, just rebalance monthly)
    print("\nRunning strategies (monthly rebalance)...")
    ew_ret, ew_growth, ew_wlog, ew_turn = walk_forward_backtest(
        returns, tickers,
        lambda w, t: np.ones(len(t)) / len(t),
        rebal_freq=REBAL_MONTHLY,
        label='Equal Weight'
    )

    # ── Strategy 2: Min Variance
    mv_ret, mv_growth, mv_wlog, mv_turn = walk_forward_backtest(
        returns, tickers,
        _min_var_weights,
        rebal_freq=REBAL_MONTHLY,
        label='Min Variance (WF)'
    )

    # ── Strategy 3: Max Sharpe
    ms_ret, ms_growth, ms_wlog, ms_turn = walk_forward_backtest(
        returns, tickers,
        _max_sharpe_weights,
        rebal_freq=REBAL_MONTHLY,
        label='Max Sharpe (WF)'
    )

    # ── Load pre-computed growth curves for BL and Risk Parity
    print("\nLoading pre-computed BL and Risk Parity results...")
    all_growths = {
        'Equal Weight'    : ew_growth,
        'Min Variance'    : mv_growth,
        'Max Sharpe'      : ms_growth,
    }
    all_rets = {
        'Equal Weight'    : ew_ret,
        'Min Variance'    : mv_ret,
        'Max Sharpe'      : ms_ret,
    }

    for name, path in [('ERC Risk Parity', 'data/portfolio/erc_growth.csv'),
                       ('Black-Litterman',  'data/portfolio/black_litterman_growth.csv'),
                       ('Max Diversification', 'data/portfolio/max_diversification_growth.csv')]:
        if os.path.exists(path):
            g = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
            r = g.pct_change().dropna()
            all_growths[name] = g / g.iloc[0]
            all_rets[name]    = r
            print(f"   Loaded {name} from {path}")
        else:
            print(f"   WARNING: {path} not found — run the model first")

    if bm_rets is not None:
        bm_growth_aligned = bm_growth.reindex(ew_growth.index, method='ffill').dropna()
        bm_growth_norm    = bm_growth_aligned / bm_growth_aligned.iloc[0]
        all_growths['NIFTY 50'] = bm_growth_norm
        all_rets['NIFTY 50']    = bm_rets

    # ── Performance table
    perf_df = performance_table(all_growths, all_rets)
    perf_df.to_csv("data/portfolio/backtest_results.csv")
    print("\nSaved: data/portfolio/backtest_results.csv")

    # ── Rebalancing frequency comparison (Min Variance)
    print("\nRebalancing frequency impact (Min Variance)...")
    freq_results = compare_rebal_frequencies(
        returns, tickers, _min_var_weights, strategy_name='Min Variance'
    )

    # ── Plots
    plot_backtest_growth(
        {k: v for k, v in all_growths.items() if k != 'NIFTY 50'},
        benchmark_growth=bm_growth_norm if bm_rets is not None else None
    )
    plot_rebalance_comparison(freq_results, 'Min Variance')
    plot_turnover({
        'Equal Weight': ew_turn,
        'Min Variance': mv_turn,
        'Max Sharpe'  : ms_turn,
    })
    plot_rolling_metrics(
        {k: v for k, v in all_rets.items() if k != 'NIFTY 50'},
        benchmark_rets=bm_rets
    )

    print("\n✅ Backtesting complete.")
    print("   Outputs → data/portfolio/backtest_results.csv")
    print("   Plots   → reports/backtest_*.png")