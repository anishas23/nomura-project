"""
performance_metrics.py — Comprehensive Performance Evaluation
=============================================================
Week 7 | Nomura Mentorship Project

Loads ALL portfolio strategies built in Weeks 3–6 and computes a full
suite of risk-adjusted performance metrics, then produces publication-
quality visualisations.

Metrics computed
----------------
  Return metrics  : Total Return, CAGR, Monthly/Annual return distribution
  Risk metrics    : Annualised Volatility, Semi-Volatility (downside only)
  Ratio metrics   : Sharpe, Sortino, Calmar, Information Ratio, Omega Ratio
  Drawdown metrics: Maximum Drawdown, Average Drawdown, Drawdown Duration,
                    Recovery Time, Underwater Plot
  Tail metrics    : VaR (95%, 99%), CVaR (95%, 99%), Skewness, Kurtosis
  Benchmark-rel.  : Alpha, Beta, Tracking Error, Up/Down Capture Ratio

Portfolios compared
-------------------
  1. Equal Weight
  2. Min Variance      (Markowitz)
  3. Max Sharpe        (Markowitz)
  4. ERC               (Risk Parity)
  5. Max Diversif.     (Risk Parity)
  6. Black-Litterman
  7. NIFTY 50 Benchmark

Run
---
    python src/performance_metrics.py

Outputs
-------
    reports/perf_summary_table.csv
    reports/perf_growth_all.png
    reports/perf_drawdown_underwater.png
    reports/perf_rolling_metrics.png
    reports/perf_monthly_heatmap.png
    reports/perf_risk_return_scatter.png
    reports/perf_tail_risk.png
    reports/perf_radar_chart.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

from utils import (
    load_returns,
    RISK_FREE_RATE_ANNUAL,
    SECTOR_MAP,
)

# ── Configuration ──────────────────────────────────────────────────
RF          = RISK_FREE_RATE_ANNUAL      # 7% p.a.
RF_DAILY    = RF / 252
PERIODS     = 252                        # trading days per year
ROLLING_WIN = 63                         # 3-month rolling window for charts
VAR_LEVELS  = [0.05, 0.01]              # 95% and 99% VaR

STRATEGY_COLORS = {
    'Equal Weight'       : '#7f7f7f',
    'Min Variance'       : '#1f77b4',
    'Max Sharpe'         : '#ff7f0e',
    'ERC'                : '#2ca02c',
    'Max Diversification': '#9467bd',
    'Black-Litterman'    : '#d62728',
    'NIFTY 50'           : '#000000',
}

os.makedirs('reports', exist_ok=True)
os.makedirs('data/portfolio', exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════

def _load_growth_csv(path: str) -> pd.Series | None:
    """Load a saved portfolio growth CSV, return None if missing."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    s  = df.iloc[:, 0].dropna()
    # Normalise to start at 1.0
    s  = s / s.iloc[0]
    return s


def load_all_portfolios(returns_df: pd.DataFrame) -> dict:
    """
    Load every portfolio strategy.

    Priority: pre-saved CSV files (from earlier weeks).
    Fallback:  reconstruct from returns_df with equal weights.

    Returns dict  strategy_name → daily_returns_Series
    """
    portfolios = {}

    growth_files = {
        'Equal Weight'       : 'data/portfolio/equal_weight_growth.csv',
        'Min Variance'       : 'data/portfolio/markowitz_growth.csv',
        'Max Sharpe'         : 'data/portfolio/sharpe_growth.csv',
        'ERC'                : 'data/portfolio/erc_growth.csv',
        'Max Diversification': 'data/portfolio/max_diversification_growth.csv',
        'Black-Litterman'    : 'data/portfolio/black_litterman_growth.csv',
    }

    for name, path in growth_files.items():
        growth = _load_growth_csv(path)
        if growth is not None:
            # Convert growth → daily returns
            ret = growth.pct_change().dropna()
            portfolios[name] = ret
            print(f"   Loaded  : {name:25s} ({len(ret)} days)")
        else:
            print(f"   Missing : {name:25s} — {path}")

    # Ensure Equal Weight always exists as baseline
    if 'Equal Weight' not in portfolios:
        n   = returns_df.shape[1]
        ew  = returns_df.dot(np.ones(n) / n)
        portfolios['Equal Weight'] = ew
        print(f"   Fallback: Equal Weight reconstructed from returns")

    return portfolios


def load_benchmark(start: pd.Timestamp, end: pd.Timestamp,
                   ref_index: pd.DatetimeIndex) -> pd.Series | None:
    """Download NIFTY 50 and align to ref_index."""
    if not HAS_YFINANCE:
        return None
    try:
        raw = yf.download('^NSEI', start=start, end=end,
                          auto_adjust=True, progress=False)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        close  = raw['Close'].squeeze().dropna()
        bm_ret = close.pct_change().dropna()
        bm_ret = bm_ret.reindex(ref_index, method='ffill').dropna()
        print(f"   Loaded  : {'NIFTY 50':25s} ({len(bm_ret)} days)")
        return bm_ret
    except Exception as e:
        print(f"   NIFTY 50 download failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# 2. METRICS ENGINE
# ══════════════════════════════════════════════════════════════════

def _cagr(ret: pd.Series) -> float:
    n_yr  = len(ret) / PERIODS
    total = (1 + ret).prod()
    return total ** (1 / n_yr) - 1 if n_yr > 0 else 0.0


def _ann_vol(ret: pd.Series) -> float:
    return ret.std() * np.sqrt(PERIODS)


def _sharpe(ret: pd.Series) -> float:
    excess = ret.mean() * PERIODS - RF
    vol    = _ann_vol(ret)
    return excess / vol if vol > 0 else 0.0


def _sortino(ret: pd.Series) -> float:
    excess   = ret - RF_DAILY
    downside = excess[excess < 0]
    if len(downside) == 0:
        return np.nan
    dd_vol = downside.std() * np.sqrt(PERIODS)
    return (ret.mean() * PERIODS - RF) / dd_vol if dd_vol > 0 else np.nan


def _max_drawdown(ret: pd.Series) -> float:
    growth = (1 + ret).cumprod()
    roll_max = growth.cummax()
    dd = (growth - roll_max) / roll_max
    return dd.min()


def _calmar(ret: pd.Series) -> float:
    cagr_ = _cagr(ret)
    mdd   = abs(_max_drawdown(ret))
    return (cagr_ - RF) / mdd if mdd > 0 else np.nan


def _avg_drawdown(ret: pd.Series) -> float:
    growth   = (1 + ret).cumprod()
    roll_max = growth.cummax()
    dd       = (growth - roll_max) / roll_max
    return dd[dd < 0].mean() if (dd < 0).any() else 0.0


def _max_drawdown_duration(ret: pd.Series) -> int:
    """Return max drawdown duration in trading days."""
    growth   = (1 + ret).cumprod()
    roll_max = growth.cummax()
    dd       = (growth - roll_max) / roll_max
    in_dd    = dd < 0
    max_dur  = 0
    cur_dur  = 0
    for val in in_dd:
        if val:
            cur_dur += 1
            max_dur  = max(max_dur, cur_dur)
        else:
            cur_dur  = 0
    return max_dur


def _var(ret: pd.Series, level: float = 0.05) -> float:
    return float(np.percentile(ret.dropna(), level * 100))


def _cvar(ret: pd.Series, level: float = 0.05) -> float:
    var = _var(ret, level)
    return ret[ret <= var].mean()


def _omega_ratio(ret: pd.Series, threshold: float = RF_DAILY) -> float:
    """Omega Ratio = E[max(R-threshold,0)] / E[max(threshold-R,0)]"""
    above = ret[ret > threshold] - threshold
    below = threshold - ret[ret <= threshold]
    denom = below.sum()
    return above.sum() / denom if denom > 0 else np.nan


def _information_ratio(ret: pd.Series,
                        benchmark: pd.Series) -> float:
    aligned   = ret.align(benchmark, join='inner')
    active    = aligned[0] - aligned[1]
    te        = active.std() * np.sqrt(PERIODS)
    ann_alpha = active.mean() * PERIODS
    return ann_alpha / te if te > 0 else np.nan


def _alpha_beta(ret: pd.Series,
                benchmark: pd.Series) -> tuple[float, float]:
    r, b = ret.align(benchmark, join='inner')
    if len(r) < 30:
        return np.nan, np.nan
    cov_mat = np.cov(r.values, b.values)
    beta    = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] > 0 else np.nan
    alpha   = (r.mean() - beta * b.mean()) * PERIODS if not np.isnan(beta) else np.nan
    return alpha, beta


def _tracking_error(ret: pd.Series, benchmark: pd.Series) -> float:
    r, b  = ret.align(benchmark, join='inner')
    active = r - b
    return active.std() * np.sqrt(PERIODS)


def _up_down_capture(ret: pd.Series,
                     benchmark: pd.Series) -> tuple[float, float]:
    r, b    = ret.align(benchmark, join='inner')
    up_bm   = b[b >= 0]
    dn_bm   = b[b <  0]
    up_port = r.loc[up_bm.index]
    dn_port = r.loc[dn_bm.index]
    up_cap  = (up_port.mean() / up_bm.mean()) if len(up_bm) > 5 and up_bm.mean() != 0 else np.nan
    dn_cap  = (dn_port.mean() / dn_bm.mean()) if len(dn_bm) > 5 and dn_bm.mean() != 0 else np.nan
    return up_cap, dn_cap


def compute_all_metrics(portfolios: dict,
                         benchmark: pd.Series | None = None) -> pd.DataFrame:
    """
    Compute the full metric suite for every strategy.
    Returns a DataFrame with strategies as rows and metrics as columns.
    """
    rows = []
    all_strategies = list(portfolios.items())
    if benchmark is not None:
        all_strategies.append(('NIFTY 50', benchmark))

    for name, ret in all_strategies:
        ret = ret.dropna()
        if len(ret) < 30:
            continue

        total_ret = (1 + ret).prod() - 1
        cagr_     = _cagr(ret)
        vol_      = _ann_vol(ret)
        sharpe_   = _sharpe(ret)
        sortino_  = _sortino(ret)
        mdd_      = _max_drawdown(ret)
        calmar_   = _calmar(ret)
        avg_dd_   = _avg_drawdown(ret)
        dd_dur_   = _max_drawdown_duration(ret)
        var95_    = _var(ret, 0.05)
        var99_    = _var(ret, 0.01)
        cvar95_   = _cvar(ret, 0.05)
        cvar99_   = _cvar(ret, 0.01)
        omega_    = _omega_ratio(ret)
        skew_     = float(ret.skew())
        kurt_     = float(ret.kurtosis())
        semi_vol_ = ret[ret < 0].std() * np.sqrt(PERIODS)

        # Benchmark-relative (skip for NIFTY 50 itself)
        ir_, alpha_, beta_, te_, up_cap_, dn_cap_ = [np.nan] * 6
        if benchmark is not None and name != 'NIFTY 50':
            ir_           = _information_ratio(ret, benchmark)
            alpha_, beta_ = _alpha_beta(ret, benchmark)
            te_           = _tracking_error(ret, benchmark)
            up_cap_, dn_cap_ = _up_down_capture(ret, benchmark)

        rows.append({
            'Strategy'              : name,
            'Total Return (%)'      : round(total_ret * 100, 2),
            'CAGR (%)'              : round(cagr_ * 100, 2),
            'Ann. Volatility (%)'   : round(vol_ * 100, 2),
            'Semi-Volatility (%)'   : round(semi_vol_ * 100, 2),
            'Sharpe Ratio'          : round(sharpe_, 3),
            'Sortino Ratio'         : round(sortino_, 3),
            'Calmar Ratio'          : round(calmar_, 3),
            'Omega Ratio'           : round(omega_, 3),
            'Max Drawdown (%)'      : round(mdd_ * 100, 2),
            'Avg Drawdown (%)'      : round(avg_dd_ * 100, 2),
            'Max DD Duration (days)': int(dd_dur_),
            'VaR 95% (daily)'       : round(var95_ * 100, 3),
            'VaR 99% (daily)'       : round(var99_ * 100, 3),
            'CVaR 95% (daily)'      : round(cvar95_ * 100, 3),
            'CVaR 99% (daily)'      : round(cvar99_ * 100, 3),
            'Skewness'              : round(skew_, 3),
            'Excess Kurtosis'       : round(kurt_, 3),
            'Alpha (ann.)'          : round(alpha_ * 100, 3) if not np.isnan(alpha_) else np.nan,
            'Beta'                  : round(beta_, 3),
            'Tracking Error (%)'    : round(te_ * 100, 3) if not np.isnan(te_) else np.nan,
            'Info Ratio'            : round(ir_, 3),
            'Up Capture (%)'        : round(up_cap_ * 100, 2) if not np.isnan(up_cap_) else np.nan,
            'Down Capture (%)'      : round(dn_cap_ * 100, 2) if not np.isnan(dn_cap_) else np.nan,
        })

    df = pd.DataFrame(rows).set_index('Strategy')
    return df


# ══════════════════════════════════════════════════════════════════
# 3. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════

def _get_color(name):
    for key, col in STRATEGY_COLORS.items():
        if key in name:
            return col
    return '#999999'


# ── 3a. Cumulative Growth ──────────────────────────────────────────

def plot_growth_all(portfolios, benchmark=None):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                              gridspec_kw={'height_ratios': [3, 1]})

    ax_growth = axes[0]
    ax_dd     = axes[1]

    for name, ret in portfolios.items():
        growth = (1 + ret).cumprod()
        color  = _get_color(name)
        ls     = '--' if name == 'Equal Weight' else '-'
        ax_growth.plot(growth.index, growth.values,
                       label=name, color=color, linewidth=1.6, linestyle=ls)
        dd = (growth - growth.cummax()) / growth.cummax()
        ax_dd.plot(dd.index, dd.values * 100,
                   color=color, linewidth=1.0, linestyle=ls, alpha=0.7)

    if benchmark is not None:
        bm_growth = (1 + benchmark).cumprod()
        bm_dd     = (bm_growth - bm_growth.cummax()) / bm_growth.cummax()
        ax_growth.plot(bm_growth.index, bm_growth.values,
                       label='NIFTY 50', color='black',
                       linewidth=2.2, linestyle=':', zorder=10)
        ax_dd.plot(bm_dd.index, bm_dd.values * 100,
                   color='black', linewidth=1.5, linestyle=':', alpha=0.9)

    ax_growth.set_title(
        "All Portfolio Strategies — Cumulative Growth vs NIFTY 50\n"
        "(rf = 7% p.a. | Ledoit-Wolf covariance | realistic constraints)",
        fontsize=12)
    ax_growth.set_ylabel("Portfolio Value (₹1 invested)")
    ax_growth.legend(fontsize=8, ncol=2)
    ax_growth.grid(True, alpha=0.25)

    ax_dd.set_title("Drawdown (%)", fontsize=10)
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_xlabel("Date")
    ax_dd.axhline(0, color='black', linewidth=0.8)
    ax_dd.fill_between(ax_dd.lines[0].get_xdata(),
                        ax_dd.lines[0].get_ydata(), 0,
                        alpha=0.05, color='red')
    ax_dd.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig('reports/perf_growth_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/perf_growth_all.png")


# ── 3b. Underwater Plot ────────────────────────────────────────────

def plot_underwater(portfolios, benchmark=None):
    n_strats = len(portfolios) + (1 if benchmark is not None else 0)
    fig, axes = plt.subplots(n_strats, 1,
                              figsize=(14, 2.5 * n_strats),
                              sharex=True)
    if n_strats == 1:
        axes = [axes]

    all_series = list(portfolios.items())
    if benchmark is not None:
        all_series.append(('NIFTY 50', benchmark))

    for ax, (name, ret) in zip(axes, all_series):
        growth = (1 + ret).cumprod()
        dd     = (growth - growth.cummax()) / growth.cummax() * 100
        color  = _get_color(name)
        ax.fill_between(dd.index, dd.values, 0,
                        alpha=0.55, color=color, linewidth=0)
        ax.plot(dd.index, dd.values, color=color, linewidth=0.8)
        ax.axhline(0, color='black', linewidth=0.6)
        mdd = dd.min()
        ax.set_ylabel(f"{name}\nMax DD: {mdd:.1f}%", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(min(dd.min() * 1.15, -1), 2)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Underwater Drawdown Plot — All Strategies",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig('reports/perf_drawdown_underwater.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/perf_drawdown_underwater.png")


# ── 3c. Rolling Sharpe, Sortino, Volatility ───────────────────────

def plot_rolling_metrics(portfolios, benchmark=None):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    all_series = dict(portfolios)
    if benchmark is not None:
        all_series['NIFTY 50'] = benchmark

    for name, ret in all_series.items():
        color = _get_color(name)
        ls    = ':' if name == 'NIFTY 50' else ('--' if name == 'Equal Weight' else '-')
        lw    = 2.0 if name == 'NIFTY 50' else 1.4

        # Rolling Sharpe
        roll_ret = ret.rolling(ROLLING_WIN).mean() * PERIODS
        roll_vol = ret.rolling(ROLLING_WIN).std() * np.sqrt(PERIODS)
        roll_sh  = (roll_ret - RF) / roll_vol.replace(0, np.nan)
        axes[0].plot(roll_sh.index, roll_sh.values,
                     label=name, color=color, linewidth=lw, linestyle=ls)

        # Rolling Sortino
        roll_dn  = ret.copy()
        roll_dn[roll_dn > 0] = 0
        roll_dn_vol = roll_dn.rolling(ROLLING_WIN).std() * np.sqrt(PERIODS)
        roll_so     = (roll_ret - RF) / roll_dn_vol.replace(0, np.nan)
        axes[1].plot(roll_so.index, roll_so.values,
                     color=color, linewidth=lw, linestyle=ls)

        # Rolling Volatility
        axes[2].plot(roll_vol.index, roll_vol.values * 100,
                     color=color, linewidth=lw, linestyle=ls)

    for ax, title in zip(axes, ['Rolling Sharpe Ratio (3M)',
                                  'Rolling Sortino Ratio (3M)',
                                  'Rolling Annualised Volatility (%) (3M)']):
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color='black', linewidth=0.7, alpha=0.5)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel(title.split('(')[0].strip())

    axes[0].legend(fontsize=8, ncol=2, loc='upper left')
    axes[-1].set_xlabel("Date")
    fig.suptitle("Rolling Risk-Adjusted Metrics — All Strategies",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('reports/perf_rolling_metrics.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/perf_rolling_metrics.png")


# ── 3d. Monthly Return Heatmap ────────────────────────────────────

def plot_monthly_heatmap(portfolios):
    """One heatmap per strategy showing calendar monthly returns."""
    strategies = list(portfolios.keys())
    n          = len(strategies)
    fig, axes  = plt.subplots(n, 1, figsize=(16, 3.2 * n))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, strategies):
        ret       = portfolios[name]
        monthly   = ret.resample('ME').apply(lambda r: (1 + r).prod() - 1)
        monthly_df = monthly.to_frame('ret')
        monthly_df['year']  = monthly_df.index.year
        monthly_df['month'] = monthly_df.index.month

        pivot = monthly_df.pivot(index='year', columns='month', values='ret') * 100
        pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun',
                          'Jul','Aug','Sep','Oct','Nov','Dec'][:len(pivot.columns)]

        vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 1)
        im   = ax.imshow(pivot.values, aspect='auto',
                         cmap='RdYlGn', vmin=-vmax, vmax=vmax,
                         interpolation='nearest')

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(f"{name} — Monthly Returns (%)", fontsize=10)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.1f}',
                            ha='center', va='center',
                            fontsize=6.5,
                            color='black' if abs(val) < vmax * 0.6 else 'white')
        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02,
                     label='Monthly Return (%)')

    plt.tight_layout()
    plt.savefig('reports/perf_monthly_heatmap.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/perf_monthly_heatmap.png")


# ── 3e. Risk-Return Scatter ────────────────────────────────────────

def plot_risk_return_scatter(metrics_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Vol vs CAGR
    ax = axes[0]
    for name, row in metrics_df.iterrows():
        color = _get_color(name)
        marker = '^' if name == 'NIFTY 50' else ('s' if name == 'Equal Weight' else 'o')
        ax.scatter(row['Ann. Volatility (%)'], row['CAGR (%)'],
                   color=color, s=160, marker=marker, zorder=5, edgecolors='white')
        ax.annotate(name.replace(' ', '\n'),
                    (row['Ann. Volatility (%)'], row['CAGR (%)']),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=7.5, color=color)

    # Plot Capital Market Line
    vols = np.linspace(0, metrics_df['Ann. Volatility (%)'].max() * 1.1, 100)
    best_sharpe = metrics_df['Sharpe Ratio'].max()
    cml_rets    = RF * 100 + best_sharpe * vols
    ax.plot(vols, cml_rets, 'k--', linewidth=1, alpha=0.4, label='CML (best Sharpe)')
    ax.scatter([0], [RF * 100], color='black', s=80, marker='D', zorder=6)
    ax.annotate(f'rf={RF*100:.0f}%', (0, RF * 100),
                textcoords='offset points', xytext=(4, -12), fontsize=8)

    ax.set_xlabel("Annualised Volatility (%)", fontsize=10)
    ax.set_ylabel("CAGR (%)", fontsize=10)
    ax.set_title("Risk-Return Trade-off", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Panel 2: Max DD vs Sharpe
    ax = axes[1]
    for name, row in metrics_df.iterrows():
        color  = _get_color(name)
        marker = '^' if name == 'NIFTY 50' else ('s' if name == 'Equal Weight' else 'o')
        ax.scatter(abs(row['Max Drawdown (%)']), row['Sharpe Ratio'],
                   color=color, s=160, marker=marker, zorder=5, edgecolors='white')
        ax.annotate(name.replace(' ', '\n'),
                    (abs(row['Max Drawdown (%)']), row['Sharpe Ratio']),
                    textcoords='offset points', xytext=(4, 4),
                    fontsize=7.5, color=color)

    ax.set_xlabel("|Max Drawdown| (%)", fontsize=10)
    ax.set_ylabel("Sharpe Ratio", fontsize=10)
    ax.set_title("Sharpe Ratio vs Maximum Drawdown", fontsize=11)
    ax.axhline(0, color='black', linewidth=0.7, alpha=0.5)
    ax.grid(True, alpha=0.25)

    plt.suptitle("Performance Space — All Strategies vs NIFTY 50",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('reports/perf_risk_return_scatter.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/perf_risk_return_scatter.png")


# ── 3f. Tail Risk Bar Chart ────────────────────────────────────────

def plot_tail_risk(metrics_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    strategies = metrics_df.index.tolist()
    colors     = [_get_color(n) for n in strategies]
    x          = np.arange(len(strategies))
    bar_kw     = dict(edgecolor='white', linewidth=0.5)

    # VaR 95%
    ax = axes[0, 0]
    vals = metrics_df['VaR 95% (daily)'].values
    ax.bar(x, abs(vals), color=colors, alpha=0.85, **bar_kw)
    ax.set_title('Daily VaR 95% (|worst 5% day|, %)', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([n.replace(' ','\n') for n in strategies], fontsize=8)
    ax.set_ylabel('%'); ax.grid(True, axis='y', alpha=0.3)

    # CVaR 95%
    ax = axes[0, 1]
    vals = metrics_df['CVaR 95% (daily)'].values
    ax.bar(x, abs(vals), color=colors, alpha=0.85, **bar_kw)
    ax.set_title('Daily CVaR 95% (Expected Shortfall, %)', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([n.replace(' ','\n') for n in strategies], fontsize=8)
    ax.set_ylabel('%'); ax.grid(True, axis='y', alpha=0.3)

    # Skewness
    ax = axes[1, 0]
    vals = metrics_df['Skewness'].values
    bar_colors = ['#2ca02c' if v > 0 else '#d62728' for v in vals]
    ax.bar(x, vals, color=bar_colors, alpha=0.85, **bar_kw)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title('Return Skewness (positive = fat right tail)', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([n.replace(' ','\n') for n in strategies], fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    # Excess Kurtosis
    ax = axes[1, 1]
    vals = metrics_df['Excess Kurtosis'].values
    ax.bar(x, vals, color=colors, alpha=0.85, **bar_kw)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--',
               label='Normal = 0')
    ax.set_title('Excess Kurtosis (>0 = fat tails)', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([n.replace(' ','\n') for n in strategies], fontsize=8)
    ax.legend(fontsize=8); ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle("Tail Risk Metrics — All Strategies",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('reports/perf_tail_risk.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/perf_tail_risk.png")


# ── 3g. Radar Chart ────────────────────────────────────────────────

def plot_radar_chart(metrics_df):
    """
    Spider/radar chart comparing strategies across 6 normalised dimensions:
      Sharpe, Sortino, Calmar, -Max DD (inverted), CAGR, -CVaR (inverted)
    """
    dims = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
            'CAGR (%)', 'Max Drawdown (%)', 'CVaR 95% (daily)']
    dim_labels = ['Sharpe', 'Sortino', 'Calmar',
                  'CAGR', 'Low\nDrawdown', 'Low\nCVaR']
    # For Max DD and CVaR, lower is better → invert for radar
    invert = {'Max Drawdown (%)': True, 'CVaR 95% (daily)': True}

    df = metrics_df[dims].copy()
    for col in dims:
        series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        mn, mx = series.min(), series.max()
        rng    = mx - mn
        if rng < 1e-9:
            df[col] = 0.5
        else:
            df[col] = (df[col] - mn) / rng
        if invert.get(col, False):
            df[col] = 1 - df[col]

    N      = len(dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9),
                            subplot_kw=dict(polar=True))

    for name, row in df.iterrows():
        vals = row[dims].tolist()
        vals += vals[:1]
        color = _get_color(name)
        ls    = ':' if name == 'NIFTY 50' else '-'
        ax.plot(angles, vals, color=color, linewidth=2.0,
                linestyle=ls, label=name)
        ax.fill(angles, vals, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, size=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7)
    ax.set_title("Strategy Performance Radar\n"
                 "(normalised 0–1 per metric; higher = better)",
                 pad=20, fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
              fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/perf_radar_chart.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/perf_radar_chart.png")


# ══════════════════════════════════════════════════════════════════
# 4. PRINT SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════

def print_metrics_table(metrics_df: pd.DataFrame):
    print("\n" + "=" * 120)
    print("COMPREHENSIVE PERFORMANCE METRICS  (rf = 7% p.a. | India 10Y G-Sec)")
    print("=" * 120)

    # Block 1: Return & Risk
    cols1 = ['Total Return (%)', 'CAGR (%)', 'Ann. Volatility (%)',
             'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Omega Ratio']
    print("\n── Return & Risk Ratios ──")
    print(metrics_df[cols1].to_string())

    # Block 2: Drawdown
    cols2 = ['Max Drawdown (%)', 'Avg Drawdown (%)', 'Max DD Duration (days)']
    print("\n── Drawdown Analysis ──")
    print(metrics_df[cols2].to_string())

    # Block 3: Tail risk
    cols3 = ['VaR 95% (daily)', 'VaR 99% (daily)',
             'CVaR 95% (daily)', 'CVaR 99% (daily)',
             'Skewness', 'Excess Kurtosis']
    print("\n── Tail Risk & Distribution ──")
    print(metrics_df[cols3].to_string())

    # Block 4: Benchmark-relative
    cols4 = ['Alpha (ann.)', 'Beta', 'Info Ratio',
             'Tracking Error (%)', 'Up Capture (%)', 'Down Capture (%)']
    bm_cols = [c for c in cols4 if c in metrics_df.columns]
    non_nan = metrics_df[bm_cols].dropna(how='all')
    if not non_nan.empty:
        print("\n── Benchmark-Relative (vs NIFTY 50) ──")
        print(non_nan.to_string())

    # Best-in-class summary
    print("\n── Best-in-Class ──")
    best_metrics = {
        'Highest Sharpe'   : metrics_df['Sharpe Ratio'].idxmax(),
        'Highest Sortino'  : metrics_df['Sortino Ratio'].idxmax(),
        'Highest Calmar'   : metrics_df['Calmar Ratio'].idxmax(),
        'Highest CAGR'     : metrics_df['CAGR (%)'].idxmax(),
        'Lowest Volatility': metrics_df['Ann. Volatility (%)'].idxmin(),
        'Smallest Max DD'  : metrics_df['Max Drawdown (%)'].idxmax(),
        'Lowest CVaR 95%'  : metrics_df['CVaR 95% (daily)'].idxmax(),
    }
    for metric, strategy in best_metrics.items():
        print(f"  {metric:22s}: {strategy}")


# ══════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 70)
    print("COMPREHENSIVE PERFORMANCE METRICS  —  Week 7")
    print("=" * 70)

    # Load underlying returns for EW fallback + alignment
    print("\nLoading base return data...")
    base_returns, tickers, n = load_returns()
    print(f"   Base data: {n} tickers | "
          f"{base_returns.index[0].date()} → {base_returns.index[-1].date()}")

    # Load all portfolio strategies
    print("\nLoading portfolio strategies...")
    portfolios = load_all_portfolios(base_returns)
    print(f"   Strategies loaded: {len(portfolios)}")

    # Determine common date range from loaded portfolios
    all_idx  = [ret.index for ret in portfolios.values()]
    start_dt = max(idx.min() for idx in all_idx)
    end_dt   = min(idx.max() for idx in all_idx)
    ref_idx  = portfolios['Equal Weight'].index

    # Download NIFTY 50 benchmark
    print("\nDownloading NIFTY 50 benchmark...")
    benchmark = load_benchmark(start_dt, end_dt, ref_idx)

    # Align all portfolios to equal-weight date range
    for name in list(portfolios.keys()):
        portfolios[name] = portfolios[name].reindex(ref_idx, method='ffill').dropna()

    # ── Compute metrics ──────────────────────────────────────────
    print("\nComputing performance metrics...")
    metrics_df = compute_all_metrics(portfolios, benchmark)
    print_metrics_table(metrics_df)

    # Save CSV
    metrics_df.to_csv('reports/perf_summary_table.csv')
    print("\nSaved: reports/perf_summary_table.csv")

    # ── Plots ────────────────────────────────────────────────────
    print("\nGenerating visualisations...")
    plot_growth_all(portfolios, benchmark)
    plot_underwater(portfolios, benchmark)
    plot_rolling_metrics(portfolios, benchmark)
    plot_monthly_heatmap(portfolios)
    plot_risk_return_scatter(metrics_df)
    plot_tail_risk(metrics_df)
    plot_radar_chart(metrics_df)

    print("\n✅ Performance metrics complete.")
    print("   CSV  → reports/perf_summary_table.csv")
    print("   Plots:")
    for f in ['perf_growth_all', 'perf_drawdown_underwater',
              'perf_rolling_metrics', 'perf_monthly_heatmap',
              'perf_risk_return_scatter', 'perf_tail_risk',
              'perf_radar_chart']:
        print(f"         reports/{f}.png")