"""
utils.py — Shared utilities for all portfolio model files
==========================================================
Imported by:
    equal_weight.py
    markowitz.py
    risk_parity.py
    black_litterman.py

Contains:
  - Global constants  (risk-free rate, BL params, rebalance freq)
  - Sector map & cyclical group
  - Market-cap proxy shares table
  - Data loading helpers
  - Ledoit-Wolf covariance estimator
  - Constraint builder  (sector caps + cyclical cap)
  - Market-cap weight computation
  - Performance metrics  (max drawdown, Sortino, Calmar)
  - Sector-stacked-bar plot helper
  - Rolling Sharpe plot helper
  - Summary table printer
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# DIRECTORY SETUP
# ------------------------------------------------------------------

os.makedirs("data/portfolio", exist_ok=True)
os.makedirs("reports", exist_ok=True)


# ------------------------------------------------------------------
# GLOBAL CONSTANTS
# ------------------------------------------------------------------

# India 10-year G-Sec yield — explicit risk-free rate
RISK_FREE_RATE_ANNUAL = 0.07
RISK_FREE_RATE_DAILY  = RISK_FREE_RATE_ANNUAL / 252

# Transaction cost — 10 bps one-way (brokerage + impact + STT for large-caps)
TRANSACTION_COST = 0.0010

# Black-Litterman parameters
LAMBDA          = 2.5    # market risk aversion (standard institutional value)
TAU             = 0.05   # uncertainty in the prior (standard in literature)

# Rolling BL schedule
REBALANCE_FREQ  = 21     # trading days (~monthly)
MOMENTUM_WINDOW = 126    # 6-month momentum signal window
MIN_HISTORY     = MOMENTUM_WINDOW + 63   # ~9 months of data before first BL solve


# ------------------------------------------------------------------
# SECTOR MAP  (verified .NS tickers)
# ------------------------------------------------------------------

SECTOR_MAP = {
    'TCS.NS'      : 'IT',          'INFY.NS'      : 'IT',
    'HCLTECH.NS'  : 'IT',          'WIPRO.NS'     : 'IT',
    'TECHM.NS'    : 'IT',          'LTIM.NS'      : 'IT',
    'HDFCBANK.NS' : 'Banking',     'ICICIBANK.NS' : 'Banking',
    'AXISBANK.NS' : 'Banking',     'KOTAKBANK.NS' : 'Banking',
    'SBIN.NS'     : 'Banking',     'INDUSINDBK.NS': 'Banking',
    'BAJFINANCE.NS': 'Financials', 'BAJAJFINSV.NS': 'Financials',
    'HDFCLIFE.NS' : 'Financials',  'SBILIFE.NS'   : 'Financials',
    'RELIANCE.NS' : 'Energy',      'ONGC.NS'      : 'Energy',
    'BPCL.NS'     : 'Energy',      'NTPC.NS'      : 'Energy',
    'POWERGRID.NS': 'Energy',      'COALINDIA.NS' : 'Energy',
    'MARUTI.NS'   : 'Auto',        'M&M.NS'       : 'Auto',
    'BAJAJ-AUTO.NS': 'Auto',       'HEROMOTOCO.NS': 'Auto',
    'EICHERMOT.NS': 'Auto',
    'SUNPHARMA.NS': 'Pharma',      'CIPLA.NS'     : 'Pharma',
    'DRREDDY.NS'  : 'Pharma',      'DIVISLAB.NS'  : 'Pharma',
    'HINDUNILVR.NS': 'FMCG',       'ITC.NS'       : 'FMCG',
    'BRITANNIA.NS': 'FMCG',        'NESTLEIND.NS' : 'FMCG',
    'LT.NS'       : 'Infra',       'ADANIPORTS.NS': 'Infra',
    'ULTRACEMCO.NS': 'Cement',     'SHREECEM.NS'  : 'Cement',
    'GRASIM.NS'   : 'Cement',
    'ADANIENT.NS' : 'Conglomerate',
    'TITAN.NS'    : 'Consumer',    'ASIANPAINT.NS': 'Consumer',
    'APOLLOHOSP.NS': 'Healthcare',
    'HINDALCO.NS' : 'Metals',      'JSWSTEEL.NS'  : 'Metals',
    'TATASTEEL.NS': 'Metals',
    'BHARTIARTL.NS': 'Telecom',
    'UPL.NS'      : 'Chemicals',
}

# Cyclical group — combined exposure capped at 40%
CYCLICAL_SECTORS = {'Auto', 'Metals', 'Energy', 'Chemicals'}

# Colour palette for sector stacked-bar charts
SECTOR_COLOR_MAP = {
    'IT'          : '#4C72B0',
    'Banking'     : '#DD8452',
    'Financials'  : '#55A868',
    'Energy'      : '#C44E52',
    'Auto'        : '#8172B2',
    'Pharma'      : '#937860',
    'FMCG'        : '#DA8BC3',
    'Infra'       : '#8C8C8C',
    'Cement'      : '#CCB974',
    'Conglomerate': '#64B5CD',
    'Consumer'    : '#B5BD61',
    'Healthcare'  : '#EDBC79',
    'Metals'      : '#A0522D',
    'Telecom'     : '#7FB5B5',
    'Chemicals'   : '#D4AC6E',
    'Other'       : '#CCCCCC',
}


# ------------------------------------------------------------------
# FREE-FLOAT SHARES TABLE  (approx., crore shares, 2023-24)
# Source: NSE Nifty 50 methodology / public disclosures.
# Adjust with latest NSE Nifty 50 factsheet if needed.
# ------------------------------------------------------------------

APPROX_FREEFLOAT_SHARES_CR = {
    'RELIANCE.NS' : 1350, 'HDFCBANK.NS'  : 760,  'ICICIBANK.NS' : 700,
    'INFY.NS'     : 420,  'TCS.NS'       : 365,  'KOTAKBANK.NS' : 199,
    'HINDUNILVR.NS': 235, 'AXISBANK.NS'  : 310,  'BHARTIARTL.NS': 560,
    'ITC.NS'      : 1250, 'SBIN.NS'      : 890,  'BAJFINANCE.NS': 60,
    'LT.NS'       : 140,  'HCLTECH.NS'   : 272,  'WIPRO.NS'     : 548,
    'SUNPHARMA.NS': 240,  'ADANIPORTS.NS': 215,  'MARUTI.NS'    : 30,
    'ULTRACEMCO.NS': 29,  'TITAN.NS'     : 89,   'NTPC.NS'      : 965,
    'POWERGRID.NS': 930,  'ONGC.NS'      : 1258, 'TECHM.NS'     : 97,
    'M&M.NS'      : 124,  'BAJAJFINSV.NS': 159,  'ASIANPAINT.NS': 96,
    'NESTLEIND.NS': 96,   'DRREDDY.NS'   : 17,   'CIPLA.NS'     : 81,
    'COALINDIA.NS': 615,  'EICHERMOT.NS' : 27,   'DIVISLAB.NS'  : 27,
    'HDFCLIFE.NS' : 202,  'SBILIFE.NS'   : 100,  'GRASIM.NS'    : 66,
    'HINDALCO.NS' : 224,  'JSWSTEEL.NS'  : 241,  'TATASTEEL.NS' : 1226,
    'BAJAJ-AUTO.NS': 29,  'HEROMOTOCO.NS': 20,   'ADANIENT.NS'  : 113,
    'SHREECEM.NS' : 4,    'APOLLOHOSP.NS': 14,   'UPL.NS'       : 76,
    'INDUSINDBK.NS': 78,  'LTIM.NS'      : 35,   'BPCL.NS'      : 433,
}


# ------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------

def load_returns():
    """
    Load and clean the daily returns matrix.
    Returns: (returns DataFrame, tickers list, n int)
    """
    returns = pd.read_csv(
        "data/processed_data/return_matrix.csv",
        index_col='date', parse_dates=True
    )
    # .fillna(method='ffill') deprecated in pandas 2.2+ → .ffill()
    returns = returns.dropna(axis=1, how='all').ffill().dropna()
    return returns, returns.columns.tolist(), returns.shape[1]


def load_prices(tickers):
    """
    Load and clean the daily price matrix, aligned to the given tickers.
    Returns: price_matrix DataFrame
    """
    prices = pd.read_csv(
        "data/processed_data/price_matrix.csv",
        index_col='date', parse_dates=True
    )
    prices = prices.dropna(axis=1, how='all').ffill()
    prices = prices[[t for t in tickers if t in prices.columns]]
    return prices


# ------------------------------------------------------------------
# LEDOIT-WOLF SHRINKAGE COVARIANCE
# ------------------------------------------------------------------

def compute_lw_cov(returns_df):
    """
    Ledoit-Wolf shrinkage covariance estimator.
    Shrinks toward scaled identity — reduces noise when n_assets is large
    relative to n_observations.

    Returns:
        cov_np  — numpy array (n x n)
        cov_df  — pandas DataFrame with ticker labels
    """
    lw = LedoitWolf()
    lw.fit(returns_df.values)
    cov_np = lw.covariance_
    cov_df = pd.DataFrame(cov_np,
                          index=returns_df.columns,
                          columns=returns_df.columns)
    return cov_np, cov_df


# ------------------------------------------------------------------
# CONSTRAINT BUILDER
# ------------------------------------------------------------------

def build_constraints(tickers, sector_map=None,
                      max_sector=0.30, max_cyclical=0.40):
    """
    Build SLSQP constraint dicts:
      - weights sum to 1
      - each sector  ≤ max_sector   (default 30%)
      - cyclical sectors combined ≤ max_cyclical  (default 40%)

    Args:
        tickers      : list of ticker strings
        sector_map   : dict mapping ticker → sector  (defaults to SECTOR_MAP)
        max_sector   : per-sector weight ceiling
        max_cyclical : combined cyclical-sector ceiling

    Returns:
        list of constraint dicts for scipy.optimize.minimize
    """
    if sector_map is None:
        sector_map = SECTOR_MAP

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]

    sectors = set(sector_map.get(t, 'Other') for t in tickers)
    for sector in sectors:
        idx = [i for i, t in enumerate(tickers)
               if sector_map.get(t, 'Other') == sector]
        if not idx:
            continue
        constraints.append({
            'type': 'ineq',
            'fun' : (lambda w, ix=idx: max_sector - np.sum(w[ix]))
        })

    cyc_idx = [i for i, t in enumerate(tickers)
               if sector_map.get(t, 'Other') in CYCLICAL_SECTORS]
    if cyc_idx:
        constraints.append({
            'type': 'ineq',
            'fun' : (lambda w, ix=cyc_idx: max_cyclical - np.sum(w[ix]))
        })

    return constraints


# ------------------------------------------------------------------
# MARKET-CAP PROXY WEIGHTS
# ------------------------------------------------------------------

def compute_market_cap_weights(tickers, price_matrix,
                               shares_map=None):
    """
    Approximate free-float market-cap weights:
        proxy_mcap_i = avg_price_i (last 63 days) * freefloat_shares_i

    Tickers not in shares_map fall back to the mean known mcap.
    Normalised to sum = 1.

    Args:
        tickers      : list of ticker strings
        price_matrix : DataFrame of daily prices
        shares_map   : dict ticker → freefloat shares (crores)
                       defaults to APPROX_FREEFLOAT_SHARES_CR

    Returns:
        numpy array of weights, length = len(tickers)
    """
    if shares_map is None:
        shares_map = APPROX_FREEFLOAT_SHARES_CR

    avg_prices = price_matrix.iloc[-63:].mean()

    mcap = {}
    for t in tickers:
        shares = shares_map.get(t, None)
        price  = avg_prices.get(t, None)
        if shares is not None and price is not None and not np.isnan(price):
            mcap[t] = price * shares
        else:
            mcap[t] = None

    known_avg = np.mean([v for v in mcap.values() if v is not None])
    for t in tickers:
        if mcap[t] is None:
            mcap[t] = known_avg

    weights = np.array([mcap[t] for t in tickers], dtype=float)
    weights /= weights.sum()
    return weights


# ------------------------------------------------------------------
# PERFORMANCE METRICS
# ------------------------------------------------------------------

def max_drawdown(growth_series):
    """
    Maximum peak-to-trough drawdown.
    Returns the most-negative value of (value - peak) / peak.
    """
    roll_max = growth_series.cummax()
    drawdown = (growth_series - roll_max) / roll_max
    return drawdown.min()


def sortino_ratio(ret_series, rf_annual=RISK_FREE_RATE_ANNUAL):
    """
    Annualised Sortino ratio using downside deviation below zero excess return.
    """
    rf_daily     = rf_annual / 252
    excess       = ret_series - rf_daily
    downside     = excess[excess < 0]
    if len(downside) == 0:
        return np.nan
    downside_std = downside.std() * np.sqrt(252)
    ann_excess   = excess.mean() * 252
    return ann_excess / downside_std if downside_std > 0 else np.nan


def calmar_ratio(ret_series, growth_series, rf_annual=RISK_FREE_RATE_ANNUAL):
    """
    Calmar ratio = (annualised return − rf) / |max drawdown|.
    """
    ann_ret = ret_series.mean() * 252
    mdd     = abs(max_drawdown(growth_series))
    return (ann_ret - rf_annual) / mdd if mdd > 0 else np.nan


# ------------------------------------------------------------------
# SUMMARY TABLE PRINTER
# ------------------------------------------------------------------

def print_summary(all_growths, all_ret_series, note=""):
    """
    Print a full performance table:
        Total Return | Ann Vol | Sharpe | Sortino | Max DD | Calmar

    Args:
        all_growths    : dict  name → cumulative growth Series
        all_ret_series : dict  name → daily return Series
        note           : optional footer string
    """
    print("\n" + "=" * 100)
    print("PERFORMANCE SUMMARY  (rf = 7% p.a. | India 10Y G-Sec)")
    if note:
        print(f"  {note}")
    print("=" * 100)
    print(f"\n{'Strategy':<22} {'Total Ret':>10} {'Ann Vol':>10} "
          f"{'Sharpe':>8} {'Sortino':>9} {'Max DD':>9} {'Calmar':>8}")
    print("-" * 82)
    for name, growth in all_growths.items():
        ret_s     = all_ret_series[name]
        total_ret = growth.iloc[-1] - 1
        ann_vol   = ret_s.std() * np.sqrt(252)
        ann_ret   = ret_s.mean() * 252
        sharpe    = (ann_ret - RISK_FREE_RATE_ANNUAL) / ann_vol if ann_vol > 0 else 0
        sortino   = sortino_ratio(ret_s)
        mdd       = max_drawdown(growth)
        calmar    = calmar_ratio(ret_s, growth)
        print(f"  {name:<20} {total_ret*100:>9.1f}%  {ann_vol*100:>9.1f}%  "
              f"{sharpe:>7.3f}  {sortino:>8.3f}  {mdd*100:>8.1f}%  {calmar:>7.3f}")


# ------------------------------------------------------------------
# PLOT HELPERS
# ------------------------------------------------------------------

def plot_growth_curves(all_growths, title, outpath,
                       dashed_keys=None):
    """
    Line chart of cumulative portfolio value for each strategy.

    Args:
        all_growths  : dict name → growth Series
        title        : chart title string
        outpath      : file path for the saved PNG
        dashed_keys  : set/list of strategy names to draw as dashed
    """
    dashed_keys = set(dashed_keys or [])
    fig, ax = plt.subplots(figsize=(12, 6))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(all_growths)))
    for i, (name, growth) in enumerate(all_growths.items()):
        ls = '--' if name in dashed_keys else '-'
        ax.plot(growth.index, growth.values, label=name,
                color=colors[i], linewidth=1.5, linestyle=ls)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Portfolio Value")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")


def plot_rolling_sharpe(all_ret_series, title, outpath,
                        window=63, dashed_keys=None):
    """
    Rolling annualised Sharpe ratio chart (default 3-month window).

    Args:
        all_ret_series : dict name → daily return Series
        title          : chart title string
        outpath        : file path for the saved PNG
        window         : rolling window in trading days
        dashed_keys    : set/list of strategy names to draw as dashed
    """
    dashed_keys = set(dashed_keys or [])
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, ret_s in all_ret_series.items():
        if len(ret_s) < window:
            continue
        roll_ret    = ret_s.rolling(window).mean() * 252
        roll_vol    = ret_s.rolling(window).std() * np.sqrt(252)
        roll_sharpe = (roll_ret - RISK_FREE_RATE_ANNUAL) / roll_vol.replace(0, np.nan)
        ls = '--' if name in dashed_keys else '-'
        ax.plot(roll_sharpe.index, roll_sharpe.values,
                label=name, linewidth=1.2, linestyle=ls)
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Sharpe Ratio (annualised)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")


def plot_sector_allocation(weight_sets, tickers, title, outpath,
                           sector_cap_line=0.30):
    """
    Stacked bar chart showing sector allocation per strategy.

    Args:
        weight_sets     : dict  label → numpy weight array
        tickers         : list of ticker strings
        title           : chart title
        outpath         : output file path
        sector_cap_line : horizontal line position (default 0.30)
    """
    all_sectors  = sorted(set(SECTOR_MAP.values()))
    strat_labels = list(weight_sets.keys())
    x            = np.arange(len(strat_labels))

    fig, ax = plt.subplots(figsize=(12, 6))
    bottoms = np.zeros(len(strat_labels))
    for sector in all_sectors:
        sec_wts = []
        for wts in weight_sets.values():
            sec_idx = [i for i, t in enumerate(tickers)
                       if SECTOR_MAP.get(t, 'Other') == sector]
            sec_wts.append(
                sum(wts[i] for i in sec_idx) * 100 if sec_idx else 0
            )
        ax.bar(x, sec_wts, bottom=bottoms,
               color=SECTOR_COLOR_MAP.get(sector, '#CCC'),
               label=sector, edgecolor='white', linewidth=0.3)
        bottoms += np.array(sec_wts)

    ax.axhline(sector_cap_line * 100, color='red', linewidth=1.0,
               linestyle='--', alpha=0.7, label=f'{sector_cap_line*100:.0f}% sector cap')
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Allocation (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(strat_labels, rotation=25, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.set_ylim(0, 105)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")


def print_sector_weights(weights, tickers, label="", warn_threshold=0.301):
    """
    Print sector-level weight breakdown for a single portfolio.
    Flags any sector exceeding warn_threshold.
    """
    if label:
        print(f"\n   Sector breakdown — {label}:")
    for sector in sorted(set(SECTOR_MAP.values())):
        sec_wt = sum(weights[i] for i, t in enumerate(tickers)
                     if SECTOR_MAP.get(t, 'Other') == sector)
        if sec_wt > 0.01:
            flag = " ⚠️ VIOLATED" if sec_wt > warn_threshold else ""
            print(f"     {sector:15s}: {sec_wt*100:.1f}%{flag}")