"""
stress_testing.py — Historical & Hypothetical Stress Testing
=============================================================

Stress events covered
─────────────────────
  1. 2008 Global Financial Crisis  (Sep 2008 – Mar 2009)
     Strategy: Download NIFTY 50 (^NSEI) for 2007-2009 via yfinance.
     Apply the NIFTY drawdown profile as a shock multiplier to each
     portfolio's current covariance structure.  This is the standard
     "historical scenario simulation" approach used in practice when
     your live data pre-dates the crisis.

  2. 2020 COVID-19 Crash  (Jan 20 – Mar 23, 2020)
     Fully in-sample — load actual portfolio returns for that window.
     Both the raw losses AND the recovery path are measured.

  3. 2022 Russia-Ukraine / Fed Tightening Selloff  (Jan – Jun 2022)
     Fully in-sample — a prolonged drawdown period useful for showing
     how each strategy fared in a slow grinding bear market vs a
     sudden crash (COVID).

  4. Hypothetical tail shocks
     ±1σ, ±2σ, ±3σ instantaneous shocks applied to each portfolio
     using the full-sample covariance matrix.  Illustrates tail risk.

Outputs
───────
  reports/stress_2008_simulation.png
  reports/stress_covid.png
  reports/stress_russia_ukraine.png
  reports/stress_tail_shocks.png
  reports/stress_summary_table.png
  data/portfolio/stress_results.csv

Run
───
  python src/stress_testing.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf

from utils import (
    load_returns, load_prices,
    compute_lw_cov,
    RISK_FREE_RATE_ANNUAL,
    SECTOR_MAP,
    max_drawdown, sortino_ratio, calmar_ratio,
)

os.makedirs("reports",           exist_ok=True)
os.makedirs("data/portfolio",    exist_ok=True)
os.makedirs("data/stress",       exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# 2008 crisis window on NIFTY (calendar dates for yfinance download)
CRISIS_2008_START = "2008-01-01"
CRISIS_2008_END   = "2009-06-30"
CRISIS_2008_SHOCK_START = "2008-09-01"   # post-Lehman peak
CRISIS_2008_SHOCK_END   = "2009-03-09"   # NIFTY trough

# In-sample stress windows (must be within your 2019-2024 data)
COVID_START  = "2020-01-20"
COVID_END    = "2020-06-30"    # include recovery
COVID_CRASH  = "2020-01-20"
COVID_TROUGH = "2020-03-23"

RU_START = "2022-01-01"
RU_END   = "2022-06-30"

# Portfolio CSV paths (outputs from your earlier models)
PORTFOLIO_FILES = {
    "Equal Weight"        : "data/portfolio/equal_weight_growth.csv",
    "Min Variance"        : "data/portfolio/markowitz_growth.csv",
    "Max Sharpe"          : "data/portfolio/sharpe_growth.csv",
    "ERC"                 : "data/portfolio/erc_growth.csv",
    "Max Diversification" : "data/portfolio/max_diversification_growth.csv",
    "Black-Litterman"     : "data/portfolio/black_litterman_growth.csv",
}

RETURN_FILES = {
    "Equal Weight"        : "data/portfolio/equal_weight_returns.csv",
    "Min Variance"        : "data/portfolio/markowitz_returns.csv",
    "Max Sharpe"          : "data/portfolio/sharpe_returns.csv",
    "ERC"                 : "data/portfolio/erc_returns.csv",
    "Max Diversification" : "data/portfolio/max_diversification_returns.csv",
    "Black-Litterman"     : "data/portfolio/black_litterman_returns.csv",
}

STRATEGY_COLORS = {
    "Equal Weight"        : "#888888",
    "Min Variance"        : "#4C72B0",
    "Max Sharpe"          : "#DD8452",
    "ERC"                 : "#55A868",
    "Max Diversification" : "#C44E52",
    "Black-Litterman"     : "#9467bd",
    "NIFTY 50"            : "#000000",
}


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_portfolio_data():
    """
    Load all available portfolio growth and return series.
    Missing files are silently skipped.
    """
    growths = {}
    rets    = {}

    for name, path in PORTFOLIO_FILES.items():
        if os.path.exists(path):
            g = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
            g = g.sort_index()
            # normalise to start at 1.0
            g = g / g.iloc[0]
            growths[name] = g
        else:
            print(f"   [SKIP] {name}: {path} not found")

    for name, path in RETURN_FILES.items():
        if os.path.exists(path):
            r = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
            rets[name] = r.sort_index()

    # If return files weren't saved separately, derive from growth
    for name, g in growths.items():
        if name not in rets:
            rets[name] = g.pct_change().dropna()

    print(f"   Loaded {len(growths)} portfolio series: {list(growths.keys())}")
    return growths, rets


def load_nifty_benchmark(start, end):
    """Download NIFTY 50 (^NSEI) daily close prices."""
    print(f"   Downloading ^NSEI  {start} → {end} ...", end=" ", flush=True)
    try:
        raw = yf.download("^NSEI", start=start, end=end,
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        close = raw["Close"].squeeze().sort_index().dropna()
        print(f"OK ({len(close)} days)")
        return close
    except Exception as e:
        print(f"FAILED — {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 1.  2008 CRISIS  —  SCENARIO SIMULATION
# ──────────────────────────────────────────────────────────────────────────────

def run_2008_simulation(returns_full):
    """
    Historical scenario simulation for the 2008 crisis.

    Method
    ------
    1. Download NIFTY 50 daily returns for the 2008 shock window.
    2. For each portfolio, project hypothetical P&L using:
           simulated_daily_ret_t  =  nifty_daily_ret_t  ×  beta_i
       where beta_i is the portfolio's beta estimated from in-sample data
       against NIFTY 50.
    3. Cumulate the projected returns over the 6-month shock window.

    This gives a realistic "how would this portfolio have fared in 2008"
    without pretending we had live NAVs back then.
    """
    print("\n── 2008 Crisis Simulation ──────────────────────────────────")

    # Download NIFTY for 2008 shock window
    nifty_2008_prices = load_nifty_benchmark(CRISIS_2008_START, CRISIS_2008_END)
    if nifty_2008_prices is None:
        print("   Cannot run 2008 simulation — NIFTY download failed.")
        return None, None

    nifty_2008_rets = nifty_2008_prices.pct_change().dropna()

    # Shock window only
    shock_rets = nifty_2008_rets.loc[CRISIS_2008_SHOCK_START:CRISIS_2008_SHOCK_END]
    nifty_shock_loss = (1 + shock_rets).prod() - 1
    print(f"   NIFTY 50 loss ({CRISIS_2008_SHOCK_START} → {CRISIS_2008_SHOCK_END}): "
          f"{nifty_shock_loss*100:.1f}%")

    # Download in-sample NIFTY (2019-2024) for beta estimation
    nifty_insample = load_nifty_benchmark("2019-01-01", "2024-12-31")
    if nifty_insample is None:
        nifty_beta_rets = None
    else:
        nifty_beta_rets = nifty_insample.pct_change().dropna()

    simulation_results = {}

    for name, port_rets in returns_full.items():
        # Estimate beta vs NIFTY using in-sample data
        if nifty_beta_rets is not None:
            aligned = pd.concat([port_rets, nifty_beta_rets], axis=1,
                                 join='inner').dropna()
            aligned.columns = ['port', 'nifty']
            if len(aligned) > 60:
                cov_mat  = np.cov(aligned['port'].values,
                                  aligned['nifty'].values)
                beta = cov_mat[0, 1] / cov_mat[1, 1]
            else:
                beta = 1.0
        else:
            beta = 1.0

        # Project 2008 shock onto this portfolio
        projected_rets  = shock_rets * beta
        projected_growth = (1 + projected_rets).cumprod()
        projected_loss   = projected_growth.iloc[-1] - 1

        simulation_results[name] = {
            'beta'           : round(beta, 3),
            'projected_loss' : round(projected_loss * 100, 2),
            'growth_series'  : projected_growth,
        }
        print(f"   {name:<22} β={beta:.3f}  projected 2008 loss: "
              f"{projected_loss*100:.1f}%")

    # Add NIFTY itself for reference
    nifty_growth = (1 + shock_rets).cumprod()
    simulation_results['NIFTY 50'] = {
        'beta': 1.0,
        'projected_loss': round(nifty_shock_loss * 100, 2),
        'growth_series': nifty_growth,
    }

    return simulation_results, shock_rets


def plot_2008_simulation(simulation_results, shock_rets):
    if simulation_results is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Cumulative growth during shock
    ax = axes[0]
    for name, res in simulation_results.items():
        g  = res['growth_series']
        g  = g / g.iloc[0]
        ls = ':' if name == 'NIFTY 50' else '-'
        lw = 2.2 if name == 'NIFTY 50' else 1.5
        ax.plot(range(len(g)), g.values,
                label=f"{name} ({res['projected_loss']:.1f}%)",
                color=STRATEGY_COLORS.get(name, '#999'),
                linewidth=lw, linestyle=ls)
    ax.axhline(1.0, color='black', linewidth=0.7, linestyle='--', alpha=0.4)
    ax.set_title("2008 Crisis — Projected Portfolio Losses\n"
                 "(Beta-adjusted simulation from NIFTY shock profile)",
                 fontsize=11)
    ax.set_xlabel(f"Trading days from {CRISIS_2008_SHOCK_START}")
    ax.set_ylabel("Normalised Value (1.0 = start of shock)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Bar chart of projected losses
    ax2 = axes[1]
    names  = list(simulation_results.keys())
    losses = [simulation_results[n]['projected_loss'] for n in names]
    betas  = [simulation_results[n]['beta']           for n in names]
    colors = [STRATEGY_COLORS.get(n, '#999') for n in names]
    x      = np.arange(len(names))
    bars   = ax2.bar(x, [abs(l) for l in losses], color=colors,
                     alpha=0.80, edgecolor='white')
    ax2.set_title("2008 Crisis — Projected Losses by Strategy\n"
                  "Lower bar = better capital protection", fontsize=11)
    ax2.set_ylabel("Projected Loss (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3)
    # Annotate with beta
    for i, (bar, beta) in enumerate(zip(bars, betas)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'β={beta:.2f}', ha='center', va='bottom', fontsize=7.5)

    plt.tight_layout()
    plt.savefig("reports/stress_2008_simulation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: reports/stress_2008_simulation.png")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  COVID-19 CRASH  (fully in-sample)
# ──────────────────────────────────────────────────────────────────────────────

def run_covid_stress(growths_full, rets_full):
    """Slice the actual portfolio returns during COVID window."""
    print("\n── COVID-19 Crash (2020)  ──────────────────────────────────")

    covid_growths = {}
    covid_rets    = {}
    results       = {}

    for name, g in growths_full.items():
        window = g.loc[COVID_START:COVID_END]
        if len(window) < 20:
            print(f"   [SKIP] {name}: insufficient data in COVID window")
            continue
        window = window / window.iloc[0]   # normalise to 1.0
        covid_growths[name] = window

        r_window = rets_full[name].loc[COVID_START:COVID_END]
        covid_rets[name] = r_window

        # Crash-only drawdown
        crash_window = g.loc[COVID_CRASH:COVID_TROUGH]
        crash_loss   = (crash_window.iloc[-1] / crash_window.iloc[0]) - 1 \
                       if len(crash_window) > 5 else np.nan

        # Recovery: days to get back to pre-crash level
        pre_crash_val = g.loc[:COVID_CRASH].iloc[-1]
        post_crash    = g.loc[COVID_TROUGH:]
        recovery_days = np.nan
        for j, (dt, val) in enumerate(post_crash.items()):
            if val >= pre_crash_val:
                recovery_days = j
                break

        max_dd_full = max_drawdown(window)
        results[name] = {
            'crash_loss_pct'   : round(crash_loss * 100, 2) if not np.isnan(crash_loss) else np.nan,
            'max_drawdown_pct' : round(max_dd_full * 100, 2),
            'recovery_days'    : recovery_days,
            'full_window_ret'  : round((window.iloc[-1] - 1) * 100, 2),
        }
        rec_str = f"{int(recovery_days)}d" if not np.isnan(recovery_days) else "Not recovered"
        print(f"   {name:<22}  crash={results[name]['crash_loss_pct']:.1f}%  "
              f"max_dd={results[name]['max_drawdown_pct']:.1f}%  "
              f"recovery={rec_str}")

    # Add NIFTY benchmark
    nifty_covid = load_nifty_benchmark(COVID_START, COVID_END)
    if nifty_covid is not None:
        ng = (nifty_covid / nifty_covid.iloc[0])
        covid_growths['NIFTY 50'] = ng
        crash_n  = nifty_covid.loc[COVID_CRASH:COVID_TROUGH]
        crash_loss_n = (crash_n.iloc[-1] / crash_n.iloc[0]) - 1 \
                       if len(crash_n) > 5 else np.nan
        results['NIFTY 50'] = {
            'crash_loss_pct'  : round(crash_loss_n * 100, 2) if not np.isnan(crash_loss_n) else np.nan,
            'max_drawdown_pct': round(max_drawdown(ng) * 100, 2),
            'recovery_days'   : np.nan,
            'full_window_ret' : round((ng.iloc[-1] - 1) * 100, 2),
        }

    return covid_growths, covid_rets, results


def plot_covid_stress(covid_growths, covid_results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for name, g in covid_growths.items():
        ls = ':' if name == 'NIFTY 50' else '-'
        lw = 2.2 if name == 'NIFTY 50' else 1.5
        ax.plot(g.index, g.values,
                label=f"{name} ({covid_results[name]['crash_loss_pct']:.1f}%)",
                color=STRATEGY_COLORS.get(name, '#999'),
                linewidth=lw, linestyle=ls)
    ax.axhline(1.0, color='black', linewidth=0.7, linestyle='--', alpha=0.4)
    ax.axvspan(pd.Timestamp(COVID_CRASH), pd.Timestamp(COVID_TROUGH),
               alpha=0.08, color='red', label='Crash window')
    ax.set_title("COVID-19 Crash & Recovery (Jan 2020 – Jun 2020)\n"
                 "Actual portfolio returns", fontsize=11)
    ax.set_ylabel("Normalised Value (1.0 = Jan 20, 2020)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Crash stats bar
    ax2 = axes[1]
    names  = [n for n in covid_results if n in covid_growths]
    losses = [abs(covid_results[n]['crash_loss_pct']) for n in names]
    max_dds = [abs(covid_results[n]['max_drawdown_pct']) for n in names]
    x, w   = np.arange(len(names)), 0.35
    colors = [STRATEGY_COLORS.get(n, '#999') for n in names]
    ax2.bar(x - w/2, losses,  w, label='Crash loss (%)',       color=colors, alpha=0.75)
    ax2.bar(x + w/2, max_dds, w, label='Max drawdown (%)', color=colors, alpha=0.45,
            edgecolor=[STRATEGY_COLORS.get(n, '#999') for n in names], linewidth=1.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
    ax2.set_title("COVID Stress — Crash Loss vs Max Drawdown", fontsize=11)
    ax2.set_ylabel("Loss (%)")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("reports/stress_covid.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: reports/stress_covid.png")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  RUSSIA-UKRAINE / FED TIGHTENING  (in-sample)
# ──────────────────────────────────────────────────────────────────────────────

def run_ru_stress(growths_full, rets_full):
    print("\n── Russia-Ukraine / Fed Tightening (Jan–Jun 2022) ─────────")

    ru_growths = {}
    results    = {}

    for name, g in growths_full.items():
        window = g.loc[RU_START:RU_END]
        if len(window) < 20:
            print(f"   [SKIP] {name}: insufficient data in R-U window")
            continue
        window = window / window.iloc[0]
        ru_growths[name] = window
        max_dd = max_drawdown(window)
        total  = window.iloc[-1] - 1
        results[name] = {
            'total_return_pct' : round(total * 100, 2),
            'max_drawdown_pct' : round(max_dd * 100, 2),
        }
        print(f"   {name:<22}  period_ret={results[name]['total_return_pct']:.1f}%  "
              f"max_dd={results[name]['max_drawdown_pct']:.1f}%")

    nifty_ru = load_nifty_benchmark(RU_START, RU_END)
    if nifty_ru is not None:
        ng = nifty_ru / nifty_ru.iloc[0]
        ru_growths['NIFTY 50'] = ng
        results['NIFTY 50'] = {
            'total_return_pct': round((ng.iloc[-1] - 1) * 100, 2),
            'max_drawdown_pct': round(max_drawdown(ng) * 100, 2),
        }

    return ru_growths, results


def plot_ru_stress(ru_growths, ru_results):
    fig, ax = plt.subplots(figsize=(13, 6))
    for name, g in ru_growths.items():
        ls = ':' if name == 'NIFTY 50' else '-'
        lw = 2.2 if name == 'NIFTY 50' else 1.5
        ax.plot(g.index, g.values,
                label=f"{name} ({ru_results[name]['total_return_pct']:.1f}%)",
                color=STRATEGY_COLORS.get(name, '#999'),
                linewidth=lw, linestyle=ls)
    ax.axhline(1.0, color='black', linewidth=0.7, linestyle='--', alpha=0.4)
    ax.set_title("Russia-Ukraine / Fed Rate Hikes (Jan–Jun 2022)\n"
                 "Slow grinding bear — tests risk parity & drawdown control",
                 fontsize=11)
    ax.set_ylabel("Normalised Value (1.0 = Jan 2022)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/stress_russia_ukraine.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: reports/stress_russia_ukraine.png")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  HYPOTHETICAL TAIL SHOCKS
# ──────────────────────────────────────────────────────────────────────────────

def run_tail_shocks(rets_full, returns_all_stocks):
    """
    Simulate instantaneous ±1σ / ±2σ / ±3σ portfolio shocks.
    σ is each portfolio's annualised daily volatility converted to a
    single-day P&L.
    """
    print("\n── Hypothetical Tail Shock Analysis ───────────────────────")
    shock_levels = [-3, -2, -1, 1, 2, 3]
    results = {}

    for name, r in rets_full.items():
        daily_vol = r.std()
        pnls = {}
        for sigma in shock_levels:
            pnl = sigma * daily_vol * 100  # as %
            pnls[f'{sigma:+d}σ'] = round(pnl, 2)
        results[name] = pnls
        print(f"   {name:<22}  daily_vol={daily_vol*100:.3f}%  "
              f"-3σ={pnls['-3σ']:.2f}%  -2σ={pnls['-2σ']:.2f}%  "
              f"-1σ={pnls['-1σ']:.2f}%")

    return results


def plot_tail_shocks(shock_results):
    names    = list(shock_results.keys())
    shock_labels = ['-3σ', '-2σ', '-1σ', '+1σ', '+2σ', '+3σ']
    x        = np.arange(len(names))
    n_shocks = len(shock_labels)
    width    = 0.13

    fig, ax = plt.subplots(figsize=(16, 6))
    shock_colors = ['#d62728','#ff7f0e','#ffbb78','#98df8a','#2ca02c','#1f77b4']

    for j, (label, color) in enumerate(zip(shock_labels, shock_colors)):
        vals = [shock_results[n][label] for n in names]
        offset = (j - n_shocks/2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.80)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title("Hypothetical Tail Shocks — Instantaneous P&L per Strategy\n"
                 "(Daily σ of each portfolio × shock magnitude)", fontsize=11)
    ax.set_ylabel("Single-day P&L (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.legend(title="Shock Level", fontsize=9, ncol=6)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/stress_tail_shocks.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: reports/stress_tail_shocks.png")


# ──────────────────────────────────────────────────────────────────────────────
# 5.  SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────────────

def build_summary_table(sim_2008, covid_results, ru_results, shock_results):
    rows = []
    all_strategies = list({**PORTFOLIO_FILES}.keys())

    for name in all_strategies:
        row = {'Strategy': name}

        # 2008
        if sim_2008 and name in sim_2008:
            row['2008 Projected Loss (%)'] = sim_2008[name]['projected_loss']
            row['2008 Beta']               = sim_2008[name]['beta']
        else:
            row['2008 Projected Loss (%)'] = np.nan
            row['2008 Beta']               = np.nan

        # COVID
        if covid_results and name in covid_results:
            row['COVID Crash Loss (%)']   = covid_results[name]['crash_loss_pct']
            row['COVID Max Drawdown (%)'] = covid_results[name]['max_drawdown_pct']
            row['COVID Recovery (days)']  = covid_results[name].get('recovery_days', np.nan)
        else:
            row['COVID Crash Loss (%)']   = np.nan
            row['COVID Max Drawdown (%)'] = np.nan
            row['COVID Recovery (days)']  = np.nan

        # Russia-Ukraine
        if ru_results and name in ru_results:
            row['R-U Period Return (%)']  = ru_results[name]['total_return_pct']
            row['R-U Max Drawdown (%)']   = ru_results[name]['max_drawdown_pct']
        else:
            row['R-U Period Return (%)']  = np.nan
            row['R-U Max Drawdown (%)']   = np.nan

        # Tail shocks
        if shock_results and name in shock_results:
            row['-3σ P&L (%)'] = shock_results[name]['-3σ']
            row['-2σ P&L (%)'] = shock_results[name]['-2σ']
        else:
            row['-3σ P&L (%)'] = np.nan
            row['-2σ P&L (%)'] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows).set_index('Strategy')
    df.to_csv("data/stress/stress_results.csv")
    print("\n   Saved: data/stress/stress_results.csv")
    return df


def plot_summary_table(df):
    """Render the summary dataframe as a formatted matplotlib table."""
    fig, ax = plt.subplots(figsize=(18, max(4, len(df) * 0.8 + 2)))
    ax.axis('off')

    cols      = df.columns.tolist()
    rows      = df.index.tolist()
    cell_data = []
    for idx in rows:
        row_data = []
        for col in cols:
            val = df.loc[idx, col]
            if pd.isna(val):
                row_data.append("—")
            elif 'days' in col.lower():
                row_data.append(f"{int(val)}d" if not np.isnan(val) else "—")
            elif 'beta' in col.lower():
                row_data.append(f"{val:.3f}")
            else:
                row_data.append(f"{val:.1f}%")
        cell_data.append(row_data)

    table = ax.table(
        cellText=cell_data,
        rowLabels=rows,
        colLabels=cols,
        cellLoc='center',
        rowLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.6)

    # Colour header row
    for j in range(len(cols)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Alternating row shading
    for i in range(len(rows)):
        for j in range(-1, len(cols)):
            if i % 2 == 1:
                table[i + 1, j].set_facecolor('#f5f5f5')

    ax.set_title("Stress Testing — Summary Table\n"
                 "(2008 simulation | COVID-2020 actual | Russia-Ukraine actual | Tail shocks)",
                 fontsize=12, pad=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig("reports/stress_summary_table.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: reports/stress_summary_table.png")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 68)
    print("STRESS TESTING FRAMEWORK")
    print("  Event 1 : 2008 Global Financial Crisis (beta-simulation)")
    print("  Event 2 : 2020 COVID-19 Crash          (actual returns)")
    print("  Event 3 : 2022 Russia-Ukraine Selloff   (actual returns)")
    print("  Event 4 : Hypothetical tail shocks      (±1σ, ±2σ, ±3σ)")
    print("=" * 68)

    # Load data
    print("\nLoading portfolio data...")
    growths_full, rets_full = load_portfolio_data()

    if not growths_full:
        print("\n[ERROR] No portfolio files found. Run your portfolio models first.")
        sys.exit(1)

    # Load full return matrix for stock-level analysis
    returns_all, tickers, n = load_returns()

    # ── 1. 2008 Simulation ────────────────────────────────────────────
    sim_2008, shock_rets_2008 = run_2008_simulation(rets_full)
    plot_2008_simulation(sim_2008, shock_rets_2008)

    # ── 2. COVID Stress ───────────────────────────────────────────────
    covid_growths, covid_rets, covid_results = run_covid_stress(growths_full, rets_full)
    plot_covid_stress(covid_growths, covid_results)

    # ── 3. Russia-Ukraine ─────────────────────────────────────────────
    ru_growths, ru_results = run_ru_stress(growths_full, rets_full)
    plot_ru_stress(ru_growths, ru_results)

    # ── 4. Tail Shocks ────────────────────────────────────────────────
    shock_results = run_tail_shocks(rets_full, returns_all)
    plot_tail_shocks(shock_results)

    # ── 5. Summary ────────────────────────────────────────────────────
    summary_df = build_summary_table(sim_2008, covid_results, ru_results, shock_results)

    print("\n\nSTRESS TEST SUMMARY")
    print("=" * 90)
    print(summary_df.to_string())
    plot_summary_table(summary_df)

    print("\n✅ Stress testing complete.")
    print("   Plots   → reports/stress_2008_simulation.png")
    print("             reports/stress_covid.png")
    print("             reports/stress_russia_ukraine.png")
    print("             reports/stress_tail_shocks.png")
    print("             reports/stress_summary_table.png")
    print("   Data    → data/stress/stress_results.csv")