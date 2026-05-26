"""
factor_attribution.py — Factor Analysis & Return Attribution
=============================================================
Week 7 deliverable: Identify sources of return and risk across all strategies.

Three-layer attribution:
  Layer 1 — MARKET ATTRIBUTION
      β (beta) to NIFTY 50, α (alpha), R²
      Jensen's Alpha, Treynor Ratio, Information Ratio

  Layer 2 — SECTOR ATTRIBUTION  (Brinson-Hood-Beebower style)
      Allocation Effect  : did we overweight winning sectors?
      Selection Effect   : did we pick better stocks within sectors?
      Interaction Effect : combined skill
      Total Active Return = Allocation + Selection + Interaction

  Layer 3 — FACTOR ATTRIBUTION  (Fama-French style with available data)
      Market Factor (MKT-RF)  : NIFTY excess return
      Size Factor (SMB proxy) : large-cap vs small-cap within universe
      Momentum Factor (MOM)   : 6-month winner vs loser spread
      Low-Vol Factor  (LV)    : low-vol vs high-vol stock spread

Additional diagnostics:
  - Rolling 63-day beta and alpha
  - Factor exposure heatmap across strategies
  - Active weight evolution over time
  - Sector allocation vs benchmark (NIFTY 50 free-float weights)

Run:
    python src/factor_attribution.py

Outputs:
    reports/factor_market_attribution.png
    reports/factor_sector_attribution.png
    reports/factor_regression.png
    reports/factor_rolling_alpha_beta.png
    reports/factor_exposure_heatmap.png
    data/portfolio/factor_attribution_results.csv
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

from scipy import stats
import yfinance as yf

from utils import (
    load_returns, load_prices,
    compute_market_cap_weights,
    RISK_FREE_RATE_ANNUAL, RISK_FREE_RATE_DAILY,
    SECTOR_MAP, SECTOR_COLOR_MAP,
    max_drawdown, sortino_ratio,
)

os.makedirs("reports",        exist_ok=True)
os.makedirs("data/portfolio", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

ROLLING_WINDOW  = 63    # 3-month rolling window for beta/alpha
MOMENTUM_WINDOW = 126   # 6-month for factor construction
VOL_WINDOW      = 63    # volatility estimation window

# Which portfolio files to load
PORTFOLIO_FILES = {
    "Equal Weight"       : "data/portfolio/equal_weight_growth.csv",
    "Min Variance"       : "data/portfolio/markowitz_growth.csv",
    "Max Sharpe"         : "data/portfolio/sharpe_growth.csv",
    "ERC"                : "data/portfolio/erc_growth.csv",
    "Max Diversification": "data/portfolio/max_diversification_growth.csv",
    "Black-Litterman"    : "data/portfolio/black_litterman_growth.csv",
}

STRATEGY_COLORS = {
    "Equal Weight"       : "#888888",
    "Min Variance"       : "#4C72B0",
    "Max Sharpe"         : "#DD8452",
    "ERC"                : "#55A868",
    "Max Diversification": "#C44E52",
    "Black-Litterman"    : "#9467bd",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_portfolio_returns():
    """Load all available portfolio growth curves and derive return series."""
    growths, rets = {}, {}
    for name, path in PORTFOLIO_FILES.items():
        if os.path.exists(path):
            g = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
            g = g.sort_index()
            g = g / g.iloc[0]
            growths[name] = g
            rets[name]    = g.pct_change().dropna()
        else:
            print(f"   [SKIP] {name}: file not found — run portfolio models first")
    print(f"   Loaded {len(growths)} strategies: {list(growths.keys())}")
    return growths, rets


def load_nifty(start, end):
    """Download NIFTY 50 (^NSEI) returns."""
    print("   Downloading NIFTY 50 benchmark...", end=" ", flush=True)
    try:
        raw = yf.download("^NSEI", start=str(start.date()),
                          end=str(end.date()), auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        close = raw["Close"].squeeze().sort_index().dropna()
        rets  = close.pct_change().dropna()
        print(f"OK ({len(rets)} days)")
        return rets
    except Exception as e:
        print(f"FAILED — {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — MARKET ATTRIBUTION (Alpha / Beta / R²)
# ─────────────────────────────────────────────────────────────────────────────

def compute_market_attribution(port_rets, bench_rets, rf=RISK_FREE_RATE_DAILY):
    """
    OLS regression of excess portfolio return on excess benchmark return.

        (Rp - Rf) = α + β(Rm - Rf) + ε

    Returns dict with alpha, beta, r_squared, treynor, information_ratio,
    tracking_error, and t-stat for alpha.
    """
    aligned = pd.concat([port_rets, bench_rets], axis=1, join="inner").dropna()
    aligned.columns = ["port", "bench"]

    excess_p = aligned["port"]  - rf
    excess_m = aligned["bench"] - rf

    slope, intercept, r_val, p_val, stderr = stats.linregress(
        excess_m.values, excess_p.values
    )

    beta         = slope
    alpha_daily  = intercept
    alpha_annual = alpha_daily * 252
    r_sq         = r_val ** 2

    residuals      = excess_p.values - (alpha_daily + beta * excess_m.values)
    tracking_error = residuals.std() * np.sqrt(252)

    ann_port  = excess_p.mean() * 252
    ann_bench = excess_m.mean() * 252

    treynor = ann_port / beta if abs(beta) > 1e-6 else np.nan
    ir      = (ann_port - ann_bench) / tracking_error if tracking_error > 1e-6 else np.nan

    # T-stat for alpha significance
    n_obs  = len(aligned)
    se_alpha = stderr * np.sqrt((excess_m ** 2).mean() / (excess_m.var() * n_obs))
    t_alpha  = alpha_daily / se_alpha if se_alpha > 1e-10 else np.nan

    return {
        "Alpha (ann %)": round(alpha_annual * 100, 3),
        "Beta"         : round(beta, 4),
        "R²"           : round(r_sq, 4),
        "Tracking Error (ann %)": round(tracking_error * 100, 3),
        "Treynor Ratio": round(treynor, 4),
        "Info Ratio"   : round(ir, 4),
        "T-stat (α)"   : round(t_alpha, 3) if not np.isnan(t_alpha) else np.nan,
        "n_obs"        : n_obs,
    }


def rolling_alpha_beta(port_rets, bench_rets, window=ROLLING_WINDOW,
                       rf=RISK_FREE_RATE_DAILY):
    """Compute rolling beta and annualised alpha over a sliding window."""
    aligned = pd.concat([port_rets, bench_rets], axis=1, join="inner").dropna()
    aligned.columns = ["port", "bench"]

    roll_beta  = []
    roll_alpha = []
    dates      = []

    for i in range(window, len(aligned)):
        sub      = aligned.iloc[i - window: i]
        ep       = sub["port"]  - rf
        em       = sub["bench"] - rf
        if em.std() < 1e-10:
            continue
        b, a, *_ = stats.linregress(em.values, ep.values)
        roll_beta.append(b)
        roll_alpha.append(a * 252)      # annualise
        dates.append(aligned.index[i])

    idx = pd.DatetimeIndex(dates)
    return (pd.Series(roll_alpha, index=idx, name="Alpha"),
            pd.Series(roll_beta,  index=idx, name="Beta"))


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — SECTOR ATTRIBUTION  (Brinson-Hood-Beebower)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sector_attribution(port_rets_all_stocks, tickers,
                                bench_weights, bench_rets_stocks):
    """
    Brinson-Hood-Beebower decomposition vs benchmark (NIFTY free-float).

    For each sector s:
        Allocation Effect  = (wp_s - wb_s) * (Rb_s - Rb)
        Selection Effect   = wb_s * (Rp_s - Rb_s)
        Interaction Effect = (wp_s - wb_s) * (Rp_s - Rb_s)
        Total Active       = sum of all three

    Inputs
    ------
    port_rets_all_stocks : daily portfolio (already invested) returns per stock
                           proxy: equal-weight within selected portfolio
    tickers              : list of tickers in portfolio
    bench_weights        : array of benchmark weights (free-float)
    bench_rets_stocks    : DataFrame of stock returns (same columns as tickers)

    Returns sector-level DataFrame with all four columns.
    """
    sectors = sorted(set(SECTOR_MAP.get(t, "Other") for t in tickers))
    n = len(tickers)

    # Portfolio weight per sector: from sector map (equal within sector proxy)
    # A real implementation would use actual portfolio weights; here we use
    # equal-weight within portfolio as an approximation when daily weights
    # are not stored.
    equal_w = np.ones(n) / n
    bench_w = bench_weights / bench_weights.sum()

    results = []
    bench_daily_ret = (bench_rets_stocks * bench_w).sum(axis=1)
    Rb_total = bench_daily_ret.mean() * 252

    for sec in sectors:
        idx_p = [i for i, t in enumerate(tickers) if SECTOR_MAP.get(t,"Other") == sec]
        if not idx_p:
            continue

        wp_s = equal_w[idx_p].sum()
        wb_s = bench_w[idx_p].sum()

        # Sector return (portfolio side = equal-weight within sector)
        sec_rets_p = bench_rets_stocks.iloc[:, idx_p].mean(axis=1)
        Rp_s = sec_rets_p.mean() * 252

        # Sector benchmark return
        sec_bench_w = bench_w[idx_p]
        sec_bench_w = sec_bench_w / sec_bench_w.sum() if sec_bench_w.sum() > 0 else sec_bench_w
        sec_rets_b = (bench_rets_stocks.iloc[:, idx_p] * sec_bench_w).sum(axis=1)
        Rb_s = sec_rets_b.mean() * 252

        alloc   = (wp_s - wb_s) * (Rb_s - Rb_total)
        select  = wb_s          * (Rp_s - Rb_s)
        interact= (wp_s - wb_s) * (Rp_s - Rb_s)

        results.append({
            "Sector"            : sec,
            "Port Weight (%)"   : round(wp_s * 100, 2),
            "Bench Weight (%)"  : round(wb_s * 100, 2),
            "Active Weight (%)" : round((wp_s - wb_s) * 100, 2),
            "Port Return (%)"   : round(Rp_s * 100, 2),
            "Bench Return (%)"  : round(Rb_s * 100, 2),
            "Allocation Effect" : round(alloc    * 100, 4),
            "Selection Effect"  : round(select   * 100, 4),
            "Interaction Effect": round(interact  * 100, 4),
            "Total Active (%)"  : round((alloc + select + interact) * 100, 4),
        })

    return pd.DataFrame(results).set_index("Sector")


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — FACTOR CONSTRUCTION  (Fama-French style)
# ─────────────────────────────────────────────────────────────────────────────

def build_factors(returns_df, prices_df, bench_rets, tickers):
    """
    Build four daily factor return series from stock-level data:

      MKT_RF  : NIFTY 50 excess return (benchmark - rf)
      MOM     : top-third 6M momentum minus bottom-third  (momentum factor)
      LV      : bottom-vol-third minus top-vol-third       (low-vol factor)
      SIZE    : small-cap proxy minus large-cap proxy
                (using rank of 63-day avg price as size proxy)

    All factors are zero-cost long-short portfolios (sum to 0).
    Returns a DataFrame of daily factor returns aligned to returns_df.
    """
    n = len(tickers)
    dates = returns_df.index

    # ── MKT_RF ───────────────────────────────────────────────────────────────
    if bench_rets is not None:
        mkt_rf = bench_rets.reindex(dates, method="ffill").fillna(0) - RISK_FREE_RATE_DAILY
    else:
        mkt_rf = returns_df.mean(axis=1) - RISK_FREE_RATE_DAILY
    mkt_rf.name = "MKT_RF"

    # ── Momentum factor  (MOM) ───────────────────────────────────────────────
    mom_factor = []
    for i in range(MOMENTUM_WINDOW, len(dates)):
        mom_window = returns_df.iloc[i - MOMENTUM_WINDOW: i]
        cum_ret    = (1 + mom_window).prod() - 1
        n_group    = max(1, n // 3)

        winners = cum_ret.nlargest(n_group).index
        losers  = cum_ret.nsmallest(n_group).index

        long_ret  = returns_df.iloc[i][winners].mean()
        short_ret = returns_df.iloc[i][losers].mean()
        mom_factor.append((dates[i], long_ret - short_ret))

    mom_series = pd.Series(
        [x[1] for x in mom_factor],
        index=[x[0] for x in mom_factor],
        name="MOM"
    )

    # ── Low-Volatility factor  (LV) ──────────────────────────────────────────
    lv_factor = []
    for i in range(VOL_WINDOW, len(dates)):
        vol_window = returns_df.iloc[i - VOL_WINDOW: i]
        ann_vol    = vol_window.std() * np.sqrt(252)
        n_group    = max(1, n // 3)

        low_vol  = ann_vol.nsmallest(n_group).index
        high_vol = ann_vol.nlargest(n_group).index

        long_ret  = returns_df.iloc[i][low_vol].mean()
        short_ret = returns_df.iloc[i][high_vol].mean()
        lv_factor.append((dates[i], long_ret - short_ret))

    lv_series = pd.Series(
        [x[1] for x in lv_factor],
        index=[x[0] for x in lv_factor],
        name="LV"
    )

    # ── Size factor  (SIZE) ──────────────────────────────────────────────────
    size_factor = []
    for i in range(63, len(dates)):
        px_window  = prices_df.iloc[max(0, i-63): i]
        avg_price  = px_window.mean()                  # price as size proxy
        n_group    = max(1, n // 3)

        small = avg_price.nsmallest(n_group).index     # low price = small-cap proxy
        large = avg_price.nlargest(n_group).index

        long_ret  = returns_df.iloc[i][small].mean()
        short_ret = returns_df.iloc[i][large].mean()
        size_factor.append((dates[i], long_ret - short_ret))

    size_series = pd.Series(
        [x[1] for x in size_factor],
        index=[x[0] for x in size_factor],
        name="SIZE"
    )

    # Align all factors to same date range
    common = mkt_rf.index
    factor_df = pd.concat([mkt_rf, mom_series, lv_series, size_series], axis=1, join="inner")
    factor_df = factor_df.dropna()

    print(f"   Factors built: {len(factor_df)} trading days")
    print(f"     MKT_RF: {factor_df['MKT_RF'].mean()*252*100:.2f}% p.a.")
    print(f"     MOM   : {factor_df['MOM'].mean()*252*100:.2f}% p.a.")
    print(f"     LV    : {factor_df['LV'].mean()*252*100:.2f}% p.a.")
    print(f"     SIZE  : {factor_df['SIZE'].mean()*252*100:.2f}% p.a.")

    return factor_df


def factor_regression(port_rets, factor_df, port_name="Portfolio"):
    """
    OLS regression of portfolio excess return on all four factors.

    (Rp - Rf) = α + β_mkt*MKT_RF + β_mom*MOM + β_lv*LV + β_size*SIZE + ε

    Returns dict of factor loadings, t-stats, alpha, and R².
    """
    aligned = pd.concat(
        [port_rets - RISK_FREE_RATE_DAILY, factor_df],
        axis=1, join="inner"
    ).dropna()

    if len(aligned) < 60:
        return None

    y = aligned.iloc[:, 0].values
    X = aligned.iloc[:, 1:].values
    X = np.column_stack([np.ones(len(X)), X])   # add intercept

    # OLS: β = (XᵀX)⁻¹Xᵀy
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    y_hat    = X @ beta
    resid    = y - y_hat
    ss_res   = (resid ** 2).sum()
    ss_tot   = ((y - y.mean()) ** 2).sum()
    r_sq     = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    n, k     = len(y), X.shape[1]
    s2       = ss_res / max(n - k, 1)
    se       = np.sqrt(np.diag(np.linalg.pinv(X.T @ X) * s2))
    t_stats  = beta / np.where(se > 1e-12, se, np.nan)

    factor_names = ["Alpha"] + list(factor_df.columns)
    result = {
        "Strategy"   : port_name,
        "Alpha (ann%)": round(beta[0] * 252 * 100, 3),
        "R²"         : round(r_sq, 4),
        "n_obs"      : n,
    }
    for i, fname in enumerate(factor_names):
        result[f"β_{fname}"]      = round(beta[i],    4)
        result[f"t_{fname}"]      = round(t_stats[i], 3)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_market_attribution(market_attr_df):
    """Bar charts: Alpha, Beta, R², Tracking Error, Information Ratio."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    metrics = [
        ("Alpha (ann %)", "Annualised Alpha (%)",        "Positive = outperformance"),
        ("Beta",          "Market Beta (β)",              "1.0 = market-neutral"),
        ("R²",            "Explained Variance (R²)",      "How much return is market-driven"),
        ("Tracking Error (ann %)", "Tracking Error (ann %)", "Lower = closer to benchmark"),
        ("Info Ratio",    "Information Ratio",            "Alpha per unit of tracking risk"),
        ("Treynor Ratio", "Treynor Ratio",                "Return per unit of market risk"),
    ]

    for ax, (col, ylabel, subtitle) in zip(axes, metrics):
        vals   = market_attr_df[col].values
        labels = market_attr_df.index.tolist()
        colors = [STRATEGY_COLORS.get(l, "#888") for l in labels]
        bars   = ax.bar(range(len(labels)), vals, color=colors, alpha=0.85,
                        edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(f"{ylabel}\n{subtitle}", fontsize=10)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + abs(vals).max() * 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle(
        "Market Attribution Analysis — All Strategies vs NIFTY 50\n"
        "(OLS regression: excess portfolio return on excess benchmark return)",
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.savefig("reports/factor_market_attribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: reports/factor_market_attribution.png")


def plot_sector_attribution(sector_attr_df):
    """Stacked bar chart: Allocation, Selection, Interaction effects."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Stacked bar (allocation + selection + interaction)
    ax = axes[0]
    sectors = sector_attr_df.index.tolist()
    x       = np.arange(len(sectors))
    alloc   = sector_attr_df["Allocation Effect"].values
    select  = sector_attr_df["Selection Effect"].values
    interact= sector_attr_df["Interaction Effect"].values

    ax.bar(x, alloc,   label="Allocation Effect",   color="#4C72B0", alpha=0.85)
    ax.bar(x, select,  label="Selection Effect",    color="#DD8452", alpha=0.85,
           bottom=alloc)
    ax.bar(x, interact,label="Interaction Effect",  color="#55A868", alpha=0.85,
           bottom=alloc + select)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(
        "Brinson-Hood-Beebower Sector Attribution\n"
        "(Equal-weight portfolio vs NIFTY free-float benchmark)",
        fontsize=11
    )
    ax.set_ylabel("Attribution Effect (% p.a.)")
    ax.set_xticks(x)
    ax.set_xticklabels(sectors, rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 2: Active weights
    ax2 = axes[1]
    active = sector_attr_df["Active Weight (%)"].values
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in active]
    ax2.barh(sectors, active, color=colors, alpha=0.80)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_title(
        "Active Sector Weights (Portfolio − Benchmark)\n"
        "Red = overweight | Blue = underweight",
        fontsize=11
    )
    ax2.set_xlabel("Active Weight (%)")
    ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("reports/factor_sector_attribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: reports/factor_sector_attribution.png")


def plot_factor_regression(factor_results_df):
    """Heatmap of factor loadings and grouped bar chart of beta magnitudes."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    beta_cols  = [c for c in factor_results_df.columns if c.startswith("β_") and c != "β_Alpha"]
    factor_names = [c.replace("β_", "") for c in beta_cols]
    strategies   = factor_results_df["Strategy"].tolist()

    beta_matrix = factor_results_df[beta_cols].values

    # Panel 1: heatmap of factor loadings
    ax = axes[0]
    im = ax.imshow(beta_matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-1.5, vmax=1.5, interpolation="nearest")
    ax.set_xticks(range(len(factor_names)))
    ax.set_xticklabels(factor_names, fontsize=10)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=9)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Factor Loading (β)", fontsize=9)

    # Annotate cells
    for i in range(len(strategies)):
        for j in range(len(factor_names)):
            ax.text(j, i, f"{beta_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(beta_matrix[i, j]) > 0.8 else "black")

    ax.set_title(
        "Factor Loadings — All Strategies\n"
        "(OLS regression on MKT_RF, MOM, LV, SIZE)",
        fontsize=11
    )

    # Panel 2: R² and Alpha bar chart
    ax2 = axes[1]
    x    = np.arange(len(strategies))
    w    = 0.35
    r2   = factor_results_df["R²"].values
    alphas = factor_results_df["Alpha (ann%)"].values
    colors = [STRATEGY_COLORS.get(s, "#888") for s in strategies]

    bars1 = ax2.bar(x - w/2, r2,     w, label="R² (factor model)",
                    color=colors, alpha=0.85)
    ax2b  = ax2.twinx()
    bars2 = ax2b.bar(x + w/2, alphas, w, label="Alpha (ann %)",
                     color=colors, alpha=0.55, hatch="///")

    ax2.set_ylabel("R² (left)")
    ax2b.set_ylabel("Annual Alpha % (right)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace(" ", "\n") for s in strategies], fontsize=8)
    ax2.set_title("Factor Model R² and Alpha by Strategy", fontsize=11)
    ax2.legend(loc="upper left",  fontsize=8)
    ax2b.legend(loc="upper right", fontsize=8)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("reports/factor_regression.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: reports/factor_regression.png")


def plot_rolling_alpha_beta(all_roll_results):
    """Four-panel: rolling alpha and beta for all strategies."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Rolling Alpha
    ax1 = axes[0]
    for name, (alpha_s, beta_s) in all_roll_results.items():
        ax1.plot(alpha_s.index, alpha_s.values,
                 label=name, color=STRATEGY_COLORS.get(name, "#888"),
                 linewidth=1.4)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax1.fill_between(ax1.get_xlim(), -0.02, 0.02, alpha=0.05, color="grey")
    ax1.set_ylabel("Rolling Alpha (annualised)")
    ax1.set_title(
        f"Rolling {ROLLING_WINDOW}-Day Alpha & Beta vs NIFTY 50\n"
        "(Expanding window | rf = 7% p.a.)",
        fontsize=12
    )
    ax1.legend(fontsize=9, ncol=3)
    ax1.grid(True, alpha=0.3)

    # Rolling Beta
    ax2 = axes[1]
    for name, (alpha_s, beta_s) in all_roll_results.items():
        ax2.plot(beta_s.index, beta_s.values,
                 label=name, color=STRATEGY_COLORS.get(name, "#888"),
                 linewidth=1.4)
    ax2.axhline(1.0, color="black", linewidth=0.8, linestyle="--",
                alpha=0.6, label="β = 1.0 (market)")
    ax2.axhline(0.0, color="grey",  linewidth=0.5, linestyle=":",  alpha=0.4)
    ax2.set_ylabel("Rolling Beta (β)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=9, ncol=3)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig("reports/factor_rolling_alpha_beta.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: reports/factor_rolling_alpha_beta.png")


def plot_factor_exposure_heatmap(factor_results_df):
    """
    Heatmap showing factor exposures across strategies, including
    t-stat significance indicators (|t| > 2 = *, |t| > 3 = **).
    """
    beta_cols  = [c for c in factor_results_df.columns if c.startswith("β_") and c != "β_Alpha"]
    tstat_cols = ["t_" + c[2:] for c in beta_cols]
    factor_names = [c.replace("β_", "") for c in beta_cols]
    strategies   = factor_results_df["Strategy"].tolist()

    beta_matrix  = factor_results_df[beta_cols].values
    tstat_matrix = factor_results_df[tstat_cols].values

    fig, ax = plt.subplots(figsize=(12, max(5, len(strategies) * 0.9 + 2)))
    im = ax.imshow(beta_matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-2, vmax=2, interpolation="nearest")

    ax.set_xticks(range(len(factor_names)))
    ax.set_xticklabels(
        [f"{f}\nFactor" for f in factor_names], fontsize=10, fontweight="bold"
    )
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=10)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Factor Exposure (β)", fontsize=10)

    for i in range(len(strategies)):
        for j in range(len(factor_names)):
            b = beta_matrix[i, j]
            t = tstat_matrix[i, j]
            stars = "**" if abs(t) > 3 else ("*" if abs(t) > 2 else "")
            txt   = f"{b:.2f}{stars}"
            color = "white" if abs(b) > 1.2 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold" if stars else "normal")

    ax.set_title(
        "Factor Exposure Heatmap — All Strategies\n"
        "(*|t|>2, **|t|>3 — statistically significant exposure)\n"
        "Blue = short factor / Red = long factor",
        fontsize=11, pad=15
    )

    plt.tight_layout()
    plt.savefig("reports/factor_exposure_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: reports/factor_exposure_heatmap.png")


def plot_cumulative_factor_returns(factor_df):
    """Show how each factor performed over time (cumulative return)."""
    fig, ax = plt.subplots(figsize=(13, 5))
    factor_colors = {
        "MKT_RF": "steelblue",
        "MOM"   : "darkorange",
        "LV"    : "green",
        "SIZE"  : "purple",
    }
    for col in factor_df.columns:
        cum = (1 + factor_df[col]).cumprod()
        ax.plot(cum.index, cum.values,
                label=col, color=factor_colors.get(col, "#888"),
                linewidth=1.8)
    ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_title(
        "Cumulative Factor Returns (2019–2024)\n"
        "MKT_RF: Market excess | MOM: Momentum | LV: Low-Volatility | SIZE: Small-cap premium",
        fontsize=11
    )
    ax.set_ylabel("Cumulative Value (₹1 invested)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig("reports/factor_cumulative_returns.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: reports/factor_cumulative_returns.png")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_attribution_summary(market_attr_df, factor_results_df):
    print("\n" + "=" * 110)
    print("FACTOR ATTRIBUTION SUMMARY")
    print("=" * 110)

    # Market attribution
    print("\nLAYER 1 — Market Attribution (vs NIFTY 50)")
    print("-" * 110)
    print(market_attr_df[[
        "Alpha (ann %)", "Beta", "R²",
        "Tracking Error (ann %)", "Info Ratio", "Treynor Ratio"
    ]].to_string())

    # Factor attribution
    print("\nLAYER 3 — Multi-Factor Attribution (4-Factor Model)")
    print("-" * 110)
    beta_cols  = [c for c in factor_results_df.columns if c.startswith("β_")]
    show_cols  = ["Strategy", "Alpha (ann%)", "R²"] + beta_cols
    show_cols  = [c for c in show_cols if c in factor_results_df.columns]
    print(factor_results_df[show_cols].to_string(index=False))
    print("=" * 110)
    print("   * = |t| > 2  (5% significance)   ** = |t| > 3  (1% significance)")
    print("   MKT_RF: market beta | MOM: momentum tilt | LV: low-vol tilt | SIZE: size tilt")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("FACTOR ATTRIBUTION ANALYSIS — Week 7")
    print("  Layer 1: Market attribution (Alpha, Beta, R², IR, Treynor)")
    print("  Layer 2: Sector attribution (Brinson-Hood-Beebower)")
    print("  Layer 3: Factor attribution (MKT_RF, MOM, LV, SIZE)")
    print("=" * 70)

    # ── Load stock-level data ────────────────────────────────────────────────
    returns_df, tickers, n = load_returns()
    prices_df = load_prices(tickers)
    print(f"\nStock data: {n} tickers | "
          f"{returns_df.index[0].date()} → {returns_df.index[-1].date()}")

    # ── Load portfolio return series ─────────────────────────────────────────
    print("\nLoading portfolio return series...")
    growths, port_rets = load_portfolio_returns()

    if not port_rets:
        print("[ERROR] No portfolio files found. Run portfolio models first.")
        raise SystemExit(1)

    # ── NIFTY 50 benchmark ───────────────────────────────────────────────────
    print("\nLoading NIFTY 50 benchmark...")
    bench_rets = load_nifty(returns_df.index[0], returns_df.index[-1])

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 1 — Market Attribution
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── Layer 1: Market Attribution ──────────────────────────────")
    market_rows = []
    for name, r in port_rets.items():
        if bench_rets is not None:
            attrs = compute_market_attribution(r, bench_rets)
        else:
            # Fallback: use equal-weight portfolio average as pseudo-benchmark
            eq_bench = returns_df.mean(axis=1)
            attrs    = compute_market_attribution(r, eq_bench)
        attrs["Strategy"] = name
        market_rows.append(attrs)
        print(f"   {name:<22} α={attrs['Alpha (ann %)']:>6}%  "
              f"β={attrs['Beta']:.3f}  R²={attrs['R²']:.3f}  "
              f"IR={attrs['Info Ratio']:.3f}")

    market_attr_df = pd.DataFrame(market_rows).set_index("Strategy")

    # Rolling alpha and beta
    print("\n   Computing rolling alpha/beta...")
    all_roll_results = {}
    for name, r in port_rets.items():
        bm = bench_rets if bench_rets is not None else returns_df.mean(axis=1)
        alpha_s, beta_s = rolling_alpha_beta(r, bm)
        all_roll_results[name] = (alpha_s, beta_s)

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 2 — Sector Attribution  (uses Equal Weight portfolio as proxy)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── Layer 2: Sector Attribution (BHB decomposition) ─────────")
    bench_weights = compute_market_cap_weights(tickers, prices_df)
    sector_attr_df = compute_sector_attribution(
        returns_df, tickers, bench_weights, returns_df
    )
    total_active = sector_attr_df["Total Active (%)"].sum()
    print(f"   Total active return (allocation + selection): {total_active:.4f}% p.a.")
    print(f"\n   Top 3 allocation contributors:")
    top3 = sector_attr_df.nlargest(3, "Allocation Effect")
    for sec, row in top3.iterrows():
        print(f"     {sec:<15} Alloc={row['Allocation Effect']:.4f}%  "
              f"Select={row['Selection Effect']:.4f}%  "
              f"Active_w={row['Active Weight (%)']:.1f}%")

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 3 — Factor Attribution
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── Layer 3: Multi-Factor Attribution ────────────────────────")
    print("\n   Building factors from stock-level data...")
    factor_df = build_factors(returns_df, prices_df, bench_rets, tickers)

    factor_rows = []
    for name, r in port_rets.items():
        res = factor_regression(r, factor_df, port_name=name)
        if res:
            factor_rows.append(res)
            stars = lambda t: "**" if abs(t) > 3 else ("*" if abs(t) > 2 else "")
            print(f"   {name:<22} α={res['Alpha (ann%)']:>6.2f}%  "
                  f"R²={res['R²']:.3f}  "
                  f"β_MKT={res['β_MKT_RF']:.3f}{stars(res['t_MKT_RF'])}  "
                  f"β_MOM={res['β_MOM']:.3f}{stars(res['t_MOM'])}  "
                  f"β_LV={res['β_LV']:.3f}{stars(res['t_LV'])}  "
                  f"β_SIZE={res['β_SIZE']:.3f}{stars(res['t_SIZE'])}")

    factor_results_df = pd.DataFrame(factor_rows)

    # ── Print full summary ───────────────────────────────────────────────────
    print_attribution_summary(market_attr_df, factor_results_df)

    # ── Save combined CSV ────────────────────────────────────────────────────
    combined = market_attr_df.merge(
        factor_results_df.set_index("Strategy"),
        left_index=True, right_index=True, how="outer"
    )
    combined.to_csv("data/portfolio/factor_attribution_results.csv")
    print("\nSaved: data/portfolio/factor_attribution_results.csv")

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_market_attribution(market_attr_df)
    plot_sector_attribution(sector_attr_df)
    plot_factor_regression(factor_results_df)
    plot_rolling_alpha_beta(all_roll_results)
    plot_factor_exposure_heatmap(factor_results_df)
    plot_cumulative_factor_returns(factor_df)

    print("\n✅ Factor attribution complete.")
    print("   Plots  → reports/factor_market_attribution.png")
    print("             reports/factor_sector_attribution.png")
    print("             reports/factor_regression.png")
    print("             reports/factor_rolling_alpha_beta.png")
    print("             reports/factor_exposure_heatmap.png")
    print("             reports/factor_cumulative_returns.png")
    print("   Data   → data/portfolio/factor_attribution_results.csv")