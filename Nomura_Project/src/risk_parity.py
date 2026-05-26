"""
risk_parity.py — Risk-Based Portfolio Construction  (v3)
=========================================================
Changes vs v2:

  ERC (v3):
    • Pure ERC weights blended with a momentum-quality tilt score
      w_final = (1 − TILT_ALPHA) × w_erc  +  TILT_ALPHA × w_tilt
      where w_tilt ranks stocks by:
          6-month momentum  ×  (1 / rolling_vol)   [quality proxy]
      TILT_ALPHA = 0.20 — preserves ERC risk discipline, adds return edge
    • All v2 improvements retained:
        log-barrier objective, multi-start (5 starts), EWMA hl=42d

  Max Diversification (v3):
    • Multi-start solver: inverse-vol, min-var proxy, 3 random Dirichlet
    • Gradient-robust objective: add tiny L2 regularisation so the
      linesearch never hits a degenerate flat region near 1/N
    • Falls back to inverse-vol (not equal weight) if all starts fail

Expected performance order: Max Diversification > ERC > Equal Weight

EWMA half-life : 42 trading days  (approx 2 months)
Rolling window : 252 trading days (approx 1 year)
Rebalance freq : monthly (calendar month-end)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from utils import (
    load_returns, load_prices,
    compute_lw_cov, build_constraints,
    RISK_FREE_RATE_ANNUAL,
    SECTOR_MAP, SECTOR_COLOR_MAP,
    print_summary, plot_growth_curves,
    print_sector_weights,
    sortino_ratio, max_drawdown, calmar_ratio,
)

# ── Configuration ──────────────────────────────────────────────────────────────
ROLLING_WINDOW   = 252
REBALANCE_FREQ   = 'ME'
EWMA_HALFLIFE    = 42

WEIGHT_MIN       = 0.00
WEIGHT_MAX       = 0.15

# ── ERC momentum-quality tilt ──────────────────────────────────────────────────
TILT_ALPHA       = 0.20   # 80% pure ERC + 20% momentum-quality tilt
MOMENTUM_WINDOW  = 126    # 6-month lookback in trading days

# ── Multi-start ────────────────────────────────────────────────────────────────
N_RANDOM_STARTS  = 3
RANDOM_SEED      = 42


# ──────────────────────────────────────────────────────────────────────────────
# EWMA COVARIANCE
# ──────────────────────────────────────────────────────────────────────────────

def _ewma_covariance(returns_window: np.ndarray, halflife: int) -> np.ndarray:
    """
    Exponentially weighted covariance with Ledoit-Wolf shrinkage.
    w_k = (1-lambda)*lambda^k, lambda = exp(-ln2/halflife), k=0 is most recent.
    """
    T, N   = returns_window.shape
    lam    = np.exp(-np.log(2) / halflife)
    lags   = np.arange(T - 1, -1, -1, dtype=float)
    wts    = (1 - lam) * (lam ** lags)
    wts   /= wts.sum()

    w_col      = wts[:, np.newaxis]
    mean       = (w_col * returns_window).sum(axis=0)
    demeaned   = returns_window - mean
    weighted_X = np.sqrt(w_col) * demeaned

    lw = LedoitWolf(assume_centered=True)
    lw.fit(weighted_X)
    return lw.covariance_


def _ewma_covariance_fullsample(returns: pd.DataFrame, halflife: int) -> np.ndarray:
    return _ewma_covariance(returns.values, halflife)


# ──────────────────────────────────────────────────────────────────────────────
# MOMENTUM-QUALITY TILT SCORE
# ──────────────────────────────────────────────────────────────────────────────

def _momentum_quality_weights(returns_window: np.ndarray, n: int) -> np.ndarray:
    """
    Compute a tilt weight vector from momentum x quality scores.

    Momentum : 6-month cumulative return
    Quality  : 1 / rolling volatility (low-vol = higher quality proxy)
    Score    : rank(momentum) x rank(quality)  — rank-based avoids outliers
    Output   : softmax-normalised, clipped to [WEIGHT_MIN, WEIGHT_MAX]
    """
    lookback   = min(MOMENTUM_WINDOW, returns_window.shape[0])
    mom_window = returns_window[-lookback:]
    momentum   = (1 + mom_window).prod(axis=0) - 1

    vol     = mom_window.std(axis=0)
    vol     = np.maximum(vol, 1e-10)
    quality = 1.0 / vol

    def rank_arr(x):
        temp  = x.argsort()
        ranks = np.empty_like(temp, dtype=float)
        ranks[temp] = np.arange(len(x))
        return ranks

    score = (rank_arr(momentum) + 1) * (rank_arr(quality) + 1)

    # Softmax with temperature = n (moderate spread)
    score_shifted = score - score.max()
    exp_score     = np.exp(score_shifted / n)
    tilt_weights  = exp_score / exp_score.sum()

    tilt_weights  = np.clip(tilt_weights, WEIGHT_MIN, WEIGHT_MAX)
    tilt_weights /= tilt_weights.sum()
    return tilt_weights


# ──────────────────────────────────────────────────────────────────────────────
# LOG-BARRIER ERC OBJECTIVE
# ──────────────────────────────────────────────────────────────────────────────

def _erc_log_barrier_objective(w, cov_np):
    """
    min  sum_i sum_j  (log RC_i - log RC_j)^2
    where RC_i = w_i * (Sigma w)_i

    Sharp gradient near equal-weight drives solver away from 1/N.
    """
    port_var = w @ cov_np @ w
    if port_var < 1e-14:
        return 0.0
    rc     = w * (cov_np @ w)
    rc_pos = np.maximum(rc, 1e-14)
    log_rc = np.log(rc_pos)
    diff   = log_rc[:, None] - log_rc[None, :]
    return float(np.sum(diff ** 2))


# ──────────────────────────────────────────────────────────────────────────────
# MULTI-START ERC SOLVER  (with momentum-quality tilt blend)
# ──────────────────────────────────────────────────────────────────────────────

def _solve_erc(cov_np, n, constraints, bounds, rng,
               prev_weights=None, tilt_weights=None):
    """
    Solve ERC from 5 starting points, keep best.
    Then blend: w_final = (1-TILT_ALPHA)*w_erc + TILT_ALPHA*w_tilt

    The tilt is applied AFTER optimisation — the pure ERC solve is
    uncontaminated and the tilt nudges the allocation toward higher-momentum,
    lower-volatility stocks to boost return without breaking risk discipline.
    """
    vols    = np.sqrt(np.maximum(np.diag(cov_np), 1e-14))
    inv_vol = (1.0 / vols) / (1.0 / vols).sum()

    starts = [np.ones(n) / n, inv_vol]
    if prev_weights is not None:
        starts.append(prev_weights / prev_weights.sum())
    for _ in range(N_RANDOM_STARTS):
        starts.append(rng.dirichlet(np.full(n, 5.0)))

    best_obj     = np.inf
    best_weights = np.ones(n) / n
    any_success  = False

    for w0 in starts:
        w0 = np.clip(w0, WEIGHT_MIN, WEIGHT_MAX)
        if w0.sum() < 1e-10:
            w0 = np.ones(n) / n
        w0 /= w0.sum()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                fun         = _erc_log_barrier_objective,
                x0          = w0,
                args        = (cov_np,),
                method      = 'SLSQP',
                bounds      = bounds,
                constraints = constraints,
                options     = {'ftol': 1e-14, 'maxiter': 3000},
            )

        if result.success:
            w_candidate = np.clip(result.x, 0, 1)
            w_candidate /= w_candidate.sum()
            obj = _erc_log_barrier_objective(w_candidate, cov_np)
            if obj < best_obj:
                best_obj     = obj
                best_weights = w_candidate
                any_success  = True

    if not any_success:
        best_weights = inv_vol  # fallback: inverse-vol, not equal weight

    # ── Momentum-quality tilt blend ───────────────────────────────────────────
    if tilt_weights is not None and TILT_ALPHA > 0:
        blended = (1 - TILT_ALPHA) * best_weights + TILT_ALPHA * tilt_weights
        blended = np.clip(blended, WEIGHT_MIN, WEIGHT_MAX)
        blended /= blended.sum()
        best_weights = blended

    return best_weights, any_success


# ──────────────────────────────────────────────────────────────────────────────
# 1. ERC — ROLLING MONTHLY REBALANCING
# ──────────────────────────────────────────────────────────────────────────────

def run_erc(returns, tickers, constraints, bounds):
    """
    Rolling ERC with monthly rebalance, EWMA-LW covariance, log-barrier
    objective, multi-start solver, and momentum-quality tilt blend.
    """
    n   = len(tickers)
    rng = np.random.RandomState(RANDOM_SEED)

    dates = returns.index
    T     = len(dates)

    rebalance_dates = pd.date_range(
        start=dates[ROLLING_WINDOW],
        end=dates[-1],
        freq=REBALANCE_FREQ,
    )
    rebalance_dates = rebalance_dates[rebalance_dates.isin(dates)]

    daily_weights   = np.zeros((T, n))
    weights_log     = {}
    prev_weights    = np.ones(n) / n
    current_weights = prev_weights.copy()
    rebalance_set   = set(rebalance_dates)
    rebalance_count = 0
    failed_count    = 0

    for i, date in enumerate(dates):
        if i < ROLLING_WINDOW:
            daily_weights[i] = prev_weights
            continue

        if date in rebalance_set:
            window_ret = returns.iloc[i - ROLLING_WINDOW: i].values

            try:
                cov_window = _ewma_covariance(window_ret, EWMA_HALFLIFE)
            except Exception:
                lw         = LedoitWolf().fit(window_ret)
                cov_window = lw.covariance_

            tilt_w = _momentum_quality_weights(window_ret, n)

            new_weights, success = _solve_erc(
                cov_window, n, constraints, bounds, rng,
                prev_weights=current_weights,
                tilt_weights=tilt_w,
            )

            if success:
                rebalance_count += 1
            else:
                failed_count += 1

            current_weights = new_weights
            weights_log[date] = current_weights.copy()

        daily_weights[i] = current_weights

    print(f"   Rolling EWMA-ERC v3: {rebalance_count} successful | "
          f"{failed_count} fallbacks")
    print(f"   EWMA half-life : {EWMA_HALFLIFE}d  |  "
          f"lambda = {np.exp(-np.log(2)/EWMA_HALFLIFE):.4f}")
    print(f"   Weight bounds  : [{WEIGHT_MIN*100:.0f}%, {WEIGHT_MAX*100:.0f}%]")
    print(f"   Tilt alpha     : {TILT_ALPHA*100:.0f}%  "
          f"(80% ERC + 20% momentum-quality)")

    weights_history = pd.DataFrame(
        {d: w for d, w in weights_log.items()},
        index=tickers,
    ).T
    weights_history.index = pd.DatetimeIndex(weights_history.index)

    ret_array     = (returns.values * daily_weights).sum(axis=1)
    ret_series    = pd.Series(ret_array, index=dates)
    growth_series = (1 + ret_series).cumprod()

    return weights_history, ret_series, growth_series


def print_erc_diagnostics(weights_history, tickers):
    final_w  = weights_history.iloc[-1].values
    first_w  = weights_history.iloc[0].values
    avg_turn = weights_history.diff().abs().sum(axis=1).mean() * 100
    zero_pos = (final_w < 1e-4).sum()
    nonzero  = final_w[final_w > 1e-6]
    log_disp = np.std(np.log(nonzero)) if len(nonzero) > 1 else 0.0

    print(f"   Weight range (final) : {final_w.min()*100:.2f}% - {final_w.max()*100:.2f}%")
    print(f"   Avg monthly turnover : {avg_turn:.2f}% of portfolio")
    print(f"   Max stock drift      : {abs(final_w - first_w).max()*100:.2f}pp")
    print(f"   Zero-weight positions: {zero_pos} / {len(final_w)}")
    print(f"   Log-dispersion       : {log_disp:.4f}  "
          f"(0.00 = equal-weight; higher = more differentiated)")


# ──────────────────────────────────────────────────────────────────────────────
# 2. MAXIMUM DIVERSIFICATION  (v3 — multi-start, gradient-robust)
# ──────────────────────────────────────────────────────────────────────────────

def run_max_diversification(returns, tickers, cov_np, constraints, bounds):
    """
    Maximise the diversification ratio:
        DR = sum(w_i * sigma_i) / sigma_p

    v3 fixes:
      - Multi-start: inverse-vol, min-var proxy, 3 random Dirichlet starts
      - L2 regularisation (lam_reg=1e-4) on portfolio variance term
        prevents the "positive directional derivative" linesearch failure
      - Falls back to inverse-vol (NOT equal weight) if all starts fail

    Why the linesearch was failing in v1/v2:
        At w = 1/N, the gradient of DR is near-zero (all assets contribute
        equally), so SLSQP found no improving direction and quit.
        Starting from inverse-vol provides a non-flat gradient.
        The L2 term ensures the Hessian stays positive definite.
    """
    n        = len(tickers)
    rng      = np.random.RandomState(RANDOM_SEED + 1)
    vol_arr  = returns.std().values
    lam_reg  = 1e-4

    def neg_dr(w):
        port_var     = w @ cov_np @ w + lam_reg * (w @ w)
        port_vol     = np.sqrt(max(port_var, 1e-14))
        weighted_vol = w @ vol_arr
        return -(weighted_vol / port_vol)

    # Inverse-vol start
    vols    = np.sqrt(np.maximum(np.diag(cov_np), 1e-14))
    inv_vol = (1.0 / vols) / (1.0 / vols).sum()

    # Min-var proxy: top-N lowest-vol stocks equal-weighted
    n_low    = max(5, n // 5)
    low_idx  = np.argsort(vols)[:n_low]
    minvar_w = np.zeros(n)
    minvar_w[low_idx] = 1.0 / n_low

    starts = [inv_vol, minvar_w]
    for _ in range(N_RANDOM_STARTS):
        starts.append(rng.dirichlet(np.full(n, 3.0)))

    best_dr      = -np.inf
    best_weights = inv_vol.copy()
    any_success  = False

    for w0 in starts:
        w0 = np.clip(w0, WEIGHT_MIN, WEIGHT_MAX)
        if w0.sum() < 1e-10:
            w0 = inv_vol.copy()
        w0 /= w0.sum()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                fun         = neg_dr,
                x0          = w0,
                method      = 'SLSQP',
                bounds      = bounds,
                constraints = constraints,
                options     = {'ftol': 1e-12, 'maxiter': 3000},
            )

        if result.success:
            w_cand = np.clip(result.x, 0, 1)
            w_cand /= w_cand.sum()
            pv  = np.sqrt(max(w_cand @ cov_np @ w_cand, 1e-14))
            dr  = (w_cand @ vol_arr) / pv
            if dr > best_dr:
                best_dr      = dr
                best_weights = w_cand
                any_success  = True

    if any_success:
        print(f"   Max Diversification v3: SUCCESS  (best DR = {best_dr:.4f})")
    else:
        print("   Max Diversification v3: all starts failed — inverse-vol fallback")
        best_weights = inv_vol

    port_vol     = np.sqrt(max(best_weights @ cov_np @ best_weights, 1e-14)) * np.sqrt(252)
    weighted_vol = best_weights @ vol_arr * np.sqrt(252)
    dr_final     = weighted_vol / port_vol
    print(f"   Final Diversification Ratio: {dr_final:.3f}")

    ret_series    = returns.dot(best_weights)
    growth_series = (1 + ret_series).cumprod()
    return best_weights, ret_series, growth_series


# ──────────────────────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────────────────────

def plot_growth_all(growth_erc, growth_md, growth_ew):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(growth_md.index,  growth_md.values,
            label='Max Diversification (v3)',
            color='green',      linewidth=2.0)
    ax.plot(growth_erc.index, growth_erc.values,
            label=f'ERC v3 (EWMA hl={EWMA_HALFLIFE}d + {TILT_ALPHA*100:.0f}% tilt)',
            color='darkorange', linewidth=1.8)
    ax.plot(growth_ew.index,  growth_ew.values,
            label='Equal Weight (ref)',
            color='grey', linewidth=1.2, linestyle='--', alpha=0.7)
    ax.set_title(
        "Risk-Based Portfolios — Cumulative Growth (v3)\n"
        f"(ERC: log-barrier + momentum-quality tilt | "
        f"MaxDiv: multi-start + L2 reg | EWMA hl={EWMA_HALFLIFE}d)",
        fontsize=11,
    )
    ax.set_ylabel("Portfolio Value (Rs 1 invested)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/risk_parity_growth.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/risk_parity_growth.png")


def plot_weight_comparison(weights_history_erc, weights_md, weights_ew, tickers):
    weights_erc = weights_history_erc.iloc[-1].values
    fig, axes   = plt.subplots(1, 2, figsize=(16, 6))

    weight_sets = {
        'Max Div (v3)'    : weights_md,
        'ERC v3\n(final)' : weights_erc,
        'Equal Weight'    : weights_ew,
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
                       if SECTOR_MAP.get(t, 'Other') == sector]
            sec_wts.append(sum(wts[i] for i in sec_idx) * 100 if sec_idx else 0)
        ax.bar(x, sec_wts, bottom=bottoms,
               color=SECTOR_COLOR_MAP.get(sector, '#CCC'),
               label=sector, edgecolor='white', linewidth=0.3)
        bottoms += np.array(sec_wts)
    ax.axhline(30, color='red', linewidth=1.0, linestyle='--',
               alpha=0.7, label='30% sector cap')
    ax.set_title("Sector Allocation — Risk-Based Models (v3)", fontsize=12)
    ax.set_ylabel("Allocation (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(strat_labels, rotation=20, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.set_ylim(0, 105)
    ax.grid(True, axis='y', alpha=0.3)

    ax = axes[1]
    erc_top10 = (pd.Series(weights_erc, index=tickers)
                 .sort_values(ascending=False).head(10))
    md_top10  = pd.Series(weights_md, index=tickers).loc[erc_top10.index]
    x2, w2   = np.arange(10), 0.35
    ax.bar(x2 - w2/2, erc_top10.values * 100, w2,
           label='ERC v3 (final month)', color='darkorange', alpha=0.85)
    ax.bar(x2 + w2/2, md_top10.values  * 100, w2,
           label='Max Div v3',           color='green',      alpha=0.85)
    ax.set_title("Top 10 Holdings — ERC v3 vs Max Div v3", fontsize=12)
    ax.set_ylabel("Weight (%)")
    ax.set_xticks(x2)
    ax.set_xticklabels([t.replace('.NS', '') for t in erc_top10.index],
                       rotation=35, ha='right', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/risk_parity_weights.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/risk_parity_weights.png")


def plot_erc_risk_contributions(weights_history, tickers):
    n        = len(tickers)
    final_w  = weights_history.iloc[-1].values
    w_series = pd.Series(final_w * 100, index=tickers).sort_values(ascending=False)
    deviation = (w_series - (100.0 / n)).abs()
    colors    = ['#d62728' if d > 2.0 else '#2ca02c' for d in deviation]
    fig, ax   = plt.subplots(figsize=(14, 5))
    ax.bar(range(n), w_series.values, color=colors, alpha=0.85, edgecolor='white')
    ax.axhline(100.0 / n, color='black', linewidth=1.5, linestyle='--',
               label=f'Equal weight = {100/n:.2f}%')
    ax.set_xticks(range(n))
    ax.set_xticklabels([t.replace('.NS', '') for t in w_series.index],
                       rotation=90, fontsize=7)
    ax.set_title(
        "ERC v3 — Final-Month Weights\n"
        f"(log-barrier + {TILT_ALPHA*100:.0f}% momentum-quality tilt | "
        f"bounds=[{WEIGHT_MIN*100:.0f}%,{WEIGHT_MAX*100:.0f}%])",
        fontsize=12,
    )
    ax.set_ylabel("Weight (%)")
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/erc_risk_contributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/erc_risk_contributions.png")


def plot_erc_weight_heatmap(weights_history, tickers):
    short_tickers = [t.replace('.NS', '') for t in tickers]
    data          = weights_history[tickers].T * 100
    n_dates       = len(weights_history)
    step          = max(1, n_dates // 12)
    xtick_idx     = list(range(0, n_dates, step))
    xtick_labels  = [weights_history.index[i].strftime('%b %Y') for i in xtick_idx]
    fig, ax       = plt.subplots(figsize=(16, max(8, len(tickers) // 4)))
    im = ax.imshow(data.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xticks(xtick_idx)
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(short_tickers, fontsize=7)
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Weight (%)", fontsize=9)
    ax.set_title(
        f"ERC v3 Rolling Weight Heatmap\n"
        f"(log-barrier + {TILT_ALPHA*100:.0f}% tilt | EWMA hl={EWMA_HALFLIFE}d | "
        f"window={ROLLING_WINDOW}d | darker = higher allocation)",
        fontsize=12,
    )
    ax.set_xlabel("Rebalance Date")
    ax.set_ylabel("Stock")
    plt.tight_layout()
    plt.savefig("reports/erc_weight_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/erc_weight_heatmap.png")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 68)
    print("RISK-BASED PORTFOLIO MODELS  (v3)")
    print(f"  1. ERC — log-barrier obj | EWMA hl={EWMA_HALFLIFE}d | "
          f"window={ROLLING_WINDOW}d | monthly")
    print(f"     Multi-start + {TILT_ALPHA*100:.0f}% momentum-quality tilt blend")
    print("  2. Max Diversification — multi-start SLSQP + L2 regularisation")
    print("=" * 68)

    returns, tickers, n = load_returns()
    prices = load_prices(tickers)

    print(f"\nData: {n} tickers | "
          f"{returns.index[0].date()} -> {returns.index[-1].date()}")

    # Equal weight baseline
    weights_ew = np.ones(n) / n
    returns_ew = returns.dot(weights_ew)
    growth_ew  = (1 + returns_ew).cumprod()

    # Full-sample EWMA-LW covariance
    print(f"\nEstimating full-sample EWMA-LW cov (hl={EWMA_HALFLIFE}d)...")
    cov_np = _ewma_covariance_fullsample(returns, EWMA_HALFLIFE)
    lam    = np.exp(-np.log(2) / EWMA_HALFLIFE)
    print(f"   lambda = {lam:.4f}  |  effective obs approx {int(1/(1-lam))} days")

    constraints = build_constraints(tickers)
    bounds      = tuple((WEIGHT_MIN, WEIGHT_MAX) for _ in range(n))

    # 1. Rolling EWMA-ERC v3
    print(f"\nRunning Rolling EWMA-ERC v3...")
    weights_history, returns_erc, growth_erc = run_erc(
        returns, tickers, constraints, bounds
    )
    growth_erc.to_csv("data/portfolio/erc_growth.csv")
    print(f"   Total return : {(growth_erc.iloc[-1]-1)*100:.1f}%")
    print(f"   Rebalances   : {len(weights_history)}")
    print_erc_diagnostics(weights_history, tickers)
    print_sector_weights(weights_history.iloc[-1].values, tickers, label="ERC v3 (final)")

    # 2. Max Diversification v3
    print("\nRunning Max Diversification v3 (multi-start + L2 reg)...")
    weights_md, returns_md, growth_md = run_max_diversification(
        returns, tickers, cov_np, constraints, bounds
    )
    growth_md.to_csv("data/portfolio/max_diversification_growth.csv")
    print(f"   Total return: {(growth_md.iloc[-1]-1)*100:.1f}%")
    print_sector_weights(weights_md, tickers, label="Max Diversification v3")

    # Performance summary
    all_growths = {
        'Max Diversification v3': growth_md,
        'ERC v3 (tilt+EWMA)'   : growth_erc,
        'Equal Weight'          : growth_ew,
    }
    all_rets = {
        'Max Diversification v3': returns_md,
        'ERC v3 (tilt+EWMA)'   : returns_erc,
        'Equal Weight'          : returns_ew,
    }
    print_summary(
        all_growths, all_rets,
        note=(f"ERC v3: log-barrier, multi-start, EWMA hl={EWMA_HALFLIFE}d, "
              f"{TILT_ALPHA*100:.0f}% momentum-quality tilt, "
              f"bounds=[{WEIGHT_MIN*100:.0f}%,{WEIGHT_MAX*100:.0f}%] | "
              "MaxDiv v3: multi-start + L2 reg | sector/cyclical caps"),
    )

    # Plots
    plot_growth_all(growth_erc, growth_md, growth_ew)
    plot_weight_comparison(weights_history, weights_md, weights_ew, tickers)
    plot_erc_risk_contributions(weights_history, tickers)
    plot_erc_weight_heatmap(weights_history, tickers)

    print("\nRisk-based models v3 complete.")
    print("   Outputs -> data/portfolio/erc_growth.csv")
    print("              data/portfolio/max_diversification_growth.csv")
    print("   Plots   -> reports/risk_parity_growth.png")
    print("              reports/risk_parity_weights.png")
    print("              reports/erc_risk_contributions.png")
    print("              reports/erc_weight_heatmap.png")