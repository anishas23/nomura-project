"""
black_litterman.py — Rolling Black-Litterman Portfolio
=======================================================
A full dynamic Black-Litterman implementation with all institutional fixes:
 
  FIX 1 — Rolling monthly re-estimation:
      BL views are re-estimated every REBALANCE_FREQ (~21) trading days.
      Each rebalance uses only data available at that point (no look-ahead).
      Produces a genuine out-of-sample return time-series instead of a
      single static in-sample snapshot.
 
  FIX 2 — Market-cap proxy equilibrium weights:
      True BL uses market-cap weights as the equilibrium portfolio.
      Approximated using free-float share counts (in crore shares) from
      NSE Nifty 50 methodology, combined with trailing 63-day avg price.
      Correctly up-weights Reliance/HDFC and down-weights smaller names.
 
  FIX 3 — Ledoit-Wolf shrinkage covariance:
      Re-estimated at every rebalance point on expanding window data.
      Reduces noise vs sample covariance when n_assets >> n_obs per window.
 
  FIX 4 — Explicit risk-free rate (India 10Y G-Sec = 7% p.a.):
      Used in implied equilibrium returns (pi = λΣw_mkt) and the BL
      optimisation objective (maximise excess return per unit of risk).
 
  FIX 5 — Transaction costs at each rebalance:
      10 bps one-way cost applied as: cost = TC × Σ|new_w − old_w|
      Deducted from portfolio return on the rebalance day.
      First trade cost = TC × 1.0 (buying from full cash position).
 
  FIX 6 — NIFTY 50 benchmark integration:
      Downloads ^NSEI from yfinance over the same date range.
      Benchmark growth, returns, Sharpe, drawdown added to all plots
      and the summary table for apples-to-apples comparison.
 
  IMPROVEMENT 1 — Composite Dual-Horizon Momentum Signal:
      signal = (0.7 × 6M_momentum + 0.3 × 12M_momentum) / annualised_vol
 
  IMPROVEMENT 2 — Stronger View Magnitude (Q scaling):
      Q_SCALE raised from 0.02 → 0.06 for meaningful posterior tilts.
 
  IMPROVEMENT 3 — Dynamic Confidence-Scaled Omega:
      Ω is signal-driven with a 20× swing range between min/max confidence.
      confidence_i = clip(|spread_i| / OMEGA_SPREAD_REF, 0, 1)
      omega_i      = base_omega_i × (1 − confidence_i × (1 − OMEGA_FLOOR))
 
  IMPROVEMENT 4 — Weak/Noisy View Suppression:
      Hard gate: views with |raw_spread| < VIEW_SIGNAL_THRESHOLD = 0.25
      are discarded. Zero-passing-view months fall back to equilibrium MV.
 
  IMPROVEMENT 5 — Blended Multi-Factor Sector Score:
      Replaces pure momentum with three-factor composite:
          score_i = 0.5 × momentum_factor_z
                  + 0.3 × low_volatility_factor_z
                  + 0.2 × trend_strength_factor_z
 
  IMPROVEMENT 6 — RELATIVE MOMENTUM:
      All momentum and low-volatility factors are now computed on
      EXCESS returns (stock/sector return − NIFTY 50 daily return)
      instead of raw absolute returns.
 
  IMPROVEMENT 7 — MARKET REGIME FILTER:
      The portfolio now detects whether the market is in a BULL, NEUTRAL,
      or BEAR/PANIC regime at each rebalance point using NIFTY 50 vs its
      own 200-day moving average and a short-term volatility spike test.
 
      Regime classification (per rebalance):
        BULL   : NIFTY > 200-DMA  AND  30-day vol < REGIME_VOL_SPIKE threshold
        NEUTRAL: NIFTY within ±REGIME_DMA_BUFFER (±2%) of 200-DMA
        BEAR   : NIFTY < 200-DMA  AND  NOT NEUTRAL
        PANIC  : NIFTY < 200-DMA  AND  30-day vol > REGIME_PANIC_VOL threshold
                 (subset of BEAR; triggers strongest defensive posture)
 
      Per-regime parameter overrides:
 
        Regime    Q_Scale  ViewGate  MaxWt   SectorCap  CyclCap  Diversify
        ────────  ───────  ────────  ─────   ─────────  ───────  ─────────
        BULL      0.06     0.25      15%     30%        40%      normal
        NEUTRAL   0.04     0.30      13%     28%        37%      slightly more
        BEAR      0.02     0.40      11%     25%        33%      increased
        PANIC     0.01     0.60      9%      22%        28%      maximum
 
      In BEAR/PANIC regimes:
        - View confidence scaling (Ω) is reduced by REGIME_OMEGA_DAMP
          (50% in BEAR, 80% in PANIC) → prior equilibrium dominates.
        - Minimum weight floor raised (more diversification).
        - Rebalance is still monthly but TC dampens unnecessary turnover.
 
  IMPROVEMENT 8 — THRESHOLD-GATED REBALANCING:
      Monthly rebalance dates are still pre-scheduled (every ~21 trading
      days), but execution only happens when the proposed weight change
      is large enough to justify incurring transaction costs.
 
      Gate condition (checked at every scheduled rebalance):
          max_stock_drift  = max(|new_w_i − old_w_i|)   for all i
          total_turnover   = Σ|new_w_i − old_w_i|        (one-way %)
 
          EXECUTE rebalance  if  max_stock_drift  ≥ REBALANCE_MAX_DRIFT
                             OR  total_turnover   ≥ REBALANCE_TURNOVER_GATE
 
          Default thresholds:
              REBALANCE_MAX_DRIFT    = 0.02  (2% single-stock drift)
              REBALANCE_TURNOVER_GATE= 0.04  (4% total one-way turnover)
 
  IMPROVEMENT 9 — EWMA COVARIANCE ESTIMATION:
      Replaces the static Ledoit-Wolf (LW) covariance with a blended
      estimator that weights recent observations more heavily:
 
          Σ_final = α × Σ_EWMA  +  (1 − α) × Σ_LW
 
      where α = EWMA_BLEND_WEIGHT (default 0.7).
 
  IMPROVEMENT 10 — POSITION CONCENTRATION LOGIC:
      Prevents alpha dilution by enforcing minimum conviction thresholds
      and concentrating allocations into the highest-scoring sectors.
 
      Two complementary mechanisms:
 
      A) TOP-N SECTOR CONCENTRATION:
         - Only sectors with blended_score ≥ CONCENTRATION_SCORE_FLOOR
           are eligible to receive overweight vs equilibrium.
         - At most TOP_N_SECTORS (default 10) sectors receive non-trivial
           allocation above the minimum floor weight.
         - Remaining sectors are capped at their equilibrium weight or
           MIN_WEIGHT_FLOOR, whichever is larger.
 
         Regime-adjusted TOP_N_SECTORS:
           BULL   : 10 sectors (broad participation in up-trend)
           NEUTRAL: 8 sectors  (moderate concentration)
           BEAR   : 6 sectors  (defensive, concentrate on quality)
           PANIC  : 4 sectors  (maximum concentration on defensives)
 
      B) MINIMUM CONVICTION THRESHOLD (per stock):
         - Each stock in the final optimised weight vector is tested
           against a CONVICTION_MIN_WEIGHT threshold.
         - Stocks falling below CONVICTION_MIN_WEIGHT are zeroed out
           (floored to 0) and their weight is redistributed to the
           surviving stocks proportionally.
         - This eliminates "rounding" allocations that add friction but
           no meaningful return contribution.
         - CONVICTION_MIN_WEIGHT is regime-dependent:
             BULL   : 1.5%  (allow broader participation)
             NEUTRAL: 1.8%  (tighten slightly)
             BEAR   : 2.2%  (concentrate on conviction names)
             PANIC  : 2.5%  (maximum conviction filtering)
 
      C) CONCENTRATION SCORE FLOOR:
         - Sectors whose blended score is below CONCENTRATION_SCORE_FLOOR
           (default 0.0 in z-score units, i.e. below-average factor rank)
           are ineligible for overweighting.
         - These sectors are forced to their equilibrium weight
           (or min_wt floor), freeing budget for high-conviction names.
         - This gate is independent of the VIEW_SIGNAL_THRESHOLD gate
           which operates on view pair spreads; this gate operates on
           absolute sector quality.
 
      D) ALPHA CONCENTRATION RATIO (diagnostic):
         - Printed at each rebalance: what fraction of total active weight
           (BL weight − market weight) is concentrated in the top quartile
           of stocks by conviction weight.
         - Target: ≥ 60% of active weight in top-quartile names.
         - Warns if concentration is too diluted.
 
      New constants:
          TOP_N_SECTORS_BASE            = 10   # max eligible sectors (BULL)
          CONCENTRATION_SCORE_FLOOR     = 0.0  # min blended z-score to overweight
          CONVICTION_MIN_WEIGHT         = 0.015 # base threshold (BULL); regime-scaled
          CONCENTRATION_WARN_THRESHOLD  = 0.60  # warn if active-weight α-ratio < this
 
      Per-regime TOP_N override:
          TOP_N_SECTORS_REGIME = {
              'BULL'   : 10,
              'NEUTRAL': 8,
              'BEAR'   : 6,
              'PANIC'  : 4,
          }
 
      Per-regime CONVICTION_MIN_WEIGHT:
          CONVICTION_MIN_WEIGHT_REGIME = {
              'BULL'   : 0.015,
              'NEUTRAL': 0.018,
              'BEAR'   : 0.022,
              'PANIC'  : 0.025,
          }
 
      Integration with existing improvements:
        - Concentration filtering is applied AFTER the BL posterior
          optimisation (solve_bl) produces raw weights.
        - It is applied BEFORE the IMPROVEMENT 8 threshold gate check,
          so the gate sees the concentrated weights (more meaningful delta).
        - The equilibrium fallback (no views passing gate) also benefits:
          low-scoring sectors are suppressed even in equilibrium-only mode.
 
View construction methodology (data-driven, not manual):
  Step 1: Detect market regime via NIFTY 200-DMA and vol-spike filter.
  Step 2: Override Q_SCALE, ViewGate, MaxWt, SectorCap per regime.
  Step 3: Compute regime-adjusted EWMA λ.
  Step 4: Estimate Σ_EWMA and Σ_LW; blend → Σ_final.
  Step 5: Compute stock-level excess-return momentum & low-vol vs NIFTY.
  Step 6: Z-score each factor cross-sectionally across sectors.
  Step 7: Blend: score = 0.5×momentum_z + 0.3×low_vol_z + 0.2×trend_z
  Step 8: Rank sectors by blended score.
  Step 9: GATE — discard any view where |raw_spread| < regime_view_gate.
  Step 10: Q = raw_spread × regime_q_scale / MOMENTUM_WINDOW_6M
  Step 11: Ω = base_omega × regime_omega_damp × (1 − conf × (1−OMEGA_FLOOR))
  Step 12: If NO views pass the gate → optimise on pi (equilibrium) only.
  Step 13: Apply concentration filter (Improvement 10):
             - Zero out stocks below CONVICTION_MIN_WEIGHT[regime]
             - Redistribute weight to surviving names proportionally
             - Cap sectors outside TOP_N_SECTORS at equilibrium weight
  Step 14: Apply threshold gate (Improvement 8) — only execute if
             concentrated drift ≥ thresholds.
 
Portfolio constraints per rebalance:
  Min/Max weight : 1% / regime_max_wt per stock
  Sector cap     : regime_sector_cap per sector
  Cyclical cap   : regime_cyclical_cap combined (Auto + Metals + Energy + Chemicals)
 
Run independently:
    python src/black_litterman.py
 
Outputs:
    data/portfolio/black_litterman_growth.csv
    reports/bl_growth.png
    reports/bl_momentum_views.png
    reports/bl_sector_tilts.png
    reports/bl_rolling_sharpe.png
    reports/bl_drawdown.png
    reports/bl_regime_history.png
    reports/bl_rebalance_activity.png
    reports/bl_ewma_diagnostics.png
    reports/bl_concentration_diagnostics.png   ← NEW (Improvement 10)
"""
 
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
 
from utils import (
    load_returns, load_prices,
    compute_lw_cov, build_constraints,
    compute_market_cap_weights,
    RISK_FREE_RATE_ANNUAL, RISK_FREE_RATE_DAILY,
    TRANSACTION_COST,
    LAMBDA, TAU,
    REBALANCE_FREQ, MOMENTUM_WINDOW, MIN_HISTORY,
    SECTOR_MAP, SECTOR_COLOR_MAP,
    print_summary, plot_growth_curves, plot_rolling_sharpe,
    print_sector_weights,
    sortino_ratio, max_drawdown, calmar_ratio,
)
 
# ------------------------------------------------------------------
# TUNABLE CONSTANTS
# ------------------------------------------------------------------
 
# Momentum blend weights — must sum to 1.0
MOMENTUM_WEIGHT_6M  = 0.7
MOMENTUM_WEIGHT_12M = 0.3
 
MOMENTUM_WINDOW_6M  = 126   # trading days in 6 months
MOMENTUM_WINDOW_12M = 252   # trading days in 12 months
TREND_DMA_WINDOW    = 200   # days for 200-DMA trend filter
 
# IMPROVEMENT 5 — blended sector score factor weights (must sum to 1.0)
SCORE_WEIGHT_MOMENTUM = 0.5   # dual-horizon vol-adjusted momentum
SCORE_WEIGHT_LOW_VOL  = 0.3   # low-volatility anomaly (negated vol)
SCORE_WEIGHT_TREND    = 0.2   # fraction of stocks above 200-DMA
 
# Q magnitude scale (BASE — overridden per regime in Improvement 7)
Q_SCALE = 0.06
 
# Omega confidence parameters (IMPROVEMENT 3)
OMEGA_SPREAD_REF = 0.15
OMEGA_FLOOR      = 0.05
 
# Weak view suppression threshold (BASE — overridden per regime in Improvement 7)
VIEW_SIGNAL_THRESHOLD = 0.25
 
# ------------------------------------------------------------------
# IMPROVEMENT 7 — REGIME FILTER CONSTANTS
# ------------------------------------------------------------------
 
REGIME_DMA_BUFFER  = 0.02   # ±2% band around 200-DMA → NEUTRAL
REGIME_VOL_SPIKE   = 0.22   # above this → NOT full bull
REGIME_PANIC_VOL   = 0.35   # above this → PANIC
REGIME_VOL_WINDOW  = 30     # trading days for short-term NIFTY vol
REGIME_DMA_WINDOW  = 200    # 200-DMA for regime detection
 
REGIME_OMEGA_DAMP = {
    'BULL'   : 1.00,
    'NEUTRAL': 0.75,
    'BEAR'   : 0.50,
    'PANIC'  : 0.20,
}
 
REGIME_PARAMS = {
    'BULL'   : dict(q_scale=0.06, view_gate=0.25, max_wt=0.15, sec_cap=0.30, cyc_cap=0.40, min_wt=0.01),
    'NEUTRAL': dict(q_scale=0.04, view_gate=0.30, max_wt=0.13, sec_cap=0.28, cyc_cap=0.37, min_wt=0.012),
    'BEAR'   : dict(q_scale=0.02, view_gate=0.40, max_wt=0.11, sec_cap=0.25, cyc_cap=0.33, min_wt=0.015),
    'PANIC'  : dict(q_scale=0.01, view_gate=0.60, max_wt=0.09, sec_cap=0.22, cyc_cap=0.28, min_wt=0.018),
}
 
# ------------------------------------------------------------------
# IMPROVEMENT 8 — THRESHOLD-GATED REBALANCING CONSTANTS
# ------------------------------------------------------------------
 
REBALANCE_MAX_DRIFT     = 0.02   # 2% single stock drift
REBALANCE_TURNOVER_GATE = 0.04   # 4% total one-way turnover
 
# ------------------------------------------------------------------
# IMPROVEMENT 9 — EWMA COVARIANCE CONSTANTS
# ------------------------------------------------------------------
 
# RiskMetrics daily decay factor (half-life ≈ 11 trading days)
EWMA_LAMBDA = 0.94
 
# Blend weight: 1.0 = pure EWMA, 0.0 = pure Ledoit-Wolf
# 0.70 balances regime-responsiveness with LW stability
EWMA_BLEND_WEIGHT = 0.70
 
# Minimum observations required before EWMA estimate is trusted
EWMA_MIN_OBS = 63
 
# Small diagonal regularisation to ensure positive-definiteness
EWMA_REG_FLOOR = 1e-6
 
# Per-regime lambda offsets — faster decay in stressed markets
EWMA_LAMBDA_REGIME_OFFSET = {
    'BULL'   :  0.00,   # λ = 0.94  half-life ≈ 11 days
    'NEUTRAL': -0.02,   # λ = 0.92  half-life ≈  8 days
    'BEAR'   : -0.04,   # λ = 0.90  half-life ≈  6 days
    'PANIC'  : -0.07,   # λ = 0.87  half-life ≈  5 days
}
 
# ------------------------------------------------------------------
# IMPROVEMENT 10 — POSITION CONCENTRATION LOGIC CONSTANTS
# ------------------------------------------------------------------
 
# Maximum number of sectors eligible for overweighting vs equilibrium.
# Sectors outside this top-N are capped at their equilibrium weight.
TOP_N_SECTORS_BASE = 10   # used in BULL regime
 
# Per-regime top-N sector limit — fewer sectors in stressed regimes
TOP_N_SECTORS_REGIME = {
    'BULL'   : 10,   # broad participation in uptrend
    'NEUTRAL': 8,    # moderate concentration
    'BEAR'   : 6,    # defensive; concentrate on quality
    'PANIC'  : 4,    # maximum concentration on defensives
}
 
# Minimum blended sector z-score to be eligible for overweight.
# Sectors with score < this floor are suppressed to equilibrium weight.
# 0.0 means "above the cross-sectional average" — sensible default.
CONCENTRATION_SCORE_FLOOR = 0.0
 
# Minimum conviction weight per stock after BL optimisation.
# Stocks below this threshold are zeroed and weight redistributed.
# Base value (BULL); scaled per regime below.
CONVICTION_MIN_WEIGHT_REGIME = {
    'BULL'   : 0.015,   # 1.5% — allow broad participation
    'NEUTRAL': 0.018,   # 1.8% — tighten slightly
    'BEAR'   : 0.022,   # 2.2% — concentrate on conviction names
    'PANIC'  : 0.025,   # 2.5% — maximum conviction filtering
}
 
# Diagnostic warning threshold: if the fraction of active weight in the
# top-quartile of stocks (by weight) falls below this, print a warning.
CONCENTRATION_WARN_THRESHOLD = 0.60   # 60% of active weight in top-quartile
 
 
# ------------------------------------------------------------------
# IMPROVEMENT 9 — EWMA COVARIANCE ESTIMATOR
# ------------------------------------------------------------------
 
def compute_ewma_cov(returns_window: pd.DataFrame,
                     regime: str = 'BULL') -> np.ndarray:
    """
    Compute an EWMA (Exponentially Weighted Moving Average) covariance
    matrix using the RiskMetrics recursive formula.
 
    Algorithm
    ---------
    Given T observations and N assets, iterate forward:
 
        Σ_t = λ × Σ_{t-1} + (1 − λ) × r_{t-1} × r_{t-1}ᵀ
 
    Starting condition: Σ_0 = sample covariance of the first
    EWMA_MIN_OBS observations (warm-up period).
 
    Parameters
    ----------
    returns_window : pd.DataFrame — T × N daily returns (expanding window)
    regime         : str          — current market regime for λ adjustment
 
    Returns
    -------
    cov_ewma : np.ndarray (N × N) — regularised EWMA covariance matrix
    """
    R = returns_window.values.copy()
    T, N = R.shape
 
    if T < EWMA_MIN_OBS:
        return np.cov(R.T) + np.eye(N) * EWMA_REG_FLOOR
 
    # Regime-adjusted lambda
    lam = EWMA_LAMBDA + EWMA_LAMBDA_REGIME_OFFSET.get(regime, 0.0)
    lam = float(np.clip(lam, 0.80, 0.99))
 
    # Demean across full window (per-asset)
    mu   = R.mean(axis=0)
    R_dm = R - mu
 
    # Warm-up: initialise Σ using first EWMA_MIN_OBS observations
    cov_t = np.cov(R_dm[:EWMA_MIN_OBS].T)
    if cov_t.ndim == 0:
        cov_t = np.array([[cov_t]])
 
    # Recursive EWMA update from warm-up end to present
    for t in range(EWMA_MIN_OBS, T):
        r_t   = R_dm[t - 1].reshape(-1, 1)
        cov_t = lam * cov_t + (1.0 - lam) * (r_t @ r_t.T)
 
    cov_ewma = cov_t + np.eye(N) * EWMA_REG_FLOOR
    return cov_ewma
 
 
def compute_blended_cov(returns_window: pd.DataFrame,
                        regime: str = 'BULL') -> tuple[np.ndarray, dict]:
    """
    Blend EWMA and Ledoit-Wolf covariance matrices.
 
    Σ_final = α × Σ_EWMA + (1 − α) × Σ_LW
 
    where α = EWMA_BLEND_WEIGHT (default 0.70).
 
    Returns
    -------
    cov_blended : np.ndarray (N × N)
    diagnostics : dict
    """
    T = len(returns_window)
 
    cov_lw, _ = compute_lw_cov(returns_window)
 
    if T >= EWMA_MIN_OBS:
        cov_ewma = compute_ewma_cov(returns_window, regime=regime)
        alpha    = EWMA_BLEND_WEIGHT
    else:
        cov_ewma = cov_lw.copy()
        alpha    = 0.0
 
    cov_blended = alpha * cov_ewma + (1.0 - alpha) * cov_lw
 
    # Ensure PD via eigen-floor
    eigvals, eigvecs = np.linalg.eigh(cov_blended)
    eigvals          = np.maximum(eigvals, EWMA_REG_FLOOR)
    cov_blended      = eigvecs @ np.diag(eigvals) @ eigvecs.T
 
    lam_eff   = EWMA_LAMBDA + EWMA_LAMBDA_REGIME_OFFSET.get(regime, 0.0)
    lam_eff   = float(np.clip(lam_eff, 0.80, 0.99))
    half_life = np.log(2) / np.log(1.0 / lam_eff)
 
    ewma_vol  = float(np.sqrt(np.diag(cov_ewma).mean())    * np.sqrt(252) * 100)
    lw_vol    = float(np.sqrt(np.diag(cov_lw).mean())      * np.sqrt(252) * 100)
    blend_vol = float(np.sqrt(np.diag(cov_blended).mean()) * np.sqrt(252) * 100)
 
    diagnostics = {
        'alpha'     : alpha,
        'lambda_eff': lam_eff,
        'half_life' : half_life,
        'ewma_vol'  : ewma_vol,
        'lw_vol'    : lw_vol,
        'blend_vol' : blend_vol,
        'n_obs'     : T,
    }
 
    return cov_blended, diagnostics
 
 
# ------------------------------------------------------------------
# IMPROVEMENT 10 — CONCENTRATION FILTER
# ------------------------------------------------------------------
 
def apply_concentration_filter(
    raw_weights: np.ndarray,
    tickers: list,
    w_market: np.ndarray,
    sector_score: pd.Series,
    regime: str = 'BULL',
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    IMPROVEMENT 10 — Position Concentration Logic.
 
    Applies two sequential filters to the raw BL-optimised weights:
 
    Filter A — Top-N Sector Gate:
        Sectors ranked below TOP_N_SECTORS_REGIME[regime] in the blended
        factor score, OR whose score is below CONCENTRATION_SCORE_FLOOR,
        are capped at their equilibrium (market-cap) weight.
        This prevents diluting alpha into low-conviction sectors.
 
    Filter B — Minimum Conviction Weight:
        Stocks with final weight < CONVICTION_MIN_WEIGHT_REGIME[regime]
        are zeroed out. Their weight is redistributed proportionally to
        the remaining survivors, then re-normalised to sum to 1.
 
    After filtering, diagnostics are computed:
        - n_stocks_pruned : number of stocks zeroed by Filter B
        - n_sectors_capped: number of sectors capped by Filter A
        - alpha_concentration_ratio: fraction of active weight in
          top-quartile of stocks (by weight).  Target ≥ 60%.
        - active_weight_before / after: total |w_i − w_mkt_i| sum
 
    Parameters
    ----------
    raw_weights  : np.ndarray  — raw BL-optimised weights (sums to 1)
    tickers      : list        — asset identifiers matching raw_weights
    w_market     : np.ndarray  — market-cap equilibrium weights
    sector_score : pd.Series   — blended sector z-scores (from compute_sector_momentum)
    regime       : str         — current market regime
    verbose      : bool        — if True, print pruning details
 
    Returns
    -------
    filtered_weights : np.ndarray (sums to 1, all ≥ 0)
    conc_diagnostics : dict — keys: n_pruned, n_capped, alpha_ratio,
                              active_before, active_after, stocks_pruned,
                              sectors_capped
    """
    n = len(raw_weights)
    w = raw_weights.copy()
 
    top_n   = TOP_N_SECTORS_REGIME.get(regime, TOP_N_SECTORS_BASE)
    min_cvx = CONVICTION_MIN_WEIGHT_REGIME.get(regime, 0.015)
 
    # ── Filter A: Top-N Sector Concentration ──────────────────────────────
    # Rank sectors by blended score (descending)
    ranked_sectors  = sector_score.sort_values(ascending=False).index.tolist()
    eligible_sectors = set(
        s for i, s in enumerate(ranked_sectors)
        if i < top_n and sector_score.get(s, 0.0) >= CONCENTRATION_SCORE_FLOOR
    )
 
    sectors_capped = []
    for i, ticker in enumerate(tickers):
        sec = SECTOR_MAP.get(ticker, 'Other')
        if sec not in eligible_sectors:
            # Cap this stock at its equilibrium (market-cap) weight
            # but never below the existing optimised weight if it's
            # already lower (avoid increasing weight in bad sectors)
            cap_wt = min(w[i], w_market[i])
            if w[i] > w_market[i]:
                w[i] = w_market[i]
                if sec not in sectors_capped:
                    sectors_capped.append(sec)
 
    # Re-normalise after sector capping
    total = w.sum()
    if total > 1e-8:
        w = w / total
 
    n_capped = len(sectors_capped)
 
    # ── Filter B: Minimum Conviction Weight ───────────────────────────────
    stocks_pruned = []
    max_iter      = 10   # safety limit for redistribution loop
 
    for _ in range(max_iter):
        below_threshold = [i for i in range(n) if 0 < w[i] < min_cvx]
        if not below_threshold:
            break
 
        # Zero out below-threshold stocks
        freed_weight = 0.0
        for i in below_threshold:
            freed_weight += w[i]
            stocks_pruned.append(tickers[i])
            w[i] = 0.0
 
        # Redistribute freed weight proportionally to survivors
        survivors  = np.array([i for i in range(n) if w[i] >= min_cvx])
        if len(survivors) == 0:
            # Edge case: all zeroed — fall back to equal weight among all
            w = np.ones(n) / n
            break
 
        survivor_total = w[survivors].sum()
        if survivor_total > 1e-8:
            for i in survivors:
                w[i] += freed_weight * (w[i] / survivor_total)
 
    # Final normalisation
    total = w.sum()
    if total > 1e-8:
        w = w / total
 
    n_pruned = len(set(stocks_pruned))
 
    # ── Diagnostics ────────────────────────────────────────────────────────
    active_before = float(np.sum(np.abs(raw_weights - w_market)))
    active_after  = float(np.sum(np.abs(w - w_market)))
 
    # Alpha concentration ratio: fraction of active weight in top quartile
    n_top_q       = max(1, n // 4)
    sorted_indices = np.argsort(w)[::-1]
    top_q_active  = float(np.sum(np.abs(w[sorted_indices[:n_top_q]] -
                                        w_market[sorted_indices[:n_top_q]])))
    alpha_ratio   = top_q_active / active_after if active_after > 1e-6 else 0.0
 
    conc_diagnostics = {
        'n_pruned'     : n_pruned,
        'n_capped'     : n_capped,
        'alpha_ratio'  : alpha_ratio,
        'active_before': active_before,
        'active_after' : active_after,
        'stocks_pruned': list(set(stocks_pruned)),
        'sectors_capped': sectors_capped,
        'top_n_used'   : top_n,
        'min_cvx_used' : min_cvx,
        'regime'       : regime,
    }
 
    if verbose:
        print(f"\n   [IMPROVEMENT 10] Concentration filter — regime={regime}")
        print(f"     Top-N eligible sectors  : {top_n}")
        print(f"     Score floor             : {CONCENTRATION_SCORE_FLOOR}")
        print(f"     Min conviction weight   : {min_cvx*100:.1f}%")
        print(f"     Sectors capped to equil : {n_capped}  {sectors_capped}")
        print(f"     Stocks pruned (< min_cvx): {n_pruned}  "
              f"{[t.replace('.NS','') for t in list(set(stocks_pruned))]}")
        print(f"     Active weight before    : {active_before*100:.2f}%")
        print(f"     Active weight after     : {active_after*100:.2f}%")
        print(f"     Alpha concentration ratio (top-quartile): {alpha_ratio*100:.1f}%"
              f"  {'✓' if alpha_ratio >= CONCENTRATION_WARN_THRESHOLD else '⚠ DILUTED'}")
 
    return w, conc_diagnostics
 
 
# ------------------------------------------------------------------
# IMPROVEMENT 7 — REGIME DETECTION
# ------------------------------------------------------------------
 
def detect_regime(nifty_rets: pd.Series, as_of_date: pd.Timestamp) -> str:
    """
    Classify the current market regime at `as_of_date` using NIFTY 50
    price-reconstructed series from returns.
 
    Returns one of: 'BULL', 'NEUTRAL', 'BEAR', 'PANIC'
    """
    if nifty_rets is None or len(nifty_rets) < REGIME_DMA_WINDOW:
        return 'BULL'
 
    window = nifty_rets.loc[nifty_rets.index <= as_of_date]
    if len(window) < REGIME_DMA_WINDOW:
        return 'BULL'
 
    nifty_price = (1 + window).cumprod() * 100
    dma_200     = nifty_price.iloc[-REGIME_DMA_WINDOW:].mean()
    current_px  = nifty_price.iloc[-1]
    deviation   = (current_px - dma_200) / dma_200
 
    vol_window = window.iloc[-REGIME_VOL_WINDOW:] if len(window) >= REGIME_VOL_WINDOW else window
    vol_30d    = vol_window.std() * np.sqrt(252)
 
    within_buffer = abs(deviation) <= REGIME_DMA_BUFFER
 
    if within_buffer:
        return 'NEUTRAL'
 
    if deviation < -REGIME_DMA_BUFFER:
        if vol_30d > REGIME_PANIC_VOL:
            return 'PANIC'
        return 'BEAR'
 
    if vol_30d > REGIME_VOL_SPIKE:
        return 'NEUTRAL'
 
    return 'BULL'
 
 
def get_regime_params(regime: str) -> dict:
    return REGIME_PARAMS.get(regime, REGIME_PARAMS['BULL'])
 
 
# ------------------------------------------------------------------
# IMPROVEMENT 8 — THRESHOLD GATE
# ------------------------------------------------------------------
 
def should_rebalance(candidate_weights: np.ndarray,
                     current_weights: np.ndarray) -> tuple[bool, float, float]:
    """
    Decide whether to execute a rebalance based on proposed weight change.
    NOTE: candidate_weights here are the CONCENTRATION-FILTERED weights
    (Improvement 10), so the drift/turnover figures are more meaningful.
    """
    delta     = candidate_weights - current_weights
    max_drift = float(np.max(np.abs(delta)))
    turnover  = float(np.sum(np.abs(delta)))
 
    execute = (max_drift >= REBALANCE_MAX_DRIFT) or (turnover >= REBALANCE_TURNOVER_GATE)
    return execute, max_drift, turnover
 
 
# ------------------------------------------------------------------
# NIFTY 50 BENCHMARK LOADER
# ------------------------------------------------------------------
 
def load_nifty_benchmark(start_date, end_date):
    """
    Download NIFTY 50 index (^NSEI) from yfinance and compute
    daily returns and cumulative growth over the given date range.
    """
    try:
        import yfinance as yf
 
        buffer_start = pd.Timestamp(start_date) - pd.Timedelta(days=10)
        print("   Downloading NIFTY 50 (^NSEI) from yfinance...", end=' ', flush=True)
 
        nifty_raw = yf.download(
            "^NSEI",
            start=str(buffer_start.date()),
            end=str(pd.Timestamp(end_date).date()),
            progress=False,
            auto_adjust=True,
        )
 
        if nifty_raw.empty:
            print("FAILED — empty data returned.")
            return None, None, False
 
        if isinstance(nifty_raw.columns, pd.MultiIndex):
            nifty_raw.columns = nifty_raw.columns.get_level_values(0)
 
        nifty_close = nifty_raw['Close'].squeeze()
        nifty_close.index = pd.to_datetime(nifty_close.index)
        nifty_close = nifty_close.sort_index().dropna()
 
        benchmark_returns = nifty_close.pct_change().dropna()
        benchmark_returns = benchmark_returns.loc[
            pd.Timestamp(start_date):pd.Timestamp(end_date)
        ]
 
        if len(benchmark_returns) < 10:
            print("FAILED — insufficient data after date filtering.")
            return None, None, False
 
        benchmark_growth = (1 + benchmark_returns).cumprod()
        benchmark_growth = benchmark_growth / benchmark_growth.iloc[0]
 
        print(f"OK  ({len(benchmark_returns)} trading days)")
        return benchmark_returns, benchmark_growth, True
 
    except Exception as e:
        print(f"FAILED — {e}")
        return None, None, False
 
 
def compute_benchmark_metrics(benchmark_returns, rf=RISK_FREE_RATE_ANNUAL, periods=252):
    """Compute standard performance metrics for the NIFTY 50 benchmark."""
    if benchmark_returns is None or len(benchmark_returns) < 10:
        return None
 
    ret       = benchmark_returns
    n_yr      = len(ret) / periods
    total_ret = (1 + ret).prod() - 1
    ann_ret   = (1 + total_ret) ** (1 / n_yr) - 1
    ann_vol   = ret.std() * np.sqrt(periods)
    sharpe    = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
 
    cum    = (1 + ret).cumprod()
    dd     = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
 
    down_ret = ret[ret < rf / periods]
    down_std = down_ret.std() * np.sqrt(periods) if len(down_ret) > 0 else 1e-10
    sortino  = (ann_ret - rf) / down_std
 
    return {
        'Total Return (%)' : round(total_ret * 100, 2),
        'Ann Return (%)'   : round(ann_ret    * 100, 2),
        'Ann Vol (%)'      : round(ann_vol    * 100, 2),
        'Sharpe Ratio'     : round(sharpe,  3),
        'Sortino Ratio'    : round(sortino, 3),
        'Max Drawdown (%)' : round(max_dd   * 100, 2),
        'Calmar Ratio'     : round(calmar,  3),
    }
 
 
# ------------------------------------------------------------------
# IMPROVEMENT 6 — RELATIVE MOMENTUM HELPER
# ------------------------------------------------------------------
 
def _align_benchmark_to_prices(prices_window: pd.DataFrame,
                                nifty_rets: pd.Series) -> pd.Series:
    if nifty_rets is None:
        return None
 
    price_dates  = prices_window.index
    nifty_window = nifty_rets.reindex(price_dates, method='ffill', limit=3).fillna(0.0)
 
    if (nifty_window != 0.0).sum() < MOMENTUM_WINDOW_6M:
        return None
 
    return nifty_window
 
 
# ------------------------------------------------------------------
# FACTOR HELPERS  (IMPROVEMENT 5 + 6)
# ------------------------------------------------------------------
 
def _zscore_series(s: pd.Series) -> pd.Series:
    std = s.std()
    if std < 1e-8:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std
 
 
def _compute_momentum_factor(prices_window: pd.DataFrame,
                              nifty_rets_aligned: pd.Series = None):
    n_rows = len(prices_window)
    if n_rows < MOMENTUM_WINDOW_6M:
        return None
 
    stock_daily_rets = prices_window.pct_change().dropna()
 
    if nifty_rets_aligned is not None:
        nifty_for_excess = nifty_rets_aligned.reindex(
            stock_daily_rets.index, method='ffill', limit=3
        ).fillna(0.0)
        excess_daily_rets = stock_daily_rets.sub(nifty_for_excess, axis=0)
    else:
        excess_daily_rets = stock_daily_rets
 
    excess_price = (1 + excess_daily_rets).cumprod()
 
    excess_6m = excess_price.iloc[-MOMENTUM_WINDOW_6M:]
    mom_6m    = (excess_6m.iloc[-1] / excess_6m.iloc[0]) - 1
 
    if n_rows >= MOMENTUM_WINDOW_12M:
        excess_12m   = excess_price.iloc[-MOMENTUM_WINDOW_12M:]
        mom_12m      = (excess_12m.iloc[-1] / excess_12m.iloc[0]) - 1
        raw_momentum = MOMENTUM_WEIGHT_6M * mom_6m + MOMENTUM_WEIGHT_12M * mom_12m
    else:
        raw_momentum = mom_6m
 
    vol_window = excess_daily_rets.iloc[-MOMENTUM_WINDOW_6M:]
    volatility = vol_window.std() * np.sqrt(252)
    volatility = volatility.replace(0, np.nan)
 
    signal = raw_momentum / volatility
    return signal.replace([np.inf, -np.inf], np.nan).dropna()
 
 
def _compute_low_vol_factor(prices_window: pd.DataFrame,
                             nifty_rets_aligned: pd.Series = None):
    n_rows = len(prices_window)
    if n_rows < MOMENTUM_WINDOW_6M:
        return None
 
    prices_6m     = prices_window.iloc[-MOMENTUM_WINDOW_6M:]
    stock_rets_6m = prices_6m.pct_change().dropna()
 
    if nifty_rets_aligned is not None:
        nifty_for_excess = nifty_rets_aligned.reindex(
            stock_rets_6m.index, method='ffill', limit=3
        ).fillna(0.0)
        excess_rets = stock_rets_6m.sub(nifty_for_excess, axis=0)
    else:
        excess_rets = stock_rets_6m
 
    ann_vol = excess_rets.std() * np.sqrt(252)
    ann_vol = ann_vol.replace(0, np.nan).dropna()
    return -ann_vol
 
 
def _compute_trend_factor(prices_window: pd.DataFrame) -> pd.Series:
    scores = {}
    for ticker in prices_window.columns:
        px = prices_window[ticker].dropna()
        if len(px) < TREND_DMA_WINDOW:
            scores[ticker] = 0.5
        else:
            dma_200        = px.iloc[-TREND_DMA_WINDOW:].mean()
            scores[ticker] = 1.0 if px.iloc[-1] > dma_200 else 0.0
    return pd.Series(scores)
 
 
def _aggregate_to_sectors(stock_scores: pd.Series, tickers: list) -> pd.Series:
    df = pd.DataFrame({'ticker': stock_scores.index,
                       'score' : stock_scores.values})
    df['sector'] = df['ticker'].map(lambda t: SECTOR_MAP.get(t, 'Other'))
    return (df.groupby('sector')['score']
              .mean()
              .sort_values(ascending=False))
 
 
# ------------------------------------------------------------------
# BLENDED SECTOR SCORE — IMPROVEMENT 5 + 6
# ------------------------------------------------------------------
 
def compute_sector_momentum(prices_window: pd.DataFrame,
                             tickers: list,
                             nifty_rets: pd.Series = None) -> pd.Series:
    """
    Blended three-factor sector score:
        score = 0.5 × momentum_z  +  0.3 × low_vol_z  +  0.2 × trend_z
    """
    n_rows = len(prices_window)
    if n_rows < MOMENTUM_WINDOW_6M:
        return None
 
    nifty_aligned = _align_benchmark_to_prices(prices_window, nifty_rets)
 
    mom_stock = _compute_momentum_factor(prices_window, nifty_aligned)
    if mom_stock is None or len(mom_stock) < 2:
        return None
 
    sector_mom_raw = _aggregate_to_sectors(mom_stock, tickers)
 
    lowvol_stock = _compute_low_vol_factor(prices_window, nifty_aligned)
    if lowvol_stock is not None and len(lowvol_stock) >= 2:
        sector_lowvol_raw = _aggregate_to_sectors(lowvol_stock, tickers)
    else:
        sector_lowvol_raw = pd.Series(0.0, index=sector_mom_raw.index)
 
    trend_stock      = _compute_trend_factor(prices_window)
    sector_trend_raw = _aggregate_to_sectors(trend_stock, tickers)
 
    all_sectors = (sector_mom_raw.index
                   .union(sector_lowvol_raw.index)
                   .union(sector_trend_raw.index))
 
    sector_mom_raw    = sector_mom_raw.reindex(all_sectors,    fill_value=0.0)
    sector_lowvol_raw = sector_lowvol_raw.reindex(all_sectors, fill_value=0.0)
    sector_trend_raw  = sector_trend_raw.reindex(all_sectors,  fill_value=0.5)
 
    z_mom    = _zscore_series(sector_mom_raw)
    z_lowvol = _zscore_series(sector_lowvol_raw)
    z_trend  = _zscore_series(sector_trend_raw)
 
    blended = (SCORE_WEIGHT_MOMENTUM * z_mom
             + SCORE_WEIGHT_LOW_VOL  * z_lowvol
             + SCORE_WEIGHT_TREND    * z_trend)
 
    return blended.sort_values(ascending=False)
 
 
# ------------------------------------------------------------------
# VIEW CONSTRUCTION HELPERS
# ------------------------------------------------------------------
 
def sector_weight_vector(sector_name, tickers, direction=1.0):
    n   = len(tickers)
    vec = np.zeros(n)
    idx = [i for i, t in enumerate(tickers)
           if SECTOR_MAP.get(t, 'Other') == sector_name]
    if idx:
        for i in idx:
            vec[i] = direction / len(idx)
    return vec
 
 
def build_omega(P, cov_local, raw_spreads, omega_damp: float = 1.0):
    """
    Build view uncertainty matrix Ω using dynamic confidence scaling.
    IMPROVEMENT 7: omega_damp multiplier applied for regime control.
    IMPROVEMENT 9: cov_local is the blended EWMA+LW matrix.
    """
    base_omega_diag = np.diag(P @ (TAU * cov_local) @ P.T)
 
    confidence = np.array([
        float(np.clip(abs(s) / OMEGA_SPREAD_REF, 0.0, 1.0))
        for s in raw_spreads
    ])
 
    scale        = 1.0 - confidence * (1.0 - OMEGA_FLOOR)
    regime_scale = 1.0 / max(omega_damp, 0.05)
    omega_diag   = np.maximum(base_omega_diag * scale * regime_scale, 1e-12)
 
    return np.diag(omega_diag)
 
 
def build_constraints_regime(tickers, regime_params: dict):
    """
    Build portfolio constraints using regime-specific sector and cyclical caps.
    """
    from utils import SECTOR_MAP
 
    n         = len(tickers)
    sec_cap   = regime_params['sec_cap']
    cyc_cap   = regime_params['cyc_cap']
    cyclicals = {'Auto', 'Metals', 'Energy', 'Chemicals'}
 
    constraints = [
        {'type': 'eq', 'fun': lambda w: w.sum() - 1.0},
    ]
 
    all_sectors = sorted(set(SECTOR_MAP.values()))
    for sector in all_sectors:
        idx = [i for i, t in enumerate(tickers)
               if SECTOR_MAP.get(t, 'Other') == sector]
        if idx:
            constraints.append({
                'type': 'ineq',
                'fun' : (lambda w, ix=idx, cap=sec_cap:
                         cap - sum(w[i] for i in ix))
            })
 
    cyc_idx = [i for i, t in enumerate(tickers)
               if SECTOR_MAP.get(t, 'Other') in cyclicals]
    if cyc_idx:
        constraints.append({
            'type': 'ineq',
            'fun' : (lambda w, ix=cyc_idx, cap=cyc_cap:
                     cap - sum(w[i] for i in ix))
        })
 
    return constraints
 
 
# ------------------------------------------------------------------
# CORE BL SOLVE — SINGLE REBALANCE POINT (IMPROVEMENTS 7 + 8 + 9 + 10)
# ------------------------------------------------------------------
 
def solve_bl(returns_window, prices_window, tickers, w_market,
             nifty_rets=None, regime='BULL'):
    """
    Solve the Black-Litterman portfolio at one rebalance point.
 
    IMPROVEMENT 9 : Uses blended EWMA+LW covariance.
    IMPROVEMENT 10: Returns raw (pre-concentration) weights. The calling
                    loop applies apply_concentration_filter() after this.
 
    Returns the candidate RAW optimal weights (np.ndarray) and the
    sector_score Series (needed by the concentration filter), or
    (None, None) on failure.
    """
    n_local = len(tickers)
 
    rp           = get_regime_params(regime)
    r_q_scale    = rp['q_scale']
    r_view_gate  = rp['view_gate']
    r_max_wt     = rp['max_wt']
    r_min_wt     = rp['min_wt']
    r_omega_damp = REGIME_OMEGA_DAMP[regime]
 
    # ── IMPROVEMENT 9: Blended EWMA+LW covariance ─────────────────────────
    cov_local, diag = compute_blended_cov(returns_window, regime=regime)
 
    pi = LAMBDA * cov_local @ w_market
 
    sector_score = compute_sector_momentum(prices_window, tickers,
                                           nifty_rets=nifty_rets)
    if sector_score is None or len(sector_score) < 4:
        return None, None
 
    ranked_sectors = sector_score.index.tolist()
    top_sectors    = ranked_sectors[:2]
    bot_sectors    = ranked_sectors[-2:]
 
    candidate_views = [
        (top_sectors[0], bot_sectors[-1]),
        (top_sectors[1], bot_sectors[-2]),
    ]
 
    passing_p      = []
    passing_q      = []
    passing_spread = []
 
    for top_s, bot_s in candidate_views:
        raw_spread = sector_score[top_s] - sector_score[bot_s]
        if abs(raw_spread) < r_view_gate:
            continue
        p_vec = (sector_weight_vector(top_s, tickers,  1.0) +
                 sector_weight_vector(bot_s,  tickers, -1.0))
        q_val = raw_spread * r_q_scale / MOMENTUM_WINDOW_6M
        passing_p.append(p_vec)
        passing_q.append(q_val)
        passing_spread.append(raw_spread)
 
    constraints_local = build_constraints_regime(tickers, rp)
    bounds_local      = tuple((r_min_wt, r_max_wt) for _ in range(n_local))
    w0                = np.ones(n_local) / n_local
 
    if not passing_p:
        # No views: optimise on equilibrium pi only
        def eq_objective(w):
            excess = w @ pi - RISK_FREE_RATE_DAILY
            return -excess + 0.5 * LAMBDA * (w @ cov_local @ w)
 
        result = minimize(
            fun=eq_objective, x0=w0, method='SLSQP',
            bounds=bounds_local, constraints=constraints_local,
            options={'ftol': 1e-12, 'maxiter': 2000}
        )
        if result.success:
            w = np.clip(result.x, 0, 1)
            return w / w.sum(), sector_score
        return None, None
 
    P     = np.array(passing_p)
    Q     = np.array(passing_q)
    omega = build_omega(P, cov_local, raw_spreads=passing_spread,
                        omega_damp=r_omega_damp)
 
    try:
        inv_tau_cov = np.linalg.inv(TAU * cov_local)
        inv_omega   = np.linalg.inv(omega)
        M           = np.linalg.inv(inv_tau_cov + P.T @ inv_omega @ P)
        bl_mu       = M @ (inv_tau_cov @ pi + P.T @ inv_omega @ Q)
    except np.linalg.LinAlgError:
        bl_mu = pi.copy()
 
    def bl_objective(w):
        excess = w @ bl_mu - RISK_FREE_RATE_DAILY
        return -excess + 0.5 * LAMBDA * (w @ cov_local @ w)
 
    result = minimize(
        fun=bl_objective, x0=w0, method='SLSQP',
        bounds=bounds_local, constraints=constraints_local,
        options={'ftol': 1e-12, 'maxiter': 2000}
    )
    if result.success:
        w = np.clip(result.x, 0, 1)
        return w / w.sum(), sector_score
    return None, None
 
 
# ------------------------------------------------------------------
# ROLLING BL PORTFOLIO ENGINE (IMPROVEMENTS 7 + 8 + 9 + 10)
# ------------------------------------------------------------------
 
def run_rolling_bl(returns, price_matrix, tickers, w_market,
                   nifty_rets=None):
    """
    Build the full rolling BL return series.
 
    IMPROVEMENT 7 : detect_regime() classifies market state per rebalance.
    IMPROVEMENT 8 : should_rebalance() gates execution on drift/turnover.
    IMPROVEMENT 9 : compute_blended_cov() uses EWMA+LW for each solve.
    IMPROVEMENT 10: apply_concentration_filter() removes low-conviction
                    positions after each BL solve, BEFORE the threshold
                    gate check (so gate sees meaningful concentrated delta).
    """
    dates           = returns.index.tolist()
    bl_weights      = None
    bl_returns_list = []
    rebalance_log   = []
    regime_log      = []
    activity_log    = []
    ewma_diag_log   = []
    conc_diag_log   = []   # NEW — Improvement 10 concentration diagnostics
 
    rebalance_dates = set()
    for i in range(MIN_HISTORY, len(dates), REBALANCE_FREQ):
        rebalance_dates.add(dates[i])
 
    print(f"   Scheduled rebalance points : {len(rebalance_dates)}")
    print(f"   Threshold gate             : max_drift≥{REBALANCE_MAX_DRIFT*100:.0f}%"
          f" OR turnover≥{REBALANCE_TURNOVER_GATE*100:.0f}% (Improvement 8)")
    print(f"   EWMA covariance            : λ={EWMA_LAMBDA} (base)  "
          f"blend={EWMA_BLEND_WEIGHT*100:.0f}% EWMA + "
          f"{(1-EWMA_BLEND_WEIGHT)*100:.0f}% LW  (Improvement 9)")
    print(f"   Concentration filter       : Top-N sectors + min conviction "
          f"(Improvement 10)")
    print(f"     BULL  top-N={TOP_N_SECTORS_REGIME['BULL']}  "
          f"min_cvx={CONVICTION_MIN_WEIGHT_REGIME['BULL']*100:.1f}%")
    print(f"     BEAR  top-N={TOP_N_SECTORS_REGIME['BEAR']}  "
          f"min_cvx={CONVICTION_MIN_WEIGHT_REGIME['BEAR']*100:.1f}%")
    print(f"     PANIC top-N={TOP_N_SECTORS_REGIME['PANIC']}  "
          f"min_cvx={CONVICTION_MIN_WEIGHT_REGIME['PANIC']*100:.1f}%")
    print(f"   Relative momentum (vs NIFTY): "
          f"{'ENABLED' if nifty_rets is not None else 'DISABLED (fallback to absolute)'}")
    print(f"   Regime filter              : "
          f"{'ENABLED' if nifty_rets is not None else 'DISABLED (no NIFTY data)'}")
 
    rebalances_solved   = 0
    rebalances_executed = 0
    rebalances_skipped  = 0
    rebalances_gated    = 0
    views_suppressed    = 0
    regime_counts       = {'BULL': 0, 'NEUTRAL': 0, 'BEAR': 0, 'PANIC': 0}
    tc_paid_total       = 0.0
    tc_saved_total      = 0.0
 
    # Concentration filter running stats
    total_pruned  = 0
    total_capped  = 0
    alpha_ratios  = []
 
    for i, date in enumerate(dates):
 
        if date in rebalance_dates and i >= MIN_HISTORY:
            returns_window = returns.iloc[:i]
            prices_window  = price_matrix.loc[price_matrix.index <= date]
 
            nifty_window = None
            if nifty_rets is not None:
                nifty_window = nifty_rets.loc[nifty_rets.index <= date]
 
            # ── IMPROVEMENT 7: Detect regime ──────────────────────────────
            regime = detect_regime(nifty_window, date)
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            regime_log.append((date, regime))
 
            r_view_gate = get_regime_params(regime)['view_gate']
            score_log   = compute_sector_momentum(prices_window, tickers,
                                                  nifty_rets=nifty_window)
 
            if score_log is not None and len(score_log) >= 4:
                ranked_log = score_log.index.tolist()
                top2_log   = ranked_log[:2]
                bot2_log   = ranked_log[-2:]
                pairs_log  = [(top2_log[0], bot2_log[-1]),
                              (top2_log[1], bot2_log[-2])]
                suppressed_this = sum(
                    1 for t, b in pairs_log
                    if abs(score_log[t] - score_log[b]) < r_view_gate
                )
                if suppressed_this > 0:
                    rebalances_gated += 1
                    views_suppressed += suppressed_this
 
            # ── IMPROVEMENT 9: Log EWMA diagnostics ──────────────────────
            _, diag = compute_blended_cov(returns_window, regime=regime)
            diag['date']   = date
            diag['regime'] = regime
            ewma_diag_log.append(diag)
 
            # ── Solve for raw BL candidate weights ────────────────────────
            candidate_raw, sector_score_solved = solve_bl(
                returns_window, prices_window, tickers, w_market,
                nifty_rets=nifty_window,
                regime=regime,
            )
 
            if candidate_raw is not None:
                rebalances_solved += 1
 
                # ── IMPROVEMENT 10: Apply concentration filter ────────────
                # Pass sector_score_solved so the filter knows which sectors
                # earned their overweight (score ≥ floor AND in top-N).
                if sector_score_solved is not None:
                    candidate_weights, conc_diag = apply_concentration_filter(
                        raw_weights   = candidate_raw,
                        tickers       = tickers,
                        w_market      = w_market,
                        sector_score  = sector_score_solved,
                        regime        = regime,
                        verbose       = False,   # set True for per-rebalance detail
                    )
                else:
                    # No sector score available — skip concentration filter
                    candidate_weights = candidate_raw
                    conc_diag = {
                        'n_pruned': 0, 'n_capped': 0, 'alpha_ratio': 0.0,
                        'active_before': 0.0, 'active_after': 0.0,
                        'stocks_pruned': [], 'sectors_capped': [],
                        'top_n_used': TOP_N_SECTORS_REGIME.get(regime, 10),
                        'min_cvx_used': CONVICTION_MIN_WEIGHT_REGIME.get(regime, 0.015),
                        'regime': regime,
                    }
 
                conc_diag['date']   = date
                conc_diag['regime'] = regime
                conc_diag_log.append(conc_diag)
 
                total_pruned += conc_diag['n_pruned']
                total_capped += conc_diag['n_capped']
                if conc_diag['alpha_ratio'] > 0:
                    alpha_ratios.append(conc_diag['alpha_ratio'])
 
                # Warn if alpha diluted at this rebalance
                if conc_diag['alpha_ratio'] < CONCENTRATION_WARN_THRESHOLD:
                    pass   # Collected; summary printed at end
 
                # ── IMPROVEMENT 8: Threshold gate ─────────────────────────
                # Gate sees CONCENTRATED weights → more honest delta
                if bl_weights is None:
                    execute      = True
                    max_drift    = 1.0
                    turnover     = 1.0
                    gate_reason  = 'FIRST'
                else:
                    execute, max_drift, turnover = should_rebalance(
                        candidate_weights, bl_weights
                    )
                    gate_reason = (
                        'EXECUTED (drift)'     if execute and max_drift >= REBALANCE_MAX_DRIFT
                        else 'EXECUTED (turn)'  if execute
                        else 'SKIPPED'
                    )
 
                hypothetical_tc = TRANSACTION_COST * turnover
 
                if execute:
                    actual_tc       = hypothetical_tc
                    tc_paid_total  += actual_tc
                    bl_weights      = candidate_weights
                    rebalances_executed += 1
 
                    if score_log is not None and len(score_log) >= 4:
                        ranked = score_log.index.tolist()
                        rebalance_log.append((date, ranked[:2], ranked[-2:], regime))
 
                    day_ret = returns.iloc[i].values @ bl_weights - actual_tc
                else:
                    actual_tc       = 0.0
                    tc_saved_total += hypothetical_tc
                    rebalances_skipped += 1
 
                activity_log.append({
                    'date'        : date,
                    'regime'      : regime,
                    'executed'    : execute,
                    'max_drift'   : max_drift,
                    'turnover'    : turnover,
                    'tc_paid'     : actual_tc,
                    'tc_saved'    : hypothetical_tc if not execute else 0.0,
                    'gate_reason' : gate_reason,
                    # Improvement 9 fields
                    'ewma_vol'    : diag['ewma_vol'],
                    'lw_vol'      : diag['lw_vol'],
                    'blend_vol'   : diag['blend_vol'],
                    'lambda_eff'  : diag['lambda_eff'],
                    'half_life'   : diag['half_life'],
                    # Improvement 10 fields
                    'n_pruned'    : conc_diag['n_pruned'],
                    'n_capped'    : conc_diag['n_capped'],
                    'alpha_ratio' : conc_diag['alpha_ratio'],
                    'active_after': conc_diag['active_after'],
                })
 
                bl_returns_list.append((date, day_ret if execute else
                                        returns.iloc[i].values @ bl_weights))
                continue
 
        if bl_weights is not None:
            day_ret = returns.iloc[i].values @ bl_weights
            bl_returns_list.append((date, day_ret))
 
    # ── Summary printout ──────────────────────────────────────────────────
    print(f"\n   Rebalances scheduled         : {len(rebalance_dates)}")
    print(f"   Rebalances solved (BL OK)    : {rebalances_solved}")
    print(f"   Rebalances EXECUTED          : {rebalances_executed} "
          f"({rebalances_executed/max(rebalances_solved,1)*100:.0f}%)")
    print(f"   Rebalances SKIPPED (gate)    : {rebalances_skipped} "
          f"({rebalances_skipped/max(rebalances_solved,1)*100:.0f}%)")
    print(f"   Rebalances w/ ≥1 view gated  : {rebalances_gated}")
    print(f"   Total individual views gated : {views_suppressed}")
 
    print(f"\n   IMPROVEMENT 8 — TC analysis:")
    print(f"     TC paid (executed)   : {tc_paid_total*10000:.1f} bps")
    print(f"     TC saved (skipped)   : {tc_saved_total*10000:.1f} bps")
    print(f"     TC saving rate       : "
          f"{tc_saved_total/(tc_paid_total+tc_saved_total+1e-12)*100:.0f}% of potential TC avoided")
 
    if activity_log:
        ex_turns   = [a['turnover'] for a in activity_log if a['executed']]
        skip_turns = [a['turnover'] for a in activity_log if not a['executed']]
        if ex_turns:
            print(f"     Avg turnover | executed : {np.mean(ex_turns)*100:.1f}%")
        if skip_turns:
            print(f"     Avg turnover | skipped  : {np.mean(skip_turns)*100:.1f}%")
 
    # IMPROVEMENT 10 — Concentration diagnostics summary
    if conc_diag_log:
        avg_alpha = np.mean(alpha_ratios) if alpha_ratios else 0.0
        n_diluted = sum(1 for r in alpha_ratios if r < CONCENTRATION_WARN_THRESHOLD)
        print(f"\n   IMPROVEMENT 10 — Concentration filter diagnostics:")
        print(f"     Total stocks pruned (all rebalances)  : {total_pruned}")
        print(f"     Total sectors capped (all rebalances) : {total_capped}")
        print(f"     Avg alpha concentration ratio         : {avg_alpha*100:.1f}%  "
              f"(target ≥ {CONCENTRATION_WARN_THRESHOLD*100:.0f}%)")
        print(f"     Rebalances below concentration target : {n_diluted} "
              f"/ {len(alpha_ratios)}")
 
        by_regime_conc = {}
        for d in conc_diag_log:
            r = d['regime']
            if r not in by_regime_conc:
                by_regime_conc[r] = {'pruned': 0, 'capped': 0, 'ratios': []}
            by_regime_conc[r]['pruned'] += d['n_pruned']
            by_regime_conc[r]['capped'] += d['n_capped']
            if d['alpha_ratio'] > 0:
                by_regime_conc[r]['ratios'].append(d['alpha_ratio'])
 
        print(f"\n     Per-regime concentration summary:")
        for r in ['BULL', 'NEUTRAL', 'BEAR', 'PANIC']:
            if r in by_regime_conc:
                rd       = by_regime_conc[r]
                avg_ar   = np.mean(rd['ratios']) * 100 if rd['ratios'] else 0.0
                top_n    = TOP_N_SECTORS_REGIME[r]
                min_cvx  = CONVICTION_MIN_WEIGHT_REGIME[r]
                print(f"       {r:<7}: top_N={top_n}  min_cvx={min_cvx*100:.1f}%  "
                      f"pruned={rd['pruned']}  capped={rd['capped']}  "
                      f"avg_α_ratio={avg_ar:.1f}%")
 
    # IMPROVEMENT 9 — EWMA diagnostics summary
    if ewma_diag_log:
        diag_df    = pd.DataFrame(ewma_diag_log)
        avg_hl     = diag_df['half_life'].mean()
        avg_ewma_v = diag_df['ewma_vol'].mean()
        avg_lw_v   = diag_df['lw_vol'].mean()
        avg_bl_v   = diag_df['blend_vol'].mean()
        print(f"\n   IMPROVEMENT 9 — EWMA Covariance diagnostics:")
        print(f"     Base λ                  : {EWMA_LAMBDA}")
        print(f"     EWMA blend weight       : {EWMA_BLEND_WEIGHT*100:.0f}% EWMA / "
              f"{(1-EWMA_BLEND_WEIGHT)*100:.0f}% LW")
        print(f"     Avg effective λ         : {diag_df['lambda_eff'].mean():.3f}")
        print(f"     Avg EWMA half-life      : {avg_hl:.1f} trading days")
        print(f"     Avg EWMA vol (ann.)     : {avg_ewma_v:.1f}%")
        print(f"     Avg LW vol (ann.)       : {avg_lw_v:.1f}%")
        print(f"     Avg Blended vol (ann.)  : {avg_bl_v:.1f}%")
        print(f"     EWMA vs LW vol divergence (avg): {abs(avg_ewma_v - avg_lw_v):.1f}%")
 
        by_regime = diag_df.groupby('regime')[['lambda_eff', 'half_life',
                                                'ewma_vol', 'lw_vol']].mean()
        print(f"\n     Per-regime EWMA stats:")
        for r in ['BULL', 'NEUTRAL', 'BEAR', 'PANIC']:
            if r in by_regime.index:
                row = by_regime.loc[r]
                print(f"       {r:<7}: λ={row['lambda_eff']:.3f}  "
                      f"HL={row['half_life']:.1f}d  "
                      f"EWMA_vol={row['ewma_vol']:.1f}%  "
                      f"LW_vol={row['lw_vol']:.1f}%")
 
    print(f"\n   IMPROVEMENT 7 — Regime breakdown across {rebalances_solved} rebalances:")
    for r in ['BULL', 'NEUTRAL', 'BEAR', 'PANIC']:
        cnt    = regime_counts.get(r, 0)
        pct    = cnt / max(rebalances_solved, 1) * 100
        params = get_regime_params(r)
        lam_r  = EWMA_LAMBDA + EWMA_LAMBDA_REGIME_OFFSET.get(r, 0.0)
        hl_r   = np.log(2) / np.log(1.0 / lam_r)
        print(f"     {r:<7}: {cnt:>3} ({pct:>4.0f}%)  "
              f"Q={params['q_scale']:.2f}  gate={params['view_gate']:.2f}  "
              f"maxWt={params['max_wt']*100:.0f}%  "
              f"Ω_damp={REGIME_OMEGA_DAMP[r]:.2f}  "
              f"λ={lam_r:.3f}  HL={hl_r:.1f}d  "
              f"top_N={TOP_N_SECTORS_REGIME[r]}  "
              f"min_cvx={CONVICTION_MIN_WEIGHT_REGIME[r]*100:.1f}%")
 
    if not bl_returns_list:
        raise RuntimeError("No BL returns generated. Check MIN_HISTORY vs data length.")
 
    bl_dates  = [x[0] for x in bl_returns_list]
    bl_vals   = [x[1] for x in bl_returns_list]
    bl_ret    = pd.Series(bl_vals, index=bl_dates, name='Black-Litterman')
    bl_growth = (1 + bl_ret).cumprod()
 
    final_weights = (bl_weights if bl_weights is not None
                     else np.ones(len(tickers)) / len(tickers))
 
    return (bl_ret, bl_growth, final_weights,
            rebalance_log, regime_log, activity_log,
            ewma_diag_log, conc_diag_log)
 
 
# ------------------------------------------------------------------
# IMPROVEMENT 10 — CONCENTRATION DIAGNOSTICS PLOT (NEW)
# ------------------------------------------------------------------
 
def plot_concentration_diagnostics(conc_diag_log, bl_growth):
    """
    IMPROVEMENT 10 — Four-panel concentration diagnostics chart:
 
      Panel 1: BL growth for reference
      Panel 2: Alpha concentration ratio per rebalance (target ≥ 60%)
      Panel 3: Stocks pruned + sectors capped per rebalance
      Panel 4: Active weight before vs after concentration filtering
    """
    if not conc_diag_log:
        print("   No concentration diagnostics available — skipping plot.")
        return
 
    REGIME_COLORS = {
        'BULL'   : '#28a745',
        'NEUTRAL': '#ffc107',
        'BEAR'   : '#fd7e14',
        'PANIC'  : '#dc3545',
    }
 
    df         = pd.DataFrame(conc_diag_log)
    df['date'] = pd.to_datetime(df['date'])
    df         = df.set_index('date').sort_index()
 
    fig, axes = plt.subplots(4, 1, figsize=(14, 14),
                             gridspec_kw={'height_ratios': [2.5, 1.5, 1.5, 2]},
                             sharex=True)
 
    # Panel 1: BL growth
    ax1 = axes[0]
    ax1.plot(bl_growth.index, bl_growth.values,
             color='steelblue', linewidth=2.0, label='BL Portfolio (concentrated)')
    ax1.set_ylabel("Portfolio Value (₹1 invested)")
    ax1.set_title(
        "IMPROVEMENT 10 — Position Concentration Diagnostics\n"
        f"(Top-N sectors per regime | Min conviction weight | "
        f"Score floor={CONCENTRATION_SCORE_FLOOR} | "
        f"Target α-ratio ≥ {CONCENTRATION_WARN_THRESHOLD*100:.0f}%)",
        fontsize=11
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)
 
    # Panel 2: Alpha concentration ratio
    ax2 = axes[1]
    bar_colors = [REGIME_COLORS.get(r, 'grey') for r in df['regime']]
    ax2.bar(df.index, df['alpha_ratio'] * 100,
            width=15, color=bar_colors, alpha=0.80)
    ax2.axhline(CONCENTRATION_WARN_THRESHOLD * 100,
                color='red', linewidth=1.2, linestyle='--', alpha=0.8,
                label=f'Target ({CONCENTRATION_WARN_THRESHOLD*100:.0f}%)')
    ax2.set_ylabel("α-Concentration Ratio (%)")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8)
    ax2.grid(True, axis='y', alpha=0.25)
 
    # Add regime patches legend on panel 2
    patches = [mpatches.Patch(color=REGIME_COLORS[r], alpha=0.8, label=r)
               for r in ['BULL', 'NEUTRAL', 'BEAR', 'PANIC']
               if r in df['regime'].values]
    ax2.legend(handles=patches + [mpatches.Patch(color='red', alpha=0.6,
               label=f'Target ({CONCENTRATION_WARN_THRESHOLD*100:.0f}%)')],
               fontsize=8, loc='lower right', ncol=5)
 
    # Panel 3: Stocks pruned and sectors capped
    ax3 = axes[2]
    x_pos = df.index
    ax3.bar(x_pos, df['n_pruned'], width=15,
            color='#e74c3c', alpha=0.75, label='Stocks pruned (< min_cvx)')
    ax3.bar(x_pos, df['n_capped'], width=15, bottom=df['n_pruned'],
            color='#f39c12', alpha=0.75, label='Sectors capped (> top-N)')
    ax3.set_ylabel("Count per Rebalance")
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, axis='y', alpha=0.25)
 
    # Panel 4: Active weight comparison
    ax4 = axes[3]
    ax4.plot(df.index, df['active_before'] * 100,
             color='#e74c3c', linewidth=1.6, linestyle='--',
             label='Active weight before filter (%)')
    ax4.plot(df.index, df['active_after'] * 100,
             color='#27ae60', linewidth=2.0, linestyle='-',
             label='Active weight after filter (%)')
    ax4.fill_between(df.index,
                     df['active_before'] * 100,
                     df['active_after']  * 100,
                     alpha=0.15, color='#e74c3c',
                     label='Weight redistributed to high-conviction')
    ax4.set_ylabel("Total Active Weight (|w − w_mkt|, %)")
    ax4.set_xlabel("Date")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.25)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
 
    plt.tight_layout()
    plt.savefig("reports/bl_concentration_diagnostics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/bl_concentration_diagnostics.png")
 
 
# ------------------------------------------------------------------
# IMPROVEMENT 9 — EWMA DIAGNOSTICS PLOT
# ------------------------------------------------------------------
 
def plot_ewma_diagnostics(ewma_diag_log, bl_growth, activity_log=None):
    """
    IMPROVEMENT 9 — Four-panel EWMA covariance diagnostics chart.
    """
    if not ewma_diag_log:
        print("   No EWMA diagnostics available — skipping plot.")
        return
 
    REGIME_COLORS_LINE = {
        'BULL'   : '#28a745',
        'NEUTRAL': '#ffc107',
        'BEAR'   : '#fd7e14',
        'PANIC'  : '#dc3545',
    }
 
    diag_df         = pd.DataFrame(ewma_diag_log)
    diag_df['date'] = pd.to_datetime(diag_df['date'])
    diag_df         = diag_df.set_index('date').sort_index()
 
    fig, axes = plt.subplots(4, 1, figsize=(14, 14),
                             gridspec_kw={'height_ratios': [2.5, 1.5, 1.5, 2]},
                             sharex=True)
 
    ax1 = axes[0]
    ax1.plot(bl_growth.index, bl_growth.values,
             color='steelblue', linewidth=2.0, label='BL Portfolio')
    ax1.set_ylabel("Portfolio Value (₹1 invested)")
    ax1.set_title(
        "IMPROVEMENT 9 — EWMA Covariance Diagnostics\n"
        f"(λ={EWMA_LAMBDA} base | blend={EWMA_BLEND_WEIGHT*100:.0f}% EWMA + "
        f"{(1-EWMA_BLEND_WEIGHT)*100:.0f}% LW | regime-adaptive λ)",
        fontsize=12
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)
 
    ax2 = axes[1]
    for regime in ['BULL', 'NEUTRAL', 'BEAR', 'PANIC']:
        mask = diag_df['regime'] == regime
        if mask.any():
            ax2.scatter(diag_df.index[mask],
                        diag_df.loc[mask, 'lambda_eff'],
                        color=REGIME_COLORS_LINE[regime],
                        s=30, label=regime, zorder=5, alpha=0.85)
    ax2.axhline(EWMA_LAMBDA, color='grey', linewidth=1.0, linestyle='--',
                alpha=0.6, label=f'Base λ={EWMA_LAMBDA}')
    ax2.set_ylabel("Effective λ")
    ax2.set_ylim(0.84, 0.97)
    ax2.legend(fontsize=8, loc='lower right', ncol=5)
    ax2.grid(True, alpha=0.25)
 
    ax3 = axes[2]
    ax3.bar(diag_df.index,
            diag_df['half_life'],
            width=15,
            color=[REGIME_COLORS_LINE.get(r, 'grey') for r in diag_df['regime']],
            alpha=0.75)
    ax3.axhline(np.log(2) / np.log(1.0 / EWMA_LAMBDA),
                color='grey', linewidth=1.0, linestyle='--', alpha=0.6,
                label=f'Base HL={np.log(2)/np.log(1/EWMA_LAMBDA):.1f}d')
    ax3.set_ylabel("Half-life (days)")
    ax3.legend(fontsize=8)
    ax3.grid(True, axis='y', alpha=0.25)
 
    ax4 = axes[3]
    ax4.plot(diag_df.index, diag_df['ewma_vol'],
             color='darkorange', linewidth=1.6, label='EWMA vol (ann. %)', linestyle='-')
    ax4.plot(diag_df.index, diag_df['lw_vol'],
             color='steelblue',  linewidth=1.6, label='LW vol (ann. %)',   linestyle='--')
    ax4.plot(diag_df.index, diag_df['blend_vol'],
             color='green',      linewidth=1.8, label='Blended vol (ann. %)', linestyle='-.')
    ax4.fill_between(diag_df.index,
                     diag_df['ewma_vol'],
                     diag_df['lw_vol'],
                     alpha=0.10, color='darkorange', label='EWMA−LW divergence')
    ax4.set_ylabel("Avg Ann. Volatility (%)")
    ax4.set_xlabel("Date")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.25)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
 
    plt.tight_layout()
    plt.savefig("reports/bl_ewma_diagnostics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/bl_ewma_diagnostics.png")
 
 
# ------------------------------------------------------------------
# EXISTING PLOTS (unchanged from Improvements 1–9)
# ------------------------------------------------------------------
 
def plot_bl_growth(bl_growth, growth_ew, benchmark_growth=None):
    """BL cumulative growth vs Equal Weight vs NIFTY 50."""
    fig, ax = plt.subplots(figsize=(13, 6))
 
    ax.plot(bl_growth.index, bl_growth.values,
            label='Black-Litterman (rolling, net TC)',
            color='steelblue', linewidth=2.0, linestyle='--')
    ax.plot(growth_ew.index, growth_ew.values,
            label='Equal Weight',
            color='grey', linewidth=1.4, alpha=0.8)
    if benchmark_growth is not None:
        bm = benchmark_growth.reindex(bl_growth.index, method='ffill').dropna()
        bm = bm / bm.iloc[0]
        ax.plot(bm.index, bm.values,
                label='NIFTY 50 (benchmark)',
                color='darkorange', linewidth=1.6, linestyle='-.')
 
    ax.set_title(
        "Black-Litterman Portfolio — Cumulative Growth\n"
        "(Rolling | EWMA+LW blended cov | Market-cap equil. | rf=7% | 10bps TC\n"
        " | Score: 0.5×rel_mom + 0.3×low-TE + 0.2×200DMA"
        f" | Regime filter | EWMA λ={EWMA_LAMBDA} blend={EWMA_BLEND_WEIGHT*100:.0f}%"
        f" | Gate: {REBALANCE_MAX_DRIFT*100:.0f}% drift / "
        f"{REBALANCE_TURNOVER_GATE*100:.0f}% turnover"
        f" | Concentration: top-N + min conviction)",
        fontsize=9
    )
    ax.set_ylabel("Portfolio Value (₹1 invested)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig("reports/bl_growth.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/bl_growth.png")
 
 
def plot_bl_rolling_sharpe(bl_ret, returns_ew, benchmark_returns=None, window=63):
    """Rolling Sharpe: BL vs Equal Weight vs NIFTY 50."""
    fig, ax = plt.subplots(figsize=(13, 5))
 
    series_to_plot = [
        ('Black-Litterman', bl_ret,     'steelblue',  '--'),
        ('Equal Weight',    returns_ew, 'grey',        '-'),
    ]
    if benchmark_returns is not None:
        series_to_plot.append(('NIFTY 50', benchmark_returns, 'darkorange', '-.'))
 
    for name, ret_s, color, ls in series_to_plot:
        if ret_s is None or len(ret_s) < window:
            continue
        ra       = ret_s.reindex(bl_ret.index).fillna(0)
        roll_ret = ra.rolling(window).mean() * 252
        roll_vol = ra.rolling(window).std()  * np.sqrt(252)
        roll_sh  = (roll_ret - RISK_FREE_RATE_ANNUAL) / roll_vol.replace(0, np.nan)
        ax.plot(roll_sh.index, roll_sh.values,
                label=name, color=color, linewidth=1.6, linestyle=ls)
 
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.set_title(
        f"Rolling {window}-Day Sharpe Ratio — BL vs Equal Weight vs NIFTY 50\n"
        f"(rf=7% | EWMA+LW cov (λ={EWMA_LAMBDA}, α={EWMA_BLEND_WEIGHT}) | "
        f"Regime filter | Gate: "
        f"{REBALANCE_MAX_DRIFT*100:.0f}% drift / {REBALANCE_TURNOVER_GATE*100:.0f}% turnover"
        f" | Concentration filter ON)",
        fontsize=10
    )
    ax.set_ylabel("Sharpe Ratio (annualised)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig("reports/bl_rolling_sharpe.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/bl_rolling_sharpe.png")
 
 
def plot_bl_drawdown(bl_ret, returns_ew, benchmark_returns=None):
    """Drawdown comparison: BL vs Equal Weight vs NIFTY 50."""
    fig, ax = plt.subplots(figsize=(13, 5))
 
    series_to_plot = [
        ('Black-Litterman', bl_ret,     'steelblue',  '--'),
        ('Equal Weight',    returns_ew, 'grey',        '-'),
    ]
    if benchmark_returns is not None:
        series_to_plot.append(('NIFTY 50', benchmark_returns, 'darkorange', '-.'))
 
    for name, ret_s, color, ls in series_to_plot:
        if ret_s is None or len(ret_s) < 10:
            continue
        ra  = ret_s.reindex(bl_ret.index).fillna(0)
        cum = (1 + ra).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax()
        ax.plot(dd.index, dd.values * 100,
                label=name, color=color, linewidth=1.5, linestyle=ls)
        ax.fill_between(dd.index, dd.values * 100, 0, alpha=0.12, color=color)
 
    ax.axhline(-15, color='red', linewidth=0.8, linestyle=':', alpha=0.7,
               label='-15% threshold')
    ax.set_title("Drawdown Comparison — BL vs Equal Weight vs NIFTY 50\n"
                 "(Concentration filter removes dilutive names → cleaner drawdown profile)",
                 fontsize=12)
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig("reports/bl_drawdown.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/bl_drawdown.png")
 
 
def plot_regime_history(regime_log, bl_growth, benchmark_growth=None):
    """
    IMPROVEMENT 7 — Regime history as coloured background bands.
    """
    if not regime_log:
        print("   No regime log available — skipping regime history plot.")
        return
 
    REGIME_COLORS = {
        'BULL'   : '#d4edda',
        'NEUTRAL': '#fff3cd',
        'BEAR'   : '#fde8d0',
        'PANIC'  : '#f8d7da',
    }
    REGIME_LINE_COLORS = {
        'BULL'   : '#28a745',
        'NEUTRAL': '#ffc107',
        'BEAR'   : '#fd7e14',
        'PANIC'  : '#dc3545',
    }
 
    regime_df = pd.DataFrame(regime_log, columns=['date', 'regime'])
    regime_df = regime_df.set_index('date').sort_index()
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=True)
 
    ax1.plot(bl_growth.index, bl_growth.values,
             label='Black-Litterman', color='steelblue', linewidth=2.0, zorder=5)
    if benchmark_growth is not None:
        bm = benchmark_growth.reindex(bl_growth.index, method='ffill').dropna()
        bm = bm / bm.iloc[0]
        ax1.plot(bm.index, bm.values,
                 label='NIFTY 50', color='darkorange', linewidth=1.5,
                 linestyle='-.', alpha=0.8, zorder=4)
 
    all_dates   = bl_growth.index
    prev_regime = None
    band_start  = None
 
    for idx, (rdate, regime) in enumerate(regime_log):
        if regime != prev_regime:
            if prev_regime is not None and band_start is not None:
                ax1.axvspan(band_start, rdate,
                            alpha=0.35, color=REGIME_COLORS[prev_regime], zorder=1)
            band_start  = rdate
            prev_regime = regime
 
    if prev_regime is not None and band_start is not None:
        ax1.axvspan(band_start, all_dates[-1],
                    alpha=0.35, color=REGIME_COLORS[prev_regime], zorder=1)
 
    patches = [mpatches.Patch(color=REGIME_COLORS[r], alpha=0.6, label=r)
               for r in ['BULL', 'NEUTRAL', 'BEAR', 'PANIC']]
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles + patches, labels + ['BULL', 'NEUTRAL', 'BEAR', 'PANIC'],
               fontsize=9, loc='upper left', ncol=2)
 
    ax1.set_ylabel("Portfolio Value (₹1 invested)")
    ax1.set_title(
        "Black-Litterman with Regime Filter — Growth & Regime History\n"
        f"(EWMA λ={EWMA_LAMBDA} blend={EWMA_BLEND_WEIGHT*100:.0f}%  |  "
        f"Gate: {REBALANCE_MAX_DRIFT*100:.0f}% drift / {REBALANCE_TURNOVER_GATE*100:.0f}% turnover  |  "
        "BULL=full | NEUTRAL=cautious | BEAR=defensive | PANIC=near equal-wt)",
        fontsize=10
    )
    ax1.grid(True, alpha=0.25, zorder=0)
 
    regime_order = {'BULL': 3, 'NEUTRAL': 2, 'BEAR': 1, 'PANIC': 0}
    regime_num   = (pd.DataFrame(regime_log, columns=['date', 'regime'])
                    .set_index('date')['regime']
                    .map(regime_order)
                    .reindex(all_dates, method='ffill')
                    .fillna(3))
 
    for r_val, r_name in zip([3, 2, 1, 0], ['BULL', 'NEUTRAL', 'BEAR', 'PANIC']):
        mask = (regime_num == r_val)
        if mask.any():
            ax2.fill_between(all_dates, 0, mask.astype(int) * (r_val + 0.8),
                             where=mask, alpha=0.7,
                             color=REGIME_LINE_COLORS[r_name],
                             label=r_name, step='post')
 
    ax2.set_yticks([0.4, 1.4, 2.4, 3.4])
    ax2.set_yticklabels(['PANIC', 'BEAR', 'NEUTRAL', 'BULL'], fontsize=8)
    ax2.set_ylabel("Regime", fontsize=9)
    ax2.set_xlabel("Date")
    ax2.grid(True, axis='x', alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
 
    plt.tight_layout()
    plt.savefig("reports/bl_regime_history.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/bl_regime_history.png")
 
 
def plot_rebalance_activity(activity_log, bl_growth):
    """
    IMPROVEMENT 8+10 — Rebalance activity chart.
    Now also shows concentration stats alongside turnover.
    """
    if not activity_log:
        print("   No activity log available — skipping rebalance activity plot.")
        return
 
    df = pd.DataFrame(activity_log)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
 
    executed = df[df['executed']]
    skipped  = df[~df['executed']]
 
    df_sorted                        = df.sort_index()
    df_sorted['cum_tc_paid']         = df_sorted['tc_paid'].cumsum()
    df_sorted['cum_tc_hypothetical'] = (df_sorted['tc_paid'] + df_sorted['tc_saved']).cumsum()
 
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10),
                                         gridspec_kw={'height_ratios': [3, 2, 2]},
                                         sharex=True)
 
    ax1.plot(bl_growth.index, bl_growth.values,
             color='steelblue', linewidth=2.0, label='BL Growth', zorder=3)
    if not executed.empty:
        for d in executed.index:
            if d in bl_growth.index:
                ax1.axvline(d, color='green', linewidth=0.5, alpha=0.4, zorder=2)
        ax1.scatter(
            [d for d in executed.index if d in bl_growth.index],
            [bl_growth.loc[d] for d in executed.index if d in bl_growth.index],
            color='green', s=18, zorder=5, label='Executed', marker='^'
        )
    if not skipped.empty:
        ax1.scatter(
            [d for d in skipped.index if d in bl_growth.index],
            [bl_growth.loc[d] for d in skipped.index if d in bl_growth.index],
            color='red', s=12, zorder=5, label='Skipped (gate)', marker='x', alpha=0.7
        )
    ax1.set_ylabel("Portfolio Value (₹1 invested)")
    ax1.set_title(
        f"IMPROVEMENTS 8+9+10 — Gated Rebalancing with EWMA Cov + Concentration Filter\n"
        f"(Gate: max_drift≥{REBALANCE_MAX_DRIFT*100:.0f}% OR turnover≥{REBALANCE_TURNOVER_GATE*100:.0f}%  |  "
        f"EWMA λ={EWMA_LAMBDA} blend={EWMA_BLEND_WEIGHT*100:.0f}%  |  "
        f"Executed={len(executed)} / Skipped={len(skipped)} of {len(df)} scheduled)",
        fontsize=10
    )
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.25)
 
    bar_dates  = df_sorted.index
    bar_vals   = df_sorted['turnover'].values * 100
    bar_colors = ['#2ca02c' if e else '#d62728' for e in df_sorted['executed']]
    ax2.bar(bar_dates, bar_vals, color=bar_colors, alpha=0.75, width=15)
    ax2.axhline(REBALANCE_TURNOVER_GATE * 100, color='purple', linewidth=1.2,
                linestyle='--', alpha=0.8,
                label=f'Turnover gate ({REBALANCE_TURNOVER_GATE*100:.0f}%)')
    ax2.axhline(REBALANCE_MAX_DRIFT * 100, color='orange', linewidth=1.0,
                linestyle=':', alpha=0.8,
                label=f'Drift gate ({REBALANCE_MAX_DRIFT*100:.0f}%)')
 
    # Overlay alpha concentration ratio on secondary y-axis if available
    if 'alpha_ratio' in df_sorted.columns:
        ax2b = ax2.twinx()
        ax2b.plot(df_sorted.index, df_sorted['alpha_ratio'] * 100,
                  color='navy', linewidth=1.2, linestyle='-.',
                  alpha=0.7, label='α-conc ratio (%)')
        ax2b.axhline(CONCENTRATION_WARN_THRESHOLD * 100, color='navy',
                     linewidth=0.8, linestyle=':', alpha=0.5)
        ax2b.set_ylabel("α-Conc Ratio (%)", fontsize=8, color='navy')
        ax2b.tick_params(axis='y', labelcolor='navy')
        ax2b.set_ylim(0, 110)
 
    ax2.set_ylabel("One-way Turnover (%)")
    ex_patch = mpatches.Patch(color='#2ca02c', alpha=0.75, label='Executed')
    sk_patch = mpatches.Patch(color='#d62728', alpha=0.75, label='Skipped')
    ax2.legend(handles=[ex_patch, sk_patch], fontsize=8, loc='upper left')
    ax2.grid(True, axis='y', alpha=0.25)
 
    ax3.plot(df_sorted.index, df_sorted['cum_tc_hypothetical'] * 10000,
             color='red', linewidth=1.5, linestyle='--', label='TC without gate (bps)')
    ax3.plot(df_sorted.index, df_sorted['cum_tc_paid'] * 10000,
             color='green', linewidth=1.8, label='TC with gate (bps)')
    ax3.fill_between(df_sorted.index,
                     df_sorted['cum_tc_paid'] * 10000,
                     df_sorted['cum_tc_hypothetical'] * 10000,
                     alpha=0.15, color='green', label='TC saved')
    ax3.set_ylabel("Cumulative TC (bps)")
    ax3.set_xlabel("Date")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.25)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
 
    plt.tight_layout()
    plt.savefig("reports/bl_rebalance_activity.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/bl_rebalance_activity.png")
 
 
def plot_final_momentum_views(price_matrix, tickers, nifty_rets=None):
    """Bar chart of blended sector score at the final rebalance."""
    lookback     = max(MOMENTUM_WINDOW_12M + 63, TREND_DMA_WINDOW + 63)
    final_prices = price_matrix.iloc[-lookback:]
 
    nifty_final = None
    if nifty_rets is not None:
        nifty_final = nifty_rets.loc[nifty_rets.index <= price_matrix.index[-1]]
 
    sector_score = compute_sector_momentum(final_prices, tickers,
                                           nifty_rets=nifty_final)
    if sector_score is None:
        print("   Insufficient data for sector score plot — skipping.")
        return
 
    final_regime    = detect_regime(nifty_final, price_matrix.index[-1])
    final_rp        = get_regime_params(final_regime)
    final_view_gate = final_rp['view_gate']
    lam_final       = EWMA_LAMBDA + EWMA_LAMBDA_REGIME_OFFSET.get(final_regime, 0.0)
    hl_final        = np.log(2) / np.log(1.0 / lam_final)
    top_n_final     = TOP_N_SECTORS_REGIME.get(final_regime, TOP_N_SECTORS_BASE)
    min_cvx_final   = CONVICTION_MIN_WEIGHT_REGIME.get(final_regime, 0.015)
 
    nifty_aligned_final = _align_benchmark_to_prices(final_prices, nifty_final)
 
    ranked = sector_score.index.tolist()
    top2   = ranked[:2]
    bot2   = ranked[-2:]
 
    # Identify eligible sectors under concentration filter
    eligible_sectors = set(
        s for i, s in enumerate(ranked)
        if i < top_n_final and sector_score.get(s, 0.0) >= CONCENTRATION_SCORE_FLOOR
    )
 
    rel_label = ("RELATIVE (excess vs NIFTY)" if nifty_aligned_final is not None
                 else "ABSOLUTE (NIFTY data unavailable)")
    print(f"\n   IMPROVEMENT 6 — Relative Momentum mode: {rel_label}")
    print(f"   IMPROVEMENT 7 — Final regime detected  : {final_regime}")
    print(f"                   Active view gate       : {final_view_gate}")
    print(f"                   Active Q scale         : {final_rp['q_scale']}")
    print(f"                   Active max weight      : {final_rp['max_wt']*100:.0f}%")
    print(f"                   Active Ω damp          : {REGIME_OMEGA_DAMP[final_regime]}")
    print(f"   IMPROVEMENT 8 — Threshold gate: drift≥{REBALANCE_MAX_DRIFT*100:.0f}%"
          f" OR turnover≥{REBALANCE_TURNOVER_GATE*100:.0f}%")
    print(f"   IMPROVEMENT 9 — EWMA: λ={lam_final:.3f} (regime-adj)  "
          f"half-life={hl_final:.1f}d  "
          f"blend={EWMA_BLEND_WEIGHT*100:.0f}% EWMA + "
          f"{(1-EWMA_BLEND_WEIGHT)*100:.0f}% LW")
    print(f"   IMPROVEMENT 10 — Concentration: top_N={top_n_final}  "
          f"min_cvx={min_cvx_final*100:.1f}%  "
          f"score_floor={CONCENTRATION_SCORE_FLOOR}")
    print(f"   Eligible sectors for overweight: {sorted(eligible_sectors)}")
    print(f"   Ineligible (capped to equil):   "
          f"{sorted(set(ranked) - eligible_sectors)}")
 
    print("\n   Blended factor breakdown (final rebalance):")
    mom_s    = _compute_momentum_factor(final_prices, nifty_aligned_final)
    lowvol_s = _compute_low_vol_factor(final_prices,  nifty_aligned_final)
    trend_s  = _compute_trend_factor(final_prices)
 
    if mom_s is not None:
        sec_mom_r  = _aggregate_to_sectors(mom_s, tickers)
        sec_lv_r   = _aggregate_to_sectors(
            lowvol_s if lowvol_s is not None
            else pd.Series(0.0, index=mom_s.index), tickers)
        sec_tr_r   = _aggregate_to_sectors(trend_s, tickers)
 
        all_sec = (sec_mom_r.index
                   .union(sec_lv_r.index)
                   .union(sec_tr_r.index))
        z_m  = _zscore_series(sec_mom_r.reindex(all_sec, fill_value=0.0))
        z_lv = _zscore_series(sec_lv_r.reindex(all_sec,  fill_value=0.0))
        z_tr = _zscore_series(sec_tr_r.reindex(all_sec,  fill_value=0.5))
 
        print(f"     {'Sector':<14} {'MomZ':>7} {'LowVolZ':>9} "
              f"{'TrendZ':>8} {'BlendedScore':>14} {'Eligible':>10}")
        print("     " + "-" * 65)
        for sec in sector_score.index:
            elig = "✓ TOP-N" if sec in eligible_sectors else "✗ capped"
            print(f"     {sec:<14} {z_m.get(sec,0.0):>7.3f} "
                  f"{z_lv.get(sec,0.0):>9.3f} "
                  f"{z_tr.get(sec,0.0):>8.3f} "
                  f"{sector_score[sec]:>14.3f}  {elig:>10}")
 
    print(f"\n   View gate + dynamic Ω — final rebalance "
          f"[regime={final_regime}, gate={final_view_gate}]:")
    for i, (top, bot) in enumerate(zip(top2, list(reversed(bot2)))):
        spread    = sector_score[top] - sector_score[bot]
        gate_pass = abs(spread) >= final_view_gate
        gate_lbl  = ("PASS ✓" if gate_pass
                     else f"SUPPRESSED ✗ (|{spread:.3f}| < {final_view_gate})")
        if gate_pass:
            conf   = float(np.clip(abs(spread) / OMEGA_SPREAD_REF, 0.0, 1.0))
            o_damp = REGIME_OMEGA_DAMP[final_regime]
            scale  = (1.0 - conf * (1.0 - OMEGA_FLOOR)) / max(o_damp, 0.05)
            conv   = ('HIGH conviction' if conf > 0.7 else
                      'MODERATE'        if conf > 0.3 else
                      'LOW / prior dominates')
            print(f"     View {i+1}: {top:12s} vs {bot:12s}  |  "
                  f"spread={spread:+.3f}  Gate: {gate_lbl}  |  "
                  f"conf={conf:.2f}  Ω_eff_scale={scale:.3f}  [{conv}]")
        else:
            print(f"     View {i+1}: {top:12s} vs {bot:12s}  |  "
                  f"spread={spread:+.3f}  Gate: {gate_lbl}")
 
    fig, ax = plt.subplots(figsize=(12, 5))
    bar_colors = []
    for s in sector_score.index:
        if s in eligible_sectors:
            bar_colors.append('#2ca02c' if sector_score[s] > 0 else '#98df8a')
        else:
            bar_colors.append('#d62728' if sector_score[s] < 0 else '#ffbb78')
 
    ax.bar(range(len(sector_score)), sector_score.values,
           color=bar_colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(sector_score)))
    ax.set_xticklabels(sector_score.index, rotation=35, ha='right', fontsize=9)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline( final_view_gate, color='purple', linewidth=0.9,
                linestyle=':', alpha=0.7,
                label=f'+view threshold ({final_view_gate}) [{final_regime}]')
    ax.axhline(-final_view_gate, color='purple', linewidth=0.9,
                linestyle=':', alpha=0.7,
                label=f'−view threshold ({final_view_gate}) [{final_regime}]')
    ax.axhline(CONCENTRATION_SCORE_FLOOR, color='navy', linewidth=1.0,
                linestyle='--', alpha=0.6,
                label=f'Concentration score floor ({CONCENTRATION_SCORE_FLOOR})')
 
    mode_label = "rel. excess vs NIFTY" if nifty_aligned_final is not None \
                 else "absolute returns"
    ax.set_title(
        f"Blended Sector Score — Final Rebalance  [{mode_label}]\n"
        f"(0.5×momentum_z + 0.3×low_vol_z + 0.2×trend_z"
        f"  |  Regime={final_regime}  |  View gate={final_view_gate}"
        f"  |  EWMA λ={lam_final:.3f}  HL={hl_final:.1f}d"
        f"  |  Top-N={top_n_final}  min_cvx={min_cvx_final*100:.1f}%)\n"
        f"Dark=eligible for overweight, Light=capped to equilibrium (Imp.10)",
        fontsize=9
    )
    ax.set_ylabel("Blended Sector Score (z-score units)")
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
 
    for i, (sec, val) in enumerate(sector_score.items()):
        offset = 0.04 if val >= 0 else -0.08
        elig   = sec in eligible_sectors
        if sec in top2:
            sp  = sector_score[sec] - sector_score[bot2[-1]]
            lbl = '▲ View+' if abs(sp) >= final_view_gate else '▲ (gated)'
            col = 'green'   if abs(sp) >= final_view_gate else 'olive'
            ax.text(i, val + offset, lbl, ha='center',
                    fontsize=8, color=col, fontweight='bold')
        elif sec in bot2:
            sp  = sector_score[top2[0]] - sector_score[sec]
            lbl = '▼ View−' if abs(sp) >= final_view_gate else '▼ (gated)'
            col = 'red'     if abs(sp) >= final_view_gate else 'salmon'
            ax.text(i, val + offset, lbl, ha='center',
                    fontsize=8, color=col, fontweight='bold')
        elif not elig:
            ax.text(i, val + offset, '⊘', ha='center',
                    fontsize=8, color='grey', alpha=0.7)
 
    plt.tight_layout()
    plt.savefig("reports/bl_momentum_views.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/bl_momentum_views.png")
 
 
def plot_bl_sector_tilts(final_weights, w_market, tickers):
    """Side-by-side: top-10 stock weights BL vs market + sector tilts."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
 
    ax = axes[0]
    bl_s   = pd.Series(final_weights, index=tickers).sort_values(ascending=False)
    mkt_s  = pd.Series(w_market, index=tickers)
    top10  = bl_s.head(10)
    mkt10  = mkt_s.loc[top10.index]
    x2, w2 = np.arange(10), 0.35
    ax.bar(x2 - w2/2, top10.values * 100, w2,
           label='Black-Litterman (final)', color='steelblue', alpha=0.85)
    ax.bar(x2 + w2/2, mkt10.values * 100, w2,
           label='Market-Cap Proxy',        color='lightgrey', alpha=0.85)
    ax.set_title("BL Final Weights vs Market-Cap Proxy\nTop 10 BL Holdings", fontsize=12)
    ax.set_ylabel("Weight (%)")
    ax.set_xticks(x2)
    ax.set_xticklabels([t.replace('.NS', '') for t in top10.index],
                       rotation=35, ha='right', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
 
    ax = axes[1]
    all_sectors = sorted(set(SECTOR_MAP.values()))
    bl_sec, mkt_sec = [], []
    for sector in all_sectors:
        idx = [i for i, t in enumerate(tickers)
               if SECTOR_MAP.get(t, 'Other') == sector]
        bl_sec.append(sum(final_weights[i] for i in idx) * 100)
        mkt_sec.append(sum(w_market[i]     for i in idx) * 100)
 
    tilts  = np.array(bl_sec) - np.array(mkt_sec)
    colors = ['#2ca02c' if t > 0 else '#d62728' for t in tilts]
    ax.bar(range(len(all_sectors)), tilts, color=colors, alpha=0.85, edgecolor='white')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(range(len(all_sectors)))
    ax.set_xticklabels(all_sectors, rotation=35, ha='right', fontsize=9)
    ax.set_title(
        "BL Sector Tilts vs Market-Cap Proxy\n"
        "(green = overweight | red = underweight)\n"
        "[Concentrated: only top-N eligible sectors overweighted — Imp.10]",
        fontsize=11)
    ax.set_ylabel("Tilt (BL weight − Market weight, %)")
    ax.grid(True, axis='y', alpha=0.3)
 
    plt.tight_layout()
    plt.savefig("reports/bl_sector_tilts.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/bl_sector_tilts.png")
 
 
# ------------------------------------------------------------------
# SUMMARY PRINTER
# ------------------------------------------------------------------
 
def print_full_summary(all_growths, all_rets, benchmark_metrics=None,
                       rf=RISK_FREE_RATE_ANNUAL, periods=252):
    """Print formatted performance table for all strategies + NIFTY 50."""
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY  (rf = 7% p.a.)")
    print("=" * 80)
 
    hdr  = f"\n{'Strategy':<25} {'Total Ret':>10} {'Ann Ret':>9} {'Ann Vol':>9} "
    hdr += f"{'Sharpe':>8} {'Sortino':>9} {'Max DD':>9} {'Calmar':>8}"
    print(hdr)
    print("-" * 80)
 
    def row(name, growth, ret_s):
        n_yr      = len(ret_s) / periods
        total_ret = growth.iloc[-1] - 1
        ann_ret   = (1 + total_ret) ** (1 / n_yr) - 1
        ann_vol   = ret_s.std() * np.sqrt(periods)
        sharpe    = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
        cum       = (1 + ret_s).cumprod()
        dd        = (cum - cum.cummax()) / cum.cummax()
        max_dd    = dd.min()
        calmar_v  = ann_ret / abs(max_dd) if max_dd != 0 else 0
        down_ret  = ret_s[ret_s < rf / periods]
        down_std  = (down_ret.std() * np.sqrt(periods)
                     if len(down_ret) > 0 else 1e-10)
        sortino_v = (ann_ret - rf) / down_std
        print(
            f"  {name:<23} {total_ret*100:>9.1f}%  {ann_ret*100:>8.1f}%"
            f"  {ann_vol*100:>7.1f}%  {sharpe:>7.3f}  {sortino_v:>8.3f}"
            f"  {max_dd*100:>8.1f}%  {calmar_v:>7.3f}"
        )
 
    for name, growth in all_growths.items():
        if (ret_s := all_rets.get(name)) is not None:
            row(name, growth, ret_s)
 
    if benchmark_metrics is not None:
        m = benchmark_metrics
        print(
            f"  {'NIFTY 50 (benchmark)':<23} {m['Total Return (%)']:>9.1f}%"
            f"  {m['Ann Return (%)']:>8.1f}%  {m['Ann Vol (%)']:>7.1f}%"
            f"  {m['Sharpe Ratio']:>7.3f}  {m['Sortino Ratio']:>8.3f}"
            f"  {m['Max Drawdown (%)']:>8.1f}%  {m['Calmar Ratio']:>7.3f}"
        )
    print("=" * 80)
 
 
# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
 
if __name__ == "__main__":
    print("=" * 72)
    print("BLACK-LITTERMAN PORTFOLIO — IMPROVEMENT 10: POSITION CONCENTRATION")
    print(f"Covariance    : {EWMA_BLEND_WEIGHT*100:.0f}% EWMA (λ={EWMA_LAMBDA}) + "
          f"{(1-EWMA_BLEND_WEIGHT)*100:.0f}% Ledoit-Wolf (blended)")
    print(f"Sector score  : 0.5×rel_mom + 0.3×low-TE + 0.2×200DMA")
    print(f"Momentum      : RELATIVE (sector excess return vs NIFTY 50)")
    print(f"Regime filter : BULL / NEUTRAL / BEAR / PANIC (NIFTY 200-DMA + vol)")
    print(f"Threshold gate: max_drift≥{REBALANCE_MAX_DRIFT*100:.0f}% OR "
          f"turnover≥{REBALANCE_TURNOVER_GATE*100:.0f}% (else skip, no TC)")
    print(f"Concentration : Top-N sectors ({TOP_N_SECTORS_REGIME}) + "
          f"min conviction ({CONVICTION_MIN_WEIGHT_REGIME}) + "
          f"score floor ({CONCENTRATION_SCORE_FLOOR})")
    print(f"\nRegime parameters (incl. EWMA λ offsets + concentration thresholds):")
    for r in ['BULL', 'NEUTRAL', 'BEAR', 'PANIC']:
        p      = REGIME_PARAMS[r]
        lam_r  = EWMA_LAMBDA + EWMA_LAMBDA_REGIME_OFFSET.get(r, 0.0)
        hl_r   = np.log(2) / np.log(1.0 / lam_r)
        top_n  = TOP_N_SECTORS_REGIME[r]
        min_cx = CONVICTION_MIN_WEIGHT_REGIME[r]
        print(f"  {r:<7}: Q={p['q_scale']}  gate={p['view_gate']}  "
              f"maxWt={p['max_wt']*100:.0f}%  Ω_damp={REGIME_OMEGA_DAMP[r]}  "
              f"λ={lam_r:.3f}  HL={hl_r:.1f}d  "
              f"top_N={top_n}  min_cvx={min_cx*100:.1f}%")
    print("=" * 72)
 
    # Load data
    returns, tickers, n = load_returns()
    price_matrix        = load_prices(tickers)
 
    print(f"\nData loaded — {n} tickers | "
          f"{returns.index[0].date()} → {returns.index[-1].date()}")
 
    total_days = len(returns)
    if total_days < MOMENTUM_WINDOW_12M + REBALANCE_FREQ:
        print(f"\n⚠  WARNING: Only {total_days} trading days available.")
        print(f"   Need ≥ {MOMENTUM_WINDOW_12M + REBALANCE_FREQ} for 12M momentum leg.")
        print(f"   Early rebalances use 6M-only momentum (automatic fallback).")
 
    # Equal weight baseline
    weights_ew = np.ones(n) / n
    returns_ew = returns.dot(weights_ew)
    growth_ew  = (1 + returns_ew).cumprod()
 
    # Market-cap proxy weights
    print("\nComputing market-cap proxy weights...")
    w_market        = compute_market_cap_weights(tickers, price_matrix)
    w_market_series = pd.Series(w_market, index=tickers)
    print("   Top 10 market-cap proxy weights:")
    print((w_market_series.sort_values(ascending=False).head(10) * 100)
          .round(2).to_string())
 
    # Load NIFTY 50
    print("\nLoading NIFTY 50 for relative momentum + regime filter...")
    nifty_returns_full, benchmark_growth, bm_ok = load_nifty_benchmark(
        start_date=returns.index[0],
        end_date=returns.index[-1],
    )
    if bm_ok:
        print(f"   NIFTY returns available: {len(nifty_returns_full)} trading days")
        print(f"   Relative momentum ENABLED")
        print(f"   Regime filter ENABLED — BULL/NEUTRAL/BEAR/PANIC classification")
    else:
        nifty_returns_full = None
        benchmark_growth   = None
        print("   NIFTY data unavailable — using absolute momentum, no regime filter")
 
    # Run rolling BL with all improvements (1–10)
    print("\nRunning rolling Black-Litterman (Improvements 1–10: Concentration)...")
    (bl_ret, bl_growth, final_weights,
     rebalance_log, regime_log,
     activity_log, ewma_diag_log,
     conc_diag_log) = run_rolling_bl(
        returns, price_matrix, tickers, w_market,
        nifty_rets=nifty_returns_full,
    )
 
    bl_growth.to_csv("data/portfolio/black_litterman_growth.csv")
    print(f"\n   Total return (net of TC): {(bl_growth.iloc[-1]-1)*100:.1f}%")
 
    # Compute benchmark metrics
    bm_ret_for_metrics = None
    if bm_ok and nifty_returns_full is not None:
        bm_ret_for_metrics = nifty_returns_full.loc[
            bl_growth.index[0]:bl_growth.index[-1]
        ]
        benchmark_metrics = compute_benchmark_metrics(bm_ret_for_metrics)
        print(f"   NIFTY 50 total return : {benchmark_metrics['Total Return (%)']:.1f}%")
        print(f"   NIFTY 50 Sharpe       : {benchmark_metrics['Sharpe Ratio']:.3f}")
 
        if benchmark_growth is not None:
            benchmark_growth = benchmark_growth.reindex(
                bl_growth.index, method='ffill'
            ).dropna()
            benchmark_growth = benchmark_growth / benchmark_growth.iloc[0]
    else:
        benchmark_metrics = None
        print("   Benchmark not available — plots will show BL + EW only.")
 
    # Sector tilts
    print("\n   Final period sector tilts vs market-cap:")
    print_sector_weights(final_weights, tickers, label="BL (final)")
    print("\n   Overweight / Underweight vs market-cap proxy:")
    for sector in sorted(set(SECTOR_MAP.values())):
        sec_idx = [i for i, t in enumerate(tickers)
                   if SECTOR_MAP.get(t, 'Other') == sector]
        if not sec_idx:
            continue
        bl_sec  = sum(final_weights[i] for i in sec_idx)
        mkt_sec = sum(w_market[i]      for i in sec_idx)
        tilt    = bl_sec - mkt_sec
        if abs(tilt) > 0.005:
            direction = "▲ OW" if tilt > 0 else "▼ UW"
            print(f"     {sector:15s}: {bl_sec*100:.1f}%  "
                  f"({direction} {abs(tilt)*100:.1f}% vs market-cap proxy)")
 
    # Final regime state
    if regime_log:
        final_regime = regime_log[-1][1]
        lam_final    = EWMA_LAMBDA + EWMA_LAMBDA_REGIME_OFFSET.get(final_regime, 0.0)
        hl_final     = np.log(2) / np.log(1.0 / lam_final)
        top_n_final  = TOP_N_SECTORS_REGIME.get(final_regime, TOP_N_SECTORS_BASE)
        min_cx_final = CONVICTION_MIN_WEIGHT_REGIME.get(final_regime, 0.015)
        print(f"\n   Current regime (last rebalance): {final_regime}")
        final_rp = get_regime_params(final_regime)
        print(f"   Active parameters: Q={final_rp['q_scale']}  "
              f"gate={final_rp['view_gate']}  maxWt={final_rp['max_wt']*100:.0f}%  "
              f"Ω_damp={REGIME_OMEGA_DAMP[final_regime]}  "
              f"λ={lam_final:.3f}  HL={hl_final:.1f}d  "
              f"top_N={top_n_final}  min_cvx={min_cx_final*100:.1f}%")
 
    # Summary table
    all_growths = {'Black-Litterman': bl_growth, 'Equal Weight': growth_ew}
    all_rets    = {'Black-Litterman': bl_ret,    'Equal Weight': returns_ew}
    print_full_summary(all_growths, all_rets, benchmark_metrics=benchmark_metrics)
 
    # Rebalance log
    if rebalance_log:
        print(f"\n   Showing last 5 rebalances (of {len(rebalance_log)} total):")
        for entry in rebalance_log[-5:]:
            date, tops, bots = entry[0], entry[1], entry[2]
            regime_label     = entry[3] if len(entry) > 3 else 'N/A'
            print(f"     {date.date()}  [{regime_label}]  "
                  f"Score+: {tops[0]}, {tops[1]}"
                  f"  |  Score−: {bots[-1]}, {bots[-2]}")
 
    # Plots
    plot_bl_growth(bl_growth, growth_ew, benchmark_growth=benchmark_growth)
    plot_final_momentum_views(price_matrix, tickers,
                               nifty_rets=nifty_returns_full)
    plot_bl_sector_tilts(final_weights, w_market, tickers)
    plot_bl_rolling_sharpe(bl_ret, returns_ew,
                           benchmark_returns=bm_ret_for_metrics)
    plot_bl_drawdown(bl_ret, returns_ew,
                     benchmark_returns=bm_ret_for_metrics)
    plot_regime_history(regime_log, bl_growth,
                        benchmark_growth=benchmark_growth)
    plot_rebalance_activity(activity_log, bl_growth)
    plot_ewma_diagnostics(ewma_diag_log, bl_growth, activity_log=activity_log)
    # NEW: Concentration diagnostics plot (Improvement 10)
    plot_concentration_diagnostics(conc_diag_log, bl_growth)
 
    print("\n✅ Black-Litterman model complete (Improvements 1–10).")
    print(f"   Covariance         : {EWMA_BLEND_WEIGHT*100:.0f}% EWMA (λ={EWMA_LAMBDA}) + "
          f"{(1-EWMA_BLEND_WEIGHT)*100:.0f}% Ledoit-Wolf")
    print(f"   Regime-adaptive λ  : BULL={EWMA_LAMBDA:.2f}  "
          f"NEUTRAL={EWMA_LAMBDA+EWMA_LAMBDA_REGIME_OFFSET['NEUTRAL']:.2f}  "
          f"BEAR={EWMA_LAMBDA+EWMA_LAMBDA_REGIME_OFFSET['BEAR']:.2f}  "
          f"PANIC={EWMA_LAMBDA+EWMA_LAMBDA_REGIME_OFFSET['PANIC']:.2f}")
    print(f"   Sector score       : 0.5×rel_mom + 0.3×low-TE + 0.2×200DMA")
    print(f"   Momentum           : RELATIVE excess returns vs NIFTY 50")
    print(f"   Threshold gate     : max_drift≥{REBALANCE_MAX_DRIFT*100:.0f}%"
          f" OR turnover≥{REBALANCE_TURNOVER_GATE*100:.0f}% (else skip, no TC)")
    print(f"   Concentration (10) : Top-N + min_cvx + score_floor=0.0")
    print(f"     BULL  → top_N={TOP_N_SECTORS_REGIME['BULL']}  "
          f"min_cvx={CONVICTION_MIN_WEIGHT_REGIME['BULL']*100:.1f}%")
    print(f"     BEAR  → top_N={TOP_N_SECTORS_REGIME['BEAR']}  "
          f"min_cvx={CONVICTION_MIN_WEIGHT_REGIME['BEAR']*100:.1f}%")
    print(f"     PANIC → top_N={TOP_N_SECTORS_REGIME['PANIC']}  "
          f"min_cvx={CONVICTION_MIN_WEIGHT_REGIME['PANIC']*100:.1f}%")
    print(f"   Outputs → data/portfolio/black_litterman_growth.csv")
    print(f"   Plots   → reports/bl_*.png  (incl. bl_concentration_diagnostics.png NEW)")