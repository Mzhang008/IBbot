"""
Pair object: encapsulates one cointegrated pair trade.

Replaces the old IV-weighted spread with:
  - Kalman-filtered hedge ratio (adaptive beta)
  - Engle-Granger ADF cointegration gate (only trade when p < threshold)
  - Z-score WITHOUT look-ahead (mu and sigma exclude the current bar)
  - Trend filter (both-leg SMA200 regime) -- replaces TrendProvider
  - Per-pair state machine with PositionMeta (filled qty/price tracked)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# =============================================================================
# Kalman filter for the hedge ratio: y_t = beta_t * x_t + v_t,
#                                    beta_t = beta_{t-1} + w_t
# =============================================================================
class KalmanHedgeRatio:
    """
    1-D Kalman filter on log-prices. delta controls how fast beta adapts:
      delta ~ 1e-5  -> almost static beta (similar to rolling OLS)
      delta ~ 1e-3  -> fast-adapting beta (regime-shift tolerant)
    R is observation noise -- start small for log-prices.
    """

    def __init__(
        self,
        delta: float = 1e-4,
        R: float = 1e-3,
        beta0: float = 1.0,
        P0: float = 1.0,
    ):
        self.Q = delta
        self.R = R
        self.beta = beta0
        self.P = P0

    def step(self, y: float, x: float) -> Tuple[float, float, float]:
        """Returns (beta_post, innovation, innovation_sigma)."""
        # Predict
        beta_pred = self.beta
        P_pred = self.P + self.Q
        # Innovation
        e = y - beta_pred * x
        S = x * P_pred * x + self.R
        # Update
        K = P_pred * x / S
        self.beta = beta_pred + K * e
        self.P = (1.0 - K * x) * P_pred
        return self.beta, e, float(np.sqrt(max(S, 1e-12)))

    def run_full(self, ys: np.ndarray, xs: np.ndarray):
        betas, resids, stds = [], [], []
        for y, x in zip(ys, xs):
            b, e, s = self.step(float(y), float(x))
            betas.append(b)
            resids.append(e)
            stds.append(s)
        return np.array(betas), np.array(resids), np.array(stds)


def calibrate_kalman_mle(
    ys: np.ndarray,
    xs: np.ndarray,
    init_delta: float = 1e-4,
    init_R: float = 1e-3,
    warmup: int = 30,
) -> Tuple[float, float, bool]:
    """
    Calibrate (delta, R) by maximizing the Gaussian log-likelihood of the
    innovation sequence:  ll = -0.5 * sum( log(2*pi*S_t) + e_t^2 / S_t )

    Returns (delta, R, converged). Falls back to init values if scipy is
    unavailable or optimisation diverges.
    """
    if not _HAS_SCIPY or len(ys) < warmup + 50:
        return init_delta, init_R, False
    ys = np.asarray(ys, dtype=float)
    xs = np.asarray(xs, dtype=float)

    def neg_ll(log_params):
        delta = float(np.exp(log_params[0]))
        R     = float(np.exp(log_params[1]))
        kf = KalmanHedgeRatio(delta=delta, R=R)
        _, resids, stds = kf.run_full(ys, xs)
        e = resids[warmup:]
        s = stds[warmup:]
        if len(e) == 0 or np.any(s <= 0) or not np.isfinite(e).all():
            return 1e10
        return float(0.5 * np.sum(np.log(2.0 * np.pi * s * s) + (e * e) / (s * s)))

    try:
        res = minimize(
            neg_ll,
            x0=[np.log(init_delta), np.log(init_R)],
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-3, "maxiter": 300},
        )
        delta_opt = float(np.exp(res.x[0]))
        R_opt = float(np.exp(res.x[1]))
        # Sanity bounds
        delta_opt = max(1e-8, min(1e-1, delta_opt))
        R_opt     = max(1e-8, min(1.0,  R_opt))
        return delta_opt, R_opt, bool(res.success)
    except Exception:
        return init_delta, init_R, False


def ou_half_life(spread: np.ndarray) -> float:
    """
    Ornstein-Uhlenbeck half-life of a mean-reverting series, in bars.

    Fit dS_t = alpha + lambda * S_{t-1} + eps.
      lambda < 0 -> mean reverting; half-life = -ln(2) / lambda.
      lambda >= 0 -> not mean reverting; return +inf.
    """
    s = np.asarray(spread, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 30:
        return float("inf")
    s_lag = s[:-1]
    ds = np.diff(s)
    n = len(s_lag)
    # Closed-form OLS for ds = a + b * s_lag
    sx, sy = s_lag.sum(), ds.sum()
    sxx = float(s_lag @ s_lag)
    sxy = float(s_lag @ ds)
    denom = n * sxx - sx * sx
    if denom == 0:
        return float("inf")
    lam = (n * sxy - sx * sy) / denom
    if lam >= 0:
        return float("inf")
    return float(-np.log(2.0) / lam)


# =============================================================================
# PositionMeta -- what got actually filled, not what was sent
# =============================================================================
@dataclass
class PositionMeta:
    direction: str                # "SHORT_SPREAD" | "LONG_SPREAD"
    entry_time: datetime
    entry_z: float
    entry_spread: float
    entry_beta: float
    qty_long: int                 # signed: + for long, - for short
    qty_short: int
    avg_price_long: float
    avg_price_short: float


# =============================================================================
# Pair
# =============================================================================
class Pair:
    """A single tradable pair with its own Kalman beta, spread, and state."""

    POS_FLAT = "FLAT"
    POS_SHORT_SPREAD = "SHORT_SPREAD"   # SELL long_sym, BUY  short_sym (Z > +entry)
    POS_LONG_SPREAD  = "LONG_SPREAD"    # BUY  long_sym, SELL short_sym (Z < -entry)

    def __init__(
        self,
        long_sym: str,
        short_sym: str,
        lookback: int = 250,
        kalman_delta: float = 1e-4,
        kalman_R: float = 1e-3,
        sma_window: int = 200,
    ):
        self.long_sym = long_sym
        self.short_sym = short_sym
        self.lookback = lookback
        self.sma_window = sma_window

        # Filled by the bot after qualifyContractsAsync
        self.long_contract = None
        self.short_contract = None

        # Bars[sym] -> DataFrame with at least a "close" column, datetime index
        self.bars: Dict[str, pd.DataFrame] = {}

        # Filter state
        self.kalman = KalmanHedgeRatio(delta=kalman_delta, R=kalman_R)
        self.beta_history: pd.Series = pd.Series(dtype=float)
        self.spread_history: pd.Series = pd.Series(dtype=float)

        # Trading state (derived from broker fills, not from order submission)
        self.state: str = self.POS_FLAT
        self.position: Optional[PositionMeta] = None

        # Cached cointegration result
        self._coint_cache: Optional[Tuple[datetime, float]] = None

    # -------------------------------------------------------------- identity
    @property
    def name(self) -> str:
        return f"{self.long_sym}/{self.short_sym}"

    # -------------------------------------------------------------- bars I/O
    def _aligned_closes(self) -> pd.DataFrame:
        l = self.bars.get(self.long_sym)
        s = self.bars.get(self.short_sym)
        if l is None or s is None or l.empty or s.empty:
            return pd.DataFrame()
        return pd.concat(
            [l["close"].rename(self.long_sym),
             s["close"].rename(self.short_sym)],
            axis=1,
        ).dropna()

    # =========================================================================
    # Hedge ratio (Kalman, adaptive)
    # =========================================================================
    def initialize_hedge_ratio(self, calibrate: bool = True) -> Tuple[float, float, bool]:
        """
        Run Kalman through ALL historical bars from scratch.
        If `calibrate` and scipy available, fit (delta, R) by MLE first.
        Returns (delta, R, calibrated_ok).
        """
        closes = self._aligned_closes()
        if len(closes) < 30:
            raise ValueError(
                f"{self.name}: need >=30 aligned bars, have {len(closes)}"
            )
        log_long  = np.log(closes[self.long_sym].values)
        log_short = np.log(closes[self.short_sym].values)

        delta, R, ok = self.kalman.Q, self.kalman.R, False
        if calibrate:
            delta, R, ok = calibrate_kalman_mle(
                log_long, log_short,
                init_delta=self.kalman.Q,
                init_R=self.kalman.R,
            )

        # Reset filter state with (possibly calibrated) params
        self.kalman = KalmanHedgeRatio(delta=delta, R=R, beta0=1.0, P0=1.0)
        betas, resids, _ = self.kalman.run_full(log_long, log_short)
        self.beta_history   = pd.Series(betas,  index=closes.index)
        self.spread_history = pd.Series(resids, index=closes.index)
        return delta, R, ok

    def update_hedge_ratio(self) -> int:
        """Step Kalman forward on bars that arrived after last update. Returns n_new."""
        closes = self._aligned_closes()
        n_new = len(closes) - len(self.beta_history)
        if n_new <= 0:
            return 0
        new_long  = np.log(closes[self.long_sym].iloc[-n_new:].values)
        new_short = np.log(closes[self.short_sym].iloc[-n_new:].values)
        new_betas, new_resids = [], []
        for y, x in zip(new_long, new_short):
            b, e, _ = self.kalman.step(float(y), float(x))
            new_betas.append(b)
            new_resids.append(e)
        new_idx = closes.index[-n_new:]
        self.beta_history   = pd.concat([self.beta_history,
                                         pd.Series(new_betas, index=new_idx)])
        self.spread_history = pd.concat([self.spread_history,
                                         pd.Series(new_resids, index=new_idx)])
        return n_new

    def current_beta(self) -> float:
        if self.beta_history.empty:
            return float("nan")
        return float(self.beta_history.iloc[-1])

    def current_spread(self) -> float:
        if self.spread_history.empty:
            return float("nan")
        return float(self.spread_history.iloc[-1])

    # =========================================================================
    # Z-score -- NO LOOK-AHEAD: mu/sigma estimated on window EXCLUDING current
    # =========================================================================
    def zscore_no_lookahead(self) -> Optional[float]:
        if len(self.spread_history) < self.lookback + 1:
            return None
        # Past `lookback` bars excluding the current observation
        window = self.spread_history.iloc[-(self.lookback + 1):-1]
        mu = window.mean()
        sigma = window.std(ddof=1)
        if sigma <= 0 or np.isnan(sigma):
            return None
        return float((self.spread_history.iloc[-1] - mu) / sigma)

    # =========================================================================
    # Cointegration gate (Engle-Granger ADF on the spread residuals)
    # =========================================================================
    def cointegration_pvalue(self) -> float:
        if not _HAS_STATSMODELS:
            # Cannot test -> conservatively assume cointegrated so bot still runs
            return 0.0
        if len(self.spread_history) < max(50, self.lookback // 2):
            return 1.0
        window = self.spread_history.iloc[-self.lookback:].dropna().values
        if len(window) < 30:
            return 1.0
        try:
            result = adfuller(window, autolag="AIC", maxlag=int(np.sqrt(len(window))))
            return float(result[1])
        except Exception:
            return 1.0

    def is_cointegrated(
        self, p_threshold: float, recheck_hours: int
    ) -> Tuple[bool, float]:
        """Cached ADF check. Returns (is_cointegrated, p_value)."""
        now = datetime.utcnow()
        if self._coint_cache is not None:
            ts, p = self._coint_cache
            if (now - ts).total_seconds() / 3600 < recheck_hours:
                return p < p_threshold, p
        p = self.cointegration_pvalue()
        self._coint_cache = (now, p)
        return p < p_threshold, p

    # =========================================================================
    # Half-life of mean reversion (cached alongside cointegration)
    # =========================================================================
    def half_life(self) -> float:
        """OU half-life in bars on the latest `lookback` spread window."""
        if len(self.spread_history) < 30:
            return float("inf")
        window = self.spread_history.iloc[-self.lookback:].values
        return ou_half_life(window)

    def is_tradable(
        self, p_threshold: float, recheck_hours: int,
        min_half_life_bars: float, max_half_life_bars: float,
    ) -> Tuple[bool, float, float]:
        """
        Combined gate: cointegration AND half-life in [min, max] range.
        Too-small half-life -> spread is noise, not a real reversion.
        Too-large half-life -> reversion is too slow to monetise within time-stop.
        Returns (tradable, p_value, half_life).
        """
        is_c, p = self.is_cointegrated(p_threshold, recheck_hours)
        hl = self.half_life()
        in_band = min_half_life_bars <= hl <= max_half_life_bars
        return (is_c and in_band), p, hl

    # =========================================================================
    # Trend adjustment -- both-leg SMA regime filter
    # =========================================================================
    def trend_adjustment(self) -> Tuple[float, str]:
        closes = self._aligned_closes()
        if len(closes) < self.sma_window:
            return 1.0, "trend:insufficient"
        l_sma = closes[self.long_sym].iloc[-self.sma_window:].mean()
        s_sma = closes[self.short_sym].iloc[-self.sma_window:].mean()
        l_up = closes[self.long_sym].iloc[-1] > l_sma
        s_up = closes[self.short_sym].iloc[-1] > s_sma
        if l_up and s_up:
            return 1.25, "trend:both_up"
        if (not l_up) and (not s_up):
            return 1.15, "trend:both_down"
        return 1.0, "trend:divergent"

    # =========================================================================
    # Sizing -- uses HEDGE RATIO so the spread itself is dollar-neutral.
    # Returns (qty_long, qty_short).
    # =========================================================================
    def hedge_neutral_sizing(
        self, capital: float, price_long: float, price_short: float,
    ) -> Tuple[int, int]:
        """
        Want: qty_long * P_L  ==  beta * qty_short * P_S
              s.t. qty_long * P_L + qty_short * P_S  ==  capital (gross)
        Then qty_long = capital / (P_L + beta * P_S * (P_L / (beta * P_S))) ...
        Simpler: fix qty_long = floor(capital / (2 * P_L)), then
                 qty_short  = round(qty_long * P_L * beta / P_S)
        Halves the gross to each leg (dollar-neutral after beta scaling).
        """
        beta = max(0.1, min(5.0, abs(self.current_beta())))
        # Allocate half capital to long-leg notional first.
        notional_long = capital / 2.0
        qty_long = max(1, int(notional_long // price_long))
        # Beta-hedged short-leg qty: short notional = beta * long notional
        notional_short = beta * qty_long * price_long
        qty_short = max(1, int(notional_short // price_short))
        return qty_long, qty_short
