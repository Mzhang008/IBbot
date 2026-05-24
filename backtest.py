"""
Vectorized backtest for the Kalman-spread + regime-gated strategy.

Standalone CLI:
    python backtest.py --long SMH --short IGV --start 2020-01-01 --end 2024-12-31

Data source:
    yfinance (free, daily/hourly bars). Hourly only goes back ~2y on Yahoo.
    Pass --interval 1d for multi-year backtests.

Models:
    - Kalman hedge ratio (same params as live bot)
    - Z-score with no look-ahead (mu, sigma exclude current bar)
    - Entry Z>2, exit |Z|<0.5, hard stop |Z|>4, time-stop = 10 bars * factor
    - Per-leg transaction cost (bps) + annualized borrow on the short leg
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

from pairs import KalmanHedgeRatio


@dataclass
class BTConfig:
    lookback: int = 250
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 4.0
    max_hold_bars: int = 240               # ~10 trading days @ hourly
    capital: float = 100_000.0
    cost_bps_per_leg: float = 1.0          # 1bp per fill (round-trip = 4bps)
    borrow_bps_annual: float = 50.0        # half-bp/day on short notional
    kalman_delta: float = 1e-4
    kalman_R: float = 1e-3


def fetch_yf(symbol: str, start: str, end: str, interval: str) -> pd.Series:
    if yf is None:
        raise ImportError("Install yfinance: pip install yfinance")
    df = yf.download(symbol, start=start, end=end, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data for {symbol} {start}->{end} {interval}")
    # yfinance can return MultiIndex columns -- flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df["Close"].rename(symbol)


def compute_kalman_spread(
    log_long: np.ndarray, log_short: np.ndarray,
    delta: float, R: float,
) -> tuple[np.ndarray, np.ndarray]:
    kf = KalmanHedgeRatio(delta=delta, R=R)
    betas, resids, _ = kf.run_full(log_long, log_short)
    return betas, resids


def zscore_no_lookahead(spread: pd.Series, lookback: int) -> pd.Series:
    """mu, sigma from PRIOR `lookback` bars, applied to the CURRENT bar."""
    mu = spread.shift(1).rolling(lookback).mean()
    sigma = spread.shift(1).rolling(lookback).std(ddof=1)
    return (spread - mu) / sigma


def backtest(
    prices_long: pd.Series,
    prices_short: pd.Series,
    cfg: BTConfig,
    bars_per_year: int = 252 * 7,    # hourly RTH default
) -> dict:
    df = pd.concat(
        [prices_long.rename("PL"), prices_short.rename("PS")], axis=1,
    ).dropna()
    if len(df) < cfg.lookback + 50:
        raise RuntimeError(f"Need >= {cfg.lookback + 50} aligned bars, have {len(df)}")

    log_PL = np.log(df["PL"].values)
    log_PS = np.log(df["PS"].values)
    betas, spread = compute_kalman_spread(
        log_PL, log_PS, cfg.kalman_delta, cfg.kalman_R,
    )
    df["beta"]   = betas
    df["spread"] = spread
    df["z"] = zscore_no_lookahead(df["spread"], cfg.lookback)
    df = df.dropna().copy()

    # Simulation state
    state = 0                              # -1 short spread, 0 flat, +1 long spread
    entry_i: Optional[int] = None
    entry_beta = 0.0
    pnl_series = np.zeros(len(df))
    trade_log = []
    cost_rate = cfg.cost_bps_per_leg / 10_000.0
    borrow_per_bar = (cfg.borrow_bps_annual / 10_000.0) / bars_per_year

    PL = df["PL"].values
    PS = df["PS"].values
    Z  = df["z"].values
    B  = df["beta"].values
    SP = df["spread"].values

    for i in range(1, len(df)):
        z_now = Z[i - 1]                   # decide on PRIOR bar -> trade THIS bar
        if state != 0:
            # MTM P&L from spread change, scaled by entry beta (locked-in hedge)
            dPL = PL[i] - PL[i - 1]
            dPS = PS[i] - PS[i - 1]
            # state +1 = long_leg long, short_leg short
            # P&L per unit capital ~ dPL/PL - beta * dPS/PS
            ret = (dPL / PL[i - 1]) - entry_beta * (dPS / PS[i - 1])
            pnl_series[i] = state * cfg.capital * ret
            # Borrow on the short leg
            pnl_series[i] -= cfg.capital * borrow_per_bar
            # Exit?
            held = i - entry_i
            exit_reason = None
            if abs(z_now) >= cfg.z_stop:
                exit_reason = "z_stop"
            elif abs(z_now) <= cfg.z_exit:
                exit_reason = "z_exit"
            elif held >= cfg.max_hold_bars:
                exit_reason = "time_stop"
            if exit_reason is not None:
                pnl_series[i] -= cfg.capital * cost_rate * 2.0   # both legs
                trade_log.append({
                    "entry_idx": entry_i, "exit_idx": i,
                    "entry_z": Z[entry_i], "exit_z": z_now,
                    "held_bars": held, "state": state,
                    "exit_reason": exit_reason,
                    "pnl": pnl_series[entry_i:i + 1].sum(),
                })
                state = 0
                entry_i = None
        else:
            if z_now > cfg.z_entry:
                state = -1
                entry_i = i
                entry_beta = B[i]
                pnl_series[i] -= cfg.capital * cost_rate * 2.0
            elif z_now < -cfg.z_entry:
                state = +1
                entry_i = i
                entry_beta = B[i]
                pnl_series[i] -= cfg.capital * cost_rate * 2.0

    pnl = pd.Series(pnl_series, index=df.index, name="pnl")
    equity = cfg.capital + pnl.cumsum()
    ret = pnl / cfg.capital
    sharpe = (ret.mean() / ret.std()) * np.sqrt(bars_per_year) if ret.std() > 0 else 0.0
    dd = (equity.cummax() - equity)
    max_dd = float(dd.max() / cfg.capital)
    total_ret = float(pnl.sum() / cfg.capital)
    trades = pd.DataFrame(trade_log)
    win_rate = float((trades["pnl"] > 0).mean()) if not trades.empty else 0.0
    return {
        "equity": equity, "pnl": pnl, "trades": trades,
        "metrics": {
            "total_return_pct": total_ret * 100,
            "sharpe_annualized": sharpe,
            "max_drawdown_pct": max_dd * 100,
            "n_trades": len(trades),
            "win_rate_pct": win_rate * 100,
            "avg_trade_pnl_$": float(trades["pnl"].mean()) if not trades.empty else 0,
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Stat-arb backtest (Kalman + Z)")
    ap.add_argument("--long",  default="SMH")
    ap.add_argument("--short", default="IGV")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end",   default=datetime.utcnow().strftime("%Y-%m-%d"))
    ap.add_argument("--interval", default="1d", choices=["1d", "1h"])
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--cost_bps", type=float, default=1.0)
    ap.add_argument("--borrow_bps", type=float, default=50.0)
    args = ap.parse_args()

    if yf is None:
        print("ERROR: yfinance is not installed. `pip install yfinance`")
        sys.exit(1)

    print(f"Fetching {args.long}, {args.short} {args.start} -> {args.end} ({args.interval})")
    pL = fetch_yf(args.long,  args.start, args.end, args.interval)
    pS = fetch_yf(args.short, args.start, args.end, args.interval)

    bars_per_year = 252 if args.interval == "1d" else 252 * 7
    cfg = BTConfig(
        lookback=args.lookback,
        capital=args.capital,
        cost_bps_per_leg=args.cost_bps,
        borrow_bps_annual=args.borrow_bps,
        max_hold_bars=20 if args.interval == "1d" else 240,
    )
    out = backtest(pL, pS, cfg, bars_per_year=bars_per_year)

    print(f"\n{'='*60}\nBacktest: {args.long} vs {args.short}  ({args.interval})")
    print(f"Window:    {pL.index[0].date()} -> {pL.index[-1].date()}  ({len(pL)} bars)\n")
    for k, v in out["metrics"].items():
        print(f"  {k:24s} {v:>10.2f}")
    if not out["trades"].empty:
        wins = out["trades"][out["trades"]["pnl"] > 0]
        loss = out["trades"][out["trades"]["pnl"] <= 0]
        print(f"\n  avg win  $: {wins['pnl'].mean() if not wins.empty else 0:>10.2f}")
        print(f"  avg loss $: {loss['pnl'].mean() if not loss.empty else 0:>10.2f}")
        by_reason = out["trades"].groupby("exit_reason")["pnl"].agg(["count", "mean", "sum"])
        print(f"\n  Exit-reason breakdown:\n{by_reason}")


if __name__ == "__main__":
    main()
