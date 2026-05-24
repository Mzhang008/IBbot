"""
Vectorized backtest for the Kalman-spread + regime-gated strategy.

Per-leg accounting:
    - Track signed share quantities for each leg independently
    - MTM P&L = qty_long * dPL + qty_short * dPS
    - Transaction cost = cost_bps * shares * fill_price per leg
    - Borrow = borrow_bps * |short_notional| pro-rated per bar
    - Hedge-neutral sizing using the Kalman beta available at decision time

Standalone CLI:
    python backtest.py --long SMH --short IGV --start 2020-01-01 --end 2025-01-01
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

from pairs import KalmanHedgeRatio, calibrate_kalman_mle, ou_half_life


@dataclass
class BTConfig:
    lookback: int = 250
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 4.0
    max_hold_bars: int = 240
    min_half_life: float = 5.0
    max_half_life: float = 120.0
    capital: float = 100_000.0
    cost_bps_per_leg: float = 1.0         # 1bp per fill
    borrow_bps_annual: float = 50.0
    kalman_delta: float = 1e-4
    kalman_R: float = 1e-3
    calibrate: bool = True


def fetch_yf(symbol: str, start: str, end: str, interval: str) -> pd.Series:
    if yf is None:
        raise ImportError("Install yfinance: pip install yfinance")
    df = yf.download(symbol, start=start, end=end, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data for {symbol} {start}->{end} {interval}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df["Close"].rename(symbol)


def zscore_no_lookahead(spread: pd.Series, lookback: int) -> pd.Series:
    mu = spread.shift(1).rolling(lookback).mean()
    sigma = spread.shift(1).rolling(lookback).std(ddof=1)
    return (spread - mu) / sigma


def _hedge_neutral_qty(
    capital: float, price_long: float, price_short: float, beta: float,
) -> tuple[int, int]:
    beta = max(0.1, min(5.0, abs(beta)))
    qty_long  = max(1, int((capital / 2.0) // price_long))
    qty_short = max(1, int((beta * qty_long * price_long) // price_short))
    return qty_long, qty_short


def backtest(
    prices_long: pd.Series,
    prices_short: pd.Series,
    cfg: BTConfig,
    bars_per_year: int = 252 * 7,
) -> dict:
    df = pd.concat(
        [prices_long.rename("PL"), prices_short.rename("PS")], axis=1,
    ).dropna()
    if len(df) < cfg.lookback + 50:
        raise RuntimeError(f"Need >= {cfg.lookback + 50} bars, have {len(df)}")

    log_PL = np.log(df["PL"].values)
    log_PS = np.log(df["PS"].values)

    delta, R, calibrated_ok = cfg.kalman_delta, cfg.kalman_R, False
    if cfg.calibrate:
        delta, R, calibrated_ok = calibrate_kalman_mle(
            log_PL, log_PS, init_delta=cfg.kalman_delta, init_R=cfg.kalman_R,
        )
    kf = KalmanHedgeRatio(delta=delta, R=R)
    betas, spread, _ = kf.run_full(log_PL, log_PS)
    df["beta"]   = betas
    df["spread"] = spread
    df["z"] = zscore_no_lookahead(df["spread"], cfg.lookback)
    # Rolling half-life on the spread (uses prior `lookback` bars)
    hl_series = pd.Series(index=df.index, dtype=float)
    sp_vals = df["spread"].values
    for i in range(cfg.lookback, len(df)):
        hl_series.iloc[i] = ou_half_life(sp_vals[i - cfg.lookback:i])
    df["half_life"] = hl_series
    df = df.dropna().copy()

    # ---- per-leg state ----
    pos_long  = 0           # signed shares
    pos_short = 0
    state = 0               # -1 short spread, 0 flat, +1 long spread
    entry_i: Optional[int] = None

    PL = df["PL"].values
    PS = df["PS"].values
    Z  = df["z"].values
    B  = df["beta"].values
    HL = df["half_life"].values

    pnl = np.zeros(len(df))
    cost_rate = cfg.cost_bps_per_leg / 10_000.0
    borrow_per_bar = (cfg.borrow_bps_annual / 10_000.0) / bars_per_year
    trade_log = []

    for i in range(1, len(df)):
        # ----- 1. MTM on current positions -----
        if pos_long != 0:
            pnl[i] += pos_long * (PL[i] - PL[i - 1])
        if pos_short != 0:
            pnl[i] += pos_short * (PS[i] - PS[i - 1])
        # ----- 2. Borrow on short notional (each leg independently) -----
        if pos_long < 0:
            pnl[i] -= abs(pos_long) * PL[i - 1] * borrow_per_bar
        if pos_short < 0:
            pnl[i] -= abs(pos_short) * PS[i - 1] * borrow_per_bar

        # ----- 3. Decision based on PRIOR bar's signal -----
        z_signal = Z[i - 1]
        hl_signal = HL[i - 1]
        tradable = (cfg.min_half_life <= hl_signal <= cfg.max_half_life)

        if state != 0:
            held = i - entry_i
            exit_reason: Optional[str] = None
            if abs(z_signal) >= cfg.z_stop:
                exit_reason = "z_stop"
            elif abs(z_signal) <= cfg.z_exit:
                exit_reason = "z_exit"
            elif held >= cfg.max_hold_bars:
                exit_reason = "time_stop"
            elif not tradable:
                exit_reason = "tradability_broke"

            if exit_reason:
                exit_cost = (abs(pos_long)  * PL[i] +
                             abs(pos_short) * PS[i]) * cost_rate
                pnl[i] -= exit_cost
                trade_log.append({
                    "entry_idx": entry_i, "exit_idx": i,
                    "entry_z": Z[entry_i], "exit_z": z_signal,
                    "held_bars": held, "state": state,
                    "exit_reason": exit_reason,
                    "qty_long": pos_long, "qty_short": pos_short,
                    "pnl": pnl[entry_i:i + 1].sum(),
                })
                pos_long = 0
                pos_short = 0
                state = 0
                entry_i = None
        else:
            if not tradable:
                continue
            if z_signal > cfg.z_entry or z_signal < -cfg.z_entry:
                direction = -1 if z_signal > 0 else +1
                ql, qs = _hedge_neutral_qty(
                    cfg.capital, PL[i], PS[i], B[i - 1],
                )
                if direction == -1:        # SHORT spread
                    pos_long  = -ql
                    pos_short = +qs
                else:                      # LONG spread
                    pos_long  = +ql
                    pos_short = -qs
                state = direction
                entry_i = i
                pnl[i] -= (ql * PL[i] + qs * PS[i]) * cost_rate

    # ---- Metrics ----
    pnl_s  = pd.Series(pnl, index=df.index, name="pnl")
    equity = cfg.capital + pnl_s.cumsum()
    ret    = pnl_s / cfg.capital
    sharpe = (ret.mean() / ret.std()) * np.sqrt(bars_per_year) if ret.std() > 0 else 0.0
    dd     = (equity.cummax() - equity)
    max_dd = float(dd.max() / cfg.capital)
    total  = float(pnl_s.sum() / cfg.capital)
    trades = pd.DataFrame(trade_log)
    win_rt = float((trades["pnl"] > 0).mean()) if not trades.empty else 0.0

    return {
        "equity": equity, "pnl": pnl_s, "trades": trades,
        "kalman": {"delta": delta, "R": R, "calibrated": calibrated_ok},
        "metrics": {
            "total_return_pct":  total * 100,
            "sharpe_annualized": sharpe,
            "max_drawdown_pct":  max_dd * 100,
            "n_trades":          len(trades),
            "win_rate_pct":      win_rt * 100,
            "avg_trade_pnl_$":   float(trades["pnl"].mean()) if not trades.empty else 0.0,
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Stat-arb backtest (Kalman + Z, per-leg)")
    ap.add_argument("--long",  default="SMH")
    ap.add_argument("--short", default="IGV")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end",   default=datetime.utcnow().strftime("%Y-%m-%d"))
    ap.add_argument("--interval", default="1d", choices=["1d", "1h"])
    ap.add_argument("--lookback",   type=int,   default=60)
    ap.add_argument("--capital",    type=float, default=100_000.0)
    ap.add_argument("--cost_bps",   type=float, default=1.0)
    ap.add_argument("--borrow_bps", type=float, default=50.0)
    ap.add_argument("--no_calibrate", action="store_true",
                    help="Skip MLE calibration of Kalman (delta, R)")
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
        calibrate=not args.no_calibrate,
    )
    out = backtest(pL, pS, cfg, bars_per_year=bars_per_year)

    print(f"\n{'='*60}\nBacktest: {args.long} vs {args.short}  ({args.interval})")
    print(f"Window:   {pL.index[0].date()} -> {pL.index[-1].date()}  ({len(pL)} bars)")
    k = out["kalman"]
    print(f"Kalman:   delta={k['delta']:.2e}  R={k['R']:.2e}  "
          f"calibrated={k['calibrated']}\n")
    for kk, vv in out["metrics"].items():
        print(f"  {kk:24s} {vv:>10.2f}")
    if not out["trades"].empty:
        wins = out["trades"][out["trades"]["pnl"] > 0]
        loss = out["trades"][out["trades"]["pnl"] <= 0]
        print(f"\n  avg win  $: {wins['pnl'].mean() if not wins.empty else 0:>10.2f}")
        print(f"  avg loss $: {loss['pnl'].mean() if not loss.empty else 0:>10.2f}")
        by_reason = out["trades"].groupby("exit_reason")["pnl"].agg(["count", "mean", "sum"])
        print(f"\n  Exit-reason breakdown:\n{by_reason}")


if __name__ == "__main__":
    main()
