"""
Mid-Frequency Statistical Arbitrage Trading Bot
Multi-pair, Kalman hedge ratio, cointegration-gated, atomic-fill execution.

================================================================================
WINDOWS / IB GATEWAY SETTINGS NOTES
================================================================================
1. Power plan: Settings > System > Power & Battery > Screen and Sleep -> "Never".
   Disable USB selective suspend; turn OFF Modern Standby if possible.
2. IB Gateway: File > Global Configuration > API > Settings:
       - Enable ActiveX and Socket Clients     [CHECKED]
       - Read-Only API                         [UNCHECKED]
       - Socket port                           4002 (Paper)
       - Trusted IPs -> add 127.0.0.1
       - Download open orders on connection    [CHECKED]
3. IB Gateway: Configure > Settings > Lock and Exit -> Auto restart outside RTH.
4. Pause Windows Update auto-reboot during market hours (Group Policy).
5. Optional: set python.exe / ibgateway.exe to High priority.
================================================================================

What this bot does, in order, every poll cycle:
  1. Pull macro regime (Congress / FRED / News / WSB) once for the loop.
  2. For each Pair:
       a. Refresh just the latest bars (not the full 60-day pull).
       b. Step the Kalman hedge ratio forward.
       c. Verify cointegration (Engle-Granger ADF, cached daily).
            -> if broken AND we hold the pair, FLATTEN it.
            -> if broken AND we are flat, SKIP.
       d. Compute Z with NO LOOK-AHEAD (mu, sigma exclude current bar).
       e. Pair-local trend filter raises Z-threshold in tech-bull regime.
       f. If in a position: check hard Z stop and time stop FIRST.
       g. Apply regime vetoes; if clean, send atomic two-leg entry / close.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ib_insync import IB, Contract, Stock, util

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from execution import (
    close_pair_atomic,
    enter_pair_atomic,
    send_marketable_limit,
    snapshot_quote,
)
from pairs import Pair, PositionMeta
from regime import RegimeAdjustment, build_default_analyzer


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class Config:
    # ---- Socket ----
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 17

    # ---- Bars / lookback ----
    bar_size: str = "1 hour"
    duration_initial: str = "180 D"      # heavy load on first connect
    duration_refresh: str = "2 D"        # small subsequent pulls
    use_rth: bool = True
    lookback: int = 250                  # bars used for mu/sigma + ADF

    # ---- Signal thresholds ----
    z_entry: float = 2.0
    z_exit:  float = 0.5
    z_stop:  float = 4.0                 # hard loss-cut
    time_stop_hours: int = 240           # ~10 trading days @ hourly

    # ---- Cointegration + half-life gate ----
    coint_pvalue_threshold: float = 0.05
    coint_recheck_hours: int = 24
    min_half_life_bars: float = 5.0      # too fast -> noise, not reversion
    max_half_life_bars: float = 120.0    # too slow -> won't monetise in time-stop

    # ---- Kalman calibration ----
    calibrate_kalman_on_init: bool = True

    # ---- Capital allocation ----
    capital_per_pair: float = 50_000.0

    # ---- Portfolio-level risk caps ----
    max_concurrent_pairs: int = 3
    max_gross_notional: float = 200_000.0

    # ---- Shortability ----
    min_shortable_shares: int = 1000     # hard veto under this
    shortability_recheck_hours: float = 4.0

    # ---- Position reconciliation ----
    reconcile_each_loop: bool = True

    # ---- Loop cadence ----
    poll_seconds: int = 3600

    # ---- Execution ----
    fill_timeout_s: float = 30.0

    # ---- Robustness ----
    market_data_type: int = 3            # 1=live 2=frozen 3=delayed 4=delayed-frozen
    max_retries: int = 5
    backoff_base: float = 2.0
    quote_timeout: float = 5.0


# Default pair universe. Add more (long, short) tuples here; each is run independently.
DEFAULT_PAIRS: List[Tuple[str, str]] = [
    ("SMH", "IGV"),
]


# =============================================================================
# TradingBot
# =============================================================================
class TradingBot:
    def __init__(self, cfg: Config, pair_specs: Optional[List[Tuple[str, str]]] = None):
        self.cfg = cfg
        self.ib = IB()
        self.log = self._setup_logger()
        specs = pair_specs or DEFAULT_PAIRS
        self._validate_pair_specs(specs)
        self.pairs: List[Pair] = [
            Pair(long_sym=l, short_sym=s, lookback=cfg.lookback)
            for l, s in specs
        ]
        self.contracts: Dict[str, Contract] = {}
        self.regime = build_default_analyzer()
        # symbol -> (timestamp, shortable_shares)
        self._shortability: Dict[str, Tuple[datetime, int]] = {}
        # IB Trade subscriptions for fill events -- bot reacts to broker reality
        self.ib.execDetailsEvent += self._on_exec_details

    @staticmethod
    def _validate_pair_specs(specs: List[Tuple[str, str]]) -> None:
        """Each symbol may appear in at most one pair. Required so the
        per-symbol broker position has unambiguous attribution during
        reconciliation."""
        seen: Dict[str, Tuple[str, str]] = {}
        for l, s in specs:
            for sym in (l, s):
                if sym in seen:
                    raise ValueError(
                        f"Symbol {sym} appears in multiple pairs "
                        f"({seen[sym]} and {(l, s)}); not supported."
                    )
                seen[sym] = (l, s)

    @staticmethod
    def _setup_logger() -> logging.Logger:
        logger = logging.getLogger("StatArbBot")
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            logger.addHandler(h)
        logger.setLevel(logging.INFO)
        return logger

    def _on_exec_details(self, trade, fill):
        """Fired by ib_insync when an execution report arrives. Pure observability."""
        self.log.info(
            "FILL %s %s %d @ %.2f (orderId=%s)",
            fill.contract.symbol, fill.execution.side, fill.execution.shares,
            fill.execution.price, trade.order.orderId,
        )

    # =========================================================================
    # Backoff wrapper
    # =========================================================================
    async def _with_retry(self, label, coro_factory, *a, **kw):
        last: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                return await coro_factory(*a, **kw)
            except Exception as exc:
                last = exc
                wait = self.cfg.backoff_base ** attempt
                self.log.warning(
                    "%s failed (%d/%d): %s -- retry in %.1fs",
                    label, attempt, self.cfg.max_retries, exc, wait,
                )
                await asyncio.sleep(wait)
        raise RuntimeError(f"{label}: exhausted retries") from last

    # =========================================================================
    # Connect / qualify
    # =========================================================================
    async def connect(self) -> None:
        async def _do():
            await self.ib.connectAsync(
                self.cfg.host, self.cfg.port, clientId=self.cfg.client_id,
            )
            self.ib.reqMarketDataType(self.cfg.market_data_type)
            self.log.info(
                "Connected %s:%d clientId=%d mktDataType=%d",
                self.cfg.host, self.cfg.port, self.cfg.client_id,
                self.cfg.market_data_type,
            )
        await self._with_retry("connect", _do)

    async def qualify_all(self) -> None:
        all_syms = set()
        for p in self.pairs:
            all_syms.update((p.long_sym, p.short_sym))

        async def _do():
            for sym in all_syms:
                c = Stock(sym, "SMART", "USD", primaryExchange="ARCA")
                q = await self.ib.qualifyContractsAsync(c)
                if not q:
                    raise RuntimeError(f"Could not qualify {sym}")
                self.contracts[sym] = q[0]
                self.log.info("Qualified %s (conId=%s)", sym, q[0].conId)
        await self._with_retry("qualify", _do)

        for p in self.pairs:
            p.long_contract  = self.contracts[p.long_sym]
            p.short_contract = self.contracts[p.short_sym]

    async def _refresh_shortability(self, symbol: str) -> int:
        """Pull current shortable-shares inventory via tick 236. Cached."""
        now = datetime.utcnow()
        cached = self._shortability.get(symbol)
        if cached is not None:
            ts, sh = cached
            age_h = (now - ts).total_seconds() / 3600.0
            if age_h < self.cfg.shortability_recheck_hours:
                return sh
        shortable = 0
        contract = self.contracts.get(symbol)
        if contract is None:
            return 0
        try:
            t = self.ib.reqMktData(contract, genericTickList="236",
                                   snapshot=False, regulatorySnapshot=False)
            # Tick 236 populates shortableShares; let it arrive
            for _ in range(20):
                await asyncio.sleep(0.25)
                val = getattr(t, "shortableShares", None)
                if val is not None and val == val:   # not-NaN check
                    shortable = int(val)
                    break
            try: self.ib.cancelMktData(contract)
            except Exception: pass
        except Exception as e:
            self.log.warning("Shortability fetch failed for %s: %s", symbol, e)
        self._shortability[symbol] = (now, shortable)
        return shortable

    async def _can_short(self, symbol: str, required_shares: int) -> Tuple[bool, int]:
        """HARD veto -- returns (ok, available)."""
        avail = await self._refresh_shortability(symbol)
        return (avail >= max(required_shares, self.cfg.min_shortable_shares)), avail

    # =========================================================================
    # History I/O -- bulk load once, then incremental refresh
    # =========================================================================
    async def _fetch_history(self, symbol: str, duration: str) -> pd.DataFrame:
        async def _do():
            bars = await self.ib.reqHistoricalDataAsync(
                self.contracts[symbol],
                endDateTime="",
                durationStr=duration,
                barSizeSetting=self.cfg.bar_size,
                whatToShow="TRADES",
                useRTH=self.cfg.use_rth,
                formatDate=1,
                keepUpToDate=False,
            )
            if not bars:
                raise RuntimeError(f"Empty bar list for {symbol}")
            df = util.df(bars).set_index("date").sort_index()
            return df
        return await self._with_retry(f"hist[{symbol}]", _do)

    async def load_initial_history(self) -> None:
        all_syms = set()
        for p in self.pairs:
            all_syms.update((p.long_sym, p.short_sym))
        tasks = {s: asyncio.create_task(
            self._fetch_history(s, self.cfg.duration_initial)) for s in all_syms}
        for s, t in tasks.items():
            df = await t
            for p in self.pairs:
                if p.long_sym == s:  p.bars[s] = df.copy()
                if p.short_sym == s: p.bars[s] = df.copy()
            self.log.info("Loaded %d %s bars (last=%s)", len(df), s, df.index[-1])
        for p in self.pairs:
            delta, R, ok = p.initialize_hedge_ratio(
                calibrate=self.cfg.calibrate_kalman_on_init,
            )
            self.log.info(
                "%s init: beta=%.3f spread=%.5f | Kalman delta=%.2e R=%.2e calibrated=%s",
                p.name, p.current_beta(), p.current_spread(), delta, R, ok,
            )

    async def refresh_pair(self, pair: Pair) -> None:
        """Append only the latest bars (~last 2 days), dedupe, sort."""
        for sym in (pair.long_sym, pair.short_sym):
            try:
                new = await self._fetch_history(sym, self.cfg.duration_refresh)
                cur = pair.bars[sym]
                merged = pd.concat([cur, new])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                pair.bars[sym] = merged
            except Exception as e:
                self.log.warning("Refresh %s failed: %s", sym, e)

    # =========================================================================
    # Per-pair processing
    # =========================================================================
    async def process_pair(
        self, pair: Pair, macro: RegimeAdjustment,
    ) -> None:
        await self.refresh_pair(pair)
        added = pair.update_hedge_ratio()
        if added == 0 and pair.state == Pair.POS_FLAT:
            self.log.debug("%s: no new bars, nothing to do", pair.name)
            return

        # ---- Combined tradability gate: cointegration + half-life band ----
        tradable, p_val, hl = pair.is_tradable(
            p_threshold=self.cfg.coint_pvalue_threshold,
            recheck_hours=self.cfg.coint_recheck_hours,
            min_half_life_bars=self.cfg.min_half_life_bars,
            max_half_life_bars=self.cfg.max_half_life_bars,
        )
        if not tradable:
            reason = []
            if p_val >= self.cfg.coint_pvalue_threshold: reason.append(f"p={p_val:.3f}")
            if hl < self.cfg.min_half_life_bars: reason.append(f"hl={hl:.1f}<min")
            if hl > self.cfg.max_half_life_bars: reason.append(f"hl={hl:.1f}>max")
            if pair.state != Pair.POS_FLAT:
                self.log.error(
                    "%s TRADABILITY BROKE (%s) while holding -> emergency flat",
                    pair.name, ",".join(reason),
                )
                await self._close_pair(pair, reason="tradability_broke")
            else:
                self.log.info("%s skip: untradable (%s)", pair.name, ",".join(reason))
            return

        # ---- Z-score (no look-ahead) ----
        z = pair.zscore_no_lookahead()
        if z is None or np.isnan(z):
            self.log.warning("%s: insufficient history for Z", pair.name)
            return

        # ---- Stops first (always evaluated before signal logic) ----
        if pair.state != Pair.POS_FLAT and pair.position is not None:
            if await self._check_stops(pair, z):
                return

        # ---- Pair-local trend filter ----
        trend_mult, trend_note = pair.trend_adjustment()

        # ---- Effective entry threshold ----
        eff_entry = self.cfg.z_entry * macro.z_entry_mult * trend_mult

        self.log.info(
            "%s | Z=%+.3f | eff=%.2f | state=%s | beta=%.3f | p_coint=%.3f | hl=%.1fb | %s",
            pair.name, z, eff_entry, pair.state,
            pair.current_beta(), p_val, hl, trend_note,
        )

        # ---- Signal logic, gated by regime + portfolio caps + shortability ----
        if pair.state == Pair.POS_FLAT:
            direction: Optional[str] = None
            if z > eff_entry and not macro.veto_short_spread:
                direction = Pair.POS_SHORT_SPREAD
            elif z < -eff_entry and not macro.veto_long_spread:
                direction = Pair.POS_LONG_SPREAD
            elif (z > eff_entry and macro.veto_short_spread) or \
                 (z < -eff_entry and macro.veto_long_spread):
                self.log.warning("%s signal Z=%.2f vetoed by regime", pair.name, z)
            if direction is not None:
                if not self._portfolio_caps_allow_new_entry(pair):
                    self.log.warning(
                        "%s entry blocked by portfolio caps", pair.name,
                    )
                    return
                ok_short = await self._verify_short_inventory(pair, direction)
                if not ok_short:
                    self.log.warning(
                        "%s entry blocked: insufficient borrow inventory", pair.name,
                    )
                    return
                await self._enter_pair(pair, direction, z, macro)
        else:
            if abs(z) < self.cfg.z_exit:
                await self._close_pair(pair, reason="z_exit")

    # =========================================================================
    # Portfolio-level risk caps
    # =========================================================================
    def _portfolio_snapshot(self) -> Tuple[int, float]:
        """Returns (n_open_pairs, gross_notional_at_entry)."""
        n_open = 0
        gross = 0.0
        for p in self.pairs:
            if p.state == Pair.POS_FLAT or p.position is None:
                continue
            n_open += 1
            gross += abs(p.position.qty_long)  * p.position.avg_price_long
            gross += abs(p.position.qty_short) * p.position.avg_price_short
        return n_open, gross

    def _portfolio_caps_allow_new_entry(self, pair: Pair) -> bool:
        n_open, gross = self._portfolio_snapshot()
        if n_open >= self.cfg.max_concurrent_pairs:
            self.log.warning(
                "Portfolio cap: max_concurrent_pairs=%d hit (open=%d)",
                self.cfg.max_concurrent_pairs, n_open,
            )
            return False
        projected = gross + self.cfg.capital_per_pair
        if projected > self.cfg.max_gross_notional:
            self.log.warning(
                "Portfolio cap: gross would be %.0f > %.0f",
                projected, self.cfg.max_gross_notional,
            )
            return False
        return True

    async def _verify_short_inventory(self, pair: Pair, direction: str) -> bool:
        """Before entry, check the symbol we're about to SELL has borrow."""
        if direction == Pair.POS_SHORT_SPREAD:
            short_sym = pair.long_sym             # we'll SELL the long-leg
            price = pair.bars[short_sym]["close"].iloc[-1]
        else:
            short_sym = pair.short_sym            # we'll SELL the short-leg
            price = pair.bars[short_sym]["close"].iloc[-1]
        # Estimate qty needed (rough; actual sizing happens later)
        est_qty = max(1, int((self.cfg.capital_per_pair / 2.0) // float(price)))
        ok, avail = await self._can_short(short_sym, est_qty)
        if not ok:
            self.log.warning(
                "Shortability veto on %s: need ~%d, have %d (min %d)",
                short_sym, est_qty, avail, self.cfg.min_shortable_shares,
            )
        return ok

    # =========================================================================
    # Reconciliation -- compare tracked state to broker truth
    # =========================================================================
    async def _reconcile_positions(self) -> None:
        """Cross-check every tracked symbol's broker position against the
        Pair's expected qty_long / qty_short. Logs ERROR on drift.

        Symbols are unique-per-pair (enforced at startup) so attribution is
        unambiguous; broker positions in non-tracked symbols are ignored.
        """
        broker: Dict[str, int] = {
            p.contract.symbol: int(p.position) for p in self.ib.positions()
        }
        for pair in self.pairs:
            for sym, expected in (
                (pair.long_sym,
                 pair.position.qty_long  if pair.position else 0),
                (pair.short_sym,
                 pair.position.qty_short if pair.position else 0),
            ):
                actual = broker.get(sym, 0)
                if actual != expected:
                    self.log.error(
                        "RECONCILE DRIFT %s/%s: tracked=%+d broker=%+d (delta=%+d)",
                        pair.name, sym, expected, actual, actual - expected,
                    )

    async def _check_stops(self, pair: Pair, z: float) -> bool:
        meta = pair.position
        if abs(z) >= self.cfg.z_stop:
            self.log.error(
                "%s HARD STOP: |Z|=%.2f >= %.2f", pair.name, abs(z), self.cfg.z_stop,
            )
            await self._close_pair(pair, reason="z_stop")
            return True
        elapsed_h = (datetime.utcnow() - meta.entry_time).total_seconds() / 3600.0
        if elapsed_h >= self.cfg.time_stop_hours:
            self.log.warning(
                "%s TIME STOP: held %.1fh >= %dh",
                pair.name, elapsed_h, self.cfg.time_stop_hours,
            )
            await self._close_pair(pair, reason="time_stop")
            return True
        return False

    # =========================================================================
    # Entry / close wrappers -- update state from FILLS, not order submission
    # =========================================================================
    async def _enter_pair(
        self, pair: Pair, direction: str, z_signal: float, macro: RegimeAdjustment,
    ) -> None:
        meta = await enter_pair_atomic(
            self.ib, pair, direction,
            capital=self.cfg.capital_per_pair,
            size_mult=macro.size_mult,
            fill_timeout_s=self.cfg.fill_timeout_s,
            quote_timeout=self.cfg.quote_timeout,
        )
        if meta is None:
            self.log.warning("%s entry aborted -> staying FLAT", pair.name)
            pair.state = Pair.POS_FLAT
            pair.position = None
            return
        meta.entry_z = z_signal
        pair.position = meta
        pair.state = direction
        self.log.info(
            "%s ENTER %s @ Z=%.2f (beta=%.3f) qty L=%+d S=%+d",
            pair.name, direction, z_signal, meta.entry_beta,
            meta.qty_long, meta.qty_short,
        )

    async def _close_pair(self, pair: Pair, reason: str) -> None:
        if pair.position is None:
            self.log.warning("%s close (%s) but no position meta", pair.name, reason)
            pair.state = Pair.POS_FLAT
            return
        ok = await close_pair_atomic(
            self.ib, pair,
            fill_timeout_s=self.cfg.fill_timeout_s,
            quote_timeout=self.cfg.quote_timeout,
        )
        if ok:
            self.log.info("%s CLOSED (%s)", pair.name, reason)
            pair.position = None
            pair.state = Pair.POS_FLAT
        else:
            self.log.error("%s CLOSE INCOMPLETE (%s) -- state kept until reconciled",
                           pair.name, reason)

    # =========================================================================
    # Emergency failsafe
    # =========================================================================
    async def flatten_portfolio(self) -> None:
        self.log.error("!!! FLATTEN_PORTFOLIO INVOKED !!!")
        # 1. Cancel everything open
        try:
            for trade in list(self.ib.openTrades()):
                try: self.ib.cancelOrder(trade.order)
                except Exception as e:
                    self.log.error("cancel %s failed: %s",
                                   getattr(trade.order, "orderId", "?"), e)
            await asyncio.sleep(1.0)
        except Exception as e:
            self.log.error("cancel sweep failed: %s", e)
        # 2. Reverse every open position in our tracked symbols
        try:
            tracked = set(self.contracts.keys())
            for pos in self.ib.positions():
                sym = pos.contract.symbol
                if sym not in tracked: continue
                qty = int(abs(pos.position))
                if qty == 0: continue
                action = "SELL" if pos.position > 0 else "BUY"
                try:
                    await send_marketable_limit(
                        self.ib, self.contracts[sym], action, qty,
                        quote_timeout=self.cfg.quote_timeout,
                    )
                except Exception as e:
                    self.log.error("Liquidate %s failed: %s", sym, e)
        except Exception as e:
            self.log.error("flatten loop failed: %s", e)
        # 3. Mark all pairs flat regardless
        for p in self.pairs:
            p.state = Pair.POS_FLAT
            p.position = None

    # =========================================================================
    # Main loop
    # =========================================================================
    async def run(self) -> None:
        await self.connect()
        await self.qualify_all()
        await self.load_initial_history()

        try:
            while True:
                try:
                    # Single macro regime read per loop, shared across pairs.
                    # Pass the first pair's bars purely as context (most providers
                    # ignore it; CongressProvider hits its own component baskets).
                    macro = await self.regime.assess(
                        self.pairs[0].bars if self.pairs else {},
                    )

                    for p in self.pairs:
                        try:
                            await self.process_pair(p, macro)
                        except Exception as e:
                            self.log.exception("Pair %s error: %s", p.name, e)

                    if self.cfg.reconcile_each_loop:
                        try:
                            await self._reconcile_positions()
                        except Exception as e:
                            self.log.warning("Reconcile failed: %s", e)

                    await asyncio.sleep(self.cfg.poll_seconds)

                except (asyncio.CancelledError, KeyboardInterrupt):
                    raise
                except Exception as loop_e:
                    self.log.exception("Loop error: %s", loop_e)
                    if not self.ib.isConnected():
                        try:
                            await self.connect()
                            await self.qualify_all()
                        except Exception as r:
                            self.log.error("Reconnect failed: %s", r)
                    await asyncio.sleep(min(60.0, self.cfg.poll_seconds))

        except (KeyboardInterrupt, asyncio.CancelledError):
            self.log.warning("Interrupt -> flattening portfolio")
            await self.flatten_portfolio()
        finally:
            if self.ib.isConnected():
                self.ib.disconnect()
            self.log.info("Disconnected.")


# =============================================================================
# Entrypoint
# =============================================================================
async def _amain() -> None:
    bot = TradingBot(Config(), DEFAULT_PAIRS)
    await bot.run()


if __name__ == "__main__":
    util.patchAsyncio()
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass
