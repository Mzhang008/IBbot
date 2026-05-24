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

    # ---- Cointegration gate ----
    coint_pvalue_threshold: float = 0.05
    coint_recheck_hours: int = 24

    # ---- Capital allocation ----
    capital_per_pair: float = 50_000.0

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
        self.pairs: List[Pair] = [
            Pair(long_sym=l, short_sym=s, lookback=cfg.lookback)
            for l, s in (pair_specs or DEFAULT_PAIRS)
        ]
        self.contracts: Dict[str, Contract] = {}
        self.regime = build_default_analyzer()
        # IB Trade subscriptions for fill events -- bot reacts to broker reality
        self.ib.execDetailsEvent += self._on_exec_details

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

        # Shortability sanity check on the leg(s) we may short
        await self._check_shortability()

    async def _check_shortability(self) -> None:
        """Log warning if a leg cannot be shorted. Strategy needs both directions."""
        try:
            for sym, contract in self.contracts.items():
                t = self.ib.reqMktData(contract, genericTickList="236",
                                       snapshot=False, regulatorySnapshot=False)
                await asyncio.sleep(2.0)
                shortable = getattr(t, "shortableShares", None)
                if shortable is not None and shortable < 1000:
                    self.log.warning(
                        "%s low shortable inventory (%s shares). Borrow may be hard.",
                        sym, shortable,
                    )
                self.ib.cancelMktData(contract)
        except Exception as e:
            self.log.warning("Shortability check failed: %s", e)

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
            p.initialize_hedge_ratio()
            self.log.info("%s init: beta=%.3f, spread=%.5f",
                          p.name, p.current_beta(), p.current_spread())

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

        # ---- Cointegration gate ----
        is_coint, p_val = pair.is_cointegrated(
            self.cfg.coint_pvalue_threshold, self.cfg.coint_recheck_hours,
        )
        if not is_coint:
            if pair.state != Pair.POS_FLAT:
                self.log.error(
                    "%s COINTEGRATION BROKE (p=%.3f) while holding -> emergency flat",
                    pair.name, p_val,
                )
                await self._close_pair(pair, reason="cointegration_broke")
            else:
                self.log.info("%s skip: not cointegrated (p=%.3f)", pair.name, p_val)
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
            "%s | Z=%+.3f | eff=%.2f | state=%s | beta=%.3f | p_coint=%.3f | %s",
            pair.name, z, eff_entry, pair.state,
            pair.current_beta(), p_val, trend_note,
        )

        # ---- Signal logic, gated by regime vetoes ----
        if pair.state == Pair.POS_FLAT:
            if z > eff_entry and not macro.veto_short_spread:
                await self._enter_pair(pair, Pair.POS_SHORT_SPREAD, z, macro)
            elif z < -eff_entry and not macro.veto_long_spread:
                await self._enter_pair(pair, Pair.POS_LONG_SPREAD, z, macro)
            elif (z > eff_entry and macro.veto_short_spread) or \
                 (z < -eff_entry and macro.veto_long_spread):
                self.log.warning("%s signal Z=%.2f vetoed by regime", pair.name, z)
        else:
            if abs(z) < self.cfg.z_exit:
                await self._close_pair(pair, reason="z_exit")

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
