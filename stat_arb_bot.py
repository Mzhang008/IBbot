"""
Mid-Frequency Statistical Arbitrage Trading Bot
SMH (semiconductors ETF) vs IGV (software ETF) -- log-spread mean-reversion
via rolling Z-score, executed against a local IB Gateway (Paper) session.

================================================================================
WINDOWS / IB GATEWAY SETTINGS NOTES
================================================================================
1. Power plan: Settings > System > Power & Battery > Screen and Sleep -> set
   both "Screen" and "Sleep" to "Never" while the bot runs. Also disable
   "USB selective suspend" under Advanced Power Settings, and turn OFF
   Modern Standby if your HW supports legacy S3 (keeps the socket alive).
2. IB Gateway: File > Global Configuration > API > Settings:
       - "Enable ActiveX and Socket Clients"  [CHECKED]
       - "Read-Only API"                      [UNCHECKED]
       - Socket port                          4002  (Paper)
       - Master API client ID                 blank or 0
       - Trusted IPs -> add 127.0.0.1
       - "Download open orders on connection" [CHECKED]
3. IB Gateway: Configure > Settings > Lock and Exit -> enable
   "Auto restart" outside of RTH to avoid the forced daily logout.
4. Pause Windows Update auto-reboot during market hours (Group Policy:
   "No auto-restart with logged on users for scheduled installations").
5. Optional: set python.exe and ibgateway.exe to High priority, pin to
   performance cores via Task Manager > Details > Set affinity.
================================================================================
"""

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from ib_insync import (
    IB,
    Contract,
    LimitOrder,
    Stock,
    Ticker,
    Trade,
    util,
)

# Optional: load .env for regime API keys (QUIVER_API_KEY, FRED_API_KEY,
# NEWSAPI_KEY). If python-dotenv isn't installed, env vars must be set
# externally -- the bot still runs.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from regime import build_default_analyzer, RegimeAdjustment


# =============================================================================
# Configuration -- all tunables centralised in one dataclass.
# =============================================================================
@dataclass
class Config:
    # ---- IB Gateway socket -------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 4002                   # 4002 = Paper Gateway, 4001 = Live
    client_id: int = 17

    # ---- Universe ----------------------------------------------------------
    symbols: Tuple[str, ...] = ("SMH", "IGV")
    long_leg: str = "SMH"              # numerator in the log-spread
    short_leg: str = "IGV"             # denominator in the log-spread

    # ---- Bars / lookback ---------------------------------------------------
    bar_size: str = "1 hour"
    duration: str = "60 D"             # ~ 60 trading days of hourly bars
    use_rth: bool = True
    lookback: int = 100                # rolling window for mu, sigma & variance

    # ---- Signal thresholds -------------------------------------------------
    z_entry: float = 2.0               # |Z| above this => open
    z_exit: float = 0.5                # |Z| below this => "approaching 0" => close

    # ---- Capital / sizing --------------------------------------------------
    capital: float = 100_000.0         # gross dollar pool split via inv-var weights

    # ---- Loop cadence ------------------------------------------------------
    poll_seconds: int = 3600           # one hour, matched to the bar size

    # ---- Robustness --------------------------------------------------------
    market_data_type: int = 3          # 1=live 2=frozen 3=delayed 4=delayed-frozen
    max_retries: int = 5
    backoff_base: float = 2.0
    quote_timeout: float = 5.0


# =============================================================================
# TradingBot
# =============================================================================
class TradingBot:
    """Object-oriented mid-frequency pair-trading bot for IB Gateway."""

    # Position-state constants. The names describe the SPREAD direction:
    #   SHORT_SPREAD = short the LONG-leg, long the SHORT-leg  (entered when Z > +entry)
    #   LONG_SPREAD  = long  the LONG-leg, short the SHORT-leg (entered when Z < -entry)
    POS_FLAT = "FLAT"
    POS_SHORT_SPREAD = "SHORT_SPREAD"
    POS_LONG_SPREAD = "LONG_SPREAD"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ib = IB()
        self.contracts: Dict[str, Contract] = {}
        self.bars: Dict[str, pd.DataFrame] = {}
        self.state: str = self.POS_FLAT
        self.log = self._configure_logger()
        # Regime layer: pulls Congress / news / WSB / FRED in parallel each
        # iteration and returns a RegimeAdjustment used to gate the signal.
        self.regime = build_default_analyzer(
            long_sym=cfg.long_leg, short_sym=cfg.short_leg,
        )

    # ------------------------------------------------------------------ logging
    @staticmethod
    def _configure_logger() -> logging.Logger:
        logger = logging.getLogger("StatArbBot")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    # =========================================================================
    # Exponential-backoff wrapper
    # =========================================================================
    async def _with_retry(self, label: str, coro_factory, *args, **kwargs):
        """
        Run an awaitable factory with exponential backoff.
        Waits 2,4,8,16,32... seconds (base ** attempt) between attempts.
        Designed for transient local-socket drops & IB pacing violations.
        """
        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                return await coro_factory(*args, **kwargs)
            except Exception as exc:  # broad on purpose: socket / pacing / API
                last_err = exc
                wait = self.cfg.backoff_base ** attempt
                self.log.warning(
                    "%s failed (attempt %d/%d): %s -- retrying in %.1fs",
                    label, attempt, self.cfg.max_retries, exc, wait,
                )
                await asyncio.sleep(wait)
        raise RuntimeError(
            f"{label}: exhausted {self.cfg.max_retries} retries"
        ) from last_err

    # =========================================================================
    # Connection & contract qualification
    # =========================================================================
    async def connect(self) -> None:
        """Async connect with backoff; sets market-data-type fallback."""
        async def _do():
            await self.ib.connectAsync(
                self.cfg.host, self.cfg.port, clientId=self.cfg.client_id
            )
            # Use delayed feed if no live entitlement (paper accounts).
            self.ib.reqMarketDataType(self.cfg.market_data_type)
            self.log.info(
                "Connected to IB Gateway %s:%d (clientId=%d, mktDataType=%d)",
                self.cfg.host, self.cfg.port, self.cfg.client_id,
                self.cfg.market_data_type,
            )
        await self._with_retry("connect", _do)

    async def qualify(self) -> None:
        """Resolve canonical conIds for each underlying symbol."""
        async def _do():
            for sym in self.cfg.symbols:
                c = Stock(sym, "SMART", "USD", primaryExchange="ARCA")
                qualified = await self.ib.qualifyContractsAsync(c)
                if not qualified:
                    raise RuntimeError(f"Could not qualify {sym}")
                self.contracts[sym] = qualified[0]
                self.log.info("Qualified %s (conId=%s)", sym, qualified[0].conId)
        await self._with_retry("qualify", _do)

    # =========================================================================
    # Async historical data ingestion
    # =========================================================================
    async def _fetch_one_history(self, symbol: str) -> pd.DataFrame:
        async def _do():
            bars = await self.ib.reqHistoricalDataAsync(
                self.contracts[symbol],
                endDateTime="",                          # now
                durationStr=self.cfg.duration,
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
        return await self._with_retry(f"history[{symbol}]", _do)

    async def load_history(self) -> None:
        """Fetch hourly bars for ALL symbols concurrently."""
        tasks = [self._fetch_one_history(s) for s in self.cfg.symbols]
        results = await asyncio.gather(*tasks)
        for sym, df in zip(self.cfg.symbols, results):
            self.bars[sym] = df
            self.log.info("Loaded %d %s bars (last=%s)", len(df), sym, df.index[-1])

    async def refresh_history(self) -> None:
        """Re-pull bars on each loop iteration; tolerant of transient errors."""
        try:
            await self.load_history()
        except Exception as exc:
            self.log.error("History refresh failed (continuing on cache): %s", exc)

    # =========================================================================
    # Quant layer -- spread, z-score, inverse-variance weights
    # =========================================================================
    def _aligned_closes(self) -> pd.DataFrame:
        long_c = self.bars[self.cfg.long_leg]["close"].rename(self.cfg.long_leg)
        short_c = self.bars[self.cfg.short_leg]["close"].rename(self.cfg.short_leg)
        return pd.concat([long_c, short_c], axis=1).dropna()

    def compute_spread(self) -> pd.Series:
        """
        Spread_t = log(P^L_t) - log(P^S_t)
        Positive spread => long leg has out-performed short leg.
        """
        closes = self._aligned_closes()
        return np.log(closes[self.cfg.long_leg]) - np.log(closes[self.cfg.short_leg])

    def compute_zscore(self, spread: pd.Series) -> float:
        """
        Z_t = (Spread_t - mu) / sigma
        mu, sigma estimated on a rolling window of length `lookback`.
        """
        window = spread.iloc[-self.cfg.lookback:]
        if len(window) < self.cfg.lookback:
            self.log.warning(
                "Insufficient history (%d/%d) for z-score; returning 0.",
                len(window), self.cfg.lookback,
            )
            return 0.0
        mu = window.mean()
        sigma = window.std(ddof=1)
        if sigma <= 0 or np.isnan(sigma):
            return 0.0
        return float((window.iloc[-1] - mu) / sigma)

    def compute_inverse_variance_weights(self) -> Dict[str, float]:
        """
        w_i = (1/sigma_i^2) / Sum_j (1/sigma_j^2)
        sigma_i := stdev of log-returns of leg i over the lookback window.
        Lower-variance leg receives MORE capital -> equalises risk contribution.
        """
        inv_vars: Dict[str, float] = {}
        for sym in self.cfg.symbols:
            log_ret = np.log(self.bars[sym]["close"]).diff().dropna()
            window = log_ret.iloc[-self.cfg.lookback:]
            var = float(window.var(ddof=1))
            inv_vars[sym] = (1.0 / var) if var > 0 else 0.0
        total = sum(inv_vars.values())
        if total <= 0:
            # Degenerate -> equal weights
            n = len(self.cfg.symbols)
            return {s: 1.0 / n for s in self.cfg.symbols}
        return {s: iv / total for s, iv in inv_vars.items()}

    # =========================================================================
    # Quoting -- snapshot bid/ask via reqTickersAsync (one-shot)
    # =========================================================================
    async def _snapshot_quote(self, symbol: str) -> Tuple[float, float, float]:
        """Return (bid, ask, mid). Falls back to last/close when book is NaN."""
        async def _do() -> Tuple[float, float, float]:
            tickers = await asyncio.wait_for(
                self.ib.reqTickersAsync(self.contracts[symbol]),
                timeout=self.cfg.quote_timeout,
            )
            if not tickers:
                raise RuntimeError(f"No ticker returned for {symbol}")
            t: Ticker = tickers[0]
            bid = t.bid if t.bid and not math.isnan(t.bid) else None
            ask = t.ask if t.ask and not math.isnan(t.ask) else None
            fallback = (
                t.last  if (t.last  and not math.isnan(t.last))  else
                t.close if (t.close and not math.isnan(t.close)) else
                None
            )
            if bid is None and fallback is not None:
                bid = fallback
            if ask is None and fallback is not None:
                ask = fallback
            if bid is None or ask is None:
                raise RuntimeError(f"No usable quote for {symbol}")
            return float(bid), float(ask), float((bid + ask) / 2.0)
        return await self._with_retry(f"quote[{symbol}]", _do)

    # =========================================================================
    # Order layer -- STRICT LimitOrder (marketable: cross the touch)
    # =========================================================================
    async def place_limit(
        self, symbol: str, action: str, quantity: int
    ) -> Optional[Trade]:
        """
        Marketable-LimitOrder:
          BUY  -> price = current ASK
          SELL -> price = current BID
        Strictly a LimitOrder (per spec); marketable price gives high fill
        probability without exposing us to a true MarketOrder slippage tail.
        """
        if quantity <= 0:
            self.log.warning("place_limit skipped: qty<=0 for %s %s", action, symbol)
            return None
        bid, ask, _mid = await self._snapshot_quote(symbol)
        price = round(ask if action == "BUY" else bid, 2)
        order = LimitOrder(
            action=action,
            totalQuantity=quantity,
            lmtPrice=price,
            tif="DAY",
        )
        trade = self.ib.placeOrder(self.contracts[symbol], order)
        self.log.info(
            "ORDER %s %d %s @ %.2f  (bid=%.2f ask=%.2f)",
            action, quantity, symbol, price, bid, ask,
        )
        return trade

    # =========================================================================
    # Position management
    # =========================================================================
    async def enter_position(self, direction: str, size_mult: float = 1.0) -> None:
        """
        Open the pair according to the spread direction:
          POS_SHORT_SPREAD -> SELL long_leg, BUY short_leg   (Z > +entry)
          POS_LONG_SPREAD  -> BUY  long_leg, SELL short_leg  (Z < -entry)
        Sizing is inverse-variance weighted, then scaled by `size_mult`
        coming from the regime layer (e.g. VIX-based de-risking).
        """
        if direction not in (self.POS_SHORT_SPREAD, self.POS_LONG_SPREAD):
            raise ValueError(direction)

        weights = self.compute_inverse_variance_weights()

        # Take both quotes BEFORE sending orders so qty is consistent.
        _, _, mid_long = await self._snapshot_quote(self.cfg.long_leg)
        _, _, mid_short = await self._snapshot_quote(self.cfg.short_leg)

        # Regime-scaled gross exposure
        gross = self.cfg.capital * max(0.0, min(1.0, size_mult))
        notional_long  = gross * weights[self.cfg.long_leg]
        notional_short = gross * weights[self.cfg.short_leg]
        qty_long  = max(1, int(notional_long  // mid_long))
        qty_short = max(1, int(notional_short // mid_short))

        self.log.info(
            "ENTRY %s | size_mult=%.2f | w=%s | notional L=%.0f S=%.0f | qty L=%d S=%d",
            direction, size_mult,
            {k: round(v, 4) for k, v in weights.items()},
            notional_long, notional_short, qty_long, qty_short,
        )

        if direction == self.POS_SHORT_SPREAD:
            # Long-leg over-performed => short it; cover with long on short-leg.
            await self.place_limit(self.cfg.long_leg, "SELL", qty_long)
            await self.place_limit(self.cfg.short_leg, "BUY",  qty_short)
        else:  # POS_LONG_SPREAD
            await self.place_limit(self.cfg.long_leg, "BUY",  qty_long)
            await self.place_limit(self.cfg.short_leg, "SELL", qty_short)

        self.state = direction

    async def close_positions(self) -> None:
        """Flatten whatever the broker reports as open for our symbols."""
        for pos in self.ib.positions():
            sym = pos.contract.symbol
            if sym not in self.cfg.symbols:
                continue
            qty = int(abs(pos.position))
            if qty == 0:
                continue
            action = "SELL" if pos.position > 0 else "BUY"
            await self.place_limit(sym, action, qty)
        self.state = self.POS_FLAT
        self.log.info("Closed pair positions; state=FLAT")

    # =========================================================================
    # Emergency failsafe
    # =========================================================================
    async def flatten_portfolio(self) -> None:
        """
        Emergency stop:
          1. Cancel every open order.
          2. Liquidate every position in tracked symbols at marketable limits.
        Each step wrapped so a partial failure does NOT abort the cascade.
        """
        self.log.error("!!! FLATTEN_PORTFOLIO INVOKED !!!")
        # ---- 1. Cancel open orders ----
        try:
            for trade in list(self.ib.openTrades()):
                try:
                    self.ib.cancelOrder(trade.order)
                except Exception as e:
                    self.log.error(
                        "Cancel failed for orderId=%s: %s",
                        getattr(trade.order, "orderId", "?"), e,
                    )
            await asyncio.sleep(1.0)
        except Exception as e:
            self.log.error("Cancel-all loop failed: %s", e)
        # ---- 2. Liquidate -----------
        try:
            await self.close_positions()
        except Exception as e:
            self.log.error("Liquidation failed: %s", e)

    # =========================================================================
    # Main control loop
    # =========================================================================
    async def run(self) -> None:
        await self.connect()
        await self.qualify()
        await self.load_history()

        try:
            while True:
                try:
                    await self.refresh_history()
                    spread = self.compute_spread()
                    z = self.compute_zscore(spread)

                    # --- Regime overlay: external data adjusts threshold/size ---
                    adj: RegimeAdjustment = await self.regime.assess(self.bars)
                    eff_entry = self.cfg.z_entry * adj.z_entry_mult

                    self.log.info(
                        "tick=%s | Z=%.3f | eff_entry=%.2f | state=%s | "
                        "size_mult=%.2f vetoS=%s vetoL=%s",
                        datetime.utcnow().isoformat(timespec="seconds"),
                        z, eff_entry, self.state,
                        adj.size_mult, adj.veto_short_spread, adj.veto_long_spread,
                    )

                    # --- Signal logic, gated by regime ---
                    if self.state == self.POS_FLAT:
                        if z > eff_entry and not adj.veto_short_spread:
                            await self.enter_position(
                                self.POS_SHORT_SPREAD, size_mult=adj.size_mult,
                            )
                        elif z < -eff_entry and not adj.veto_long_spread:
                            await self.enter_position(
                                self.POS_LONG_SPREAD, size_mult=adj.size_mult,
                            )
                        else:
                            if (z > eff_entry and adj.veto_short_spread) or \
                               (z < -eff_entry and adj.veto_long_spread):
                                self.log.warning(
                                    "Signal Z=%.2f suppressed by regime veto.", z,
                                )
                    else:
                        if abs(z) < self.cfg.z_exit:
                            await self.close_positions()

                    await asyncio.sleep(self.cfg.poll_seconds)

                except (asyncio.CancelledError, KeyboardInterrupt):
                    raise
                except Exception as loop_exc:
                    # Transient drop -> log, attempt reconnect, continue.
                    self.log.exception("Loop iteration failed: %s", loop_exc)
                    if not self.ib.isConnected():
                        try:
                            await self.connect()
                            await self.qualify()
                        except Exception as recon_e:
                            self.log.error("Reconnect failed: %s", recon_e)
                    await asyncio.sleep(min(60.0, self.cfg.poll_seconds))

        except (KeyboardInterrupt, asyncio.CancelledError):
            self.log.warning("Interrupt received -> flattening portfolio.")
            await self.flatten_portfolio()
        finally:
            if self.ib.isConnected():
                self.ib.disconnect()
            self.log.info("Disconnected. Shutdown complete.")


# =============================================================================
# Entrypoint
# =============================================================================
async def _amain() -> None:
    bot = TradingBot(Config())
    await bot.run()


if __name__ == "__main__":
    util.patchAsyncio()       # required for nested loop usage on Windows
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass
