"""
Atomic two-leg execution with watchdog.

Solves the race condition in the old enter_position():
   send leg A -> network blip -> leg B fails -> naked directional exposure.

Strategy:
   1. Fire both LimitOrders in the SAME tick (no awaits between placements).
   2. Wait up to `fill_timeout_s` for both to fill.
   3. If only one filled, IMMEDIATELY:
        - Cancel the unfilled leg.
        - Unwind any partial fill on the filled leg at marketable limit.
   4. Compute PositionMeta from the actual Trade.fills, not the order params.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from ib_insync import IB, Contract, LimitOrder, Ticker, Trade

from pairs import Pair, PositionMeta

log = logging.getLogger("Execution")


# =============================================================================
# Quote helper -- snapshot via reqTickersAsync, fallback to last/close
# =============================================================================
async def snapshot_quote(
    ib: IB, contract: Contract, timeout: float = 5.0,
) -> Tuple[float, float, float]:
    tickers = await asyncio.wait_for(
        ib.reqTickersAsync(contract), timeout=timeout,
    )
    if not tickers:
        raise RuntimeError(f"No ticker for {contract.symbol}")
    t: Ticker = tickers[0]
    bid = t.bid if t.bid and not math.isnan(t.bid) else None
    ask = t.ask if t.ask and not math.isnan(t.ask) else None
    fb = (
        t.last  if (t.last  and not math.isnan(t.last))  else
        t.close if (t.close and not math.isnan(t.close)) else None
    )
    if bid is None and fb is not None: bid = fb
    if ask is None and fb is not None: ask = fb
    if bid is None or ask is None:
        raise RuntimeError(f"No usable quote for {contract.symbol}")
    return float(bid), float(ask), float((bid + ask) / 2.0)


def _filled_qty(trade: Trade) -> int:
    return int(sum(f.execution.shares for f in trade.fills))


def _avg_fill_price(trade: Trade) -> float:
    total_sh = sum(f.execution.shares for f in trade.fills)
    if total_sh == 0:
        return 0.0
    total_val = sum(f.execution.shares * f.execution.price for f in trade.fills)
    return float(total_val / total_sh)


async def _wait_for_terminal(trade: Trade, deadline: float) -> None:
    """Wait until the trade is done OR deadline reached, without sleep-polling."""
    while time.monotonic() < deadline and not trade.isDone():
        # Use ib.waitOnUpdate via asyncio.sleep with short tick
        await asyncio.sleep(0.25)


# =============================================================================
# Send a marketable limit (BUY at ask, SELL at bid). Strictly LimitOrder.
# =============================================================================
async def send_marketable_limit(
    ib: IB,
    contract: Contract,
    action: str,
    quantity: int,
    quote_timeout: float = 5.0,
) -> Optional[Trade]:
    if quantity <= 0:
        return None
    bid, ask, _ = await snapshot_quote(ib, contract, timeout=quote_timeout)
    price = round(ask if action == "BUY" else bid, 2)
    order = LimitOrder(action=action, totalQuantity=quantity,
                       lmtPrice=price, tif="DAY")
    trade = ib.placeOrder(contract, order)
    log.info("ORDER %s %d %s @ %.2f (bid=%.2f ask=%.2f)",
             action, quantity, contract.symbol, price, bid, ask)
    return trade


# =============================================================================
# Atomic pair entry
# =============================================================================
async def enter_pair_atomic(
    ib: IB,
    pair: Pair,
    direction: str,
    capital: float,
    size_mult: float,
    fill_timeout_s: float = 30.0,
    quote_timeout: float = 5.0,
) -> Optional[PositionMeta]:
    """
    Returns PositionMeta on success (both legs filled, even partially), or
    None if the pair could not be entered cleanly (e.g. one leg unwound).
    """
    if direction == Pair.POS_SHORT_SPREAD:
        action_long, action_short = "SELL", "BUY"
    elif direction == Pair.POS_LONG_SPREAD:
        action_long, action_short = "BUY", "SELL"
    else:
        raise ValueError(direction)

    # Fresh quotes for both legs (in parallel)
    (bL, aL, _), (bS, aS, _) = await asyncio.gather(
        snapshot_quote(ib, pair.long_contract,  quote_timeout),
        snapshot_quote(ib, pair.short_contract, quote_timeout),
    )

    # Beta-hedge-neutral sizing, then regime scale
    gross = capital * max(0.0, min(1.0, size_mult))
    qty_long, qty_short = pair.hedge_neutral_sizing(
        gross,
        price_long=aL if action_long == "BUY" else bL,
        price_short=aS if action_short == "BUY" else bS,
    )

    price_long  = round(aL if action_long  == "BUY" else bL, 2)
    price_short = round(aS if action_short == "BUY" else bS, 2)

    order_long  = LimitOrder(action_long,  qty_long,  price_long,  tif="DAY")
    order_short = LimitOrder(action_short, qty_short, price_short, tif="DAY")

    # ----- Fire BOTH in the SAME tick, no await between them -----
    trade_long  = ib.placeOrder(pair.long_contract,  order_long)
    trade_short = ib.placeOrder(pair.short_contract, order_short)
    log.info(
        "%s atomic-entry %s | L=%s %d @ %.2f | S=%s %d @ %.2f",
        pair.name, direction,
        action_long,  qty_long,  price_long,
        action_short, qty_short, price_short,
    )

    # ----- Watchdog -----
    deadline = time.monotonic() + fill_timeout_s
    await asyncio.gather(
        _wait_for_terminal(trade_long,  deadline),
        _wait_for_terminal(trade_short, deadline),
    )

    fl, fs = _filled_qty(trade_long), _filled_qty(trade_short)
    pl, ps = _avg_fill_price(trade_long), _avg_fill_price(trade_short)

    # ----- Imbalance handling -----
    if fl > 0 and fs == 0:
        log.error("%s LEG IMBALANCE: long filled %d, short 0 -- unwinding long",
                  pair.name, fl)
        try: ib.cancelOrder(order_short)
        except Exception as e: log.error("cancel short failed: %s", e)
        await asyncio.sleep(0.5)
        unwind = "SELL" if action_long == "BUY" else "BUY"
        await send_marketable_limit(ib, pair.long_contract, unwind, fl, quote_timeout)
        return None

    if fs > 0 and fl == 0:
        log.error("%s LEG IMBALANCE: short filled %d, long 0 -- unwinding short",
                  pair.name, fs)
        try: ib.cancelOrder(order_long)
        except Exception as e: log.error("cancel long failed: %s", e)
        await asyncio.sleep(0.5)
        unwind = "SELL" if action_short == "BUY" else "BUY"
        await send_marketable_limit(ib, pair.short_contract, unwind, fs, quote_timeout)
        return None

    if fl == 0 and fs == 0:
        log.warning("%s atomic-entry: NO fills within %.0fs -- cancelling both",
                    pair.name, fill_timeout_s)
        for o in (order_long, order_short):
            try: ib.cancelOrder(o)
            except Exception: pass
        return None

    # Both legs filled (possibly partially) -- record actual fills
    meta = PositionMeta(
        direction=direction,
        entry_time=datetime.utcnow(),
        entry_z=float("nan"),       # filled in by caller
        entry_spread=pair.current_spread(),
        entry_beta=pair.current_beta(),
        qty_long=fl  if action_long  == "BUY"  else -fl,
        qty_short=fs if action_short == "BUY"  else -fs,
        avg_price_long=pl,
        avg_price_short=ps,
    )
    log.info(
        "%s ENTRY confirmed: L=%+d @ %.2f | S=%+d @ %.2f",
        pair.name, meta.qty_long, pl, meta.qty_short, ps,
    )
    return meta


# =============================================================================
# Atomic pair close -- reverses meta.qty_long and meta.qty_short exactly
# =============================================================================
async def close_pair_atomic(
    ib: IB,
    pair: Pair,
    fill_timeout_s: float = 30.0,
    quote_timeout: float = 5.0,
) -> bool:
    """Returns True if both legs successfully closed."""
    if pair.position is None:
        log.warning("%s close: no PositionMeta", pair.name)
        return True

    meta = pair.position
    # Reverse signs
    qty_long  = abs(meta.qty_long)
    qty_short = abs(meta.qty_short)
    action_long  = "SELL" if meta.qty_long  > 0 else "BUY"
    action_short = "SELL" if meta.qty_short > 0 else "BUY"

    (bL, aL, _), (bS, aS, _) = await asyncio.gather(
        snapshot_quote(ib, pair.long_contract,  quote_timeout),
        snapshot_quote(ib, pair.short_contract, quote_timeout),
    )
    price_long  = round(aL if action_long  == "BUY" else bL, 2)
    price_short = round(aS if action_short == "BUY" else bS, 2)

    order_long  = LimitOrder(action_long,  qty_long,  price_long,  tif="DAY")
    order_short = LimitOrder(action_short, qty_short, price_short, tif="DAY")

    trade_long  = ib.placeOrder(pair.long_contract,  order_long)
    trade_short = ib.placeOrder(pair.short_contract, order_short)
    log.info(
        "%s atomic-close | L=%s %d @ %.2f | S=%s %d @ %.2f",
        pair.name,
        action_long,  qty_long,  price_long,
        action_short, qty_short, price_short,
    )

    deadline = time.monotonic() + fill_timeout_s
    await asyncio.gather(
        _wait_for_terminal(trade_long,  deadline),
        _wait_for_terminal(trade_short, deadline),
    )
    fl, fs = _filled_qty(trade_long), _filled_qty(trade_short)
    if fl < qty_long or fs < qty_short:
        log.error("%s CLOSE INCOMPLETE: long %d/%d short %d/%d",
                  pair.name, fl, qty_long, fs, qty_short)
        return False
    return True
