"""
Regime analyzer + external-data providers for the SMH/IGV stat-arb bot.

The base Z-score logic is mean-reversion. In the current bull market this is
dangerous because:
   - Congressional buying in NVDA/MSFT/etc is informed flow leaning long
   - WSB / Twitter pile-ons drive multi-day momentum that fades Z-fading bots
   - AI/semi concentration means a Z-spike is often *real* repricing, not noise

This module produces a RegimeAdjustment that the bot consumes to:
   - Raise the entry threshold (z_entry_mult > 1) when conditions warn against
     fading the move
   - Shrink position size (size_mult < 1) in fragile or high-vol regimes
   - VETO a side entirely when informed/retail flow strongly contradicts it

All providers are independent, run concurrently, and degrade gracefully.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import aiohttp
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Hard-coded top holdings (refresh every ~quarter from issuer prospectus).
# These are the names through which the macro/sentiment/flow signals propagate
# back into the SMH/IGV spread.
# -----------------------------------------------------------------------------
SMH_TOP_HOLDINGS = [
    "NVDA", "TSM", "AVGO", "AMD", "ASML",
    "QCOM", "AMAT", "MU",   "LRCX", "MRVL",
]
IGV_TOP_HOLDINGS = [
    "ORCL", "MSFT", "CRM",  "ADBE", "INTU",
    "PLTR", "NOW",  "SHOP", "SNOW", "FTNT",
]


# =============================================================================
# RegimeAdjustment -- the contract between regime layer and trading loop
# =============================================================================
@dataclass
class RegimeAdjustment:
    z_entry_mult: float = 1.0          # multiplier on |Z| entry threshold
    size_mult: float = 1.0             # multiplier on capital allocation
    veto_short_spread: bool = False    # block SELL long_leg + BUY  short_leg
    veto_long_spread:  bool = False    # block BUY  long_leg + SELL short_leg
    notes: List[str] = field(default_factory=list)

    def merge(self, other: "RegimeAdjustment") -> "RegimeAdjustment":
        return RegimeAdjustment(
            z_entry_mult=self.z_entry_mult * other.z_entry_mult,
            size_mult=self.size_mult * other.size_mult,
            veto_short_spread=self.veto_short_spread or other.veto_short_spread,
            veto_long_spread=self.veto_long_spread or other.veto_long_spread,
            notes=self.notes + other.notes,
        )


# =============================================================================
# Base provider interface
# =============================================================================
class RegimeProvider(ABC):
    name: str = "base"

    @property
    @abstractmethod
    def enabled(self) -> bool: ...

    @abstractmethod
    async def assess(
        self,
        session: aiohttp.ClientSession,
        bars: Dict[str, pd.DataFrame],
    ) -> RegimeAdjustment: ...


# =============================================================================
# 1. Trend filter (no external API, uses bar history)
# -----------------------------------------------------------------------------
# In a strong tech bull market (both legs above SMA200), short-spread trades
# fade an uptrend -- raise the bar. Symmetric handling for downtrends.
# =============================================================================
class TrendProvider(RegimeProvider):
    name = "trend"

    def __init__(self, long_sym: str = "SMH", short_sym: str = "IGV",
                 sma_window: int = 200):
        self.long_sym = long_sym
        self.short_sym = short_sym
        self.sma_window = sma_window

    @property
    def enabled(self) -> bool:
        return True

    async def assess(self, session, bars):
        adj = RegimeAdjustment()
        long_close  = bars.get(self.long_sym,  pd.DataFrame()).get("close")
        short_close = bars.get(self.short_sym, pd.DataFrame()).get("close")
        if long_close is None or short_close is None:
            return adj
        if len(long_close)  < self.sma_window: return adj
        if len(short_close) < self.sma_window: return adj

        long_sma  = long_close.iloc[-self.sma_window:].mean()
        short_sma = short_close.iloc[-self.sma_window:].mean()
        long_up   = long_close.iloc[-1]  > long_sma
        short_up  = short_close.iloc[-1] > short_sma

        if long_up and short_up:
            # Tech bull regime -- bias against fading the over-performer
            adj.z_entry_mult *= 1.25
            adj.notes.append("trend:both_uptrend->z_mult*1.25")
        elif (not long_up) and (not short_up):
            adj.z_entry_mult *= 1.15
            adj.notes.append("trend:both_downtrend->z_mult*1.15")
        return adj


# =============================================================================
# 2. Congressional flow (Quiver preferred, senate/house-stock-watcher fallback)
# -----------------------------------------------------------------------------
# Tracks net buy minus sell over a 30d window across each ETF's top holdings.
# Significant net buying = informed bullish flow -> veto shorting that leg.
# =============================================================================
class CongressProvider(RegimeProvider):
    name = "congress"
    QUIVER_BASE = "https://api.quiverquant.com/beta"
    SSW_URL = ("https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com"
               "/aggregate/all_transactions.json")
    HSW_URL = ("https://house-stock-watcher-data.s3-us-west-2.amazonaws.com"
               "/data/all_transactions.json")

    def __init__(self, quiver_api_key: Optional[str] = None,
                 lookback_days: int = 30, sig_threshold: int = 5):
        self.quiver_api_key = quiver_api_key
        self.lookback_days = lookback_days
        self.sig_threshold = sig_threshold
        self._ssw_cache: Optional[pd.DataFrame] = None
        self._ssw_cache_ts: float = 0.0

    @property
    def enabled(self) -> bool:
        return True  # free fallback always works

    async def _fetch_quiver(self, session, ticker) -> Optional[List[dict]]:
        if not self.quiver_api_key:
            return None
        headers = {"Authorization": f"Bearer {self.quiver_api_key}",
                   "Accept": "application/json"}
        url = f"{self.QUIVER_BASE}/historical/congresstrading/{ticker}"
        try:
            async with session.get(url, headers=headers, timeout=10) as r:
                if r.status != 200:
                    return None
                return await r.json()
        except Exception:
            return None

    async def _fetch_ssw_bulk(self, session) -> Optional[pd.DataFrame]:
        # Cache the bulk file for 1 hour to be polite.
        now = datetime.utcnow().timestamp()
        if self._ssw_cache is not None and (now - self._ssw_cache_ts) < 3600:
            return self._ssw_cache
        rows: List[dict] = []
        for url in (self.SSW_URL, self.HSW_URL):
            try:
                async with session.get(url, timeout=30) as r:
                    if r.status == 200:
                        rows.extend(await r.json())
            except Exception:
                continue
        if not rows:
            return None
        df = pd.DataFrame(rows)
        # Normalise the two schemas (senate vs house) into one.
        date_col = "transaction_date" if "transaction_date" in df.columns else "transactionDate"
        type_col = "type" if "type" in df.columns else "transactionType"
        df = df.rename(columns={date_col: "date", type_col: "type"})
        df["date"]   = pd.to_datetime(df["date"], errors="coerce")
        df["ticker"] = df.get("ticker", pd.Series([None]*len(df))).astype(str).str.upper()
        df = df.dropna(subset=["date", "ticker", "type"])
        self._ssw_cache    = df
        self._ssw_cache_ts = now
        return df

    @staticmethod
    def _quiver_net(records: List[dict], lookback_days: int) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        net = 0
        for r in records:
            dt_s = r.get("TransactionDate") or r.get("transaction_date") or ""
            ttype = (r.get("Transaction") or r.get("type") or "").lower()
            try:
                dt = pd.to_datetime(dt_s, utc=True)
            except Exception:
                continue
            if dt < cutoff:
                continue
            if "purchase" in ttype or "buy" in ttype:
                net += 1
            elif "sale" in ttype or "sell" in ttype:
                net -= 1
        return net

    async def assess(self, session, bars):
        adj = RegimeAdjustment()
        smh_net, igv_net = 0, 0

        if self.quiver_api_key:
            tasks_smh = [self._fetch_quiver(session, t) for t in SMH_TOP_HOLDINGS]
            tasks_igv = [self._fetch_quiver(session, t) for t in IGV_TOP_HOLDINGS]
            res_smh = await asyncio.gather(*tasks_smh, return_exceptions=False)
            res_igv = await asyncio.gather(*tasks_igv, return_exceptions=False)
            for recs in res_smh:
                if recs: smh_net += self._quiver_net(recs, self.lookback_days)
            for recs in res_igv:
                if recs: igv_net += self._quiver_net(recs, self.lookback_days)
        else:
            df = await self._fetch_ssw_bulk(session)
            if df is None or df.empty:
                return adj
            cutoff = datetime.utcnow() - timedelta(days=self.lookback_days)
            recent = df[df["date"] >= cutoff].copy()
            buys = recent["type"].str.lower().str.contains("purchase|buy", na=False)
            recent["sign"] = np.where(buys, 1, -1)
            smh_net = int(recent[recent["ticker"].isin(SMH_TOP_HOLDINGS)]["sign"].sum())
            igv_net = int(recent[recent["ticker"].isin(IGV_TOP_HOLDINGS)]["sign"].sum())

        adj.notes.append(f"congress:smh_net={smh_net},igv_net={igv_net}")

        sig = self.sig_threshold
        # Heavy net buying in long-leg components -> don't short the long leg
        if smh_net >= sig:
            adj.veto_short_spread = True
            adj.notes.append("congress:smh_net_buy->veto_short_spread")
        if igv_net >= sig:
            adj.veto_long_spread = True
            adj.notes.append("congress:igv_net_buy->veto_long_spread")
        if smh_net <= -sig:
            adj.veto_long_spread = True
            adj.notes.append("congress:smh_net_sell->veto_long_spread")
        if igv_net <= -sig:
            adj.veto_short_spread = True
            adj.notes.append("congress:igv_net_sell->veto_short_spread")
        return adj


# =============================================================================
# 3. FRED -- macro overlay (VIX + financial conditions)
# =============================================================================
class FredProvider(RegimeProvider):
    name = "fred"
    BASE = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def _last_value(self, session, series_id: str) -> Optional[float]:
        params = {
            "series_id":  series_id,
            "api_key":    self.api_key,
            "file_type":  "json",
            "sort_order": "desc",
            "limit":      5,
        }
        try:
            async with session.get(self.BASE, params=params, timeout=10) as r:
                if r.status != 200:
                    return None
                j = await r.json()
                for obs in j.get("observations", []):
                    v = obs.get("value")
                    if v not in (None, ".", ""):
                        return float(v)
                return None
        except Exception:
            return None

    async def assess(self, session, bars):
        adj = RegimeAdjustment()
        vix, nfci = await asyncio.gather(
            self._last_value(session, "VIXCLS"),
            self._last_value(session, "NFCI"),
        )
        adj.notes.append(f"fred:vix={vix},nfci={nfci}")
        if vix is not None:
            if vix < 13:
                # Complacent regime -- mean reversion fragile, shrink size
                adj.size_mult *= 0.75
                adj.notes.append("fred:vix<13->size*0.75")
            elif vix > 25:
                adj.size_mult *= 0.5
                adj.notes.append("fred:vix>25->size*0.5")
        if nfci is not None and nfci > 0:
            # Tight conditions -- require stronger signal
            adj.z_entry_mult *= 1.15
            adj.notes.append("fred:nfci>0->z_mult*1.15")
        return adj


# =============================================================================
# 4. News sentiment via NewsAPI (lexicon scorer)
# -----------------------------------------------------------------------------
# Crude but useful: extreme one-sided news flow on a leg => don't fade it.
# =============================================================================
class NewsProvider(RegimeProvider):
    name = "news"
    BASE = "https://newsapi.org/v2/everything"

    BULL = re.compile(
        r"\b(rally|surge|soar|beat|jumps?|gains?|record|strong|upgrade|"
        r"breakthrough|bullish|outperform|tops?|blowout)\b", re.I,
    )
    BEAR = re.compile(
        r"\b(crash|plunge|fall|miss(ed)?|weak|drops?|declines?|downgrade|"
        r"warning|bearish|underperform|bubble|risk|tumbl|sell-?off)\b", re.I,
    )
    SMH_Q = ("(semiconductor OR \"AI chip\" OR Nvidia OR TSMC OR "
             "ASML OR AMD OR Broadcom)")
    IGV_Q = ("(\"enterprise software\" OR \"AI software\" OR Microsoft OR "
             "Oracle OR Salesforce OR Palantir OR ServiceNow)")

    def __init__(self, api_key: Optional[str], hours_back: int = 24):
        self.api_key    = api_key
        self.hours_back = hours_back

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def _score(self, session, query: str) -> float:
        since = (datetime.utcnow() - timedelta(hours=self.hours_back))\
                    .strftime("%Y-%m-%dT%H:%M:%S")
        params = {
            "q": query, "from": since, "language": "en",
            "sortBy": "publishedAt", "pageSize": 100, "apiKey": self.api_key,
        }
        try:
            async with session.get(self.BASE, params=params, timeout=10) as r:
                if r.status != 200:
                    return 0.0
                j = await r.json()
                bull = bear = 0
                for a in j.get("articles", []):
                    txt = (a.get("title") or "") + " " + (a.get("description") or "")
                    bull += len(self.BULL.findall(txt))
                    bear += len(self.BEAR.findall(txt))
                tot = bull + bear
                return (bull - bear) / tot if tot > 0 else 0.0
        except Exception:
            return 0.0

    async def assess(self, session, bars):
        adj = RegimeAdjustment()
        smh_s, igv_s = await asyncio.gather(
            self._score(session, self.SMH_Q),
            self._score(session, self.IGV_Q),
        )
        diff = smh_s - igv_s
        adj.notes.append(f"news:smh={smh_s:+.2f},igv={igv_s:+.2f},diff={diff:+.2f}")
        if smh_s > 0.4:
            adj.veto_short_spread = True
            adj.notes.append("news:smh_extreme_bull->veto_short_spread")
        if igv_s > 0.4:
            adj.veto_long_spread = True
            adj.notes.append("news:igv_extreme_bull->veto_long_spread")
        if abs(diff) > 0.3:
            adj.z_entry_mult *= 1.15
            adj.notes.append("news:asymmetric->z_mult*1.15")
        return adj


# =============================================================================
# 5. Reddit / WSB pile-on detection via Quiver
# =============================================================================
class WSBProvider(RegimeProvider):
    name = "wsb"
    QUIVER_BASE = "https://api.quiverquant.com/beta"

    def __init__(self, quiver_api_key: Optional[str],
                 pile_on_threshold: int = 500):
        self.quiver_api_key   = quiver_api_key
        self.pile_on_threshold = pile_on_threshold

    @property
    def enabled(self) -> bool:
        return bool(self.quiver_api_key)

    async def assess(self, session, bars):
        adj = RegimeAdjustment()
        headers = {"Authorization": f"Bearer {self.quiver_api_key}"}
        try:
            async with session.get(
                f"{self.QUIVER_BASE}/live/wallstreetbets",
                headers=headers, timeout=10,
            ) as r:
                if r.status != 200:
                    return adj
                data = await r.json()
        except Exception:
            return adj

        idx = {row.get("Ticker", "").upper(): row for row in data
               if "Ticker" in row}
        smh_m = sum(idx.get(t, {}).get("Mentions", 0) for t in SMH_TOP_HOLDINGS)
        igv_m = sum(idx.get(t, {}).get("Mentions", 0) for t in IGV_TOP_HOLDINGS)
        adj.notes.append(f"wsb:smh={smh_m},igv={igv_m}")

        if smh_m > self.pile_on_threshold:
            adj.veto_short_spread = True
            adj.notes.append("wsb:smh_pile_on->veto_short_spread")
        if igv_m > self.pile_on_threshold:
            adj.veto_long_spread = True
            adj.notes.append("wsb:igv_pile_on->veto_long_spread")
        return adj


# =============================================================================
# Orchestrator
# =============================================================================
class RegimeAnalyzer:
    """Runs all enabled providers in parallel and merges their adjustments."""

    def __init__(self, providers: List[RegimeProvider]):
        self.providers = [p for p in providers if p.enabled]
        self.log = logging.getLogger("Regime")
        self.log.info("Regime providers active: %s",
                      [p.name for p in self.providers])

    async def assess(self, bars: Dict[str, pd.DataFrame]) -> RegimeAdjustment:
        if not self.providers:
            return RegimeAdjustment()
        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(
                *[p.assess(session, bars) for p in self.providers],
                return_exceptions=True,
            )
        merged = RegimeAdjustment()
        for p, r in zip(self.providers, results):
            if isinstance(r, Exception):
                self.log.warning("Provider %s failed: %s", p.name, r)
                continue
            merged = merged.merge(r)
        self.log.info(
            "Regime: z_mult=%.2f size_mult=%.2f veto_S=%s veto_L=%s | notes=%s",
            merged.z_entry_mult, merged.size_mult,
            merged.veto_short_spread, merged.veto_long_spread,
            "; ".join(merged.notes),
        )
        return merged


def build_default_analyzer(long_sym: str = "SMH",
                           short_sym: str = "IGV") -> RegimeAnalyzer:
    """Read env vars, wire up whatever providers are configured."""
    quiver  = os.getenv("QUIVER_API_KEY") or None
    fred    = os.getenv("FRED_API_KEY")   or None
    newsapi = os.getenv("NEWSAPI_KEY")    or None
    providers: List[RegimeProvider] = [
        TrendProvider(long_sym=long_sym, short_sym=short_sym, sma_window=200),
        CongressProvider(quiver_api_key=quiver, lookback_days=30),
        FredProvider(api_key=fred),
        NewsProvider(api_key=newsapi, hours_back=24),
        WSBProvider(quiver_api_key=quiver),
    ]
    return RegimeAnalyzer(providers)
