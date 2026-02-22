"""
scanner.py - Symbol Scanner and Ranker
Scores symbols using EMA trend strength, ATR expansion, momentum,
volume spike, and relative strength vs BTC. Returns ranked list of top candidates.
"""
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import ta

from config import get_config
from logger import get_logger

log = get_logger("scanner")


@dataclass
class SymbolScore:
    symbol: str
    score: float = 0.0
    ema_score: float = 0.0
    atr_score: float = 0.0
    momentum_score: float = 0.0
    volume_score: float = 0.0
    rs_btc_score: float = 0.0       # Relative strength vs BTC (0 = weakest, 1 = strongest)
    atr_pct_rank: float = 0.5       # ATR percentile rank [0, 1] — 0=compressed, 1=expanded
    regime: str = "RANGING"
    current_price: float = 0.0
    atr: float = 0.0


def _compute_ema_score(close: pd.Series, fast: int, slow: int) -> float:
    """EMA trend strength normalised to [0, 1] via tanh."""
    if len(close) < slow + 5:
        return 0.0
    ema_fast = ta.trend.EMAIndicator(close=close, window=fast).ema_indicator()
    ema_slow = ta.trend.EMAIndicator(close=close, window=slow).ema_indicator()
    if ema_fast.empty or ema_slow.empty:
        return 0.0
    diff = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / (ema_slow.iloc[-1] + 1e-10)
    return float((np.tanh(diff * 50) + 1) / 2)


def _compute_atr_expansion(df: pd.DataFrame, period: int) -> Tuple[float, float]:
    """Return (atr_expansion_score [0,1], atr_percentile_rank [0,1])."""
    if len(df) < period + 5:
        return 0.0, 0.5
    ind = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=period
    )
    s = ind.average_true_range().dropna()
    if len(s) < period:
        return 0.0, 0.5
    current = float(s.iloc[-1])
    median = float(s.median())
    expansion_score = float(min((current / (median + 1e-10)) / 3.0, 1.0))
    pct_rank = float(s.rank(pct=True).iloc[-1])
    return expansion_score, pct_rank


def _compute_momentum_score(close: pd.Series, period: int) -> float:
    """Rate-of-change momentum normalised to [0, 1]."""
    if len(close) < period + 2:
        return 0.0
    roc = (close.iloc[-1] - close.iloc[-period]) / (close.iloc[-period] + 1e-10)
    return float((np.tanh(roc * 30) + 1) / 2)


def _compute_volume_spike(df: pd.DataFrame, multiplier: float) -> float:
    """Volume of last bar vs average. Returns [0, 1]."""
    if len(df) < 20:
        return 0.5
    avg_vol = df["volume"].iloc[-20:-1].mean()
    if avg_vol == 0:
        return 0.5
    ratio = df["volume"].iloc[-1] / avg_vol
    return float(min(ratio / multiplier, 1.0))


def _compute_rs_btc(symbol_close: pd.Series, btc_close: Optional[pd.Series], period: int = 20) -> float:
    """
    Relative strength vs BTC: (symbol_roc - btc_roc) normalised to [0, 1].
    Returns 0.5 if BTC data unavailable.
    """
    if btc_close is None or len(btc_close) < period + 2 or len(symbol_close) < period + 2:
        return 0.5
    sym_roc = (float(symbol_close.iloc[-1]) - float(symbol_close.iloc[-period])) / (float(symbol_close.iloc[-period]) + 1e-10)
    btc_roc = (float(btc_close.iloc[-1]) - float(btc_close.iloc[-period])) / (float(btc_close.iloc[-period]) + 1e-10)
    relative = sym_roc - btc_roc
    return float((np.tanh(relative * 30) + 1) / 2)


def score_symbol(
    symbol: str,
    df: pd.DataFrame,
    btc_close: Optional[pd.Series] = None,
) -> SymbolScore:
    """Compute composite score for a single symbol."""
    cfg = get_config().trading
    close = df["close"]

    ema_score = _compute_ema_score(close, cfg.ema_fast, cfg.ema_slow)
    atr_score, atr_pct_rank = _compute_atr_expansion(df, cfg.atr_period)
    mom_score = _compute_momentum_score(close, cfg.momentum_period)
    vol_score = _compute_volume_spike(df, cfg.volume_spike_multiplier)
    rs_score  = _compute_rs_btc(close, btc_close)

    # Weighted composite — RS vs BTC replaces the old 0% remainder
    composite = (
        0.30 * ema_score
        + 0.20 * atr_score
        + 0.20 * mom_score
        + 0.15 * vol_score
        + 0.15 * rs_score
    )

    ind = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=cfg.atr_period
    )
    atr_s = ind.average_true_range().dropna()
    atr_val = float(atr_s.iloc[-1]) if len(atr_s) > 0 else 0.0

    return SymbolScore(
        symbol=symbol,
        score=composite,
        ema_score=ema_score,
        atr_score=atr_score,
        momentum_score=mom_score,
        volume_score=vol_score,
        rs_btc_score=rs_score,
        atr_pct_rank=atr_pct_rank,
        current_price=float(close.iloc[-1]),
        atr=atr_val,
    )


class Scanner:
    """
    Scans the universe, fetches data, scores and ranks symbols.
    Includes BTC relative strength for each symbol.
    """

    def __init__(self, data_feed) -> None:
        self._feed = data_feed
        self._cfg = get_config().trading
        self._last_scores: List[SymbolScore] = []
        self._lock = threading.Lock()

    def scan(self, symbols: List[str]) -> List[SymbolScore]:
        """Score all symbols and return sorted list (best first)."""
        log.info("Scanning %d symbols …", len(symbols))
        ohlcv_map = self._feed.bulk_fetch(symbols, max_workers=10)

        # Fetch BTC reference for RS calculation
        btc_close: Optional[pd.Series] = None
        btc_df = ohlcv_map.get("BTCUSDT")
        if btc_df is None:
            btc_df = self._feed.get_ohlcv("BTCUSDT")
        if btc_df is not None and not btc_df.empty:
            btc_close = btc_df["close"].reset_index(drop=True)

        scores: List[SymbolScore] = []
        for sym, df in ohlcv_map.items():
            try:
                sym_btc = btc_close if sym != "BTCUSDT" else None
                s = score_symbol(sym, df, btc_close=sym_btc)
                if s.current_price < self._cfg.min_entry_price:
                    log.debug("Skipping %s: price %.10f below min_entry_price", sym, s.current_price)
                    continue
                scores.append(s)
            except Exception as e:
                log.warning("Score error for %s: %s", sym, e)

        scores.sort(key=lambda x: x.score, reverse=True)

        with self._lock:
            self._last_scores = scores

        top = scores[:self._cfg.top_n_symbols]
        log.info(
            "Top symbols: %s",
            [(s.symbol, round(s.score, 3), f"rs={s.rs_btc_score:.2f}", f"atr_pct={s.atr_pct_rank:.2f}")
             for s in top],
        )
        # Warn on extreme new-entrant symbols: rs≥0.90 AND atr_pct≥0.90 = momentum spike + volatility spike
        # These symbols may be running catalysts; allow scanning but flag them.
        for s in top:
            if s.rs_btc_score >= 0.90 and s.atr_pct_rank >= 0.90:
                log.warning(
                    "⚠️  Extreme symbol [%s]: rs=%.2f atr_pct=%.2f — momentum/volatility spike, treat entries with caution",
                    s.symbol, s.rs_btc_score, s.atr_pct_rank,
                )
        return scores

    def get_top_n(self, n: Optional[int] = None) -> List[SymbolScore]:
        with self._lock:
            n = n or self._cfg.top_n_symbols
            return self._last_scores[:n]

    def get_score(self, symbol: str) -> Optional[SymbolScore]:
        with self._lock:
            for s in self._last_scores:
                if s.symbol == symbol:
                    return s
        return None

