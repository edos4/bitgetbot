"""
scanner.py - Symbol Scanner and Ranker
Scores symbols using EMA trend strength, ATR expansion, momentum,
and volume spike. Returns ranked list of top candidates.
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
    regime: str = "RANGING"
    current_price: float = 0.0
    atr: float = 0.0


def _compute_ema_score(close: pd.Series, fast: int, slow: int) -> float:
    """
    EMA trend strength: (EMA_fast - EMA_slow) / EMA_slow
    Normalized to [0, 1] via tanh.
    """
    if len(close) < slow + 5:
        return 0.0
    ema_fast = ta.trend.EMAIndicator(close=close, window=fast).ema_indicator()
    ema_slow = ta.trend.EMAIndicator(close=close, window=slow).ema_indicator()
    if ema_fast.empty or ema_slow.empty:
        return 0.0
    diff = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / (ema_slow.iloc[-1] + 1e-10)
    # tanh maps to (-1, 1); shift to (0, 1)
    return float((np.tanh(diff * 50) + 1) / 2)


def _compute_atr_expansion(df: pd.DataFrame, period: int) -> float:
    """
    ATR expansion: current ATR vs recent median.
    Returns ratio (>1 = expanding, normalized).
    """
    if len(df) < period + 5:
        return 0.0
    ind = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=period
    )
    s = ind.average_true_range().dropna()
    if len(s) < period:
        return 0.0
    current = s.iloc[-1]
    median = s.median()
    if median == 0:
        return 0.0
    ratio = current / median
    # Cap at 3x and normalize to [0, 1]
    return float(min(ratio / 3.0, 1.0))


def _compute_momentum_score(close: pd.Series, period: int) -> float:
    """
    10-period rate-of-change momentum.
    Returns normalized [0, 1].
    """
    if len(close) < period + 2:
        return 0.0
    roc = (close.iloc[-1] - close.iloc[-period]) / (close.iloc[-period] + 1e-10)
    return float((np.tanh(roc * 30) + 1) / 2)


def _compute_volume_spike(df: pd.DataFrame, multiplier: float) -> float:
    """Volume of last bar vs average. Returns 1.0 if spike, else scaled value."""
    if len(df) < 20:
        return 0.5
    avg_vol = df["volume"].iloc[-20:-1].mean()
    if avg_vol == 0:
        return 0.5
    ratio = df["volume"].iloc[-1] / avg_vol
    return float(min(ratio / multiplier, 1.0))


def score_symbol(symbol: str, df: pd.DataFrame) -> SymbolScore:
    """Compute composite score for a single symbol."""
    cfg = get_config().trading
    close = df["close"]

    ema_score = _compute_ema_score(close, cfg.ema_fast, cfg.ema_slow)
    atr_score = _compute_atr_expansion(df, cfg.atr_period)
    mom_score = _compute_momentum_score(close, cfg.momentum_period)
    vol_score = _compute_volume_spike(df, cfg.volume_spike_multiplier)

    # Weighted composite
    composite = (
        0.35 * ema_score
        + 0.25 * atr_score
        + 0.25 * mom_score
        + 0.15 * vol_score
    )

    # ATR value (raw)
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
        current_price=float(close.iloc[-1]),
        atr=atr_val,
    )


class Scanner:
    """
    Scans the universe, fetches data, scores and ranks symbols.
    Uses a thread pool for parallel data fetching.
    """

    def __init__(self, data_feed) -> None:
        self._feed = data_feed
        self._cfg = get_config().trading
        self._last_scores: List[SymbolScore] = []
        self._lock = threading.Lock()

    def scan(self, symbols: List[str]) -> List[SymbolScore]:
        """
        Score all symbols and return sorted list (best first).
        """
        log.info("Scanning %d symbols …", len(symbols))
        ohlcv_map = self._feed.bulk_fetch(symbols, max_workers=10)

        scores: List[SymbolScore] = []
        for sym, df in ohlcv_map.items():
            try:
                s = score_symbol(sym, df)
                scores.append(s)
            except Exception as e:
                log.warning("Score error for %s: %s", sym, e)

        scores.sort(key=lambda x: x.score, reverse=True)

        with self._lock:
            self._last_scores = scores

        top = scores[:self._cfg.top_n_symbols]
        log.info("Top symbols: %s", [(s.symbol, round(s.score, 3)) for s in top])
        return scores

    def get_top_n(self, n: Optional[int] = None) -> List[SymbolScore]:
        """Return top-N from last scan result."""
        with self._lock:
            n = n or self._cfg.top_n_symbols
            return self._last_scores[:n]

    def get_score(self, symbol: str) -> Optional[SymbolScore]:
        with self._lock:
            for s in self._last_scores:
                if s.symbol == symbol:
                    return s
        return None
