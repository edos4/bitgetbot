"""
strategy.py - Trading Signal Generation
Generates entry/exit signals based on the detected market regime.

TRENDING:   EMA crossover + ATR stop + trailing stop
RANGING:    Bollinger Band mean reversion
HIGH_VOL:   Trending signals but 50% position size reduction
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import ta

from config import get_config
from regime import Regime, detect_regime
from logger import get_logger

log = get_logger("strategy")


class SignalDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class Signal:
    symbol: str
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    regime: Regime
    size_multiplier: float = 1.0   # 0.5 for HIGH_VOL
    atr: float = 0.0
    reason: str = ""


def generate_signal(symbol: str, df: pd.DataFrame) -> Signal:
    """
    Main entry point: detect regime then apply matching strategy.
    Returns a Signal (direction=NONE if no setup).
    """
    cfg = get_config().trading
    regime = detect_regime(df)
    close = df["close"]
    current_price = float(close.iloc[-1])

    # ATR (used by all regimes)
    atr_ind = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=cfg.atr_period
    )
    atr_series = atr_ind.average_true_range().dropna()
    atr = float(atr_series.iloc[-1]) if len(atr_series) > 0 else current_price * 0.01

    size_mult = cfg.high_vol_size_reduction if regime == Regime.HIGH_VOLATILITY else 1.0

    if regime in (Regime.TRENDING, Regime.HIGH_VOLATILITY):
        sig = _ema_crossover_signal(symbol, df, current_price, atr, regime)
    else:
        sig = _bb_mean_reversion_signal(symbol, df, current_price, atr, regime)

    sig.size_multiplier = size_mult
    return sig


def _ema_crossover_signal(
    symbol: str, df: pd.DataFrame, price: float, atr: float, regime: Regime
) -> Signal:
    """
    EMA 9/21 crossover signal.
    LONG:  fast crosses above slow and price > slow
    SHORT: fast crosses below slow and price < slow
    Stop:  ATR × multiplier from entry
    """
    cfg = get_config().trading

    close = df["close"]
    ema_fast = ta.trend.EMAIndicator(close=close, window=cfg.ema_fast).ema_indicator()
    ema_slow = ta.trend.EMAIndicator(close=close, window=cfg.ema_slow).ema_indicator()

    if len(ema_fast) < 3 or len(ema_slow) < 3:
        return _no_signal(symbol, price, atr, regime)

    fast_cur, fast_prev = ema_fast.iloc[-1], ema_fast.iloc[-2]
    slow_cur, slow_prev = ema_slow.iloc[-1], ema_slow.iloc[-2]

    stop_distance = atr * cfg.atr_stop_multiplier

    # Bullish crossover
    if fast_prev <= slow_prev and fast_cur > slow_cur and price > slow_cur:
        stop = price - stop_distance
        tp = price + stop_distance * 2  # 2R target
        log.debug("%s EMA LONG signal @ %.4f stop=%.4f", symbol, price, stop)
        return Signal(
            symbol=symbol, direction=SignalDirection.LONG,
            entry_price=price, stop_loss=stop, take_profit=tp,
            regime=regime, atr=atr, reason="EMA_CROSSOVER_BULL"
        )

    # Bearish crossover
    if fast_prev >= slow_prev and fast_cur < slow_cur and price < slow_cur:
        stop = price + stop_distance
        tp = price - stop_distance * 2
        log.debug("%s EMA SHORT signal @ %.4f stop=%.4f", symbol, price, stop)
        return Signal(
            symbol=symbol, direction=SignalDirection.SHORT,
            entry_price=price, stop_loss=stop, take_profit=tp,
            regime=regime, atr=atr, reason="EMA_CROSSOVER_BEAR"
        )

    return _no_signal(symbol, price, atr, regime)


def _bb_mean_reversion_signal(
    symbol: str, df: pd.DataFrame, price: float, atr: float, regime: Regime
) -> Signal:
    """
    Bollinger Band mean reversion.
    LONG:  price touches lower band + prior candle was below band
    SHORT: price touches upper band + prior candle was above band
    Stop:  outside the band by 0.5 ATR
    """
    cfg = get_config().trading
    close = df["close"]

    bb = ta.volatility.BollingerBands(
        close=close, window=cfg.bb_period, window_dev=cfg.bb_std
    )
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_mid = bb.bollinger_mavg()

    if len(bb_upper) < 3:
        return _no_signal(symbol, price, atr, regime)

    upper_cur = bb_upper.iloc[-1]
    lower_cur = bb_lower.iloc[-1]
    mid_cur = bb_mid.iloc[-1]

    # Long: price pierced lower band and came back
    if close.iloc[-2] < bb_lower.iloc[-2] and price >= lower_cur:
        stop = lower_cur - atr * 0.5
        tp = mid_cur
        log.debug("%s BB LONG signal @ %.4f stop=%.4f tp=%.4f", symbol, price, stop, tp)
        return Signal(
            symbol=symbol, direction=SignalDirection.LONG,
            entry_price=price, stop_loss=stop, take_profit=tp,
            regime=regime, atr=atr, reason="BB_REVERSION_BULL"
        )

    # Short: price pierced upper band and came back
    if close.iloc[-2] > bb_upper.iloc[-2] and price <= upper_cur:
        stop = upper_cur + atr * 0.5
        tp = mid_cur
        log.debug("%s BB SHORT signal @ %.4f stop=%.4f tp=%.4f", symbol, price, stop, tp)
        return Signal(
            symbol=symbol, direction=SignalDirection.SHORT,
            entry_price=price, stop_loss=stop, take_profit=tp,
            regime=regime, atr=atr, reason="BB_REVERSION_BEAR"
        )

    return _no_signal(symbol, price, atr, regime)


def _no_signal(symbol: str, price: float, atr: float, regime: Regime) -> Signal:
    return Signal(
        symbol=symbol, direction=SignalDirection.NONE,
        entry_price=price, stop_loss=0.0, take_profit=None,
        regime=regime, atr=atr, reason="NO_SETUP"
    )


def compute_trailing_stop(
    direction: SignalDirection,
    current_price: float,
    highest_price: float,  # for LONG: max since entry | for SHORT: min since entry
    atr: float,
) -> float:
    """Calculate a trailing stop price."""
    cfg = get_config().trading
    mult = cfg.atr_trailing_multiplier
    if direction == SignalDirection.LONG:
        return highest_price - atr * mult
    else:
        return highest_price + atr * mult
