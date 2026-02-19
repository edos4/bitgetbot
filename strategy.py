"""Adaptive multi-strategy signal generation with regime awareness."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np
import ta

from config import get_config
from regime import Regime, RegimeState, detect_regime
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
    size_multiplier: float = 1.0
    atr: float = 0.0
    reason: str = ""
    strategy: str = "NONE"
    confidence: float = 0.0
    kelly_fraction: float = 1.0


def generate_signal(symbol: str, df: pd.DataFrame, scanner_score: float = 0.5) -> Signal:
    """Blend trend-following and mean-reversion signals, weighted by regime."""
    cfg = get_config().trading
    regime_state = detect_regime(df)
    close = df["close"]
    current_price = float(close.iloc[-1])
    scanner_score = float(np.clip(scanner_score, 0.0, 1.0))

    atr = _compute_atr(df)
    size_mult = cfg.high_vol_size_reduction if regime_state.primary == Regime.HIGH_VOLATILITY else 1.0

    candidates = [
        _trend_following_signal(symbol, df, current_price, atr, regime_state, scanner_score, size_mult),
        _mean_reversion_signal(symbol, df, current_price, atr, regime_state, scanner_score, size_mult),
    ]

    actionable = [c for c in candidates if c.direction != SignalDirection.NONE]
    if actionable:
        best = max(actionable, key=lambda s: s.confidence)
        log.debug(
            "%s signal selected | strat=%s | conf=%.2f | regime=%s | reason=%s",
            symbol, best.strategy, best.confidence, best.regime.value, best.reason,
        )
        return best

    # No strategy fired — propagate highest-confidence rejection reason for diagnostics
    fallback = max(candidates, key=lambda s: s.confidence)
    if fallback.reason:
        log.debug("%s no-trade | top_reason=%s | conf=%.2f", symbol, fallback.reason, fallback.confidence)
    return fallback


def _trend_following_signal(
    symbol: str,
    df: pd.DataFrame,
    price: float,
    atr: float,
    regime_state: RegimeState,
    scanner_score: float,
    size_mult: float,
) -> Signal:
    cfg = get_config().trading
    close = df["close"]
    ema_fast = ta.trend.EMAIndicator(close=close, window=cfg.ema_fast).ema_indicator()
    ema_slow = ta.trend.EMAIndicator(close=close, window=cfg.ema_slow).ema_indicator()

    if ema_fast.empty or ema_slow.empty:
        return _no_signal(symbol, price, atr, regime_state.primary, "EMA_NOT_READY", "TREND")

    ema_fast_cur = float(ema_fast.iloc[-1])
    ema_slow_cur = float(ema_slow.iloc[-1])
    ema_diff = ema_fast_cur - ema_slow_cur
    price_z = (price - ema_slow_cur) / (atr + 1e-10)

    trend_weight = max(regime_state.trend_score, 1e-3)
    confidence = _blend_confidence(trend_weight, scanner_score, regime_state)

    if abs(price_z) < cfg.trend_breakout_z or np.isclose(ema_diff, 0.0, atol=1e-8):
        return _no_signal(symbol, price, atr, regime_state.primary, "TREND_BREAKOUT_INACTIVE", "TREND", confidence)

    direction = SignalDirection.LONG if ema_diff > 0 else SignalDirection.SHORT
    if direction == SignalDirection.LONG and price_z < cfg.trend_breakout_z:
        return _no_signal(symbol, price, atr, regime_state.primary, "TREND_Z_BELOW_LONG", "TREND", confidence)
    if direction == SignalDirection.SHORT and price_z > -cfg.trend_breakout_z:
        return _no_signal(symbol, price, atr, regime_state.primary, "TREND_Z_ABOVE_SHORT", "TREND", confidence)

    stop_mult = cfg.atr_stop_trend_multiplier
    target_mult = cfg.atr_target_trend_multiplier
    if regime_state.primary == Regime.HIGH_VOLATILITY:
        stop_mult *= 1.2
        target_mult *= 1.1

    if direction == SignalDirection.LONG:
        stop = price - atr * stop_mult
        target = price + atr * target_mult
    else:
        stop = price + atr * stop_mult
        target = price - atr * target_mult

    kelly = _kelly_from_confidence(confidence)
    return Signal(
        symbol=symbol,
        direction=direction,
        entry_price=price,
        stop_loss=stop,
        take_profit=target,
        regime=regime_state.primary,
        atr=atr,
        reason="TREND_BREAKOUT",
        strategy="TREND",
        confidence=confidence,
        kelly_fraction=kelly,
        size_multiplier=size_mult,
    )


def _mean_reversion_signal(
    symbol: str,
    df: pd.DataFrame,
    price: float,
    atr: float,
    regime_state: RegimeState,
    scanner_score: float,
    size_mult: float,
) -> Signal:
    cfg = get_config().trading
    close = df["close"]
    bb = ta.volatility.BollingerBands(close=close, window=cfg.bb_period, window_dev=cfg.bb_std)
    bb_mid = bb.bollinger_mavg()
    bb_up = bb.bollinger_hband()
    bb_low = bb.bollinger_lband()

    if bb_mid.empty or bb_up.empty or bb_low.empty:
        return _no_signal(symbol, price, atr, regime_state.primary, "BB_NOT_READY", "MEAN_REVERSION")

    mid = float(bb_mid.iloc[-1])
    up = float(bb_up.iloc[-1])
    low = float(bb_low.iloc[-1])
    band_width = (up - mid) + 1e-10
    z_score = (price - mid) / band_width

    range_weight = max(regime_state.range_score, 1e-3)
    confidence = _blend_confidence(range_weight, scanner_score, regime_state)

    direction = SignalDirection.NONE
    reason = "BB_ZSCORE_NEUTRAL"
    stop = price
    target = mid

    if z_score <= -cfg.mean_reversion_z_entry:
        direction = SignalDirection.LONG
        stop = price - atr * cfg.atr_stop_range_multiplier
        target = price + atr * cfg.atr_target_range_multiplier
        reason = "BB_ZSCORE_LONG"
    elif z_score >= cfg.mean_reversion_z_entry:
        direction = SignalDirection.SHORT
        stop = price + atr * cfg.atr_stop_range_multiplier
        target = price - atr * cfg.atr_target_range_multiplier
        reason = "BB_ZSCORE_SHORT"

    if direction == SignalDirection.NONE:
        return _no_signal(symbol, price, atr, regime_state.primary, reason, "MEAN_REVERSION", confidence)

    # Exit nearer to mid if z-score collapses quickly
    if abs(z_score) <= cfg.mean_reversion_z_exit:
        target = mid

    kelly = _kelly_from_confidence(confidence)
    return Signal(
        symbol=symbol,
        direction=direction,
        entry_price=price,
        stop_loss=stop,
        take_profit=target,
        regime=regime_state.primary,
        atr=atr,
        reason=reason,
        strategy="MEAN_REVERSION",
        confidence=confidence,
        kelly_fraction=kelly,
        size_multiplier=size_mult,
    )


def _no_signal(
    symbol: str,
    price: float,
    atr: float,
    regime: Regime,
    reason: str,
    strategy: str,
    confidence: float = 0.0,
) -> Signal:
    return Signal(
        symbol=symbol,
        direction=SignalDirection.NONE,
        entry_price=price,
        stop_loss=0.0,
        take_profit=None,
        regime=regime,
        atr=atr,
        reason=reason,
        strategy=strategy,
        confidence=confidence,
        kelly_fraction=0.0,
    )


def _compute_atr(df: pd.DataFrame) -> float:
    cfg = get_config().trading
    atr_ind = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=cfg.atr_period
    )
    atr_series = atr_ind.average_true_range().dropna()
    if len(atr_series) == 0:
        close = df["close"]
        return float(close.iloc[-1]) * 0.01
    return float(atr_series.iloc[-1])


def _blend_confidence(weight: float, scanner_score: float, regime_state: RegimeState) -> float:
    cfg = get_config().trading
    alpha = cfg.signal_confidence_alpha
    base = (1 - alpha) * weight + alpha * scanner_score
    if regime_state.primary == Regime.HIGH_VOLATILITY and not regime_state.expansion:
        base *= 0.85  # penalize choppy volatility spikes
    return float(np.clip(base, 0.0, 1.0))


def _kelly_from_confidence(confidence: float) -> float:
    cfg = get_config().trading
    return float(np.clip(confidence, cfg.kelly_fraction_floor, cfg.kelly_fraction_cap))


def compute_trailing_stop(
    direction: SignalDirection,
    current_price: float,
    highest_price: float,
    atr: float,
) -> float:
    cfg = get_config().trading
    mult = cfg.atr_trailing_multiplier
    if direction == SignalDirection.LONG:
        return highest_price - atr * mult
    return highest_price + atr * mult
