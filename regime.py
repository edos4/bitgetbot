"""
regime.py - Market Regime Detection
Classifies current market state as TRENDING, RANGING, or HIGH_VOLATILITY.
Used to select the appropriate strategy variant.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np
import ta

from config import get_config
from logger import get_logger

log = get_logger("regime")


class Regime(str, Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"


@dataclass
class RegimeState:
    primary: Regime
    trend_score: float
    range_score: float
    volatility_ratio: float
    atr_ratio: float
    adx: float
    expansion: bool
    contraction: bool


def detect_regime(df: pd.DataFrame) -> RegimeState:
    """Probability-aware regime classification with volatility diagnostics."""
    cfg = get_config().trading
    min_bars_needed = max(
        cfg.volatility_long_window + 5,
        cfg.ema_slow + 5,
        cfg.atr_period * 2,
    )
    if len(df) < min_bars_needed:
        log.debug(
            "Not enough bars for regime detection (%d < %d), defaulting to RANGING",
            len(df), min_bars_needed,
        )
        return RegimeState(
            primary=Regime.RANGING,
            trend_score=0.2,
            range_score=0.8,
            volatility_ratio=1.0,
            atr_ratio=1.0,
            adx=0.0,
            expansion=False,
            contraction=True,
        )

    try:
        close = df["close"]
        atr_indicator = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=close,
            window=cfg.atr_period
        )
        atr_series = atr_indicator.average_true_range().dropna()
        current_atr = float(atr_series.iloc[-1]) if len(atr_series) > 0 else 0.0
        median_atr = float(atr_series.median()) if len(atr_series) > 0 else 1.0
        atr_ratio = current_atr / (median_atr + 1e-10)

        ema_fast = ta.trend.EMAIndicator(close=close, window=cfg.ema_fast).ema_indicator()
        ema_slow = ta.trend.EMAIndicator(close=close, window=cfg.ema_slow).ema_indicator()
        if ema_fast.empty or ema_slow.empty:
            trend_diff = 0.0
        else:
            trend_diff = abs(float(ema_fast.iloc[-1] - ema_slow.iloc[-1]))
        trend_denom = current_atr if current_atr > 0 else max(abs(float(close.iloc[-1])), 1.0)
        dvr = trend_diff / (trend_denom + 1e-10)
        trend_score = float(np.tanh(dvr))  # maps to (0,1)
        range_score = float(max(0.0, 1.0 - trend_score))

        returns = close.pct_change().dropna()
        short_std = returns.rolling(cfg.volatility_short_window).std().dropna()
        long_std = returns.rolling(cfg.volatility_long_window).std().dropna()
        if len(short_std) == 0 or len(long_std) == 0:
            vol_ratio = 1.0
        else:
            vol_ratio = float(short_std.iloc[-1] / (long_std.iloc[-1] + 1e-10))
        expansion = vol_ratio > cfg.vol_expansion_ratio
        contraction = vol_ratio < cfg.vol_contraction_ratio

        adx_indicator = ta.trend.ADXIndicator(
            high=df["high"], low=df["low"], close=close,
            window=cfg.adx_period
        )
        adx_series = adx_indicator.adx().dropna()
        adx = float(adx_series.iloc[-1]) if len(adx_series) > 0 else 0.0

        primary = Regime.RANGING
        if atr_ratio > cfg.atr_volatility_multiplier or expansion:
            primary = Regime.HIGH_VOLATILITY
        elif trend_score >= cfg.regime_trend_threshold or adx > cfg.adx_trending_threshold:
            primary = Regime.TRENDING
        elif trend_score <= cfg.regime_range_threshold or contraction:
            primary = Regime.RANGING

        log.debug(
            "RegimeState: primary=%s trend=%.2f range=%.2f atr_ratio=%.2f vol_ratio=%.2f adx=%.1f",
            primary.value, trend_score, range_score, atr_ratio, vol_ratio, adx,
        )
        return RegimeState(
            primary=primary,
            trend_score=trend_score,
            range_score=range_score,
            volatility_ratio=vol_ratio,
            atr_ratio=atr_ratio,
            adx=adx,
            expansion=expansion,
            contraction=contraction,
        )

    except Exception as e:
        log.error("Regime detection error: %s", e)
        return RegimeState(
            primary=Regime.RANGING,
            trend_score=0.5,
            range_score=0.5,
            volatility_ratio=1.0,
            atr_ratio=1.0,
            adx=0.0,
            expansion=False,
            contraction=False,
        )


def compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Return the latest ADX value."""
    if len(df) < period * 2 + 5:
        return 0.0
    try:
        ind = ta.trend.ADXIndicator(
            high=df["high"], low=df["low"], close=df["close"], window=period
        )
        s = ind.adx().dropna()
        if len(s) > 0:
            val = float(s.iloc[-1])
            return val if not np.isnan(val) else 0.0
        return 0.0
    except Exception:
        return 0.0


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Return the latest ATR value."""
    if len(df) < period + 5:
        return 0.0
    try:
        ind = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=period
        )
        s = ind.average_true_range().dropna()
        if len(s) > 0:
            val = float(s.iloc[-1])
            return val if not np.isnan(val) else 0.0
        return 0.0
    except Exception:
        return 0.0
