"""
regime.py - Market Regime Detection
Classifies current market state as TRENDING, RANGING, or HIGH_VOLATILITY.
Used to select the appropriate strategy variant.
"""
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


def detect_regime(df: pd.DataFrame) -> Regime:
    """
    Classify market regime from OHLCV dataframe.

    Logic:
    1. Compute ATR; if ATR > 2× median ATR → HIGH_VOLATILITY
    2. Compute ADX; if ADX > threshold → TRENDING
    3. Default → RANGING
    """
    cfg = get_config().trading
    # ADX needs roughly 2x the period for proper calculation
    min_bars_needed = max(cfg.adx_period * 2, cfg.atr_period) + 10
    if len(df) < min_bars_needed:
        log.debug("Not enough bars for regime detection (%d < %d), defaulting to RANGING", len(df), min_bars_needed)
        return Regime.RANGING

    try:
        # ATR
        atr_indicator = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"],
            window=cfg.atr_period
        )
        atr_series = atr_indicator.average_true_range().dropna()
        if len(atr_series) < 10:
            return Regime.RANGING

        current_atr = atr_series.iloc[-1]
        median_atr = atr_series.median()

        if median_atr > 0 and current_atr > cfg.atr_volatility_multiplier * median_atr:
            log.debug("Regime: HIGH_VOLATILITY (ATR=%.4f, median=%.4f)", current_atr, median_atr)
            return Regime.HIGH_VOLATILITY

        # ADX - needs more bars than ATR
        if len(df) < cfg.adx_period * 2 + 5:
            log.debug("Not enough bars for ADX, defaulting to RANGING")
            return Regime.RANGING
            
        adx_indicator = ta.trend.ADXIndicator(
            high=df["high"], low=df["low"], close=df["close"],
            window=cfg.adx_period
        )
        adx_series = adx_indicator.adx().dropna()
        if len(adx_series) == 0:
            log.debug("ADX series empty, defaulting to RANGING")
            return Regime.RANGING

        adx = adx_series.iloc[-1]
        if np.isnan(adx):
            log.debug("ADX is NaN, defaulting to RANGING")
            return Regime.RANGING
            
        if adx > cfg.adx_trending_threshold:
            log.debug("Regime: TRENDING (ADX=%.2f)", adx)
            return Regime.TRENDING

        log.debug("Regime: RANGING (ADX=%.2f)", adx)
        return Regime.RANGING

    except Exception as e:
        log.error("Regime detection error: %s", e)
        return Regime.RANGING


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
