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
    if len(df) < cfg.adx_period + 5:
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

        # ADX
        adx_indicator = ta.trend.ADXIndicator(
            high=df["high"], low=df["low"], close=df["close"],
            window=cfg.adx_period
        )
        adx_series = adx_indicator.adx().dropna()
        if len(adx_series) == 0:
            return Regime.RANGING

        adx = adx_series.iloc[-1]
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
    try:
        ind = ta.trend.ADXIndicator(
            high=df["high"], low=df["low"], close=df["close"], window=period
        )
        s = ind.adx().dropna()
        return float(s.iloc[-1]) if len(s) > 0 else 0.0
    except Exception:
        return 0.0


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Return the latest ATR value."""
    try:
        ind = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=period
        )
        s = ind.average_true_range().dropna()
        return float(s.iloc[-1]) if len(s) > 0 else 0.0
    except Exception:
        return 0.0
