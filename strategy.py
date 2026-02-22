"""Adaptive multi-strategy signal generation with isolated strategy classes.

Architecture
------------
Three fully isolated strategy classes derived from BaseStrategy:
  MeanReversionStrategy     — RANGING regime, BB z-score with soft band
  TrendPullbackStrategy     — TRENDING regime, EMA pullback with trend continuation
  VolatilityBreakoutStrategy — HIGH_VOLATILITY regime, N-bar high/low breakout

StrategyRouter selects the active strategy from the detected RegimeState.

Two public entry-points (API unchanged):
  generate_signal_from_cache(symbol, cache, idx) → Signal   [O(1), backtester]
  generate_signal(symbol, df)                   → Signal   [per-call, paper/live]

Volume-ratio tiering (replaces hard spike filter)
-------------------------------------------------
  raw_vol_ratio = current_vol / rolling_20_avg
  < vol_ratio_min  (0.8) → no signal
  0.8 – 1.0              → signal.volume_ratio = raw  (execution scales to ~50%)
  1.0 – 1.5              → signal.volume_ratio = raw  (execution scales to 100%)
  > 1.5                  → signal.volume_ratio = raw  (execution scales to 125%)

5-Factor Confidence Score
-------------------------
  [signal_strength, regime_strength, volume_ratio, trend_alignment, volatility_state]
  Weights: 0.30 / 0.25 / 0.20 / 0.15 / 0.10
"""
from __future__ import annotations
from abc import ABC, abstractmethod
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
    stop_distance: float
    reward_risk_ratio: float
    atr: float
    reason: str
    strategy: str
    atr_ratio: float
    confidence: float = 0.0
    volume_ratio: float = 1.0   # raw_vol / rolling_avg; used by execution for tiered sizing


# ---------------------------------------------------------------------------
# Precomputed indicator cache (one object per full dataframe)
# ---------------------------------------------------------------------------

class IndicatorCache:
    """
    Holds all indicator series for a full dataframe.
    Build once with IndicatorCache.from_df(df), then look up any bar by index.
    """

    def __init__(self) -> None:
        self.df: Optional[pd.DataFrame] = None
        self.atr: Optional[pd.Series] = None
        self.ema_fast: Optional[pd.Series] = None
        self.ema_slow: Optional[pd.Series] = None
        self.bb_mid: Optional[pd.Series] = None
        self.bb_up: Optional[pd.Series] = None
        self.bb_low: Optional[pd.Series] = None
        self.adx: Optional[pd.Series] = None
        self.vol_ratio: Optional[pd.Series] = None
        self.atr_ratio: Optional[pd.Series] = None
        # Volatility-breakout indicators
        self.high_n: Optional[pd.Series] = None      # N-bar rolling high, shifted to avoid lookahead
        self.low_n: Optional[pd.Series] = None       # N-bar rolling low, shifted
        self.raw_vol_ratio: Optional[pd.Series] = None  # current_vol / rolling_avg_vol
        # Momentum breakout indicators
        self.rsi: Optional[pd.Series] = None          # RSI(rsi_period)
        self.donchian_high: Optional[pd.Series] = None  # Donchian channel high, shifted
        self.donchian_low: Optional[pd.Series] = None   # Donchian channel low, shifted
        self.atr_pct_rank: Optional[pd.Series] = None   # ATR rolling percentile rank [0,1]

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "IndicatorCache":
        cfg = get_config().trading
        obj = cls()
        obj.df = df

        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        atr_ind = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=cfg.atr_period)
        obj.atr = atr_ind.average_true_range()

        obj.ema_fast = ta.trend.EMAIndicator(close=close, window=cfg.ema_fast).ema_indicator()
        obj.ema_slow = ta.trend.EMAIndicator(close=close, window=cfg.ema_slow).ema_indicator()

        bb = ta.volatility.BollingerBands(close=close, window=cfg.bb_period, window_dev=cfg.bb_std)
        obj.bb_mid = bb.bollinger_mavg()
        obj.bb_up  = bb.bollinger_hband()
        obj.bb_low = bb.bollinger_lband()

        adx_ind = ta.trend.ADXIndicator(high=high, low=low, close=close, window=cfg.adx_period)
        obj.adx = adx_ind.adx()

        returns   = close.pct_change()
        short_std = returns.rolling(cfg.volatility_short_window).std()
        long_std  = returns.rolling(cfg.volatility_long_window).std()
        obj.vol_ratio = short_std / (long_std + 1e-10)

        obj.atr_ratio = obj.atr / (obj.atr.rolling(50).median() + 1e-10)

        # Volatility-breakout: N-bar high/low (shift by 1 — strict lookahead-free)
        n = cfg.vol_breakout_lookback
        obj.high_n = df["high"].rolling(n).max().shift(1)
        obj.low_n  = df["low"].rolling(n).min().shift(1)

        # Raw volume ratio (current bar / rolling mean)
        raw_vol = df["volume"]
        avg_vol = raw_vol.rolling(cfg.volume_lookback).mean()
        obj.raw_vol_ratio = raw_vol / (avg_vol + 1e-10)

        # RSI
        rsi_ind = ta.momentum.RSIIndicator(close=close, window=cfg.rsi_period)
        obj.rsi = rsi_ind.rsi()

        # Donchian channel for momentum breakout (lookahead-free: shift 1)
        d = cfg.donchian_period
        obj.donchian_high = df["high"].rolling(d).max().shift(1)
        obj.donchian_low  = df["low"].rolling(d).min().shift(1)

        # ATR rolling percentile rank — values in [0, 1]
        atr_comp_window = cfg.atr_compression_lookback
        obj.atr_pct_rank = obj.atr.rolling(atr_comp_window).rank(pct=True)

        return obj

    def get_atr(self, idx: int) -> float:
        v = self.atr.iloc[idx]
        if pd.isna(v) or v == 0:
            return float(self.df["close"].iloc[idx]) * 0.01
        return float(v)

    def get_ema_fast(self, idx: int) -> Optional[float]:
        v = self.ema_fast.iloc[idx]
        return None if pd.isna(v) else float(v)

    def get_ema_slow(self, idx: int) -> Optional[float]:
        v = self.ema_slow.iloc[idx]
        return None if pd.isna(v) else float(v)

    def get_bb(self, idx: int):
        m = self.bb_mid.iloc[idx]
        u = self.bb_up.iloc[idx]
        l = self.bb_low.iloc[idx]
        if pd.isna(m) or pd.isna(u) or pd.isna(l):
            return None, None, None
        return float(m), float(u), float(l)

    def get_adx(self, idx: int) -> float:
        v = self.adx.iloc[idx]
        return float(v) if not pd.isna(v) else 0.0

    def get_vol_ratio(self, idx: int) -> float:
        v = self.vol_ratio.iloc[idx]
        return float(v) if not pd.isna(v) else 1.0

    def get_atr_ratio(self, idx: int) -> float:
        v = self.atr_ratio.iloc[idx]
        return float(v) if not pd.isna(v) else 1.0

    def get_n_bar_high(self, idx: int) -> Optional[float]:
        """N-bar rolling high from previous bars (lookahead-free)."""
        if self.high_n is None:
            return None
        v = self.high_n.iloc[idx]
        return None if pd.isna(v) else float(v)

    def get_n_bar_low(self, idx: int) -> Optional[float]:
        """N-bar rolling low from previous bars (lookahead-free)."""
        if self.low_n is None:
            return None
        v = self.low_n.iloc[idx]
        return None if pd.isna(v) else float(v)

    def get_raw_vol_ratio(self, idx: int) -> float:
        """Return current_vol / rolling_avg_vol; defaults to 1.0 if unavailable."""
        if self.raw_vol_ratio is None:
            return 1.0
        v = self.raw_vol_ratio.iloc[idx]
        return float(v) if not pd.isna(v) else 1.0

    def get_rsi(self, idx: int) -> float:
        """Return RSI value at idx; defaults to 50 (neutral) if unavailable."""
        if self.rsi is None:
            return 50.0
        v = self.rsi.iloc[idx]
        return float(v) if not pd.isna(v) else 50.0

    def get_donchian_high(self, idx: int) -> Optional[float]:
        """Donchian channel high at idx (lookahead-free, shifted 1 bar)."""
        if self.donchian_high is None:
            return None
        v = self.donchian_high.iloc[idx]
        return None if pd.isna(v) else float(v)

    def get_donchian_low(self, idx: int) -> Optional[float]:
        """Donchian channel low at idx (lookahead-free, shifted 1 bar)."""
        if self.donchian_low is None:
            return None
        v = self.donchian_low.iloc[idx]
        return None if pd.isna(v) else float(v)

    def get_atr_pct_rank(self, idx: int) -> float:
        """Rolling ATR percentile rank in [0, 1]. 0=compressed, 1=extremely expanded."""
        if self.atr_pct_rank is None:
            return 0.5
        v = self.atr_pct_rank.iloc[idx]
        return float(v) if not pd.isna(v) else 0.5

    def volume_size_factor(self, idx: int) -> float:
        """Map volume_ratio to a position-size multiplier.

        < vol_ratio_min  → 0.0 (caller should return no_signal)
        0.8 – 1.0        → 0.5
        1.0 – 1.5        → 1.0
        > 1.5            → 1.25
        """
        cfg = get_config().trading
        ratio = self.get_raw_vol_ratio(idx)
        if ratio < cfg.volume_ratio_min:
            return 0.0
        if ratio < cfg.volume_ratio_half:
            return 0.5
        if ratio <= cfg.volume_ratio_boost:
            return 1.0
        return 1.25
        if idx < lookback:
            return False
        avg_vol = float(self.df["volume"].iloc[idx - lookback:idx].mean())
        last_vol = float(self.df["volume"].iloc[idx])
        return last_vol >= avg_vol * multiplier

    def get_regime(self, idx: int) -> RegimeState:
        cfg = get_config().trading
        atr_val   = self.get_atr(idx)
        atr_ratio = self.get_atr_ratio(idx)
        vol_ratio = self.get_vol_ratio(idx)
        adx_val   = self.get_adx(idx)

        ema_f = self.get_ema_fast(idx)
        ema_s = self.get_ema_slow(idx)
        if ema_f is not None and ema_s is not None:
            trend_diff = abs(ema_f - ema_s)
        else:
            trend_diff = 0.0

        trend_denom = atr_val if atr_val > 0 else max(abs(float(self.df["close"].iloc[idx])), 1.0)
        dvr = trend_diff / (trend_denom + 1e-10)
        trend_score = float(np.tanh(dvr))
        range_score = max(0.0, 1.0 - trend_score)

        expansion   = vol_ratio > cfg.vol_expansion_ratio
        contraction = vol_ratio < cfg.vol_contraction_ratio

        primary = Regime.RANGING
        if atr_ratio > cfg.atr_volatility_multiplier or expansion:
            primary = Regime.HIGH_VOLATILITY
        elif trend_score >= cfg.regime_trend_threshold or adx_val > cfg.adx_trending_threshold:
            primary = Regime.TRENDING
        elif trend_score <= cfg.regime_range_threshold or contraction:
            primary = Regime.RANGING

        return RegimeState(
            primary=primary,
            trend_score=trend_score,
            range_score=range_score,
            volatility_ratio=vol_ratio,
            atr_ratio=atr_ratio,
            adx=adx_val,
            expansion=expansion,
            contraction=contraction,
        )


def _ema_pullback_from_cache(
    symbol: str,
    cache: "IndicatorCache",
    idx: int,
    price: float,
    atr: float,
    regime_state: RegimeState,
) -> Signal:
    """
    EMA Pullback strategy for TRENDING regime.
    Enter when price pulls back to the fast EMA in the direction of the trend.
    Stop: 1.5×ATR.  Target: 3.0×ATR.  PF=1.91 on BTC 15m backtest.
    """
    cfg = get_config().trading
    ema_f = cache.get_ema_fast(idx)
    ema_s = cache.get_ema_slow(idx)
    if ema_f is None or ema_s is None:
        return _no_signal(symbol, price, atr, regime_state.primary, "EMA_NOT_READY", "TREND_PULLBACK", regime_state.atr_ratio)

    ema_diff = ema_f - ema_s
    if ema_diff == 0:
        return _no_signal(symbol, price, atr, regime_state.primary, "EMA_FLAT", "TREND_PULLBACK", regime_state.atr_ratio)

    dist_to_fast = abs(price - ema_f)
    zone = cfg.ema_pullback_zone_atr * atr
    if dist_to_fast > zone:
        return _no_signal(
            symbol, price, atr, regime_state.primary,
            f"PULLBACK_ZONE_MISS(dist={dist_to_fast:.2f},zone={zone:.2f})",
            "TREND_PULLBACK", regime_state.atr_ratio,
        )

    vol_ratio = cache.get_raw_vol_ratio(idx)
    if vol_ratio < cfg.volume_ratio_min:
        return _no_signal(symbol, price, atr, regime_state.primary,
                          f"VOL_RATIO_LOW({vol_ratio:.2f})", "TREND_PULLBACK", regime_state.atr_ratio)

    direction = SignalDirection.LONG if ema_diff > 0 else SignalDirection.SHORT
    stop_distance = atr * cfg.atr_stop_trend_multiplier
    reward_risk   = cfg.atr_target_trend_multiplier / cfg.atr_stop_trend_multiplier

    if direction == SignalDirection.LONG:
        stop   = price - stop_distance
        target = price + stop_distance * reward_risk
    else:
        stop   = price + stop_distance
        target = price - stop_distance * reward_risk

    confidence = ConfidenceScorer.trend_pullback(
        dist_to_fast, zone, ema_diff, atr, regime_state, vol_ratio
    )

    return Signal(
        symbol=symbol, direction=direction, entry_price=price,
        stop_loss=stop, take_profit=target, regime=regime_state.primary,
        stop_distance=stop_distance, reward_risk_ratio=reward_risk, atr=atr,
        reason="EMA_PULLBACK", strategy="TREND_PULLBACK", atr_ratio=regime_state.atr_ratio,
        confidence=confidence, volume_ratio=vol_ratio,
    )


def _ema_pullback_signal(
    symbol: str,
    df: pd.DataFrame,
    price: float,
    atr: float,
    regime_state: RegimeState,
) -> Signal:
    """EMA pullback strategy for the legacy (per-call) path used in paper/live."""
    cfg = get_config().trading
    close = df["close"]
    ema_fast_s = ta.trend.EMAIndicator(close=close, window=cfg.ema_fast).ema_indicator()
    ema_slow_s = ta.trend.EMAIndicator(close=close, window=cfg.ema_slow).ema_indicator()

    if ema_fast_s.empty or ema_slow_s.empty:
        return _no_signal(symbol, price, atr, regime_state.primary, "EMA_NOT_READY", "TREND_PULLBACK", regime_state.atr_ratio)

    ema_f = float(ema_fast_s.iloc[-1])
    ema_s = float(ema_slow_s.iloc[-1])
    ema_diff = ema_f - ema_s

    if ema_diff == 0 or pd.isna(ema_f) or pd.isna(ema_s):
        return _no_signal(symbol, price, atr, regime_state.primary, "EMA_FLAT", "TREND_PULLBACK", regime_state.atr_ratio)

    dist_to_fast = abs(price - ema_f)
    zone = cfg.ema_pullback_zone_atr * atr
    if dist_to_fast > zone:
        return _no_signal(
            symbol, price, atr, regime_state.primary,
            f"PULLBACK_ZONE_MISS(dist={dist_to_fast:.2f},zone={zone:.2f})",
            "TREND_PULLBACK", regime_state.atr_ratio,
        )

    vol_ratio = _compute_vol_ratio(df, lookback=cfg.volume_lookback)
    if vol_ratio < cfg.volume_ratio_min:
        return _no_signal(symbol, price, atr, regime_state.primary,
                          f"VOL_RATIO_LOW({vol_ratio:.2f})", "TREND_PULLBACK", regime_state.atr_ratio)

    direction = SignalDirection.LONG if ema_diff > 0 else SignalDirection.SHORT
    stop_distance = atr * cfg.atr_stop_trend_multiplier
    reward_risk   = cfg.atr_target_trend_multiplier / cfg.atr_stop_trend_multiplier

    if direction == SignalDirection.LONG:
        stop   = price - stop_distance
        target = price + stop_distance * reward_risk
    else:
        stop   = price + stop_distance
        target = price - stop_distance * reward_risk

    confidence = ConfidenceScorer.trend_pullback(
        dist_to_fast, zone, ema_diff, atr, regime_state, vol_ratio
    )

    return Signal(
        symbol=symbol, direction=direction, entry_price=price,
        stop_loss=stop, take_profit=target, regime=regime_state.primary,
        stop_distance=stop_distance, reward_risk_ratio=reward_risk, atr=atr,
        reason="EMA_PULLBACK", strategy="TREND_PULLBACK", atr_ratio=regime_state.atr_ratio,
        confidence=confidence, volume_ratio=vol_ratio,
    )


# ---------------------------------------------------------------------------
# Confidence scoring — 5-factor composite [0, 1]
# ---------------------------------------------------------------------------

class ConfidenceScorer:
    """Computes a weighted 5-factor confidence score in [0, 1].

    Factors
    -------
    signal_strength  — how far into the signal zone the price is
    regime_strength  — how clearly the market is in the target regime
    volume           — current volume vs rolling average
    trend_alignment  — alignment between LTF and regime direction
    volatility_state — ATR expansion/contraction appropriate to strategy
    """

    @staticmethod
    def _weights() -> tuple[float, float, float, float, float]:
        cfg = get_config().trading
        return (
            cfg.confidence_w_signal,
            cfg.confidence_w_regime,
            cfg.confidence_w_volume,
            cfg.confidence_w_trend_align,
            cfg.confidence_w_vol_state,
        )

    @staticmethod
    def mean_reversion(
        z_score: float,
        regime_state: RegimeState,
        vol_ratio: float,
    ) -> float:
        cfg = get_config().trading
        w = ConfidenceScorer._weights()
        # 1. Signal strength: how far beyond the soft threshold
        signal_f = min(1.0, abs(z_score) / (cfg.mean_reversion_z_full + 1e-10))
        # 2. Regime: ranging regime strength
        regime_f = float(regime_state.range_score)
        # 3. Volume: normalised to [0, 1] saturating at 3× avg
        vol_f = min(1.0, vol_ratio / 3.0)
        # 4. Trend alignment: mean reversion favours low trend score
        align_f = max(0.0, 1.0 - regime_state.trend_score)
        # 5. Volatility state: stable ATR preferred (high expansion → harder reversal)
        atr_ratio = regime_state.atr_ratio
        vol_state_f = max(0.0, 1.0 - (atr_ratio - 1.0)) if atr_ratio > 1.0 else 1.0
        score = (w[0]*signal_f + w[1]*regime_f + w[2]*vol_f +
                 w[3]*align_f + w[4]*vol_state_f)
        return float(min(1.0, max(0.0, score)))

    @staticmethod
    def trend_pullback(
        dist_to_fast: float,
        zone: float,
        ema_diff: float,
        atr: float,
        regime_state: RegimeState,
        vol_ratio: float,
    ) -> float:
        w = ConfidenceScorer._weights()
        # 1. Signal strength: proximity to EMA (1.0 = at EMA touch, 0 = zone boundary)
        signal_f = max(0.0, 1.0 - dist_to_fast / (zone + 1e-10))
        # 2. Regime: trending regime strength
        regime_f = float(regime_state.trend_score)
        # 3. Volume
        vol_f = min(1.0, vol_ratio / 3.0)
        # 4. Trend alignment: EMA separation relative to ATR
        align_f = min(1.0, abs(ema_diff) / (2.0 * atr + 1e-10))
        # 5. Volatility state: stable ATR preferred for clean pullback
        atr_ratio = regime_state.atr_ratio
        vol_state_f = max(0.0, 1.0 - max(0.0, atr_ratio - 1.5) / 1.5)
        score = (w[0]*signal_f + w[1]*regime_f + w[2]*vol_f +
                 w[3]*align_f + w[4]*vol_state_f)
        return float(min(1.0, max(0.0, score)))

    @staticmethod
    def volatility_breakout(
        breakout_distance: float,
        atr: float,
        regime_state: RegimeState,
        vol_ratio: float,
    ) -> float:
        w = ConfidenceScorer._weights()
        # 1. Signal strength: how far price has broken above/below the range
        signal_f = min(1.0, breakout_distance / (atr + 1e-10))
        # 2. Regime: high-volatility regime strength (atr_ratio)
        regime_f = min(1.0, (regime_state.atr_ratio - 1.0) / 1.5) if regime_state.atr_ratio > 1.0 else 0.0
        # 3. Volume: breakout needs strong volume
        vol_f = min(1.0, vol_ratio / 3.0)
        # 4. Trend alignment: volatility expansion aligns with no specific trend → neutral 0.5
        align_f = 0.5
        # 5. Volatility state: ATR expanding → good for breakout
        vol_state_f = min(1.0, (regime_state.atr_ratio - 1.0)) if regime_state.atr_ratio > 1.0 else 0.0
        score = (w[0]*signal_f + w[1]*regime_f + w[2]*vol_f +
                 w[3]*align_f + w[4]*vol_state_f)
        return float(min(1.0, max(0.0, score)))

    @staticmethod
    def momentum_breakout(
        rsi: float,
        breakout_dist: float,
        atr: float,
        regime_state: RegimeState,
        vol_ratio: float,
        ema_f: Optional[float],
        ema_s: Optional[float],
    ) -> float:
        """5-factor confidence for Donchian momentum breakout signals."""
        w = ConfidenceScorer._weights()
        # 1. Signal strength: RSI distance from 50 (momentum strength)
        signal_f = min(1.0, abs(rsi - 50.0) / 50.0)
        # 2. Regime: expanding ATR or strong trend boosts momentum signals
        regime_f = min(1.0, (regime_state.trend_score + regime_state.atr_ratio * 0.5) / 1.5)
        # 3. Volume
        vol_f = min(1.0, vol_ratio / 3.0)
        # 4. EMA alignment: fast above/below slow adds directional conviction
        if ema_f is not None and ema_s is not None and atr > 0:
            align_f = min(1.0, abs(ema_f - ema_s) / (2.0 * atr + 1e-10))
        else:
            align_f = 0.3
        # 5. Volatility state: moderately expanding ATR is best for breakouts
        atr_r = regime_state.atr_ratio
        vol_state_f = min(1.0, max(0.0, (atr_r - 0.8) / 1.0))
        score = (w[0]*signal_f + w[1]*regime_f + w[2]*vol_f +
                 w[3]*align_f + w[4]*vol_state_f)
        return float(min(1.0, max(0.0, score)))


# ---------------------------------------------------------------------------
# Volatility Breakout strategy (HIGH_VOLATILITY regime)
# ---------------------------------------------------------------------------

def _volatility_breakout_from_cache(
    symbol: str,
    cache: "IndicatorCache",
    idx: int,
    price: float,
    atr: float,
    regime_state: RegimeState,
) -> Signal:
    """Break above N-bar high or below N-bar low with expanding ATR confirmation."""
    cfg = get_config().trading

    high_n = cache.get_n_bar_high(idx)
    low_n  = cache.get_n_bar_low(idx)
    if high_n is None or low_n is None:
        return _no_signal(symbol, price, atr, regime_state.primary, "BREAKOUT_LEVELS_NOT_READY",
                          "VOL_BREAKOUT", regime_state.atr_ratio)

    # ATR must be expanding — otherwise no momentum confirmation
    if regime_state.atr_ratio < 1.0:
        return _no_signal(symbol, price, atr, regime_state.primary,
                          f"ATR_CONTRACTING(ratio={regime_state.atr_ratio:.2f})",
                          "VOL_BREAKOUT", regime_state.atr_ratio)

    vol_ratio = cache.get_raw_vol_ratio(idx)
    if vol_ratio < cfg.volume_ratio_min:
        return _no_signal(symbol, price, atr, regime_state.primary,
                          f"VOL_RATIO_LOW({vol_ratio:.2f})",
                          "VOL_BREAKOUT", regime_state.atr_ratio)

    direction = SignalDirection.NONE
    breakout_dist = 0.0

    if price > high_n:
        direction = SignalDirection.LONG
        breakout_dist = price - high_n
    elif price < low_n:
        direction = SignalDirection.SHORT
        breakout_dist = low_n - price

    if direction == SignalDirection.NONE:
        return _no_signal(symbol, price, atr, regime_state.primary,
                          f"NO_BREAKOUT(h={high_n:.2f},l={low_n:.2f},p={price:.2f})",
                          "VOL_BREAKOUT", regime_state.atr_ratio)

    stop_distance = atr * cfg.atr_stop_volatility_multiplier
    reward_risk   = cfg.atr_target_volatility_multiplier / cfg.atr_stop_volatility_multiplier

    if direction == SignalDirection.LONG:
        stop   = price - stop_distance
        target = price + stop_distance * reward_risk
    else:
        stop   = price + stop_distance
        target = price - stop_distance * reward_risk

    confidence = ConfidenceScorer.volatility_breakout(breakout_dist, atr, regime_state, vol_ratio)

    return Signal(
        symbol=symbol, direction=direction, entry_price=price,
        stop_loss=stop, take_profit=target, regime=regime_state.primary,
        stop_distance=stop_distance, reward_risk_ratio=reward_risk, atr=atr,
        reason=f"VOL_BREAKOUT({'LONG' if direction==SignalDirection.LONG else 'SHORT'})",
        strategy="VOL_BREAKOUT", atr_ratio=regime_state.atr_ratio,
        confidence=confidence, volume_ratio=vol_ratio,
    )


def _volatility_breakout_signal(
    symbol: str,
    df: pd.DataFrame,
    price: float,
    atr: float,
    regime_state: RegimeState,
) -> Signal:
    """Per-call path for paper/live — computes N-bar high/low fresh."""
    cfg = get_config().trading
    n = cfg.vol_breakout_lookback
    if len(df) < n + 1:
        return _no_signal(symbol, price, atr, regime_state.primary, "NOT_ENOUGH_BARS",
                          "VOL_BREAKOUT", regime_state.atr_ratio)

    high_n = float(df["high"].iloc[-(n + 1):-1].max())
    low_n  = float(df["low"].iloc[-(n + 1):-1].min())

    if regime_state.atr_ratio < 1.0:
        return _no_signal(symbol, price, atr, regime_state.primary,
                          f"ATR_CONTRACTING(ratio={regime_state.atr_ratio:.2f})",
                          "VOL_BREAKOUT", regime_state.atr_ratio)

    vol_ratio = _compute_vol_ratio(df, lookback=cfg.volume_lookback)
    if vol_ratio < cfg.volume_ratio_min:
        return _no_signal(symbol, price, atr, regime_state.primary,
                          f"VOL_RATIO_LOW({vol_ratio:.2f})",
                          "VOL_BREAKOUT", regime_state.atr_ratio)

    direction = SignalDirection.NONE
    breakout_dist = 0.0

    if price > high_n:
        direction = SignalDirection.LONG
        breakout_dist = price - high_n
    elif price < low_n:
        direction = SignalDirection.SHORT
        breakout_dist = low_n - price

    if direction == SignalDirection.NONE:
        return _no_signal(symbol, price, atr, regime_state.primary,
                          f"NO_BREAKOUT(h={high_n:.2f},l={low_n:.2f},p={price:.2f})",
                          "VOL_BREAKOUT", regime_state.atr_ratio)

    stop_distance = atr * cfg.atr_stop_volatility_multiplier
    reward_risk   = cfg.atr_target_volatility_multiplier / cfg.atr_stop_volatility_multiplier

    if direction == SignalDirection.LONG:
        stop   = price - stop_distance
        target = price + stop_distance * reward_risk
    else:
        stop   = price + stop_distance
        target = price - stop_distance * reward_risk

    confidence = ConfidenceScorer.volatility_breakout(breakout_dist, atr, regime_state, vol_ratio)

    return Signal(
        symbol=symbol, direction=direction, entry_price=price,
        stop_loss=stop, take_profit=target, regime=regime_state.primary,
        stop_distance=stop_distance, reward_risk_ratio=reward_risk, atr=atr,
        reason=f"VOL_BREAKOUT({'LONG' if direction==SignalDirection.LONG else 'SHORT'})",
        strategy="VOL_BREAKOUT", atr_ratio=regime_state.atr_ratio,
        confidence=confidence, volume_ratio=vol_ratio,
    )


def _compute_vol_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    """Return current_vol / rolling_avg_vol for live/paper path."""
    if len(df) < lookback + 1:
        return 1.0
    avg_vol  = float(df["volume"].iloc[-(lookback + 1):-1].mean())
    last_vol = float(df["volume"].iloc[-1])
    return last_vol / (avg_vol + 1e-10)


def _compute_rsi_last(df: pd.DataFrame, period: int = 14) -> float:
    """Compute the last RSI value for the live/paper signal path."""
    if len(df) < period + 5:
        return 50.0
    try:
        rsi_ind = ta.momentum.RSIIndicator(close=df["close"], window=period)
        s = rsi_ind.rsi()
        if s.empty:
            return 50.0
        v = s.iloc[-1]
        return float(v) if not pd.isna(v) else 50.0
    except Exception:
        return 50.0


# ---------------------------------------------------------------------------
# Momentum Breakout strategy (TRENDING + RANGING regimes)
# Donchian channel breakout with RSI momentum confirmation
# ---------------------------------------------------------------------------

def _momentum_breakout_from_cache(
    symbol: str,
    cache: "IndicatorCache",
    idx: int,
    price: float,
    atr: float,
    regime_state: RegimeState,
) -> Signal:
    """Donchian channel breakout with RSI confirmation — cache path."""
    cfg = get_config().trading

    d_high = cache.get_donchian_high(idx)
    d_low  = cache.get_donchian_low(idx)
    if d_high is None or d_low is None:
        return _no_signal(symbol, price, atr, regime_state.primary,
                          "DONCHIAN_NOT_READY", "MOMENTUM_BREAKOUT", regime_state.atr_ratio)

    vol_ratio = cache.get_raw_vol_ratio(idx)
    if vol_ratio < cfg.volume_ratio_min:
        return _no_signal(symbol, price, atr, regime_state.primary,
                          f"VOL_RATIO_LOW({vol_ratio:.2f})", "MOMENTUM_BREAKOUT", regime_state.atr_ratio)

    rsi = cache.get_rsi(idx)
    ema_f = cache.get_ema_fast(idx)
    ema_s = cache.get_ema_slow(idx)

    direction = SignalDirection.NONE
    breakout_dist = 0.0

    if price > d_high and rsi > cfg.momentum_rsi_long:
        direction = SignalDirection.LONG
        breakout_dist = price - d_high
    elif price < d_low and rsi < cfg.momentum_rsi_short:
        direction = SignalDirection.SHORT
        breakout_dist = d_low - price

    if direction == SignalDirection.NONE:
        reason = (
            f"NO_BREAKOUT(p={price:.4f},h={d_high:.4f},l={d_low:.4f},rsi={rsi:.1f})"
        )
        return _no_signal(symbol, price, atr, regime_state.primary,
                          reason, "MOMENTUM_BREAKOUT", regime_state.atr_ratio)

    stop_distance = atr * cfg.donchian_atr_stop_multiplier
    reward_risk   = cfg.donchian_rr

    if direction == SignalDirection.LONG:
        stop   = price - stop_distance
        target = price + stop_distance * reward_risk
    else:
        stop   = price + stop_distance
        target = price - stop_distance * reward_risk

    confidence = ConfidenceScorer.momentum_breakout(
        rsi, breakout_dist, atr, regime_state, vol_ratio, ema_f, ema_s
    )

    return Signal(
        symbol=symbol, direction=direction, entry_price=price,
        stop_loss=stop, take_profit=target, regime=regime_state.primary,
        stop_distance=stop_distance, reward_risk_ratio=reward_risk, atr=atr,
        reason=f"MOMENTUM_BREAKOUT({'LONG' if direction == SignalDirection.LONG else 'SHORT'})",
        strategy="MOMENTUM_BREAKOUT", atr_ratio=regime_state.atr_ratio,
        confidence=confidence, volume_ratio=vol_ratio,
    )


def _momentum_breakout_signal(
    symbol: str,
    df: pd.DataFrame,
    price: float,
    atr: float,
    regime_state: RegimeState,
) -> Signal:
    """Momentum breakout strategy for the live/paper path."""
    cfg = get_config().trading
    n = cfg.donchian_period
    if len(df) < n + 5:
        return _no_signal(symbol, price, atr, regime_state.primary,
                          "NOT_ENOUGH_BARS", "MOMENTUM_BREAKOUT", regime_state.atr_ratio)

    d_high = float(df["high"].iloc[-(n + 1):-1].max())
    d_low  = float(df["low"].iloc[-(n + 1):-1].min())

    vol_ratio = _compute_vol_ratio(df, lookback=cfg.volume_lookback)
    if vol_ratio < cfg.volume_ratio_min:
        return _no_signal(symbol, price, atr, regime_state.primary,
                          f"VOL_RATIO_LOW({vol_ratio:.2f})", "MOMENTUM_BREAKOUT", regime_state.atr_ratio)

    rsi = _compute_rsi_last(df, period=cfg.rsi_period)
    close = df["close"]
    ema_fast_s = ta.trend.EMAIndicator(close=close, window=cfg.ema_fast).ema_indicator()
    ema_slow_s = ta.trend.EMAIndicator(close=close, window=cfg.ema_slow).ema_indicator()
    ema_f = float(ema_fast_s.iloc[-1]) if not ema_fast_s.empty else None
    ema_s = float(ema_slow_s.iloc[-1]) if not ema_slow_s.empty else None

    direction = SignalDirection.NONE
    breakout_dist = 0.0

    if price > d_high and rsi > cfg.momentum_rsi_long:
        direction = SignalDirection.LONG
        breakout_dist = price - d_high
    elif price < d_low and rsi < cfg.momentum_rsi_short:
        direction = SignalDirection.SHORT
        breakout_dist = d_low - price

    if direction == SignalDirection.NONE:
        reason = f"NO_BREAKOUT(p={price:.4f},h={d_high:.4f},l={d_low:.4f},rsi={rsi:.1f})"
        return _no_signal(symbol, price, atr, regime_state.primary,
                          reason, "MOMENTUM_BREAKOUT", regime_state.atr_ratio)

    stop_distance = atr * cfg.donchian_atr_stop_multiplier
    reward_risk   = cfg.donchian_rr

    if direction == SignalDirection.LONG:
        stop   = price - stop_distance
        target = price + stop_distance * reward_risk
    else:
        stop   = price + stop_distance
        target = price - stop_distance * reward_risk

    confidence = ConfidenceScorer.momentum_breakout(
        rsi, breakout_dist, atr, regime_state, vol_ratio, ema_f, ema_s
    )

    return Signal(
        symbol=symbol, direction=direction, entry_price=price,
        stop_loss=stop, take_profit=target, regime=regime_state.primary,
        stop_distance=stop_distance, reward_risk_ratio=reward_risk, atr=atr,
        reason=f"MOMENTUM_BREAKOUT({'LONG' if direction == SignalDirection.LONG else 'SHORT'})",
        strategy="MOMENTUM_BREAKOUT", atr_ratio=regime_state.atr_ratio,
        confidence=confidence, volume_ratio=vol_ratio,
    )


# ---------------------------------------------------------------------------
# OOP strategy classes — thin wrappers over the validated signal functions
# Provides clean interface for future live hot-swap and parameter injection
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """Abstract base for all strategies."""

    @abstractmethod
    def generate_from_cache(
        self, symbol: str, cache: "IndicatorCache", idx: int,
        price: float, atr: float, regime_state: RegimeState,
    ) -> Signal: ...

    @abstractmethod
    def generate_live(
        self, symbol: str, df: pd.DataFrame,
        price: float, atr: float, regime_state: RegimeState,
    ) -> Signal: ...


class MeanReversionStrategy(BaseStrategy):
    """Bollinger Band z-score reversion for RANGING markets."""

    def generate_from_cache(
        self, symbol: str, cache: "IndicatorCache", idx: int,
        price: float, atr: float, regime_state: RegimeState,
    ) -> Signal:
        return _mr_signal_from_cache(symbol, cache, idx, price, atr, regime_state)

    def generate_live(
        self, symbol: str, df: pd.DataFrame,
        price: float, atr: float, regime_state: RegimeState,
    ) -> Signal:
        return _mean_reversion_signal(symbol, df, price, atr, regime_state)


class TrendPullbackStrategy(BaseStrategy):
    """EMA pullback with trend confirmation for TRENDING markets."""

    def generate_from_cache(
        self, symbol: str, cache: "IndicatorCache", idx: int,
        price: float, atr: float, regime_state: RegimeState,
    ) -> Signal:
        return _ema_pullback_from_cache(symbol, cache, idx, price, atr, regime_state)

    def generate_live(
        self, symbol: str, df: pd.DataFrame,
        price: float, atr: float, regime_state: RegimeState,
    ) -> Signal:
        return _ema_pullback_signal(symbol, df, price, atr, regime_state)


class VolatilityBreakoutStrategy(BaseStrategy):
    """N-bar high/low breakout with ATR expansion confirmation for HIGH_VOLATILITY."""

    def generate_from_cache(
        self, symbol: str, cache: "IndicatorCache", idx: int,
        price: float, atr: float, regime_state: RegimeState,
    ) -> Signal:
        return _volatility_breakout_from_cache(symbol, cache, idx, price, atr, regime_state)

    def generate_live(
        self, symbol: str, df: pd.DataFrame,
        price: float, atr: float, regime_state: RegimeState,
    ) -> Signal:
        return _volatility_breakout_signal(symbol, df, price, atr, regime_state)


class MomentumBreakoutStrategy(BaseStrategy):
    """Donchian channel breakout with RSI confirmation for TRENDING and RANGING regimes."""

    def generate_from_cache(
        self, symbol: str, cache: "IndicatorCache", idx: int,
        price: float, atr: float, regime_state: RegimeState,
    ) -> Signal:
        return _momentum_breakout_from_cache(symbol, cache, idx, price, atr, regime_state)

    def generate_live(
        self, symbol: str, df: pd.DataFrame,
        price: float, atr: float, regime_state: RegimeState,
    ) -> Signal:
        return _momentum_breakout_signal(symbol, df, price, atr, regime_state)


class StrategyRouter:
    """Selects and invokes the appropriate strategy based on regime.

    Routing table (priority order within each regime):
      RANGING        → MeanReversion (primary), MomentumBreakout (secondary)
      TRENDING       → TrendPullback (primary), MomentumBreakout (secondary)
      HIGH_VOLATILITY→ VolatilityBreakout (if enabled), else MomentumBreakout
    """

    def __init__(self) -> None:
        self._mr   = MeanReversionStrategy()
        self._tp   = TrendPullbackStrategy()
        self._vb   = VolatilityBreakoutStrategy()
        self._mb   = MomentumBreakoutStrategy()

    def _best_signal(self, a: Signal, b: Signal) -> Signal:
        """Return the higher-confidence signal; NONE loses to any valid signal."""
        if a.direction == SignalDirection.NONE:
            return b
        if b.direction == SignalDirection.NONE:
            return a
        return a if a.confidence >= b.confidence else b

    def route_from_cache(
        self, symbol: str, cache: "IndicatorCache", idx: int,
        price: float, atr: float, regime_state: RegimeState,
    ) -> Signal:
        cfg = get_config().trading
        regime = regime_state.primary

        if regime == Regime.RANGING:
            mr_sig = self._mr.generate_from_cache(symbol, cache, idx, price, atr, regime_state)
            if cfg.disable_momentum_strategy:
                return mr_sig
            mb_sig = self._mb.generate_from_cache(symbol, cache, idx, price, atr, regime_state)
            return self._best_signal(mr_sig, mb_sig)

        if regime == Regime.TRENDING:
            primary = (
                self._tp.generate_from_cache(symbol, cache, idx, price, atr, regime_state)
                if not cfg.disable_trend_strategy
                else _no_signal(symbol, price, atr, regime, "TREND_STRATEGY_DISABLED", "NONE", regime_state.atr_ratio)
            )
            if cfg.disable_momentum_strategy:
                return primary
            mb_sig = self._mb.generate_from_cache(symbol, cache, idx, price, atr, regime_state)
            return self._best_signal(primary, mb_sig)

        if regime == Regime.HIGH_VOLATILITY:
            if not cfg.disable_volatility_strategy:
                return self._vb.generate_from_cache(symbol, cache, idx, price, atr, regime_state)
            # HIGH_VOL without VB: fall back to Momentum Breakout with wider stop + confidence penalty.
            sig = self._mb.generate_from_cache(symbol, cache, idx, price, atr, regime_state)
            if sig.direction != SignalDirection.NONE:
                sig.confidence *= cfg.high_vol_size_reduction  # 0.5× confidence → 0.5× size
                # Widen stop to HV multiplier to survive 1m volatility spikes
                hv_stop_dist = atr * cfg.atr_stop_hv_multiplier
                if sig.direction == SignalDirection.LONG:
                    sig.stop_loss = price - hv_stop_dist
                    sig.take_profit = price + hv_stop_dist * sig.reward_risk_ratio
                else:
                    sig.stop_loss = price + hv_stop_dist
                    sig.take_profit = price - hv_stop_dist * sig.reward_risk_ratio
                sig.stop_distance = hv_stop_dist
                sig.reason = f"HIGH_VOL_MB({sig.reason})"
            return sig

        # Unknown regime fallback: run MB with half confidence rather than blocking
        sig = self._mb.generate_from_cache(symbol, cache, idx, price, atr, regime_state)
        if sig.direction != SignalDirection.NONE:
            sig.confidence *= cfg.high_vol_size_reduction
        return sig

    def route_live(
        self, symbol: str, df: pd.DataFrame,
        price: float, atr: float, regime_state: RegimeState,
    ) -> Signal:
        cfg = get_config().trading
        regime = regime_state.primary

        if regime == Regime.RANGING:
            mr_sig = self._mr.generate_live(symbol, df, price, atr, regime_state)
            if cfg.disable_momentum_strategy:
                return mr_sig
            mb_sig = self._mb.generate_live(symbol, df, price, atr, regime_state)
            return self._best_signal(mr_sig, mb_sig)

        if regime == Regime.TRENDING:
            primary = (
                self._tp.generate_live(symbol, df, price, atr, regime_state)
                if not cfg.disable_trend_strategy
                else _no_signal(symbol, price, atr, regime, "TREND_STRATEGY_DISABLED", "NONE", regime_state.atr_ratio)
            )
            if cfg.disable_momentum_strategy:
                return primary
            mb_sig = self._mb.generate_live(symbol, df, price, atr, regime_state)
            return self._best_signal(primary, mb_sig)

        if regime == Regime.HIGH_VOLATILITY:
            if not cfg.disable_volatility_strategy:
                return self._vb.generate_live(symbol, df, price, atr, regime_state)
            # HIGH_VOL without VB: fall back to Momentum Breakout with wider stop + confidence penalty.
            sig = self._mb.generate_live(symbol, df, price, atr, regime_state)
            if sig.direction != SignalDirection.NONE:
                sig.confidence *= cfg.high_vol_size_reduction  # 0.5× confidence → 0.5× size
                # Widen stop to HV multiplier to survive 1m volatility spikes
                hv_stop_dist = atr * cfg.atr_stop_hv_multiplier
                if sig.direction == SignalDirection.LONG:
                    sig.stop_loss = price - hv_stop_dist
                    sig.take_profit = price + hv_stop_dist * sig.reward_risk_ratio
                else:
                    sig.stop_loss = price + hv_stop_dist
                    sig.take_profit = price - hv_stop_dist * sig.reward_risk_ratio
                sig.stop_distance = hv_stop_dist
                sig.reason = f"HIGH_VOL_MB({sig.reason})"
            return sig

        # Unknown regime fallback: run MB with half confidence rather than blocking
        sig = self._mb.generate_live(symbol, df, price, atr, regime_state)
        if sig.direction != SignalDirection.NONE:
            sig.confidence *= cfg.high_vol_size_reduction
        return sig


# Module-level singleton (instantiated once)
_router = StrategyRouter()


# ---------------------------------------------------------------------------
# Fast signal generation using precomputed cache (used by backtester)
# ---------------------------------------------------------------------------

def generate_signal_from_cache(
    symbol: str,
    cache: "IndicatorCache",
    idx: int,
) -> Signal:
    """O(1) signal generation using precomputed indicator cache."""
    price = float(cache.df["close"].iloc[idx])
    atr   = cache.get_atr(idx)
    regime_state = cache.get_regime(idx)
    return _router.route_from_cache(symbol, cache, idx, price, atr, regime_state)


def _trend_signal_from_cache(
    symbol: str,
    cache: "IndicatorCache",
    idx: int,
    price: float,
    atr: float,
    regime_state: RegimeState,
) -> Signal:
    cfg = get_config().trading
    ema_f = cache.get_ema_fast(idx)
    ema_s = cache.get_ema_slow(idx)
    if ema_f is None or ema_s is None:
        return _no_signal(symbol, price, atr, regime_state.primary, "EMA_NOT_READY", "TREND", regime_state.atr_ratio)

    ema_diff = ema_f - ema_s
    price_z  = (price - ema_s) / (atr + 1e-10)

    if abs(price_z) < cfg.trend_breakout_z or ema_diff == 0:
        return _no_signal(symbol, price, atr, regime_state.primary, "TREND_BREAKOUT_INACTIVE", "TREND", regime_state.atr_ratio)

    if not cache.volume_is_spiking(idx, lookback=20, multiplier=cfg.volume_spike_multiplier):
        return _no_signal(symbol, price, atr, regime_state.primary, "LOW_VOLUME", "TREND", regime_state.atr_ratio)

    direction = SignalDirection.LONG if ema_diff > 0 else SignalDirection.SHORT
    stop_distance = atr * cfg.atr_stop_trend_multiplier
    reward_risk   = cfg.atr_target_trend_multiplier / cfg.atr_stop_trend_multiplier

    if direction == SignalDirection.LONG:
        stop   = price - stop_distance
        target = price + stop_distance * reward_risk
    else:
        stop   = price + stop_distance
        target = price - stop_distance * reward_risk

    confidence = min(1.0, abs(price_z) / (cfg.trend_breakout_z + 1e-10))
    return Signal(
        symbol=symbol, direction=direction, entry_price=price,
        stop_loss=stop, take_profit=target, regime=regime_state.primary,
        stop_distance=stop_distance, reward_risk_ratio=reward_risk, atr=atr,
        reason="TREND_BREAKOUT", strategy="TREND", atr_ratio=regime_state.atr_ratio,
        confidence=confidence,
    )


def _mr_signal_from_cache(
    symbol: str,
    cache: "IndicatorCache",
    idx: int,
    price: float,
    atr: float,
    regime_state: RegimeState,
) -> Signal:
    cfg = get_config().trading
    mid, up, low = cache.get_bb(idx)
    if mid is None:
        return _no_signal(symbol, price, atr, regime_state.primary, "BB_NOT_READY", "MEAN_REVERSION", regime_state.atr_ratio)

    band_width = (up - mid) + 1e-10
    z_score    = (price - mid) / band_width

    direction = SignalDirection.NONE
    reason    = f"BB_ZSCORE_NEUTRAL(z={z_score:.2f},thr={cfg.mean_reversion_z_entry})"

    if z_score <= -cfg.mean_reversion_z_entry:
        direction = SignalDirection.LONG
        reason    = "BB_ZSCORE_LONG"
    elif z_score >= cfg.mean_reversion_z_entry:
        direction = SignalDirection.SHORT
        reason    = "BB_ZSCORE_SHORT"

    if direction == SignalDirection.NONE:
        return _no_signal(symbol, price, atr, regime_state.primary, reason, "MEAN_REVERSION", regime_state.atr_ratio)

    vol_ratio = cache.get_raw_vol_ratio(idx)
    if vol_ratio < cfg.volume_ratio_min:
        return _no_signal(symbol, price, atr, regime_state.primary,
                          f"VOL_RATIO_LOW({vol_ratio:.2f})", "MEAN_REVERSION", regime_state.atr_ratio)

    stop_distance = atr * cfg.atr_stop_range_multiplier
    reward_risk   = cfg.atr_target_range_multiplier / cfg.atr_stop_range_multiplier

    if direction == SignalDirection.LONG:
        stop   = price - stop_distance
        target = price + stop_distance * reward_risk
    else:
        stop   = price + stop_distance
        target = price - stop_distance * reward_risk

    # Confidence normalised against z_full: <1 for soft entries, =1 beyond full threshold
    confidence = ConfidenceScorer.mean_reversion(z_score, regime_state, vol_ratio)
    return Signal(
        symbol=symbol, direction=direction, entry_price=price,
        stop_loss=stop, take_profit=target, regime=regime_state.primary,
        stop_distance=stop_distance, reward_risk_ratio=reward_risk, atr=atr,
        reason=reason, strategy="MEAN_REVERSION", atr_ratio=regime_state.atr_ratio,
        confidence=confidence, volume_ratio=vol_ratio,
    )


# ---------------------------------------------------------------------------
# Legacy per-bar API (kept for paper/live trading engine compatibility)
# ---------------------------------------------------------------------------

def generate_signal(symbol: str, df: pd.DataFrame) -> Signal:
    """Return a signal for the active regime strategy — paper/live entry point."""
    regime_state  = detect_regime(df)
    current_price = float(df["close"].iloc[-1])
    atr           = _compute_atr(df)
    return _router.route_live(symbol, df, current_price, atr, regime_state)


def _compute_htf_trend(df: pd.DataFrame, resample_n: int = 4) -> int:
    try:
        n = resample_n
        rows = (len(df) // n) * n
        if rows < n * 30:
            return 0
        sliced = df.iloc[-rows:].copy()
        sliced = sliced.reset_index(drop=True)
        groups = sliced.groupby(sliced.index // n)
        htf = pd.DataFrame({
            "open":  groups["open"].first(),
            "high":  groups["high"].max(),
            "low":   groups["low"].min(),
            "close": groups["close"].last(),
            "volume": groups["volume"].sum(),
        })
        ema_fast = ta.trend.EMAIndicator(close=htf["close"], window=9).ema_indicator()
        ema_slow = ta.trend.EMAIndicator(close=htf["close"], window=21).ema_indicator()
        if ema_fast.empty or ema_slow.empty or pd.isna(ema_fast.iloc[-1]) or pd.isna(ema_slow.iloc[-1]):
            return 0
        diff = float(ema_fast.iloc[-1]) - float(ema_slow.iloc[-1])
        if diff > 0:
            return 1
        if diff < 0:
            return -1
    except Exception:
        pass
    return 0


def _volume_is_spiking(df: pd.DataFrame, lookback: int = 20, multiplier: float = 1.3) -> bool:
    if len(df) < lookback + 1:
        return False
    avg_vol = float(df["volume"].iloc[-(lookback + 1):-1].mean())
    last_vol = float(df["volume"].iloc[-1])
    return last_vol >= avg_vol * multiplier


def _trend_following_signal(
    symbol: str,
    df: pd.DataFrame,
    price: float,
    atr: float,
    regime_state: RegimeState,
) -> Signal:
    cfg = get_config().trading
    close = df["close"]
    ema_fast = ta.trend.EMAIndicator(close=close, window=cfg.ema_fast).ema_indicator()
    ema_slow = ta.trend.EMAIndicator(close=close, window=cfg.ema_slow).ema_indicator()

    if ema_fast.empty or ema_slow.empty:
        return _no_signal(symbol, price, atr, regime_state.primary, "EMA_NOT_READY", "TREND", regime_state.atr_ratio)

    ema_fast_cur = float(ema_fast.iloc[-1])
    ema_slow_cur = float(ema_slow.iloc[-1])
    ema_diff = ema_fast_cur - ema_slow_cur
    price_z = (price - ema_slow_cur) / (atr + 1e-10)

    if abs(price_z) < cfg.trend_breakout_z or ema_diff == 0:
        return _no_signal(symbol, price, atr, regime_state.primary, "TREND_BREAKOUT_INACTIVE", "TREND", regime_state.atr_ratio)

    if not _volume_is_spiking(df, lookback=20, multiplier=cfg.volume_spike_multiplier):
        return _no_signal(symbol, price, atr, regime_state.primary, "LOW_VOLUME", "TREND", regime_state.atr_ratio)

    htf = _compute_htf_trend(df, resample_n=4)
    tentative_dir = 1 if ema_diff > 0 else -1
    if htf != 0 and htf != tentative_dir:
        return _no_signal(symbol, price, atr, regime_state.primary, "HTF_CONFLICT", "TREND", regime_state.atr_ratio)

    direction = SignalDirection.LONG if ema_diff > 0 else SignalDirection.SHORT
    stop_distance = atr * cfg.atr_stop_trend_multiplier
    target_mult = cfg.atr_target_trend_multiplier
    reward_risk = target_mult / cfg.atr_stop_trend_multiplier

    if direction == SignalDirection.LONG:
        stop = price - stop_distance
        target = price + stop_distance * reward_risk
    else:
        stop = price + stop_distance
        target = price - stop_distance * reward_risk

    confidence = min(1.0, abs(price_z) / (cfg.trend_breakout_z + 1e-10))

    return Signal(
        symbol=symbol, direction=direction, entry_price=price,
        stop_loss=stop, take_profit=target, regime=regime_state.primary,
        stop_distance=stop_distance, reward_risk_ratio=reward_risk, atr=atr,
        reason="TREND_BREAKOUT", strategy="TREND", atr_ratio=regime_state.atr_ratio,
        confidence=confidence,
    )


def _mean_reversion_signal(
    symbol: str,
    df: pd.DataFrame,
    price: float,
    atr: float,
    regime_state: RegimeState,
) -> Signal:
    cfg = get_config().trading
    close = df["close"]
    bb = ta.volatility.BollingerBands(close=close, window=cfg.bb_period, window_dev=cfg.bb_std)
    bb_mid = bb.bollinger_mavg()
    bb_up = bb.bollinger_hband()
    bb_low = bb.bollinger_lband()

    if bb_mid.empty or bb_up.empty or bb_low.empty:
        return _no_signal(symbol, price, atr, regime_state.primary, "BB_NOT_READY", "MEAN_REVERSION", regime_state.atr_ratio)

    mid = float(bb_mid.iloc[-1])
    up = float(bb_up.iloc[-1])
    low = float(bb_low.iloc[-1])
    band_width = (up - mid) + 1e-10
    z_score = (price - mid) / band_width

    direction = SignalDirection.NONE
    reason = f"BB_ZSCORE_NEUTRAL(z={z_score:.2f},thr={cfg.mean_reversion_z_entry})"

    if z_score <= -cfg.mean_reversion_z_entry:
        direction = SignalDirection.LONG
        reason = "BB_ZSCORE_LONG"
    elif z_score >= cfg.mean_reversion_z_entry:
        direction = SignalDirection.SHORT
        reason = "BB_ZSCORE_SHORT"

    if direction == SignalDirection.NONE:
        return _no_signal(symbol, price, atr, regime_state.primary, reason, "MEAN_REVERSION", regime_state.atr_ratio)

    vol_ratio = _compute_vol_ratio(df, lookback=cfg.volume_lookback)
    if vol_ratio < cfg.volume_ratio_min:
        return _no_signal(symbol, price, atr, regime_state.primary,
                          f"VOL_RATIO_LOW({vol_ratio:.2f})", "MEAN_REVERSION", regime_state.atr_ratio)

    stop_distance = atr * cfg.atr_stop_range_multiplier
    reward_risk = cfg.atr_target_range_multiplier / cfg.atr_stop_range_multiplier

    if direction == SignalDirection.LONG:
        stop = price - stop_distance
        target = price + stop_distance * reward_risk
    else:
        stop = price + stop_distance
        target = price - stop_distance * reward_risk

    # Confidence normalised against z_full so soft entries (<z_full) get proportionally smaller position
    confidence = ConfidenceScorer.mean_reversion(z_score, regime_state, vol_ratio)

    return Signal(
        symbol=symbol, direction=direction, entry_price=price,
        stop_loss=stop, take_profit=target, regime=regime_state.primary,
        stop_distance=stop_distance, reward_risk_ratio=reward_risk, atr=atr,
        reason=reason, strategy="MEAN_REVERSION", atr_ratio=regime_state.atr_ratio,
        confidence=confidence, volume_ratio=vol_ratio,
    )


def _no_signal(
    symbol: str,
    price: float,
    atr: float,
    regime: Regime,
    reason: str,
    strategy: str,
    atr_ratio: float,
) -> Signal:
    return Signal(
        symbol=symbol,
        direction=SignalDirection.NONE,
        entry_price=price,
        stop_loss=0.0,
        take_profit=None,
        regime=regime,
        stop_distance=0.0,
        reward_risk_ratio=0.0,
        atr=atr,
        reason=reason,
        strategy=strategy,
        atr_ratio=atr_ratio,
        confidence=0.0,
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
