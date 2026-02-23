"""
config.py - Central Configuration for Bitget Futures Trading Engine
All tuneable parameters live here. Edit this file to customize behavior.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class APIConfig:
    api_key: str = field(default_factory=lambda: os.getenv("BITGET_API_KEY", ""))
    secret_key: str = field(default_factory=lambda: os.getenv("BITGET_SECRET_KEY", ""))
    passphrase: str = field(default_factory=lambda: os.getenv("BITGET_PASSPHRASE", ""))
    base_url: str = "https://api.bitget.com"
    ws_url: str = "wss://ws.bitget.com/v2/ws/public"
    ws_private_url: str = "wss://ws.bitget.com/v2/ws/private"
    # WARNING: Only disable SSL verification for development if you have SSL inspection/proxy issues
    # Never use this in production with real funds
    disable_ssl_verification: bool = field(default_factory=lambda: os.getenv("DISABLE_SSL_VERIFICATION", "false").lower() == "true")


@dataclass
class TradingConfig:
    # ---- Mode ----
    mode: str = field(default_factory=lambda: os.getenv("TRADING_MODE", "paper"))
    # Options: "paper" | "live"

    # ---- Universe / Symbol Selection ----
    min_volume_24h_usdt: float = 20_000_000   # $20M threshold; low-quality tokens excluded via symbol_blocklist
    top_n_symbols: int = 20                    # Rank top 20 by volume for 1m scanning
    symbol_blocklist: list = field(default_factory=lambda: [
        "XAUUSDT",    # Gold: 1m ATR too small vs price (sub-pip moves)
        "AZTECUSDT",  # Low-cap: thin liquidity, outsized losses vs model
        "ESPUSDT",    # Low-cap small: same pattern as AZTEC
        "YGGUSDT",    # Low-cap gaming token: wide spreads
        "SIRENUSDT",  # Micro-cap: occasionally appears, always problematic
        "KITEUSDT",   # Low-cap: appeared in run 4, blocklisted preemptively
    ])  # Expand as new problem tokens appear in logs
    scan_interval_seconds: int = 30            # 30s sub-candle scanning on 1m timeframe
    max_concurrent_positions: int = 5           # Hard cap on open positions (tight to prevent clustering)
    max_new_trades_per_cycle: int = 3           # Max new entries per single scan cycle (raised: 2 was blocking valid signals when 2 fill simultaneously)
    max_same_direction_positions: int = 4       # Soft cap: up to 4 same-direction positions; decay ladder applied in execution_engine
    same_direction_risk_decay: float = 0.5      # Each additional same-direction position scales new entry risk by this factor
    atr_compression_ratio: float = 0.7          # ATR below this fraction of median = compressed market → halve position size

    # ---- Risk ----
    risk_per_trade_pct: float = 0.005          # 0.5% base risk per trade (prop-desk standard)
    max_risk_per_trade_pct: float = 0.01       # Absolute hard cap per trade (1%)
    portfolio_heat_cap_pct: float = 0.03       # Max 3% total open risk at any time
    max_correlated_base_pct: float = 0.015     # Max 1.5% exposure per base asset (BTC, ETH, …)
    max_daily_loss_pct: float = 0.03           # Stop ALL trading after 3% daily drawdown
    max_consecutive_losses: int = 3            # Halve size after N consecutive losses
    max_directional_exposure_pct: float = 0.015 # Max 1.5% risk on long side / 1.5% on short side (tightened)
    max_notional_per_trade_x: float = 0.30     # Per-trade notional ≤ 30% equity (was 2.5× — allowed $21k notional on $10k account)
    max_total_notional_x: float = 1.0           # Total portfolio notional ≤ 100% equity (was 5.0×)
    correlation_threshold: float = 0.75        # Threshold for correlation-based scaling
    directional_corr_penalty_threshold: float = 0.80  # Corr above this + same direction → halve risk

    # ---- Execution ----
    default_leverage: int = 3
    order_retry_attempts: int = 3
    order_retry_delay_seconds: float = 1.5
    slippage_bps: float = 3.0                  # 1m market orders: slightly wider spread
    maker_fee_pct: float = 0.0002             # 0.02% maker (limit order)
    taker_fee_pct: float = 0.0002             # use maker fee - strategy enters with limit orders

    # ---- Strategy Parameters ----
    ema_fast: int = 9
    ema_slow: int = 21
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0
    atr_trailing_multiplier: float = 2.5
    bb_period: int = 20
    bb_std: float = 2.0
    momentum_period: int = 10
    ema_price_buffer_pct: float = 0.001       # Allow EMA entries within 0.1% of slow EMA
    bb_touch_buffer_pct: float = 0.001        # Allow BB touches within 0.1% of the band
    log_decision_trace: bool = field(default_factory=lambda: os.getenv("LOG_DECISION_TRACE", "false").lower() == "true")
    mean_reversion_z_entry: float = 1.2        # Lowered from 1.5: overnight market z stuck at 0.1–0.99 with 1.5 (0 trades in 1hr); 1.2 gives occasional signal
    mean_reversion_z_full: float = 1.2        # Full-size threshold aligned with entry
    mean_reversion_z_exit: float = 0.25       # Exit bias when price reverts near mid band
    trend_breakout_z: float = 2.0             # (legacy breakout — not used; kept for compat)
    disable_trend_strategy: bool = True       # TREND_PULLBACK disabled: losing strategy in all 6 live sessions; largest aggregate loss source across TRENDING regime
    disable_volatility_strategy: bool = True   # VOL_BREAKOUT disabled — full BT shows PF≈1.0, Sharpe=-8 even at best params
    disable_momentum_strategy: bool = True    # DISABLED: 7.1% WR over 14 live trades — Donchian 1m breakouts are noise
    ema_pullback_zone_atr: float = 1.0        # Default zone (fallback / HIGH_VOLATILITY)
    ema_pullback_zone_atr_trending: float = 0.8  # Tighter zone in trend: only catch clean retests
    ema_pullback_zone_atr_ranging: float = 1.8   # Wider zone in range: price oscillates further from EMA
    atr_stop_trend_multiplier: float = 2.0    # Widened for 1m noise: 2.0 ATR stop
    atr_stop_range_multiplier: float = 2.0    # Widened from 1.6: 1m noise eats tight stops; 2.0 gives MR room to breathe
    atr_stop_hv_multiplier: float = 2.5       # Extra-wide stop for HIGH_VOLATILITY regime
    atr_target_trend_multiplier: float = 4.0  # Trend take-profit: 4.0 stop keeps 2:1 RR
    atr_target_range_multiplier: float = 3.0  # Reduced from 5.0: no trade reached 5.0x target in 25 bars; 3.0/2.0 = 1.5:1 RR, needs WR>40%
    signal_confidence_alpha: float = 0.35     # (legacy) reserved for future weighting
    # ---- Position Management ----
    enable_partial_close: bool = False         # Disabled: partial close caps wins at ~0.5R; let positions run to full target
    partial_exit_r: float = 1.0               # Take partial profit at 1R (one full risk unit)
    partial_exit_fraction: float = 0.50       # Close 50% at partial exit; trail remainder
    breakeven_trigger_r: float = 9999.0       # DISABLED: re-enabling at 1.5R converts potential 2.5R wins into $0 BE exits; let positions run to target
    stagnant_exit_bars: int = 25              # 25 × 1m bars = 25 min; 20 was too fast, 40 held losers too long
    stagnant_exit_r_threshold: float = 0.2   # Exit if unrealized < this many R after N bars
    candle_seconds: int = 60                  # 1m candles = 60 seconds (for time-based calc)
    # ---- Volatility Breakout ----
    vol_breakout_lookback: int = 20           # N-bar high/low for breakout detection
    atr_stop_volatility_multiplier: float = 2.5   # Wider stop for volatile breakout entries
    atr_target_volatility_multiplier: float = 5.0  # Target keeps 2:1 RR with 2.5 stop
    volume_spike_multiplier: float = 1.0      # Legacy hard threshold; volume_ratio tiers take priority
    # Tiered volume-ratio position-sizing (replaces hard volume block)
    # Lowered thresholds for 1m: micro breakouts rarely show 1.5× volume spikes
    volume_ratio_min: float = 0.05            # Below → skip entry (only truly zero-volume bars blocked)
    volume_ratio_half: float = 0.60           # 0.05–0.60 → 50% position size
    volume_ratio_boost: float = 1.50          # 0.60–1.50 → 100%; above → 125% (confirmed momentum)
    volume_lookback: int = 20                 # Rolling window for average volume

    # ---- Regime Detection ----
    adx_period: int = 14
    adx_trending_threshold: float = 25.0
    atr_volatility_multiplier: float = 2.0    # ATR > N× median → HIGH_VOL
    high_vol_size_reduction: float = 0.50     # Reduce size by 50% in HIGH_VOL
    regime_trend_threshold: float = 0.6       # Trend score above → TRENDING
    regime_range_threshold: float = 0.4       # Trend score below → RANGING
    volatility_short_window: int = 20
    volatility_long_window: int = 60
    vol_expansion_ratio: float = 1.15
    vol_contraction_ratio: float = 0.9

    # ---- Analytics ----
    sharpe_risk_free_rate: float = 0.04       # Annual risk-free rate
    equity_curve_export_path: str = "logs/equity_curve.csv"
    trade_journal_export_path: str = "logs/trade_journal.csv"

    # ---- Signal Validation Thresholds ----
    min_entry_price: float = 0.0001           # Reject sub-cent tokens (PEPE-class: float precision breaks sizing)
    min_stop_fraction: float = 0.00015         # Price-% stop floor: 0.015% — BTC 1m ATR-based stops are $10-20 ($68k); 0.05% floor ($34) rejected BTC every cycle; 0.015% = $10.2 threshold
    min_atr_fraction: float = 0.00005         # ATR ≥ 0.005% of price; 0.01% caused borderline BTC rejections ($6.75 vs $6.79)
    min_stop_atr_multiple: float = 1.5        # Stop distance ≥ 1.5×ATR (prevents noise-level stops)
    rsi_momentum_long_min: float = 55.0       # TREND_PULLBACK LONG in TRENDING: RSI must be > this (momentum confirm)
    btc_gate_min_correlation: float = 0.6     # BTC EMA gate only applies if symbol’s correlation to BTC ≥ this; low-corr alts are exempt
    trend_pullback_trending_min_conf: float = 0.70  # TREND_PULLBACK in TRENDING regime requires higher confidence (4 sessions -PF; raise bar)
    min_signal_confidence_ranging: float = 0.55     # RANGING: lowered 0.60→0.55; 0.60–0.65 band is best performer (PF=8); capture more of it
    rs_exit_drop_threshold: float = 0.40      # Close position if symbol RS drops > this from entry-time RS (momentum collapse)
    min_order_notional_usdt: float = 5.1      # Bitget minimum order value (rejects silently below this)
    # ---- Session Gate ----
    trading_hours_start: int = 0               # 0 = no gate (test mode); set to 9 + trading_hours_end=21 for active-hour guard
    trading_hours_end: int = 24               # 24 = no gate (test mode); covers full day
    # ---- Portfolio Heat Gate ----
    max_portfolio_heat_pct: float = 0.005     # Block new entries if open unrealized loss > 0.5% of equity
    # ---- Symbol Blacklist ----
    symbol_blacklist: List[str] = field(default_factory=lambda: ["PIPPINUSDT"])  # Symbols permanently excluded; PIPPIN: losses on both long+short sides across 5 sessions
    min_signal_confidence: float = 0.45       # Reject signals below this confidence; conf=0.32 is noise territory on 1m
    # ---- Expected Value (EV) Filter ----
    min_ev_threshold: float = 0.0             # Reject trades where rolling EV ≤ this
    min_ev_sample: int = 30                   # Enforce EV filter after ≥ 30 closed trades (Bayesian prior handles cold-start)

    # ---- Correlation Window ----
    correlation_lookback: int = 100           # Periods for rolling correlation

    # ---- OHLCV fetch ----
    candle_limit: int = 300                   # 300 × 1m = 5 hours of look-back window
    candle_granularity: str = "1m"           # 1-minute candles for intraday frequency
    candle_periods_per_day: int = 1440        # 1440 × 1m per day (Sharpe annualization)

    # ---- RSI Parameters ----
    rsi_period: int = 14
    rsi_oversold: float = 35.0               # MR long: tightened — require deeper oversold for better MR entries
    rsi_overbought: float = 65.0             # MR short: tightened — require deeper overbought for better MR entries
    momentum_rsi_long: float = 55.0          # Momentum breakout long: RSI must exceed
    momentum_rsi_short: float = 45.0         # Momentum breakout short: RSI must be below
    # ---- Donchian / Momentum Breakout ----
    donchian_period: int = 20                # N-bar channel for momentum breakout entries
    donchian_atr_stop_multiplier: float = 2.0  # Widened stop for 1m momentum breakout
    donchian_rr: float = 2.0                 # Reward-risk for momentum breakout
    # ---- ATR Compression Detection ----
    atr_compression_lookback: int = 50       # Window for ATR percentile ranking
    atr_compression_pct: float = 35.0        # ATR below this percentile = compressed
    # ---- Strategy Stats ----
    kelly_fraction_cap: float = 0.25          # Fractional Kelly cap multiplier
    kelly_fraction_floor: float = 0.0         # Minimum allocation per approved trade
    max_trades_per_window: int = 8            # Raised: allow more trades per window on 1m
    trade_frequency_window_seconds: int = 300 # Reduced: 5-min rolling window
    symbol_cooldown_seconds: int = 180        # Per-symbol re-entry cooldown (3 min)
    portfolio_variance_cap: float = 0.08      # Variance ceiling for position weights
    strategy_stats_window: int = 100          # Rolling trade window per strategy (100 for better Kelly estimates)
    # ---- Confidence Scoring Weights ----
    # 5-factor confidence: [signal_strength, regime, volume, trend_align, volatility_state]
    confidence_w_signal: float = 0.30
    confidence_w_regime: float = 0.25
    confidence_w_volume: float = 0.20
    confidence_w_trend_align: float = 0.15
    confidence_w_vol_state: float = 0.10


@dataclass
class NotificationConfig:
    discord_webhook: Optional[str] = field(
        default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL")
    )
    notify_on_trade: bool = True
    notify_on_regime_change: bool = True
    notify_on_daily_loss_hit: bool = True
    notify_on_error: bool = True


@dataclass
class AppConfig:
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_dir: str = "logs"


# Singleton accessor
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def reload_config() -> AppConfig:
    global _config
    _config = AppConfig()
    return _config
