"""
config.py - Central Configuration for Bitget Futures Trading Engine
All tuneable parameters live here. Edit this file to customize behavior.
"""
import os
from dataclasses import dataclass, field
from typing import Optional
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
    min_volume_24h_usdt: float = 50_000_000   # Minimum 24h volume to be eligible
    top_n_symbols: int = 5                     # Number of symbols to trade concurrently
    scan_interval_seconds: int = 60            # How often to re-rank and scan
    max_concurrent_positions: int = 5          # Hard cap on open positions

    # ---- Risk ----
    risk_per_trade_pct: float = 0.01           # 1% of equity per trade
    portfolio_heat_cap_pct: float = 0.05       # Max 5% total open risk
    max_daily_loss_pct: float = 0.04           # Stop trading after 4% daily drawdown
    max_consecutive_losses: int = 3            # Pause after N consecutive losses
    correlation_threshold: float = 0.80        # Block trades if pair correlation > this

    # ---- Execution ----
    default_leverage: int = 3
    order_retry_attempts: int = 3
    order_retry_delay_seconds: float = 1.5
    slippage_bps: float = 5.0                  # Paper mode: basis points of slippage
    maker_fee_pct: float = 0.0002             # 0.02%
    taker_fee_pct: float = 0.0006             # 0.06%

    # ---- Strategy Parameters ----
    ema_fast: int = 9
    ema_slow: int = 21
    atr_period: int = 14
    atr_stop_multiplier: float = 1.5
    atr_trailing_multiplier: float = 2.0
    bb_period: int = 20
    bb_std: float = 2.0
    momentum_period: int = 10
    volume_spike_multiplier: float = 1.5      # Volume must be N× avg to confirm signal
    ema_price_buffer_pct: float = 0.001       # Allow EMA entries within 0.1% of slow EMA
    bb_touch_buffer_pct: float = 0.001        # Allow BB touches within 0.1% of the band
    log_decision_trace: bool = field(default_factory=lambda: os.getenv("LOG_DECISION_TRACE", "false").lower() == "true")
    mean_reversion_z_entry: float = 1.25      # Entry when |z-score| exceeds this value
    mean_reversion_z_exit: float = 0.25       # Exit bias when price reverts near mid band
    trend_breakout_z: float = 0.65            # Minimum z-score for breakout confirmation
    atr_stop_trend_multiplier: float = 1.8    # ATR multiplier for trend stops
    atr_stop_range_multiplier: float = 1.2    # ATR multiplier for range stops
    atr_target_trend_multiplier: float = 2.5  # Trend targets aim for >2R
    atr_target_range_multiplier: float = 0.8  # Range targets capture partial mean reversion
    signal_confidence_alpha: float = 0.35     # Weight for scanner score inside confidence blend

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

    # ---- Correlation Window ----
    correlation_lookback: int = 50            # Periods for rolling correlation

    # ---- OHLCV fetch ----
    candle_limit: int = 200                   # Number of candles to fetch per symbol
    candle_granularity: str = "15m"          # Candle interval

    # ---- Portfolio / Allocation ----
    kelly_fraction_cap: float = 0.65          # Max fractional Kelly allocation
    kelly_fraction_floor: float = 0.25        # Minimum allocation per approved trade
    max_trades_per_hour: int = 8              # Hard cap for trade frequency
    portfolio_variance_cap: float = 0.08      # Variance ceiling for position weights


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
