"""
backtester.py - Historical Backtesting Engine
Replays historical OHLCV data through the strategy and risk layers
to generate a backtest report without touching the exchange.

Usage:
    from backtester import Backtester
    bt = Backtester(initial_equity=10000)
    results = bt.run(symbol="BTCUSDT", start="2024-01-01", end="2024-06-01")
    bt.print_results()
"""
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import uuid

from config import get_config
from strategy import generate_signal, generate_signal_from_cache, IndicatorCache, SignalDirection, compute_trailing_stop
from metrics import MetricsTracker, TradeRecord
from logger import get_logger

log = get_logger("backtester")


@dataclass
class BacktestTrade:
    symbol: str
    direction: str
    entry_price: float
    entry_bar: int
    exit_price: float = 0.0
    exit_bar: int = 0
    quantity: float = 0.0
    pnl: float = 0.0
    fees: float = 0.0
    exit_reason: str = ""
    regime: str = ""


@dataclass
class BacktestState:
    open_trade: Optional[BacktestTrade] = None
    highest_price: float = 0.0
    lowest_price: float = float("inf")
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_stop: float = 0.0
    atr: float = 0.0
    entry_price: float = 0.0
    stop_distance: float = 0.0
    partial_done: bool = False


class Backtester:
    def __init__(self, initial_equity: float = 10_000.0) -> None:
        self._cfg = get_config().trading
        self._equity = initial_equity
        self._initial_equity = initial_equity
        self._trades: List[BacktestTrade] = []
        self._equity_curve: List[float] = [initial_equity]
        self._metrics = MetricsTracker()
        self._consecutive_losses = 0
        self._daily_start_equity = initial_equity

    def run(self, df: pd.DataFrame, symbol: str = "BACKTEST") -> Dict:
        """
        Run backtest on a single symbol OHLCV dataframe.
        Requires at minimum columns: open, high, low, close, volume.
        """
        log.info("Starting backtest: %s | %d bars", symbol, len(df))
        warmup = max(self._cfg.ema_slow, self._cfg.bb_period, self._cfg.atr_period,
                     self._cfg.volatility_long_window) + 5
        state = BacktestState()
        current_day: Optional[str] = None  # tracks calendar day for daily resets

        # Precompute all indicators once on the full dataframe.
        # Allow injecting a pre-built cache (e.g. for parameter sweeps).
        if hasattr(self, '_prebuilt_cache') and self._prebuilt_cache is not None:
            cache = self._prebuilt_cache
        else:
            log.info("Precomputing indicators on %d bars...", len(df))
            import time as _time
            t0 = _time.time()
            cache = IndicatorCache.from_df(df)
            log.info("Indicators precomputed in %.2fs", _time.time() - t0)

        for i in range(warmup, len(df)):
            current_bar = df.iloc[i]
            price = float(current_bar["close"])
            high = float(current_bar["high"])
            low = float(current_bar["low"])

            # ---- Daily risk-limit reset ----
            bar_day = str(df.index[i])[:10]  # "YYYY-MM-DD"
            if bar_day != current_day:
                current_day = bar_day
                self._consecutive_losses = 0
                self._daily_start_equity = self._equity

            # ---- Manage open trade ----
            if state.open_trade:
                trade = state.open_trade

                # ---- Partial exit: lock profit at partial_exit_r × stop_distance ----
                if not state.partial_done and state.stop_distance > 0 and self._cfg.partial_exit_fraction > 0:
                    pe_threshold = state.stop_distance * self._cfg.partial_exit_r
                    triggered = (
                        (trade.direction == "LONG" and high >= state.entry_price + pe_threshold) or
                        (trade.direction == "SHORT" and low <= state.entry_price - pe_threshold)
                    )
                    if triggered:
                        partial_qty = trade.quantity * self._cfg.partial_exit_fraction
                        partial_exit_price = (
                            state.entry_price + pe_threshold if trade.direction == "LONG"
                            else state.entry_price - pe_threshold
                        )
                        if trade.direction == "LONG":
                            partial_pnl = (partial_exit_price - trade.entry_price) * partial_qty
                        else:
                            partial_pnl = (trade.entry_price - partial_exit_price) * partial_qty
                        partial_fee = partial_exit_price * partial_qty * self._cfg.taker_fee_pct
                        partial_pnl -= partial_fee
                        self._equity += partial_pnl
                        self._equity_curve.append(self._equity)
                        partial_record = BacktestTrade(
                            symbol=trade.symbol,
                            direction=trade.direction,
                            entry_price=trade.entry_price,
                            entry_bar=trade.entry_bar,
                            exit_price=partial_exit_price,
                            exit_bar=i,
                            quantity=partial_qty,
                            pnl=partial_pnl,
                            fees=partial_fee,
                            exit_reason="PARTIAL_EXIT",
                            regime=trade.regime,
                        )
                        self._trades.append(partial_record)
                        trade.quantity -= partial_qty
                        # Move stop to lock in partial profit minus 1 ATR buffer
                        # prevents full retracement from erasing gains
                        if trade.direction == "LONG":
                            new_stop = state.entry_price + pe_threshold - state.stop_distance
                            state.stop_loss = max(state.stop_loss, new_stop)
                        else:
                            new_stop = state.entry_price - pe_threshold + state.stop_distance
                            state.stop_loss = min(state.stop_loss, new_stop)
                        state.partial_done = True
                        log.debug("BT partial exit %s @ %.4f | pnl=%.2f | new_stop=%.4f", trade.symbol, partial_exit_price, partial_pnl, state.stop_loss)

                # ---- Fixed stop-loss and take-profit checks ----
                if trade.direction == "LONG":
                    if low <= state.stop_loss:
                        self._close_trade(trade, i, state.stop_loss, "STOP_LOSS", state)
                        continue
                    if state.take_profit and high >= state.take_profit:
                        self._close_trade(trade, i, state.take_profit, "TAKE_PROFIT", state)
                        continue
                else:  # SHORT
                    if high >= state.stop_loss:
                        self._close_trade(trade, i, state.stop_loss, "STOP_LOSS", state)
                        continue
                    if state.take_profit and low <= state.take_profit:
                        self._close_trade(trade, i, state.take_profit, "TAKE_PROFIT", state)
                        continue

            # ---- Look for entry ----
            if state.open_trade is None:
                if self._consecutive_losses >= self._cfg.max_consecutive_losses:
                    continue
                daily_loss = (self._equity - self._daily_start_equity) / self._daily_start_equity
                if daily_loss <= -self._cfg.max_daily_loss_pct:
                    continue

                signal = generate_signal_from_cache(symbol, cache, i)
                if signal.direction == SignalDirection.NONE:
                    continue

                stop_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
                if stop_pct <= 0:
                    continue

                # Base position USD from risk-per-trade
                pos_usd = (self._equity * self._cfg.risk_per_trade_pct) / stop_pct
                pos_usd = min(pos_usd, self._equity * 0.20)

                # Apply tiered volume-ratio sizing (mirrors execution_engine logic)
                vol_ratio = signal.volume_ratio
                if vol_ratio < self._cfg.volume_ratio_min:
                    continue   # below hard floor — skip (should never reach here but safety guard)
                elif vol_ratio < self._cfg.volume_ratio_half:
                    pos_usd *= 0.50    # weak volume → half size
                elif vol_ratio > self._cfg.volume_ratio_boost:
                    pos_usd *= 1.25    # strong volume → boost size
                # else 1.0–1.5 → full size (no adjustment)

                # Scale position by signal confidence (floor 0.25)
                if signal.confidence < 1.0:
                    pos_usd *= max(signal.confidence, 0.25)

                qty = (pos_usd * self._cfg.default_leverage) / signal.entry_price

                entry_fee = signal.entry_price * qty * self._cfg.taker_fee_pct
                self._equity -= entry_fee

                trade = BacktestTrade(
                    symbol=symbol,
                    direction=signal.direction.value,
                    entry_price=signal.entry_price,
                    entry_bar=i,
                    quantity=qty,
                    regime=signal.regime.value,
                    fees=entry_fee,          # start with entry fee; exit fees accumulated later
                )
                state.open_trade = trade
                state.stop_loss = signal.stop_loss
                state.take_profit = signal.take_profit or 0.0
                state.highest_price = signal.entry_price
                state.lowest_price = signal.entry_price
                state.trailing_stop = 0.0 if signal.direction == SignalDirection.LONG else float("inf")
                state.atr = signal.atr
                state.entry_price = signal.entry_price
                state.stop_distance = signal.stop_distance
                state.partial_done = False

        # Force close at end
        if state.open_trade:
            last_price = float(df.iloc[-1]["close"])
            self._close_trade(state.open_trade, len(df) - 1, last_price, "END_OF_DATA", state)

        return self._compute_results()

    def _close_trade(
        self,
        trade: BacktestTrade,
        bar_idx: int,
        exit_price: float,
        reason: str,
        state: BacktestState,
    ) -> None:
        trade.exit_price = exit_price
        trade.exit_bar = bar_idx
        trade.exit_reason = reason

        if trade.direction == "LONG":
            gross = (exit_price - trade.entry_price) * trade.quantity
        else:
            gross = (trade.entry_price - exit_price) * trade.quantity

        exit_fee = exit_price * trade.quantity * self._cfg.taker_fee_pct
        trade.fees += exit_fee
        # pnl = gross price move minus exit fee; entry fee is a sunk cost already
        # reflected in equity but we keep it in trade.fees for reporting
        trade.pnl = gross - exit_fee

        self._equity += gross - exit_fee  # equity already had entry fee removed at open
        self._equity_curve.append(self._equity)
        self._trades.append(trade)

        if trade.pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        state.open_trade = None
        log.debug("BT close %s @ %.4f | PnL=%.2f | %s", trade.symbol, exit_price, trade.pnl, reason)

    def _compute_results(self) -> Dict:
        if not self._trades:
            return {"status": "no_trades"}

        pnls = [t.pnl for t in self._trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        equity_arr = np.array(self._equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        drawdowns = (peak - equity_arr) / (peak + 1e-10)
        max_dd = float(np.max(drawdowns))

        returns = np.diff(equity_arr) / (equity_arr[:-1] + 1e-10)
        sharpe = 0.0
        if len(returns) > 5 and np.std(returns) > 0:
            ann_factor = np.sqrt(96 * 365)
            sharpe = float(np.mean(returns) / np.std(returns) * ann_factor)

        profit_factor = abs(sum(wins)) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")

        return {
            "total_trades": len(self._trades),
            "win_rate": round(len(wins) / len(pnls), 4),
            "profit_factor": round(profit_factor, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd, 4),
            "total_pnl": round(sum(pnls), 2),
            "final_equity": round(self._equity, 2),
            "initial_equity": round(self._initial_equity, 2),
            "return_pct": round((self._equity - self._initial_equity) / self._initial_equity, 4),
            "avg_win": round(np.mean(wins) if wins else 0, 2),
            "avg_loss": round(np.mean(losses) if losses else 0, 2),
        }

    def print_results(self) -> None:
        results = self._compute_results()
        print("\n" + "=" * 55)
        print("  BACKTEST RESULTS")
        print("=" * 55)
        for k, v in results.items():
            print(f"  {k:<30} {v}")
        print("=" * 55 + "\n")

    def export_trades(self, path: str = "logs/backtest_trades.csv") -> None:
        import csv, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not self._trades:
            return
        fields = ["symbol", "direction", "entry_price", "entry_bar",
                  "exit_price", "exit_bar", "quantity", "pnl", "fees", "exit_reason", "regime"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for t in self._trades:
                w.writerow({k: getattr(t, k) for k in fields})
        log.info("Backtest trades exported → %s", path)
