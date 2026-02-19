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
from strategy import generate_signal, SignalDirection, compute_trailing_stop
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
        warmup = max(self._cfg.ema_slow, self._cfg.bb_period, self._cfg.atr_period) + 5
        state = BacktestState()

        for i in range(warmup, len(df)):
            window = df.iloc[:i].copy()
            current_bar = df.iloc[i]
            price = float(current_bar["close"])
            high = float(current_bar["high"])
            low = float(current_bar["low"])

            # ---- Manage open trade ----
            if state.open_trade:
                trade = state.open_trade
                # Update extremes for trailing stop
                if trade.direction == "LONG":
                    state.highest_price = max(state.highest_price, high)
                    trail_stop = compute_trailing_stop(
                        SignalDirection.LONG, price, state.highest_price,
                        window["close"].iloc[-1] * 0.01  # rough ATR
                    )
                    state.trailing_stop = max(state.trailing_stop, trail_stop) if state.trailing_stop else trail_stop

                    # Check stops
                    if low <= state.stop_loss or low <= state.trailing_stop:
                        exit_price = min(state.stop_loss, price)
                        self._close_trade(trade, i, exit_price, "STOP_LOSS", state)
                        continue
                    if state.take_profit and high >= state.take_profit:
                        self._close_trade(trade, i, state.take_profit, "TAKE_PROFIT", state)
                        continue
                else:  # SHORT
                    state.lowest_price = min(state.lowest_price, low)
                    trail_stop = compute_trailing_stop(
                        SignalDirection.SHORT, price, state.lowest_price,
                        window["close"].iloc[-1] * 0.01
                    )
                    state.trailing_stop = min(state.trailing_stop, trail_stop) if state.trailing_stop else trail_stop

                    if high >= state.stop_loss or high >= state.trailing_stop:
                        exit_price = max(state.stop_loss, price)
                        self._close_trade(trade, i, exit_price, "STOP_LOSS", state)
                        continue
                    if state.take_profit and low <= state.take_profit:
                        self._close_trade(trade, i, state.take_profit, "TAKE_PROFIT", state)
                        continue

            # ---- Look for entry ----
            if state.open_trade is None:
                # Risk checks
                if self._consecutive_losses >= self._cfg.max_consecutive_losses:
                    continue
                daily_loss = (self._equity - self._daily_start_equity) / self._daily_start_equity
                if daily_loss <= -self._cfg.max_daily_loss_pct:
                    continue

                signal = generate_signal(symbol, window)
                if signal.direction == SignalDirection.NONE:
                    continue

                stop_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
                if stop_pct <= 0:
                    continue

                pos_usd = (self._equity * self._cfg.risk_per_trade_pct) / stop_pct
                pos_usd = min(pos_usd, self._equity * 0.20) * signal.size_multiplier
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
                )
                state.open_trade = trade
                state.stop_loss = signal.stop_loss
                state.take_profit = signal.take_profit or 0.0
                state.highest_price = signal.entry_price
                state.lowest_price = signal.entry_price
                state.trailing_stop = 0.0

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
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity

        fee = exit_price * trade.quantity * self._cfg.taker_fee_pct
        trade.fees = fee
        trade.pnl -= fee

        self._equity += trade.pnl
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
