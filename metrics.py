"""
metrics.py - Performance Analytics
Computes Sharpe ratio, win rate, profit factor, max drawdown,
and exports trade journal to CSV.
"""
import os
import csv
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from config import get_config
from logger import get_logger

log = get_logger("metrics")


@dataclass
class TradeRecord:
    trade_id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    fees: float
    entry_time: str
    exit_time: str
    duration_seconds: float
    regime: str
    exit_reason: str
    stop_loss: float
    take_profit: Optional[float]


class MetricsTracker:
    """
    Collects trade records and computes performance statistics.
    Thread-safe. Exports to CSV on demand.
    """

    def __init__(self) -> None:
        self._cfg = get_config()
        self._lock = threading.Lock()
        self._trades: List[TradeRecord] = []
        self._daily_returns: List[float] = []
        self._equity_snapshots: List[float] = []

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def record_trade(self, trade: TradeRecord) -> None:
        with self._lock:
            self._trades.append(trade)
        log.info(
            "Trade recorded: %s %s | PnL=%.2f (%.2f%%) | reason=%s",
            trade.symbol, trade.direction, trade.pnl, trade.pnl_pct * 100, trade.exit_reason
        )

    def record_equity(self, equity: float) -> None:
        with self._lock:
            if self._equity_snapshots:
                prev = self._equity_snapshots[-1]
                if prev > 0:
                    ret = (equity - prev) / prev
                    self._daily_returns.append(ret)
            self._equity_snapshots.append(equity)

    # ------------------------------------------------------------------ #
    # Analytics
    # ------------------------------------------------------------------ #

    def compute_stats(self) -> Dict:
        with self._lock:
            trades = list(self._trades)
            equity = list(self._equity_snapshots)
            returns = list(self._daily_returns)

        if not trades:
            return {"status": "no_trades"}

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate = len(wins) / len(pnls) if pnls else 0.0

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0

        # Sharpe
        sharpe = self._compute_sharpe(returns)

        # Max drawdown
        max_dd = self._compute_max_drawdown(equity)

        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        stats = {
            "total_trades": len(trades),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd, 4),
            "total_pnl": round(sum(pnls), 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "expectancy": round(expectancy, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
        }
        return stats

    def _compute_sharpe(self, returns: List[float]) -> float:
        if len(returns) < 5:
            return 0.0
        cfg = get_config().trading
        arr = np.array(returns)
        # Annualize: 96 × 15-min periods per day, 365 days
        periods_per_year = 96 * 365
        excess = arr - (cfg.sharpe_risk_free_rate / periods_per_year)
        std = np.std(excess)
        if std == 0:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(periods_per_year))

    def _compute_max_drawdown(self, equity: List[float]) -> float:
        if len(equity) < 2:
            return 0.0
        arr = np.array(equity)
        peak = np.maximum.accumulate(arr)
        drawdown = (peak - arr) / (peak + 1e-10)
        return float(np.max(drawdown))

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def export_trade_journal(self, path: Optional[str] = None) -> str:
        path = path or get_config().trading.trade_journal_export_path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        with self._lock:
            trades = list(self._trades)

        if not trades:
            log.warning("No trades to export")
            return path

        fieldnames = list(asdict(trades[0]).keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in trades:
                writer.writerow(asdict(t))

        log.info("Trade journal exported: %d trades → %s", len(trades), path)
        return path

    def export_equity_curve(self, path: Optional[str] = None) -> str:
        path = path or get_config().trading.equity_curve_export_path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        with self._lock:
            equity = list(self._equity_snapshots)

        df = pd.DataFrame({"equity": equity})
        df.to_csv(path, index=True)
        log.info("Equity curve exported → %s", path)
        return path

    def print_summary(self) -> None:
        stats = self.compute_stats()
        if stats.get("status") == "no_trades":
            log.info("No trades recorded yet")
            return
        log.info("=" * 50)
        log.info("PERFORMANCE SUMMARY")
        log.info("=" * 50)
        for k, v in stats.items():
            log.info("  %-25s : %s", k, v)
        log.info("=" * 50)
