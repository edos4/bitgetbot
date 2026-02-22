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
    strategy: str
    confidence: float
    kelly_fraction: float


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
        self._starting_balance: Optional[float] = None

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
            else:
                self._starting_balance = equity
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

        starting = self._starting_balance if self._starting_balance is not None else (equity[0] if equity else 0.0)
        ending = equity[-1] if equity else starting

        stats = {
            "starting_balance": round(starting, 2),
            "ending_balance": round(ending, 2),
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

    def compute_regime_stats(self) -> Dict:
        """Return per-regime performance breakdown."""
        with self._lock:
            trades = list(self._trades)

        if not trades:
            return {}

        regimes: Dict[str, list] = {}
        for t in trades:
            regimes.setdefault(t.regime, []).append(t.pnl)

        result: Dict[str, Dict] = {}
        for regime, pnls in regimes.items():
            wins   = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            gp = sum(wins)
            gl = abs(sum(losses)) if losses else 0.0
            result[regime] = {
                "n_trades": len(pnls),
                "win_rate":       round(len(wins) / len(pnls), 4),
                "profit_factor":  round(gp / gl, 4) if gl > 0 else float("inf"),
                "total_pnl":      round(sum(pnls), 2),
                "avg_win":        round(float(np.mean(wins)), 2) if wins else 0.0,
                "avg_loss":       round(float(np.mean(losses)), 2) if losses else 0.0,
            }
        return result

    def print_regime_summary(self) -> None:
        stats = self.compute_regime_stats()
        if not stats:
            return
        log.info("=" * 55)
        log.info("REGIME PERFORMANCE BREAKDOWN")
        log.info("=" * 55)
        for regime, s in stats.items():
            log.info(
                "  %-18s  n=%3d  WR=%.1f%%  PF=%.3f  PnL=%+.2f",
                regime, s["n_trades"], s["win_rate"] * 100,
                s["profit_factor"] if s["profit_factor"] != float("inf") else 999.0,
                s["total_pnl"],
            )
        log.info("=" * 55)

    def _compute_sharpe(self, returns: List[float]) -> float:
        if len(returns) < 5:
            return 0.0
        cfg = get_config().trading
        arr = np.array(returns)
        # Annualize based on configured candle granularity
        periods_per_year = getattr(cfg, "candle_periods_per_day", 96) * 365
        excess = arr - (cfg.sharpe_risk_free_rate / periods_per_year)
        std = np.std(excess)
        if abs(std) < 1e-10:
            return 0.0
        raw = float(np.mean(excess) / std * np.sqrt(periods_per_year))
        return max(-100.0, min(100.0, raw))

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
