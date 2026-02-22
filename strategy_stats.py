"""Rolling per-strategy performance statistics and Kelly sizing helper."""
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict
import math

from config import get_config


@dataclass
class StrategySnapshot:
    win_rate: float
    avg_r: float
    expectancy: float
    profit_factor: float
    sharpe: float
    max_drawdown: float
    sample_size: int


class StrategyStats:
    def __init__(self, window: int) -> None:
        self._window = window
        self._r_values: Deque[float] = deque(maxlen=window)
        self._pnl_values: Deque[float] = deque(maxlen=window)
        self._equity_curve: Deque[float] = deque(maxlen=window)
        self._equity_curve.append(0.0)

    def record(self, r_multiple: float, pnl: float) -> None:
        if len(self._equity_curve) == self._equity_curve.maxlen:
            # Maintain cumulative sequence by shifting to keep relative changes meaningful
            baseline = self._equity_curve[0]
            self._equity_curve = deque((val - baseline for val in self._equity_curve), maxlen=self._window)
        self._r_values.append(r_multiple)
        self._pnl_values.append(pnl)
        next_equity = self._equity_curve[-1] + pnl
        self._equity_curve.append(next_equity)

    def snapshot(self) -> StrategySnapshot:
        sample = len(self._r_values)
        if sample == 0:
            return StrategySnapshot(0.5, 1.0, 0.0, 1.0, 0.0, 0.0, 0)

        wins = [r for r in self._r_values if r > 0]
        losses = [r for r in self._r_values if r <= 0]
        win_rate = len(wins) / sample if sample else 0.0
        avg_r = sum(self._r_values) / sample if sample else 0.0
        expectancy = avg_r
        profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")
        sharpe = self._compute_sharpe()
        max_dd = self._compute_drawdown()
        return StrategySnapshot(win_rate, avg_r, expectancy, profit_factor, sharpe, max_dd, sample)

    def kelly_fraction(self, reward_risk_ratio: float, min_sample: int = 20) -> float:
        cfg = get_config().trading
        snap = self.snapshot()
        if snap.sample_size < min_sample:
            # Insufficient history — fall back to conservative 50% win-rate assumption
            p = 0.50
        else:
            p = snap.win_rate
        b = max(reward_risk_ratio, 0.01)
        q = 1 - p
        f_star = (b * p - q) / b
        if f_star <= 0:
            return 0.0
        fractional = f_star * cfg.kelly_fraction_cap
        return min(fractional, cfg.max_risk_per_trade_pct)

    def _compute_sharpe(self) -> float:
        n = len(self._r_values)
        if n < 5:
            return 0.0
        mean_r = sum(self._r_values) / n
        variance = sum((r - mean_r) ** 2 for r in self._r_values) / max(n - 1, 1)
        std = math.sqrt(variance)
        return (mean_r / std) if std > 0 else 0.0

    def _compute_drawdown(self) -> float:
        if len(self._equity_curve) < 2:
            return 0.0
        peaks = []
        peak = -float("inf")
        dd = 0.0
        for val in self._equity_curve:
            peak = max(peak, val)
            if peak != 0:
                dd = max(dd, (peak - val) / abs(peak))
        return dd


class StrategyStatsManager:
    def __init__(self) -> None:
        cfg = get_config().trading
        self._window = cfg.strategy_stats_window
        self._stats: Dict[str, StrategyStats] = {}

    def record_trade(self, strategy: str, r_multiple: float, pnl: float) -> None:
        stats = self._stats.setdefault(strategy, StrategyStats(self._window))
        stats.record(r_multiple, pnl)

    def get_snapshot(self, strategy: str) -> StrategySnapshot:
        stats = self._stats.setdefault(strategy, StrategyStats(self._window))
        return stats.snapshot()

    def get_kelly_fraction(self, strategy: str, reward_risk_ratio: float) -> float:
        stats = self._stats.setdefault(strategy, StrategyStats(self._window))
        return stats.kelly_fraction(reward_risk_ratio)