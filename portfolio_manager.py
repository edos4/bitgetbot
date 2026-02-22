"""
portfolio_manager.py - Portfolio State and Correlation Management
Tracks open positions, computes rolling correlation matrix,
and identifies correlated pairs that should not both be held.
"""
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from config import get_config
from strategy import SignalDirection
from strategy_stats import StrategyStatsManager
from logger import get_logger

log = get_logger("portfolio")


@dataclass
class Position:
    symbol: str
    direction: SignalDirection
    entry_price: float
    quantity: float          # contracts / base units
    stop_loss: float
    take_profit: Optional[float]
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    highest_price: float = 0.0   # for trailing stop (LONG)
    lowest_price: float = float("inf")   # for trailing stop (SHORT)
    unrealized_pnl: float = 0.0
    atr: float = 0.0
    leverage: int = 3
    regime: str = "TRENDING"
    strategy: str = "UNKNOWN"
    signal_confidence: float = 0.0
    kelly_fraction: float = 0.0
    stop_distance: float = 0.0
    reward_risk_ratio: float = 0.0
    risk_fraction: float = 0.0
    risk_dollar: float = 0.0
    # Position management state flags
    partial_taken: bool = False       # True once 50% partial profit has been locked at 1R
    breakeven_set: bool = False       # True once stop has been moved to entry price after 1R

    def update_price(self, current_price: float) -> None:
        if self.direction == SignalDirection.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
            if current_price > self.highest_price:
                self.highest_price = current_price
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
            if current_price < self.lowest_price:
                self.lowest_price = current_price


class PortfolioManager:
    """
    Tracks all open positions, updates PnL, and manages correlation.
    Thread-safe.
    """

    def __init__(self, data_feed, risk_manager, stats_manager: StrategyStatsManager) -> None:
        self._feed = data_feed
        self._risk = risk_manager
        self._cfg = get_config().trading
        self._stats = stats_manager
        self._lock = threading.Lock()
        self._positions: Dict[str, Position] = {}    # symbol → Position
        self._equity: float = 0.0
        self._correlations: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------ #
    # Position management
    # ------------------------------------------------------------------ #

    def open_position(self, pos: Position) -> None:
        with self._lock:
            self._positions[pos.symbol] = pos
        risk_usd = abs(pos.entry_price - pos.stop_loss) * pos.quantity
        self._risk.register_trade_open(pos.symbol, risk_usd, direction=pos.direction.value)
        log.info("Position opened: %s %s @ %.4f qty=%.4f stop=%.4f",
                 pos.symbol, pos.direction.value, pos.entry_price, pos.quantity, pos.stop_loss)

    def close_position(self, symbol: str, exit_price: float, reason: str = "") -> Optional[Position]:
        with self._lock:
            pos = self._positions.pop(symbol, None)
        if pos is None:
            return None

        if pos.direction == SignalDirection.LONG:
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        self._risk.register_trade_close(symbol, pnl, reason)
        risk_dollar = pos.risk_dollar if pos.risk_dollar > 0 else pos.stop_distance * pos.quantity
        r_multiple = pnl / (risk_dollar + 1e-10)
        self._stats.record_trade(pos.strategy, r_multiple, pnl)
        log.info("Position closed: %s @ %.4f | PnL=%.2f | reason=%s",
                 symbol, exit_price, pnl, reason)
        return pos

    def get_position(self, symbol: str) -> Optional[Position]:
        with self._lock:
            return self._positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        with self._lock:
            return dict(self._positions)

    def get_open_symbols(self) -> List[str]:
        with self._lock:
            return list(self._positions.keys())

    def count_open(self) -> int:
        with self._lock:
            return len(self._positions)

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update unrealized PnL for all open positions."""
        with self._lock:
            for symbol, pos in self._positions.items():
                price = prices.get(symbol)
                if price:
                    pos.update_price(price)

    def update_stop_loss(self, symbol: str, new_stop: float) -> None:
        with self._lock:
            if symbol in self._positions:
                self._positions[symbol].stop_loss = new_stop

    def set_equity(self, equity: float) -> None:
        self._equity = equity
        self._risk.register_equity_update(equity)

    def get_equity(self) -> float:
        return self._equity

    # ------------------------------------------------------------------ #
    # Correlation management
    # ------------------------------------------------------------------ #

    def compute_correlation_blocks(self, symbols: List[str]) -> Dict[str, List[str]]:
        """
        Compute rolling correlation matrix for all candidate symbols.
        Returns dict: symbol → [list of highly correlated symbols].
        """
        lookback = self._cfg.correlation_lookback
        threshold = self._cfg.correlation_threshold

        close_matrix = self._feed.get_close_matrix(symbols)
        if close_matrix.empty or len(close_matrix) < lookback:
            return {}

        # Use last N rows
        data = close_matrix.tail(lookback)
        # Drop columns with all-NaN
        data = data.dropna(axis=1, how="all")
        # Forward fill then drop remaining NaN
        data = data.ffill().dropna(axis=1)

        if data.shape[1] < 2:
            return {}

        try:
            corr = data.corr()
        except Exception as e:
            log.error("Correlation computation failed: %s", e)
            return {}

        self._correlations = {
            sym: {other: float(corr.loc[sym, other]) for other in corr.columns if other != sym}
            for sym in corr.columns
        }

        blocks: Dict[str, List[str]] = {}
        syms = list(corr.columns)
        for i, sym_a in enumerate(syms):
            correlated = []
            for j, sym_b in enumerate(syms):
                if i != j:
                    val = corr.loc[sym_a, sym_b]
                    if pd.notna(val) and abs(val) >= threshold:
                        correlated.append(sym_b)
            if correlated:
                blocks[sym_a] = correlated

        # Log high correlations
        reported = set()
        for sym, corr_list in blocks.items():
            key = frozenset([sym] + corr_list)
            if key not in reported:
                log.debug("High correlation: %s ↔ %s", sym, corr_list)
                reported.add(key)

        return blocks

    def check_stop_and_target(self, symbol: str, current_price: float) -> Tuple[bool, str]:
        """
        Check if stop or target is hit.
        Returns (should_close, reason).
        """
        with self._lock:
            pos = self._positions.get(symbol)
        if not pos:
            return False, ""

        if pos.direction == SignalDirection.LONG:
            if current_price <= pos.stop_loss:
                return True, "STOP_LOSS"
            if pos.take_profit and current_price >= pos.take_profit:
                return True, "TAKE_PROFIT"
        else:
            if current_price >= pos.stop_loss:
                return True, "STOP_LOSS"
            if pos.take_profit and current_price <= pos.take_profit:
                return True, "TAKE_PROFIT"

        return False, ""

    def get_total_unrealized_pnl(self) -> float:
        with self._lock:
            return sum(p.unrealized_pnl for p in self._positions.values())

    def get_total_notional(self) -> float:
        """Sum of abs(entry_price * quantity) across all open positions."""
        with self._lock:
            return sum(abs(p.entry_price * p.quantity) for p in self._positions.values())

    def get_max_correlation(self, symbol: str) -> float:
        with self._lock:
            row = self._correlations.get(symbol, {})
            open_syms = set(self._positions.keys())
        if not row:
            return 0.0
        vals = [abs(v) for other, v in row.items() if other in open_syms]
        return max(vals) if vals else 0.0

    # ------------------------------------------------------------------ #
    # Portfolio variance guard
    # ------------------------------------------------------------------ #

    def breaches_variance_cap(self, position_usd: float, entry_price: float, atr: float) -> bool:
        equity = max(self._equity, 1.0)
        new_weight = position_usd / (equity + 1e-10)
        new_vol = self._vol_ratio(atr, entry_price)

        with self._lock:
            weights = [
                (pos.entry_price * pos.quantity) / (equity + 1e-10)
                for pos in self._positions.values()
            ]
            vols = [self._vol_ratio(pos.atr, pos.entry_price) for pos in self._positions.values()]

        weights.append(new_weight)
        vols.append(new_vol)

        variance = sum((w * v) ** 2 for w, v in zip(weights, vols))
        exceeds = variance > self._cfg.portfolio_variance_cap
        if exceeds:
            log.info(
                "Variance cap hit: var=%.4f cap=%.4f | weights=%s",
                variance,
                self._cfg.portfolio_variance_cap,
                [round(w, 3) for w in weights],
            )
        return exceeds

    @staticmethod
    def _vol_ratio(atr: float, price: float) -> float:
        price = max(price, 1e-8)
        atr = atr if atr and atr > 0 else price * 0.01
        return atr / price
