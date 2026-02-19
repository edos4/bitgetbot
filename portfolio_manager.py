"""
portfolio_manager.py - Portfolio State and Correlation Management
Tracks open positions, computes rolling correlation matrix,
and identifies correlated pairs that should not both be held.
"""
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from config import get_config
from strategy import SignalDirection
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
    entry_time: datetime = field(default_factory=datetime.utcnow)
    highest_price: float = 0.0   # for trailing stop (LONG)
    lowest_price: float = float("inf")   # for trailing stop (SHORT)
    unrealized_pnl: float = 0.0
    atr: float = 0.0
    leverage: int = 3
    regime: str = "TRENDING"

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

    def __init__(self, data_feed, risk_manager) -> None:
        self._feed = data_feed
        self._risk = risk_manager
        self._cfg = get_config().trading
        self._lock = threading.Lock()
        self._positions: Dict[str, Position] = {}    # symbol → Position
        self._equity: float = 0.0

    # ------------------------------------------------------------------ #
    # Position management
    # ------------------------------------------------------------------ #

    def open_position(self, pos: Position) -> None:
        with self._lock:
            self._positions[pos.symbol] = pos
        risk_usd = abs(pos.entry_price - pos.stop_loss) * pos.quantity
        self._risk.register_trade_open(pos.symbol, risk_usd)
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

        self._risk.register_trade_close(symbol, pnl)
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

        # Update risk manager
        self._risk.update_correlation_blocks(blocks)

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
