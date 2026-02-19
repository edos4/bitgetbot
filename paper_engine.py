"""
paper_engine.py - Paper Trading Simulation Engine
Simulates order fills with realistic slippage, fees, and equity tracking.
Mirrors the interface of the live execution engine.
"""
import time
import uuid
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from config import get_config
from strategy import SignalDirection
from logger import get_logger

log = get_logger("paper_engine")


@dataclass
class PaperFill:
    order_id: str
    symbol: str
    direction: SignalDirection
    quantity: float
    fill_price: float
    fee_paid: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PaperEngine:
    """
    Fully simulated execution engine.
    - Applies configurable slippage and fees
    - Tracks equity curve with every fill
    - Thread-safe
    """

    def __init__(self, initial_equity: float = 10_000.0) -> None:
        self._cfg = get_config().trading
        self._lock = threading.Lock()
        self._equity = initial_equity
        self._initial_equity = initial_equity
        self._fills: List[PaperFill] = []
        self._equity_curve: List[Dict] = [
            {"ts": datetime.now(timezone.utc).isoformat(), "equity": initial_equity}
        ]

    # ------------------------------------------------------------------ #
    # Order placement
    # ------------------------------------------------------------------ #

    def market_buy(
        self, symbol: str, quantity: float, mid_price: float
    ) -> PaperFill:
        """Simulate a market buy (long open or short close)."""
        return self._fill(symbol, SignalDirection.LONG, quantity, mid_price, is_buy=True)

    def market_sell(
        self, symbol: str, quantity: float, mid_price: float
    ) -> PaperFill:
        """Simulate a market sell (short open or long close)."""
        return self._fill(symbol, SignalDirection.SHORT, quantity, mid_price, is_buy=False)

    def _fill(
        self,
        symbol: str,
        direction: SignalDirection,
        quantity: float,
        mid_price: float,
        is_buy: bool,
    ) -> PaperFill:
        slip_bps = self._cfg.slippage_bps / 10_000.0
        # Buys fill above mid, sells fill below mid
        slippage = mid_price * slip_bps * (1 if is_buy else -1)
        fill_price = mid_price + slippage

        fee = fill_price * quantity * self._cfg.taker_fee_pct

        order_id = str(uuid.uuid4())[:8]
        fill = PaperFill(
            order_id=order_id,
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            fill_price=fill_price,
            fee_paid=fee,
        )

        with self._lock:
            self._fills.append(fill)

        log.info(
            "PAPER FILL %s | %s %s | qty=%.4f | price=%.4f | fee=%.2f",
            order_id, "BUY" if is_buy else "SELL", symbol, quantity, fill_price, fee
        )
        return fill

    # ------------------------------------------------------------------ #
    # Equity tracking
    # ------------------------------------------------------------------ #

    def record_trade_pnl(self, pnl: float, fees: float) -> None:
        with self._lock:
            self._equity += pnl - fees
            self._equity_curve.append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "equity": self._equity,
            })

    def get_equity(self) -> float:
        with self._lock:
            return self._equity

    def get_equity_curve(self) -> pd.DataFrame:
        with self._lock:
            return pd.DataFrame(self._equity_curve)

    def get_fills(self) -> List[PaperFill]:
        with self._lock:
            return list(self._fills)

    def export_equity_curve(self, path: str) -> None:
        df = self.get_equity_curve()
        df.to_csv(path, index=False)
        log.info("Equity curve exported to %s", path)

    # ------------------------------------------------------------------ #
    # Simulated position sizing (contracts calculation)
    # ------------------------------------------------------------------ #

    def compute_quantity(
        self,
        position_size_usd: float,
        entry_price: float,
        leverage: int,
    ) -> float:
        """
        qty = (position_size_usd * leverage) / entry_price
        Rounds to 1 decimal place (adjust per symbol precision in live).
        """
        if entry_price <= 0:
            return 0.0
        qty = (position_size_usd * leverage) / entry_price
        return round(max(qty, 0.001), 4)
