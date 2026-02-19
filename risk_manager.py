"""
risk_manager.py - Risk Management Engine
Handles position sizing, portfolio heat tracking, daily loss cutoffs,
consecutive loss tracking, correlation checks, and leverage control.
"""
import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Set
import numpy as np

from config import get_config
from logger import get_logger

log = get_logger("risk_manager")


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    position_size_usd: float = 0.0
    contracts: float = 0.0
    leverage: int = 3


class RiskManager:
    """
    Centralizes all pre-trade and portfolio risk checks.
    Thread-safe.
    """

    def __init__(self) -> None:
        self._cfg = get_config().trading
        self._lock = threading.Lock()

        # Daily tracking
        self._today: date = date.today()
        self._daily_pnl: float = 0.0
        self._starting_equity: float = 0.0

        # Consecutive loss tracking
        self._consecutive_losses: int = 0

        # Active risk: symbol → $ at risk
        self._active_risk: Dict[str, float] = {}

        # Blocked symbols
        self._blocked_symbols: Set[str] = set()

        # Correlation state (updated by portfolio manager)
        self._correlation_clusters: Dict[str, str] = {}   # symbol → cluster_id
        self._active_cluster_symbols: Dict[str, str] = {} # cluster_id → active symbol

    # ------------------------------------------------------------------ #
    # Main check
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        symbol: str,
        stop_loss_pct: float,       # % distance from entry to stop
        current_equity: float,
        existing_positions: List[str],
        correlated_with: Optional[List[str]] = None,
    ) -> RiskDecision:
        """
        Full pre-trade risk check.
        Returns RiskDecision with approved flag and reason.
        """
        with self._lock:
            self._refresh_daily(current_equity)

            # 1. Symbol blocked
            if symbol in self._blocked_symbols:
                return RiskDecision(False, f"{symbol} is blocked")

            # 2. Max concurrent positions
            if len(existing_positions) >= self._cfg.max_concurrent_positions:
                return RiskDecision(False, f"Max concurrent positions reached ({self._cfg.max_concurrent_positions})")

            # 3. Daily loss cutoff
            daily_loss_pct = -self._daily_pnl / (self._starting_equity + 1e-10)
            if daily_loss_pct >= self._cfg.max_daily_loss_pct:
                return RiskDecision(False, f"Daily loss cap hit ({daily_loss_pct:.1%})")

            # 4. Consecutive loss cutoff
            if self._consecutive_losses >= self._cfg.max_consecutive_losses:
                return RiskDecision(
                    False,
                    f"Consecutive losses: {self._consecutive_losses} ≥ {self._cfg.max_consecutive_losses}"
                )

            # 5. Correlation check
            if correlated_with:
                for existing_sym in correlated_with:
                    if existing_sym in existing_positions:
                        return RiskDecision(
                            False, f"{symbol} highly correlated with active position {existing_sym}"
                        )

            # 6. Portfolio heat cap
            total_heat = sum(self._active_risk.values())
            risk_dollar = current_equity * self._cfg.risk_per_trade_pct
            heat_pct = (total_heat + risk_dollar) / (current_equity + 1e-10)
            if heat_pct > self._cfg.portfolio_heat_cap_pct:
                return RiskDecision(
                    False, f"Portfolio heat cap exceeded ({heat_pct:.1%} > {self._cfg.portfolio_heat_cap_pct:.1%})"
                )

            # 7. Calculate position size
            if stop_loss_pct <= 0:
                return RiskDecision(False, "Invalid stop loss distance (zero or negative)")

            position_size_usd = self._size_position(current_equity, stop_loss_pct)
            leverage = self._cfg.default_leverage

            log.info(
                "Risk OK: %s | size=$%.2f | heat=%.1f%% | consc_losses=%d",
                symbol, position_size_usd, heat_pct * 100, self._consecutive_losses
            )
            return RiskDecision(
                approved=True,
                reason="OK",
                position_size_usd=position_size_usd,
                leverage=leverage,
            )

    def _size_position(self, equity: float, stop_pct: float) -> float:
        """
        Kelly-inspired fixed-fractional sizing.
        risk_dollar = equity × risk_per_trade_pct
        position = risk_dollar / stop_pct
        But cap at equity × 0.2 per position.
        """
        risk_dollar = equity * self._cfg.risk_per_trade_pct
        position_size = risk_dollar / stop_pct
        max_position = equity * 0.20
        return min(position_size, max_position)

    # ------------------------------------------------------------------ #
    # State updates
    # ------------------------------------------------------------------ #

    def register_trade_open(self, symbol: str, risk_usd: float) -> None:
        with self._lock:
            self._active_risk[symbol] = risk_usd

    def register_trade_close(self, symbol: str, pnl: float) -> None:
        with self._lock:
            self._active_risk.pop(symbol, None)
            self._daily_pnl += pnl
            if pnl < 0:
                self._consecutive_losses += 1
                log.warning("Consecutive losses: %d", self._consecutive_losses)
            else:
                self._consecutive_losses = 0

    def register_equity_update(self, equity: float) -> None:
        with self._lock:
            if self._starting_equity == 0.0:
                self._starting_equity = equity

    def update_correlation_blocks(self, blocked: Dict[str, List[str]]) -> None:
        """
        Update which symbols are correlated with which.
        blocked = {symbol: [correlated_symbols, ...]}
        """
        with self._lock:
            self._correlation_clusters = blocked

    def get_correlated_with(self, symbol: str) -> List[str]:
        with self._lock:
            return self._correlation_clusters.get(symbol, [])

    def block_symbol(self, symbol: str, reason: str = "") -> None:
        with self._lock:
            self._blocked_symbols.add(symbol)
            log.warning("Symbol blocked: %s | reason: %s", symbol, reason)

    def unblock_symbol(self, symbol: str) -> None:
        with self._lock:
            self._blocked_symbols.discard(symbol)

    def reset_consecutive_losses(self) -> None:
        with self._lock:
            self._consecutive_losses = 0
            log.info("Consecutive loss counter reset")

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _refresh_daily(self, equity: float) -> None:
        """Reset daily stats at the start of a new day."""
        today = date.today()
        if today != self._today:
            log.info("New trading day — resetting daily PnL (was %.2f)", self._daily_pnl)
            self._today = today
            self._daily_pnl = 0.0
            self._starting_equity = equity
            self._consecutive_losses = 0  # Reset on new day

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def get_portfolio_heat(self, equity: float) -> float:
        with self._lock:
            return sum(self._active_risk.values()) / (equity + 1e-10)

    def get_daily_pnl(self) -> float:
        with self._lock:
            return self._daily_pnl

    def get_consecutive_losses(self) -> int:
        with self._lock:
            return self._consecutive_losses
