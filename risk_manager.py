"""
risk_manager.py - Risk Management Engine
Handles position sizing, portfolio heat tracking, daily loss cutoffs,
consecutive loss tracking, correlation checks, and leverage control.
"""
import threading
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

from config import get_config
from logger import get_logger

log = get_logger("risk_manager")


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    risk_fraction: float = 0.0
    risk_dollar: float = 0.0
    leverage: int = 3


def _base_asset(symbol: str) -> str:
    """Extract base asset from trading pair (e.g. BTCUSDT → BTC)."""
    for quote in ("USDT", "USD", "BUSD", "USDC", "PERP"):
        if symbol.upper().endswith(quote):
            return symbol.upper()[: -len(quote)]
    return symbol[:3].upper()


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

        # Per-base-asset risk: base ("BTC", "ETH") → $ at risk across all correlated pairs
        self._base_risk: Dict[str, float] = {}

        # Blocked symbols
        self._blocked_symbols: set[str] = set()

    # ------------------------------------------------------------------ #
    # Main check
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        symbol: str,
        requested_risk_fraction: float,
        current_equity: float,
        existing_positions: List[str],
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

            # 4. Consecutive losses → risk reduction
            risk_fraction = min(requested_risk_fraction, self._cfg.max_risk_per_trade_pct)
            if self._consecutive_losses >= self._cfg.max_consecutive_losses:
                risk_fraction *= 0.5

            if risk_fraction <= 0:
                return RiskDecision(False, "Risk fraction collapsed after adjustments")

            risk_dollar = current_equity * risk_fraction

            # 5. Portfolio heat cap — scale down to remaining headroom rather than hard-block
            total_heat = sum(self._active_risk.values())
            heat_cap_dollars = self._cfg.portfolio_heat_cap_pct * current_equity
            remaining_heat = heat_cap_dollars - total_heat

            if remaining_heat <= 0:
                return RiskDecision(
                    False,
                    f"Portfolio heat cap fully consumed "
                    f"({total_heat / current_equity:.1%} ≥ {self._cfg.portfolio_heat_cap_pct:.1%})"
                )

            if risk_dollar > remaining_heat:
                log.info(
                    "Heat cap sizing down %s: $%.2f → $%.2f (headroom=%.1f%%)",
                    symbol, risk_dollar, remaining_heat,
                    remaining_heat / current_equity * 100,
                )
                risk_dollar = remaining_heat
                risk_fraction = risk_dollar / (current_equity + 1e-10)

            # Sanity: don't enter a position too tiny to be meaningful (< 0.1% equity)
            if risk_fraction < 0.001:
                return RiskDecision(False, "Scaled risk fraction too small after heat-cap adjustment (<0.1%)")

            # 6. Per-base-asset exposure cap
            base = _base_asset(symbol)
            base_used = self._base_risk.get(base, 0.0)
            base_cap  = self._cfg.max_correlated_base_pct * current_equity
            if base_used + risk_dollar > base_cap:
                available = base_cap - base_used
                if available <= 0:
                    return RiskDecision(
                        False,
                        f"Per-base cap hit for {base}: "
                        f"{base_used/current_equity:.1%} ≥ {self._cfg.max_correlated_base_pct:.1%}",
                    )
                log.info(
                    "Per-base cap sizing down %s (%s): $%.2f → $%.2f",
                    symbol, base, risk_dollar, available,
                )
                risk_dollar = available
                risk_fraction = risk_dollar / (current_equity + 1e-10)

            if risk_fraction < 0.001:
                return RiskDecision(False, "Scaled risk fraction too small after base-cap adjustment (<0.1%)")

            heat_pct = (total_heat + risk_dollar) / (current_equity + 1e-10)
            log.info(
                "Risk OK: %s | heat=%.1f%% | base_risk(%s)=%.1f%% | consc_losses=%d",
                symbol, heat_pct * 100,
                _base_asset(symbol),
                (self._base_risk.get(_base_asset(symbol), 0.0) + risk_dollar) / (current_equity + 1e-10) * 100,
                self._consecutive_losses,
            )
            return RiskDecision(
                approved=True,
                reason="OK",
                risk_fraction=risk_fraction,
                risk_dollar=risk_dollar,
                leverage=self._cfg.default_leverage,
            )

    # ------------------------------------------------------------------ #
    # State updates
    # ------------------------------------------------------------------ #

    def register_trade_open(self, symbol: str, risk_usd: float) -> None:
        with self._lock:
            self._active_risk[symbol] = risk_usd
            base = _base_asset(symbol)
            self._base_risk[base] = self._base_risk.get(base, 0.0) + risk_usd

    def register_trade_close(self, symbol: str, pnl: float, exit_reason: str = "") -> None:
        with self._lock:
            risk = self._active_risk.pop(symbol, 0.0)
            # Remove from per-base tracker
            base = _base_asset(symbol)
            self._base_risk[base] = max(0.0, self._base_risk.get(base, 0.0) - risk)
            self._daily_pnl += pnl
            # SHUTDOWN closes are forced exits, not a strategy failure — skip streak counter
            if exit_reason.upper() == "SHUTDOWN":
                return
            if pnl < 0:
                self._consecutive_losses += 1
                log.warning("Consecutive losses: %d", self._consecutive_losses)
            else:
                self._consecutive_losses = 0

    def register_equity_update(self, equity: float) -> None:
        with self._lock:
            if self._starting_equity == 0.0:
                self._starting_equity = equity

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
