"""
universe.py - Symbol Universe Discovery and Filtering
Fetches all USDT perpetual pairs from Bitget, filters by volume,
and maintains the tradeable universe.
"""
import time
from typing import Dict, List, Optional
import threading

from config import get_config
from logger import get_logger

log = get_logger("universe")


class UniverseManager:
    """
    Discovers and maintains the set of eligible trading symbols.
    Refreshes periodically. Thread-safe.
    """

    def __init__(self, rest_client) -> None:
        self._rest = rest_client
        self._cfg = get_config().trading
        self._lock = threading.Lock()
        self._universe: List[str] = []
        self._ticker_data: Dict[str, Dict] = {}   # symbol → ticker dict
        self._last_refresh: float = 0.0
        self._refresh_interval: int = 300         # seconds

    def refresh(self) -> List[str]:
        """Fetch all tickers, filter by volume, return eligible symbols."""
        log.info("Refreshing symbol universe …")
        try:
            tickers = self._rest.get_usdt_perpetuals()
        except Exception as e:
            log.error("Failed to fetch universe: %s", e)
            if not self._universe:
                # Fallback to known liquid symbols when API is blocked
                log.warning("API unreachable — using fallback universe [BTCUSDT, ETHUSDT]")
                with self._lock:
                    self._universe = ["BTCUSDT", "ETHUSDT"]
                    self._last_refresh = time.time()
            return self._universe  # return stale/fallback

        eligible: List[str] = []
        ticker_map: Dict[str, Dict] = {}

        for t in tickers:
            symbol = t.get("symbol", "")
            try:
                vol_24h = float(t.get("quoteVolume", 0) or t.get("usdtVolume", 0) or 0)
            except ValueError:
                vol_24h = 0.0

            if vol_24h >= self._cfg.min_volume_24h_usdt:
                eligible.append(symbol)
                ticker_map[symbol] = t

        with self._lock:
            self._universe = eligible
            self._ticker_data = ticker_map
            self._last_refresh = time.time()

        log.info("Universe: %d symbols qualify (min vol $%.0fM)",
                 len(eligible), self._cfg.min_volume_24h_usdt / 1e6)
        return eligible

    def get_universe(self) -> List[str]:
        with self._lock:
            return list(self._universe)

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        with self._lock:
            return self._ticker_data.get(symbol)

    def get_all_tickers(self) -> Dict[str, Dict]:
        with self._lock:
            return dict(self._ticker_data)

    def needs_refresh(self) -> bool:
        return time.time() - self._last_refresh > self._refresh_interval

    def get_24h_volume(self, symbol: str) -> float:
        t = self.get_ticker(symbol)
        if not t:
            return 0.0
        try:
            return float(t.get("quoteVolume", 0) or t.get("usdtVolume", 0) or 0)
        except (ValueError, TypeError):
            return 0.0
