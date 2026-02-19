"""
data_feed.py - OHLCV Data Feed with Caching
Fetches candle data from Bitget REST and caches it in memory.
In paper mode, serves the same data (no difference in candle fetching).
"""
import time
import threading
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from config import get_config
from logger import get_logger

log = get_logger("data_feed")


def candles_to_df(raw: List[List]) -> pd.DataFrame:
    """Convert raw Bitget candle list to a typed DataFrame."""
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume", "quote_volume"])
    df = df.astype({
        "ts": "int64",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
        "quote_volume": "float64",
    })
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.sort_values("ts").reset_index(drop=True)
    return df


class DataFeed:
    """
    Fetches and caches OHLCV data.
    Provides helpers for the scanner and strategy layers.
    """

    CACHE_TTL = 30  # seconds before a symbol's data is considered stale

    def __init__(self, rest_client) -> None:
        self._rest = rest_client
        self._cfg = get_config().trading
        self._cache: Dict[str, Dict] = {}  # symbol → {"df": df, "fetched_at": ts}
        self._lock = threading.Lock()

    def get_ohlcv(self, symbol: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """Return OHLCV dataframe for symbol, using cache if fresh."""
        with self._lock:
            entry = self._cache.get(symbol)
            if entry and not force_refresh:
                age = time.time() - entry["fetched_at"]
                if age < self.CACHE_TTL:
                    return entry["df"]

        try:
            raw = self._rest.get_candles(
                symbol,
                granularity=self._cfg.candle_granularity,
                limit=self._cfg.candle_limit,
            )
            df = candles_to_df(raw)
            if df.empty:
                log.warning("Empty candle data for %s", symbol)
                return None

            with self._lock:
                self._cache[symbol] = {"df": df, "fetched_at": time.time()}
            return df

        except Exception as e:
            log.error("Failed to fetch candles for %s: %s", symbol, e)
            # Return stale if available
            with self._lock:
                entry = self._cache.get(symbol)
                return entry["df"] if entry else None

    def bulk_fetch(self, symbols: List[str], max_workers: int = 8) -> Dict[str, pd.DataFrame]:
        """Fetch candles for multiple symbols using a thread pool."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: Dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="feed") as ex:
            future_map = {ex.submit(self.get_ohlcv, sym): sym for sym in symbols}
            for future in as_completed(future_map):
                sym = future_map[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        results[sym] = df
                except Exception as e:
                    log.error("bulk_fetch error for %s: %s", sym, e)
        return results

    def get_close_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """
        Return a DataFrame of close prices for all symbols.
        Columns = symbols, rows = candles (aligned by position).
        """
        frames: Dict[str, pd.Series] = {}
        for sym in symbols:
            df = self.get_ohlcv(sym)
            if df is not None and not df.empty:
                frames[sym] = df["close"].reset_index(drop=True)
        if not frames:
            return pd.DataFrame()
        return pd.DataFrame(frames)

    def invalidate(self, symbol: str) -> None:
        with self._lock:
            self._cache.pop(symbol, None)

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
