"""Parameter grid search optimizer."""
import itertools, logging
import pandas as pd
logging.disable(logging.CRITICAL)

from config import reload_config
from backtester import Backtester

df = pd.read_csv("logs/cached_btcusdt_15m.csv", index_col=0, parse_dates=True)
print(f"Loaded {len(df)} bars for optimization")

results = []

for stop, tgt, pr, z, vol_mult in itertools.product(
    [1.5, 2.0, 2.5],            # atr_stop_trend
    [3.0, 4.0],                  # atr_target_trend
    [0.8, 1.0],                  # partial_exit_r
    [0.8, 1.0, 1.2],             # trend_breakout_z
    [1.1, 1.3],                  # volume_spike_multiplier
):
    cfg = reload_config().trading
    cfg.atr_stop_trend_multiplier = stop
    cfg.atr_target_trend_multiplier = tgt
    cfg.atr_stop_range_multiplier = round(stop * 0.7, 2)
    cfg.atr_target_range_multiplier = round(tgt * 0.7, 2)
    cfg.partial_exit_r = pr
    cfg.trend_breakout_z = z
    cfg.volume_spike_multiplier = vol_mult

    bt = Backtester(initial_equity=10_000.0)
    r = bt.run(df, "BTCUSDT")
    pf = r.get("profit_factor", 0.0)
    trades = r.get("total_trades", 0)
    wr = r.get("win_rate", 0.0)
    ret = r.get("return_pct", 0.0)
    results.append((pf, trades, wr, ret, stop, tgt, pr, z, vol_mult))
    if trades >= 10:
        print(f"stop={stop} tgt={tgt} pr={pr} z={z} vol={vol_mult} | trades={trades} wr={wr:.2f} pf={pf:.4f} ret={ret:+.4f}")

# Filter to configs with >= 10 trades, sort by pf
valid = [r for r in results if r[1] >= 10]
valid.sort(reverse=True)
print(f"\n=== TOP 10 CONFIGS (>=10 trades) ===")
for pf, trades, wr, ret, stop, tgt, pr, z, vol_mult in valid[:10]:
    print(f"  stop={stop} tgt={tgt} pr={pr} z={z} vol={vol_mult} | trades={trades} wr={wr:.2f} pf={pf:.4f} ret={ret:+.4f}")

if not valid:
    print("\n=== ALL RESULTS (sorted by pf) ===")
    results.sort(reverse=True)
    for pf, trades, wr, ret, stop, tgt, pr, z, vol_mult in results[:10]:
        print(f"  stop={stop} tgt={tgt} pr={pr} z={z} vol={vol_mult} | trades={trades} wr={wr:.2f} pf={pf:.4f} ret={ret:+.4f}")
