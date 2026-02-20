"""MR-only parameter optimizer — runs in ~3-4 minutes."""
import itertools, logging, time
import pandas as pd
logging.disable(logging.CRITICAL)

from config import reload_config
from backtester import Backtester
import strategy as strat
from strategy import SignalDirection, _no_signal

# Disable trend-following
orig_tf = strat._trend_following_signal
def no_tf(s, df, p, a, r): return _no_signal(s, p, a, r.primary, "X", "TREND", r.atr_ratio)
strat._trend_following_signal = no_tf

df = pd.read_csv("logs/cached_btcusdt_15m.csv", index_col=0, parse_dates=True)
print(f"Loaded {len(df)} bars (MR only mode)")

best = []
t_start = time.time()

for stop, tgt, pr in itertools.product(
    [0.7, 0.8, 1.0, 1.2],          # atr_stop_range
    [2.0, 2.5, 3.0, 3.5],           # atr_target_range
    [0.5, 0.7, 0.9],                 # partial_exit_r
):
    if tgt <= stop * 1.5:            # skip bad R:R upfront
        continue
    cfg = reload_config().trading
    cfg.atr_stop_range_multiplier = stop
    cfg.atr_target_range_multiplier = tgt
    cfg.partial_exit_r = pr

    bt = Backtester(10_000.0)
    r = bt.run(df, "BTCUSDT")
    pf = r.get("profit_factor", 0.0)
    n = r.get("total_trades", 0)
    wr = r.get("win_rate", 0.0)
    ret = r.get("return_pct", 0.0)
    aw = r.get("avg_win", 0.0)
    al = r.get("avg_loss", 0.0)
    best.append((pf, n, wr, ret, aw, al, stop, tgt, pr))
    if n >= 20:
        print(f"stop={stop} tgt={tgt} pr={pr} | n={n} wr={wr:.2f} pf={pf:.4f} ret={ret:+.4f} aw=${aw:.1f} al=${al:.1f}")

elapsed = time.time() - t_start
print(f"\nCompleted {len(best)} combos in {elapsed:.0f}s")

valid = [b for b in best if b[1] >= 20]
valid.sort(reverse=True)
print("\n=== TOP 10 MR CONFIGS (>=20 trades) ===")
for pf, n, wr, ret, aw, al, stop, tgt, pr in valid[:10]:
    print(f"  stop={stop} tgt={tgt} pr={pr} | n={n} wr={wr:.2f} pf={pf:.4f} ret={ret:+.4f} aw=${aw:.1f} al=${al:.1f}")

strat._trend_following_signal = orig_tf
