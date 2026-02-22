"""
analyze_confidence.py — Confidence band audit across all sessions.

Groups trades from logs/trade_journal.csv by confidence band and reports
WR, PF, avg_win, avg_loss, expectancy per band. Excludes partial closes.

Usage:
    python3 analyze_confidence.py [--journal logs/trade_journal.csv]
"""
import argparse
import csv
import os
from collections import defaultdict


BANDS = [
    (0.00, 0.45, "0.00–0.45"),
    (0.45, 0.50, "0.45–0.50"),
    (0.50, 0.55, "0.50–0.55"),
    (0.55, 0.60, "0.55–0.60"),
    (0.60, 0.65, "0.60–0.65"),
    (0.65, 0.70, "0.65–0.70"),
    (0.70, 0.80, "0.70–0.80"),
    (0.80, 1.01, "0.80+   "),
]


def band_label(conf: float) -> str:
    for lo, hi, label in BANDS:
        if lo <= conf < hi:
            return label
    return "UNKNOWN"


def analyse(path: str) -> None:
    if not os.path.isfile(path):
        print(f"No trade journal found at {path}. Run at least one session first.")
        return

    with open(path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("Trade journal is empty.")
        return

    full_trades = [r for r in rows if r.get("is_partial", "False") != "True"]
    print(f"Total rows: {len(rows)}  |  Full exits (excl. partials): {len(full_trades)}\n")

    by_band: dict = defaultdict(list)
    by_strategy: dict = defaultdict(list)
    by_regime: dict = defaultdict(list)

    for r in full_trades:
        conf = float(r["confidence"])
        pnl = float(r["pnl"])
        strat = r.get("strategy", "?")
        regime = r.get("regime", "?")
        by_band[band_label(conf)].append(pnl)
        by_strategy[strat].append(pnl)
        by_regime[regime].append(pnl)

    def stats(pnls: list) -> dict:
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins) / len(pnls) * 100 if pnls else 0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        exp = sum(pnls) / len(pnls) if pnls else 0
        return dict(n=len(pnls), wr=wr, pf=pf, avg_win=avg_win,
                    avg_loss=avg_loss, exp=exp, total=sum(pnls))

    # ---- Confidence bands ----
    print("=== BY CONFIDENCE BAND ===")
    header = f"{'Band':<12}  {'N':>5}  {'WR%':>6}  {'PF':>6}  {'AvgWin':>8}  {'AvgLoss':>9}  {'Exp/trade':>10}  {'Total':>9}"
    print(header)
    print("-" * len(header))
    for _, _, label in BANDS:
        pnls = by_band.get(label, [])
        if not pnls:
            continue
        s = stats(pnls)
        pf_str = f"{s['pf']:.3f}" if s['pf'] != float("inf") else "  ∞"
        print(f"{label:<12}  {s['n']:>5}  {s['wr']:>5.1f}%  {pf_str:>6}  "
              f"{s['avg_win']:>+8.2f}  {s['avg_loss']:>+9.2f}  {s['exp']:>+10.2f}  {s['total']:>+9.2f}")

    # ---- Strategy breakdown ----
    print("\n=== BY STRATEGY ===")
    print(header.replace("Band", "Strategy   "))
    print("-" * len(header))
    for strat, pnls in sorted(by_strategy.items()):
        s = stats(pnls)
        pf_str = f"{s['pf']:.3f}" if s['pf'] != float("inf") else "  ∞"
        print(f"{strat:<12}  {s['n']:>5}  {s['wr']:>5.1f}%  {pf_str:>6}  "
              f"{s['avg_win']:>+8.2f}  {s['avg_loss']:>+9.2f}  {s['exp']:>+10.2f}  {s['total']:>+9.2f}")

    # ---- Regime breakdown ----
    print("\n=== BY REGIME ===")
    print(header.replace("Band", "Regime     "))
    print("-" * len(header))
    for regime, pnls in sorted(by_regime.items()):
        s = stats(pnls)
        pf_str = f"{s['pf']:.3f}" if s['pf'] != float("inf") else "  ∞"
        print(f"{regime:<12}  {s['n']:>5}  {s['wr']:>5.1f}%  {pf_str:>6}  "
              f"{s['avg_win']:>+8.2f}  {s['avg_loss']:>+9.2f}  {s['exp']:>+10.2f}  {s['total']:>+9.2f}")

    print(f"\nNOTE: Only {len(full_trades)} full-exit trades available. "
          "Statistical significance requires ≥30 per band. Run more sessions to build sample.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", default="logs/trade_journal.csv")
    args = parser.parse_args()
    analyse(args.journal)
