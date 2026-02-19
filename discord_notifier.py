"""
discord_notifier.py - Discord Webhook Notifications
Sends real-time alerts for trades, regime changes, errors, and daily summaries.
Non-blocking (runs in background thread). Fails silently if webhook is not configured.
"""
import threading
import requests
from datetime import datetime, timezone
from typing import Optional, Dict

from config import get_config
from logger import get_logger

log = get_logger("discord")


class DiscordNotifier:
    def __init__(self) -> None:
        cfg = get_config()
        self._webhook = cfg.notifications.discord_webhook
        self._cfg_n = cfg.notifications
        self._enabled = bool(self._webhook)
        if not self._enabled:
            log.info("Discord notifications disabled (no webhook URL)")

    def _send(self, payload: Dict) -> None:
        if not self._enabled:
            return
        try:
            resp = requests.post(self._webhook, json=payload, timeout=5)
            resp.raise_for_status()
        except Exception as e:
            log.warning("Discord send failed: %s", e)

    def _send_async(self, payload: Dict) -> None:
        t = threading.Thread(target=self._send, args=(payload,), daemon=True)
        t.start()

    def _embed(self, title: str, description: str, color: int, fields: Optional[list] = None) -> Dict:
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "Bitget Trading Engine"},
        }
        if fields:
            embed["fields"] = fields
        return {"embeds": [embed]}

    # ------------------------------------------------------------------ #
    # Public notification methods
    # ------------------------------------------------------------------ #

    def trade_opened(
        self,
        symbol: str,
        direction: str,
        entry: float,
        stop: float,
        size_usd: float,
        regime: str,
        pnl: float = 0.0,
        pnl_pct: float = 0.0,
        strategy: str = "",
        confidence: float = 0.0,
    ) -> None:
        if not self._cfg_n.notify_on_trade:
            return
        color = 0x00FF00 if direction == "LONG" else 0xFF4444
        emoji = "📈" if direction == "LONG" else "📉"
        self._send_async(self._embed(
            title=f"{emoji} Trade Opened: {symbol}",
            description=f"**{direction}** @ `{entry:.4f}`",
            color=color,
            fields=[
                {"name": "Stop Loss", "value": f"`{stop:.4f}`", "inline": True},
                {"name": "Size", "value": f"`${size_usd:.2f}`", "inline": True},
                {"name": "PnL", "value": f"`${pnl:+.2f}` ({pnl_pct:+.2f}%)", "inline": True},
                {"name": "Regime", "value": f"`{regime}`", "inline": True},
                {"name": "Strategy", "value": f"`{strategy or 'NA'}`", "inline": True},
                {"name": "Confidence", "value": f"`{confidence:.2f}`", "inline": True},
            ],
        ))

    def trade_closed(
        self,
        symbol: str,
        direction: str,
        entry: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        reason: str,
    ) -> None:
        if not self._cfg_n.notify_on_trade:
            return
        color = 0x00FF00 if pnl >= 0 else 0xFF4444
        emoji = "✅" if pnl >= 0 else "❌"
        self._send_async(self._embed(
            title=f"{emoji} Trade Closed: {symbol}",
            description=f"**{direction}** | Entry `{entry:.4f}` → Exit `{exit_price:.4f}`",
            color=color,
            fields=[
                {"name": "PnL", "value": f"`${pnl:+.2f}` ({pnl_pct:+.2f}%)", "inline": True},
                {"name": "Reason", "value": f"`{reason}`", "inline": True},
            ],
        ))

    def regime_change(self, symbol: str, old_regime: str, new_regime: str) -> None:
        if not self._cfg_n.notify_on_regime_change:
            return
        self._send_async(self._embed(
            title=f"🔀 Regime Change: {symbol}",
            description=f"`{old_regime}` → `{new_regime}`",
            color=0xFFAA00,
        ))

    def daily_loss_hit(self, loss_pct: float, equity: float) -> None:
        if not self._cfg_n.notify_on_daily_loss_hit:
            return
        self._send_async(self._embed(
            title="⛔ Daily Loss Cap Hit",
            description=f"Trading halted. Daily loss: **{loss_pct:.1%}**",
            color=0xFF0000,
            fields=[{"name": "Current Equity", "value": f"`${equity:.2f}`", "inline": True}],
        ))

    def error(self, message: str) -> None:
        if not self._cfg_n.notify_on_error:
            return
        self._send_async(self._embed(
            title="⚠️ Engine Error",
            description=f"```{message[:1000]}```",
            color=0xFF8800,
        ))

    def daily_summary(self, stats: Dict) -> None:
        fields = [
            {"name": k, "value": f"`{v}`", "inline": True}
            for k, v in stats.items()
        ]
        self._send_async(self._embed(
            title="📊 Daily Performance Summary",
            description="End-of-day statistics",
            color=0x0099FF,
            fields=fields[:25],  # Discord limit
        ))
