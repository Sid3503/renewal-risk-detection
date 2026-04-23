"""Compute support-based risk signals from support_tickets.csv."""
import pandas as pd
from models.signals import SupportSignal


def compute_support_signals(tickets_df: pd.DataFrame) -> dict[str, SupportSignal]:
    """Compute SupportSignal for every account_id that has tickets."""
    signals: dict[str, SupportSignal] = {}

    for account_id, group in tickets_df.groupby("account_id"):
        p1 = group[group["priority"] == "P1"]
        open_t = group[group["status"] == "open"]
        escalated = group[group["status"] == "escalated"]
        avg_res = group["resolution_time_hours"].mean()

        signals[str(account_id)] = SupportSignal(
            account_id=str(account_id),
            total_tickets=len(group),
            p1_count=len(p1),
            open_tickets=len(open_t),
            escalated_tickets=len(escalated),
            avg_resolution_hours=round(float(avg_res), 1) if not pd.isna(avg_res) else 0.0,
            has_unresolved_p1=len(p1[p1["status"].isin(["open", "escalated"])]) > 0,
        )

    return signals
