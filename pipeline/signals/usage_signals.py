"""
Compute usage-based risk signals from usage_metrics.csv.

Key signals:
  - MoM change in active_users (last two months)
  - 3-month linear slope (downward trend detection)
  - Drop flag (MoM > 30% decline)
  - Near-zero usage (active_users <= 2 in latest month)
  - SDK version (latest month)
"""
import pandas as pd
import numpy as np
from models.signals import UsageSignal


def compute_usage_signals(usage_df: pd.DataFrame) -> dict[str, UsageSignal]:
    """Compute UsageSignal for every account_id in usage_metrics."""
    signals: dict[str, UsageSignal] = {}

    for account_id, group in usage_df.groupby("account_id"):
        group = group.sort_values("month")
        users = group["active_users"].values
        latest = int(users[-1]) if len(users) > 0 else 0

        if len(users) >= 2 and users[-2] > 0:
            mom_change = (users[-1] - users[-2]) / users[-2] * 100
        else:
            mom_change = 0.0

        last3 = users[-3:] if len(users) >= 3 else users
        if len(last3) >= 2:
            slope = float(np.polyfit(range(len(last3)), last3, 1)[0])
        else:
            slope = 0.0

        sdk = str(group.iloc[-1]["sdk_version"]) if len(group) > 0 else "unknown"

        signals[str(account_id)] = UsageSignal(
            account_id=str(account_id),
            latest_active_users=latest,
            mom_change_pct=round(mom_change, 2),
            three_month_slope=round(slope, 3),
            usage_drop_flag=mom_change < -30,
            near_zero_usage=latest <= 2,
            sdk_version=sdk,
        )

    return signals
