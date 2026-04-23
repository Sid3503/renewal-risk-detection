"""
Cross-reference account SDK versions against changelog deprecation events.

Non-obvious insight:
  Accounts on deprecated SDK v3.x face security patch cutoff April 30 2026
  AND REST API v2 sunset — even if no CSM has flagged it.
  This module surfaces that invisible technical churn risk.
"""
from datetime import date, datetime
from models.signals import UsageSignal, ChangelogSignal
from pipeline.scoring.weight_config import DEPRECATED_SDK_VERSIONS, DEPRECATED_SDK_DEADLINE


def _is_deprecated(sdk_version: str) -> bool:
    """Check if SDK version is a deprecated v3.x release.

    Uses prefix matching to avoid false positives like v4.3.1 matching '3.1'.
    """
    import re
    v = sdk_version.lower().strip()
    # Match v3.x.y, 3.x.y, or the literal token "v3.x"
    return bool(re.match(r"^v?3[\.\-]", v)) or v in ("v3.x", "3.x")


def _days_to_deadline() -> int:
    """Days from today to the deprecated SDK deadline."""
    deadline = datetime.strptime(DEPRECATED_SDK_DEADLINE, "%Y-%m-%d").date()
    return (deadline - date.today()).days


def compute_changelog_signals(
    usage_signals: dict[str, UsageSignal],
) -> dict[str, ChangelogSignal]:
    """Build ChangelogSignal for every account by checking its SDK version."""
    signals: dict[str, ChangelogSignal] = {}
    days_left = _days_to_deadline()

    for account_id, usage in usage_signals.items():
        deprecated = _is_deprecated(usage.sdk_version)
        signals[account_id] = ChangelogSignal(
            account_id=account_id,
            sdk_version=usage.sdk_version,
            is_deprecated=deprecated,
            deprecation_deadline=DEPRECATED_SDK_DEADLINE if deprecated else None,
            days_to_deadline=days_left if deprecated else None,
            affected_features=(
                ["REST API v2 (sunset April 30)", "SDK v3 security patches stop April 30"]
                if deprecated
                else []
            ),
        )

    return signals
