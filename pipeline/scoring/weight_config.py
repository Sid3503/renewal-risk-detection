"""
Signal weights and tier thresholds — all scoring logic lives here.
Change behavior by editing this file, never by touching scorer logic.
"""

SIGNAL_WEIGHTS: dict[str, float] = {
    "usage_drop_flag": 2.5,
    "near_zero_usage": 3.0,
    "negative_mom_trend": 1.5,
    "p1_tickets": 2.0,
    "unresolved_p1": 2.5,
    "escalated_tickets": 1.5,
    "low_nps": 2.0,
    "detractor_nps": 1.0,
    "competitor_mentioned": 3.0,
    "deprecated_sdk": 2.5,
    "sdk_deadline_30d": 1.5,
    "budget_cut": 2.0,
    "exec_escalation": 2.0,
    "migration_risk": 3.0,
    "renewal_threatened": 4.0,
    "missed_meetings": 1.0,
}

TIER_THRESHOLDS: dict[str, float] = {
    "High": 7.0,
    "Medium": 3.5,
}

# CSM signals below this confidence are weighted at 50%
CSM_CONFIDENCE_THRESHOLD: float = 0.6

# Accounts renewing within this many days are included in the analysis
RENEWAL_WINDOW_DAYS: int = 90

# SDK versions considered deprecated per changelog (April 30 2026 deadline)
DEPRECATED_SDK_VERSIONS: list[str] = [
    "v3.0", "v3.1", "v3.2", "v3.x", "3.0", "3.1", "3.2",
    "v3.0.0", "v3.1.0", "v3.2.0",
]
DEPRECATED_SDK_DEADLINE: str = "2026-04-30"

# Competitor names to detect in text (NPS verbatim + CSM notes)
COMPETITOR_SIGNALS: list[str] = [
    "strapi",
    "sanity",
    "contentful",
    "hygraph",
    "kontent.ai",
    "kontent",
    "builder.io",
    "wordpress",
    "drupal",
    "prismic",
]
