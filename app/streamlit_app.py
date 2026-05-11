"""
Renewal Risk Intelligence Dashboard — Streamlit UI.

Pages (sidebar navigation):
  🏠 Dashboard       — KPIs + ranked risk table + account drill-down
  🧠 How It Works    — pipeline explanation, agentic flow, LLM role
  📚 Signal Reference — full signal weight table + tier definitions
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from pipeline import run_pipeline
from models.account import RiskReport, RiskTier
from pipeline.scoring.weight_config import SIGNAL_WEIGHTS, TIER_THRESHOLDS

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Renewal Risk Intelligence",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Smooth transitions on all metric cards */
[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
    transition: box-shadow .2s;
}
[data-testid="metric-container"]:hover { box-shadow: 0 4px 16px rgba(0,0,0,.08); }

/* Tier badge pills */
.tier-high   { color:#dc2626; background:#fef2f2; border:1px solid #fecaca;
               border-radius:20px; padding:2px 10px; font-weight:600; font-size:13px; }
.tier-medium { color:#d97706; background:#fffbeb; border:1px solid #fde68a;
               border-radius:20px; padding:2px 10px; font-weight:600; font-size:13px; }
.tier-low    { color:#16a34a; background:#f0fdf4; border:1px solid #bbf7d0;
               border-radius:20px; padding:2px 10px; font-weight:600; font-size:13px; }

/* Pipeline node cards on How It Works page */
.pipeline-node {
    background: #f0f9ff; border: 2px solid #bae6fd;
    border-radius: 12px; padding: 16px; text-align: center;
    font-weight: 600; font-size: 14px; color: #0369a1;
}
.pipeline-node.llm {
    background: #faf5ff; border-color: #e9d5ff; color: #7c3aed;
}

/* Tooltip helper icon */
.tooltip-icon { color:#94a3b8; font-size:12px; cursor:help; }

/* Signal row in reference table */
.signal-row { padding:6px 0; border-bottom:1px solid #f1f5f9; }

/* Loading stage rows */
.stage-row {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 0; font-size: 15px;
}
.stage-dot {
    width:10px; height:10px; border-radius:50%;
    background:#cbd5e1; flex-shrink:0;
}
.stage-dot.active { background:#3b82f6; animation: pulse-dot 1s infinite; }
.stage-dot.done   { background:#22c55e; }

@keyframes pulse-dot {
    0%,100% { transform: scale(1); opacity:1; }
    50%     { transform: scale(1.5); opacity:.7; }
}

/* Animated data-flow banner on loading screen */
@keyframes flow-right {
    0%   { left: -40%; }
    100% { left: 110%; }
}
.flow-bar {
    position: relative; overflow: hidden;
    height: 4px; background: #e2e8f0; border-radius: 4px; margin: 4px 0;
}
.flow-bar::after {
    content: "";
    position: absolute; top: 0; height: 100%; width: 40%;
    background: linear-gradient(90deg, transparent, #3b82f6, transparent);
    animation: flow-right 1.4s ease-in-out infinite;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TIER_BADGE  = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
TIER_CLASS  = {"High": "tier-high", "Medium": "tier-medium", "Low": "tier-low"}
TIER_COLOR  = {"High": "#dc2626",   "Medium": "#d97706",      "Low": "#16a34a"}

# ── Pipeline runner — module-level cache survives page nav, refresh, new tabs ──

@st.cache_data(show_spinner=False)
def _cached_pipeline(run_key: int = 0) -> list[RiskReport]:
    """Run pipeline once per run_key. Cached at module level — not per session.
    Increment run_key (via the Re-run button) to force a fresh execution."""
    return run_pipeline()


_PIPELINE_STAGES = [
    ("📥", "Stage 1 — Load & Reconcile",     "6 files (5 CSV/TXT + changelog.md) → DataFrames · LLM extracts deprecation registry"),
    ("⚡", "Stage 2 — Signal Computation",    "Usage trends · support escalations · NPS · SDK cross-ref"),
    ("🤖", "Stage 3 — LLM CSM Extraction",   "LLM parses messy CSM notes → structured signals + confidence"),
    ("📊", "Stage 4 — Risk Scoring",          "Weighted signal sum → High / Medium / Low tier assignment"),
    ("✍️", "Stage 5 — LLM Explanation",      "Plain-English briefing + recommended action per account"),
]

_STAGE_ROW_CSS = (
    "display:flex;align-items:flex-start;gap:12px;"
    "padding:10px 14px;border-bottom:1px solid #f1f5f9;"
)
_ICON_CSS  = "font-size:22px;flex-shrink:0;line-height:1.4"
_TITLE_CSS = "font-weight:600;font-size:14px;color:#0f172a;margin:0 0 2px"
_DESC_CSS  = "font-size:12px;color:#64748b;margin:0"


def _stage_rows_html() -> str:
    rows = "".join(
        f"<div style='{_STAGE_ROW_CSS}'>"
        f"  <div style='{_ICON_CSS}'>{icon}</div>"
        f"  <div>"
        f"    <p style='{_TITLE_CSS}'>{title}</p>"
        f"    <p style='{_DESC_CSS}'>{detail}</p>"
        f"  </div>"
        f"</div>"
        for icon, title, detail in _PIPELINE_STAGES
    )
    return (
        "<div style='border:1px solid #e2e8f0;border-radius:10px;"
        "overflow:hidden;background:#fff;margin-top:8px'>"
        f"{rows}"
        "</div>"
    )


def get_reports() -> list[RiskReport]:
    """Return pipeline reports.

    Fast path: module-level @cache_data hit → returns in <100 ms even after a
    browser refresh or opening a new tab.
    Slow path: cache miss (first ever run or after Re-run) → shows an animated
    st.status() panel listing all 5 pipeline stages while blocking.
    """
    run_key = st.session_state.get("run_key", 0)

    # Session-local shortcut — avoids even the @cache_data lookup on hot reruns
    # (sidebar filter changes, expander toggles, etc.)
    if st.session_state.get("_loaded_key") == run_key and "reports" in st.session_state:
        return st.session_state["reports"]

    with st.status("⏳ Running pipeline — this takes ~60–90 s on first run…", expanded=True) as status:
        st.markdown(
            "<p style='color:#64748b;font-size:13px;margin:0 0 4px'>"
            "All 5 stages will execute sequentially. Results are cached after this run.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(_stage_rows_html(), unsafe_allow_html=True)
        st.markdown(
            "<div class='flow-bar' style='margin-top:12px'></div>",
            unsafe_allow_html=True,
        )

        reports = _cached_pipeline(run_key)

        status.update(
            label="✅ Pipeline complete — results cached for this session.",
            state="complete",
            expanded=False,
        )

    st.session_state["reports"]     = reports
    st.session_state["_loaded_key"] = run_key
    return reports


_NODE_STYLE    = ("background:#f0f9ff;border:2px solid #bae6fd;border-radius:12px;"
                   "padding:14px 18px;text-align:center;font-weight:600;"
                   "font-size:13px;color:#0369a1;min-width:90px;")
_NODE_LLM_STYLE= ("background:#faf5ff;border:2px solid #e9d5ff;border-radius:12px;"
                   "padding:14px 18px;text-align:center;font-weight:600;"
                   "font-size:13px;color:#7c3aed;min-width:90px;")
_ARROW         = "<div style='color:#94a3b8;font-size:22px;padding:0 4px'>→</div>"


def _node(icon: str, label: str, llm: bool = False) -> str:
    """Return an inline-styled pipeline node div."""
    style = _NODE_LLM_STYLE if llm else _NODE_STYLE
    return f"<div style='{style}'>{icon}<br>{label}</div>"


def _show_loading_splash() -> None:
    """Animated splash shown once before the pipeline starts."""
    nodes = _ARROW.join([
        _node("📥", "Load Data"),
        _node("⚡", "Signals"),
        _node("🤖", "LLM Extract", llm=True),
        _node("📊", "Score"),
        _node("✍️", "LLM Explain", llm=True),
    ])
    st.markdown(f"""
    <div style="text-align:center;padding:32px 0 16px">
      <div style="font-size:52px;animation:none">⚠️</div>
      <h2 style="margin:8px 0 4px;color:#0f172a">Renewal Risk Intelligence</h2>
      <p style="color:#64748b;font-size:15px;margin-bottom:28px">
        Analysing your 90-day renewal window — hang tight while the pipeline runs.
      </p>
    </div>

    <div style="display:flex;justify-content:center;align-items:center;
                flex-wrap:wrap;gap:6px;margin-bottom:20px">
      {nodes}
    </div>

    <div style="display:flex;justify-content:center;gap:28px;
                font-size:13px;color:#64748b;margin-bottom:8px">
      <span>🗂️ 6 data sources</span>
      <span>📋 120 accounts</span>
      <span>🔍 30 in renewal window</span>
      <span>✨ 3 LLM passes</span>
    </div>
    """, unsafe_allow_html=True)
    st.info(
        "💡 **How this works:** Deterministic signal scoring (no LLM for risk scores) "
        "+ LLM for changelog parsing, CSM note extraction, and plain-English briefings. "
        "First run: ~60–90 s. Subsequent interactions: instant (session-cached).",
        icon=None,
    )
    st.markdown("---")
    st.markdown("**Pipeline progress**")


def _show_start_screen() -> None:
    """Landing hero shown before the user triggers the pipeline."""
    nodes = _ARROW.join([
        _node("📥", "Load Data"),
        _node("⚡", "Signals"),
        _node("🤖", "LLM Extract", llm=True),
        _node("📊", "Score"),
        _node("✍️", "LLM Explain", llm=True),
    ])
    st.markdown(f"""
    <div style="text-align:center;padding:48px 0 24px">
      <div style="font-size:64px;margin-bottom:12px">⚠️</div>
      <h1 style="margin:0 0 8px;color:#0f172a;font-size:2.2rem">Renewal Risk Intelligence</h1>
      <p style="color:#64748b;font-size:16px;max-width:560px;margin:0 auto 32px">
        Automatically surface which accounts are at risk of churning — before it's too late.
        Deterministic scoring across 5 data sources, with LLM-generated plain-English briefings.
      </p>
    </div>

    <div style="display:flex;justify-content:center;align-items:center;
                flex-wrap:wrap;gap:6px;margin-bottom:28px">
      {nodes}
    </div>

    <div style="display:flex;justify-content:center;gap:32px;
                font-size:13px;color:#64748b;margin-bottom:36px;flex-wrap:wrap">
      <span>🗂️ 6 data sources</span>
      <span>📋 120 accounts</span>
      <span>🔍 ~30 in renewal window</span>
      <span>✨ 3 LLM passes</span>
      <span>⏱️ ~60–90 s first run</span>
    </div>
    """, unsafe_allow_html=True)

    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        if st.button(
            "🚀 Start Analysis",
            use_container_width=True,
            type="primary",
            help="Run the full pipeline: load data, compute risk signals, call the LLM, and render the dashboard.",
        ):
            st.session_state["analysis_started"] = True
            st.rerun()

    st.markdown("")
    st.info(
        "💡 **First run takes ~60–90 s** (LLM calls for all accounts in the renewal window). "
        "Subsequent page interactions are instant — results are session-cached.",
        icon=None,
    )


# ── Sidebar navigation ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚠️ Renewal Risk")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "🧠 How It Works", "📚 Signal Reference"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if st.button("🔄 Re-run pipeline", help="Clear cache and re-run the full pipeline"):
        _cached_pipeline.clear()
        st.session_state.pop("reports", None)
        st.session_state.pop("_loaded_key", None)
        st.session_state["run_key"] = st.session_state.get("run_key", 0) + 1
        st.session_state["analysis_started"] = False
        st.rerun()
    st.caption("Scores are deterministic weighted sums.\nLLM writes explanations only.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Dashboard":
    st.title("⚠️ Renewal Risk Intelligence")
    st.caption("Contentstack BizOps · Accounts renewing in the next 90 days · LLM-powered risk briefings")

    if not st.session_state.get("analysis_started", False):
        _show_start_screen()
        st.stop()

    reports = get_reports()

    if not reports:
        st.error("Pipeline returned no reports. Check that data/raw/ files exist and LLM_API_KEY is set in .env.")
        st.stop()

    # ── Sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("#### Filters")
        selected_tiers = st.multiselect(
            "Risk Tier",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
            help="Filter by computed risk tier. Tier is determined by the weighted signal score.",
        )
        selected_csms = st.multiselect(
            "CSM",
            sorted({r.csm_name for r in reports}),
            default=sorted({r.csm_name for r in reports}),
            help="Filter to accounts owned by specific Customer Success Managers.",
        )
        selected_industries = st.multiselect(
            "Industry",
            sorted({r.industry for r in reports}),
            default=sorted({r.industry for r in reports}),
            help="Filter by account industry vertical.",
        )

    filtered = [
        r for r in reports
        if r.tier.value in selected_tiers
        and r.csm_name in selected_csms
        and r.industry in selected_industries
    ]

    # ── KPI row ───────────────────────────────────────────────────────────────
    high   = [r for r in filtered if r.tier == RiskTier.HIGH]
    medium = [r for r in filtered if r.tier == RiskTier.MEDIUM]
    low    = [r for r in filtered if r.tier == RiskTier.LOW]
    arr_high   = sum(r.arr for r in high)
    arr_medium = sum(r.arr for r in medium)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Accounts in window",
        len(filtered),
        help="Number of accounts whose contract renews within the next 90 days.",
    )
    c2.metric(
        "🔴 High Risk",
        len(high),
        help="Accounts with a risk score ≥ 7.0. Require immediate CSM action this week.",
    )
    c3.metric(
        "🟡 Medium Risk",
        len(medium),
        help="Accounts with a risk score between 3.5 and 7.0. Proactive outreach recommended.",
    )
    c4.metric(
        "🟢 Low Risk",
        len(low),
        help="Accounts with a risk score below 3.5. Maintain regular cadence.",
    )
    c5.metric(
        "ARR at High Risk",
        f"${arr_high:,.0f}",
        delta=f"+${arr_medium:,.0f} medium exposure",
        delta_color="off",
        help="Sum of ARR for all High-risk accounts. This is the immediate revenue at risk.",
    )

    st.markdown("---")

    # ── Risk table ────────────────────────────────────────────────────────────
    st.subheader(
        "Ranked Risk Table",
        help="Accounts sorted by risk score (highest first). Click any account name below to drill down.",
    )

    rows = []
    for r in filtered:
        badge = TIER_BADGE.get(r.tier.value, "")
        rows.append({
            "": badge,
            "Account": r.account_name,
            "Tier": r.tier.value,
            "Score": r.raw_score,
            "ARR": f"${r.arr:,.0f}",
            "Days Left": r.days_to_renewal,
            "Top Signal": r.top_signal,
            "Plan": r.plan_tier,
            "CSM": r.csm_name,
            "Industry": r.industry,
        })

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
            column_config={
                "":        st.column_config.TextColumn("", width=30),
                "Score":   st.column_config.NumberColumn("Score", format="%.1f",
                    help="Weighted sum of all risk signals that fired for this account."),
                "Days Left": st.column_config.NumberColumn("Days Left",
                    help="Calendar days until the contract renewal date."),
                "Top Signal": st.column_config.TextColumn("Top Signal",
                    help="The single highest-weighted signal driving this account's risk score."),
                "Plan":    st.column_config.TextColumn("Plan",
                    help="Contentstack plan tier: Starter / Growth / Enterprise."),
            },
        )
        with st.expander("ℹ️ How to read this table"):
            st.markdown("""
| Column | What it means |
|--------|--------------|
| **Score** | Weighted sum of all risk signals. Not a percentage — higher = more risk signals fired. Max possible is ~35. |
| **Top Signal** | The single signal contributing the most weight to this account's score. |
| **Days Left** | Days until contract end date. Accounts with < 30 days left need urgent action regardless of score. |
| **Tier** | 🔴 High ≥ 7.0 · 🟡 Medium ≥ 3.5 · 🟢 Low < 3.5 |
""")
    else:
        st.info("No accounts match the current filters.")

    # ── Account detail ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Account Detail", help="Select any account to see the full LLM-generated risk briefing.")

    account_options = [r.account_name for r in filtered]
    if not account_options:
        st.info("Adjust filters to see accounts.")
        st.stop()

    selected_name = st.selectbox(
        "Select account",
        account_options,
        help="Choose an account to see its full risk briefing, contributing signals, and recommended action.",
    )
    report = next((r for r in filtered if r.account_name == selected_name), None)

    if report:
        tier_color = TIER_COLOR.get(report.tier.value, "#374151")
        badge = TIER_BADGE.get(report.tier.value, "")

        col_title, col_meta = st.columns([3, 2])
        with col_title:
            st.markdown(f"### {badge} {report.account_name}")
            st.markdown(
                f"<span class='{TIER_CLASS.get(report.tier.value, '')}'>"
                f"{report.tier.value} Risk</span>"
                f"&nbsp; Score: <b>{report.raw_score:.1f}</b>"
                f"&nbsp;·&nbsp; ARR: <b>${report.arr:,.0f}</b>"
                f"&nbsp;·&nbsp; Renews in <b>{report.days_to_renewal} days</b>",
                unsafe_allow_html=True,
            )
        with col_meta:
            st.markdown(
                f"**Plan:** {report.plan_tier} &nbsp;·&nbsp; "
                f"**CSM:** {report.csm_name} &nbsp;·&nbsp; "
                f"**Industry:** {report.industry}"
            )

        st.divider()
        left, right = st.columns([3, 2])

        with left:
            st.markdown(
                "**Why this account is at risk**",
                help="LLM-generated explanation grounded in the pre-computed risk signals. "
                     "The LLM does NOT decide the tier — it explains the signals the scorer already found.",
            )
            st.write(report.explanation)
            st.markdown("")
            st.markdown(
                "**Recommended action**",
                help="Specific action for the named CSM to take this week. Generated by the LLM "
                     "from account context — not a generic template.",
            )
            st.info(report.recommended_action)

        with right:
            st.markdown(
                "**Signal breakdown**",
                help="Every signal that fired for this account, sorted by weight contribution. "
                     "The bar shows each signal's share of the total score.",
            )
            if report.contributing_signals:
                total = report.raw_score or 1.0
                _SOURCE_ICON = {
                    "Usage": "📈", "usage": "📈",
                    "P1": "🎫", "ticket": "🎫", "Unresolved": "🎫", "escalated": "🎫",
                    "NPS": "⭐", "nps": "⭐", "Low NPS": "⭐", "detractor": "⭐",
                    "Competitor": "🔍",
                    "SDK": "🔧", "deprecated": "🔧",
                    "Budget": "💰", "Executive": "👔", "Migration": "🔄",
                    "churn": "⚠️", "Explicit": "⚠️",
                    "meeting": "📅",
                }
                def _signal_icon(label: str) -> str:
                    for k, v in _SOURCE_ICON.items():
                        if k.lower() in label.lower():
                            return v
                    return "•"

                breakdown_rows = []
                for label, w in zip(report.contributing_signals, report.contributing_signal_weights):
                    breakdown_rows.append({
                        "Signal": f"{_signal_icon(label)}  {label}",
                        "pts": w,
                        "% of score": round(w / total, 3),
                    })
                st.dataframe(
                    pd.DataFrame(breakdown_rows),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "pts": st.column_config.NumberColumn("pts", format="%.1f",
                            help="Weight points this signal contributed to the total score."),
                        "% of score": st.column_config.ProgressColumn(
                            "% of score", min_value=0, max_value=1, format="%.0%%",
                            help="This signal's share of the total raw score."),
                    },
                )
            else:
                st.caption("No significant signals fired.")

            # ── Pipeline transparency panel ────────────────────────────────────
            st.markdown("")
            st.markdown(
                "**Pipeline features used**",
                help="Which pipeline features contributed to analysing this account.",
            )
            features: list[tuple[str, str, str]] = []

            # Always present: deterministic signal computation
            features.append(("⚡", "Deterministic scoring", "Weighted sum of all fired signals — no LLM involvement in the score"))

            # SDK cross-reference
            if any("deprecated SDK" in s or "SDK deadline" in s for s in report.contributing_signals):
                features.append(("🔧", "Changelog × usage cross-reference", "SDK version from usage_metrics.csv matched against engineering changelog deprecation registry"))

            # CSM LLM extraction
            if report.csm_confidence is not None:
                conf_pct = int(report.csm_confidence * 100)
                features.append(("🤖", f"LLM CSM extraction (confidence: {conf_pct}%)",
                    "Raw call notes parsed by LLM with CoT reasoning → structured risk flags"))

            # NPS translation
            if report.nps_verbatim_translated:
                features.append(("🌐", "NPS verbatim translated (non-English → English)",
                    f'Original: "{report.nps_verbatim_translated[:80]}…"' if len(report.nps_verbatim_translated) > 80
                    else f'Translated: "{report.nps_verbatim_translated}"'))

            # LLM explanation
            if report.tier in (RiskTier.HIGH, RiskTier.MEDIUM):
                features.append(("✍️", "LLM briefing generated", "Score + signals sent to LLM for plain-English explanation and specific recommended action"))

            _F_STYLE = ("background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;"
                        "padding:8px 12px;margin-bottom:6px;font-size:12px")
            _F_ICON_CSS = "font-size:16px;margin-right:8px"
            _F_TITLE_CSS = "font-weight:600;color:#0f172a"
            _F_DESC_CSS = "color:#64748b;margin-top:2px"
            for icon, title, desc in features:
                st.markdown(
                    f"<div style='{_F_STYLE}'>"
                    f"<span style='{_F_ICON_CSS}'>{icon}</span>"
                    f"<span style='{_F_TITLE_CSS}'>{title}</span>"
                    f"<div style='{_F_DESC_CSS}'>{desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── SDK insight panel ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(
        "🔍 Non-Obvious Insight: SDK Deprecation Risk",
        help="Accounts on deprecated SDK v3.x face a hard security-patch cutoff on April 30 2026. "
             "This risk is invisible in the CRM — it only surfaces by cross-referencing usage_metrics.csv "
             "with the engineering changelog. A pure CRM-based system would miss it entirely.",
    )
    st.markdown(
        "SDK **v3.x** loses security patches on **April 30 2026** (REST API v2 also sunsets). "
        "The accounts below are in the renewal window **and** on a deprecated SDK. "
        "Some have no CSM note mentioning it — that's the invisible risk."
    )

    sdk_rows = []
    for r in reports:
        sdk_signal = next((s for s in r.contributing_signals if "deprecated SDK" in s), None)
        if sdk_signal:
            csm_aware = any(
                kw in r.explanation.lower()
                for kw in ["sdk", "v3", "deprecated", "migration", "endpoint"]
            )
            sdk_rows.append({
                "Account":        r.account_name,
                "Tier":           r.tier.value,
                "Risk Score":     r.raw_score,
                "ARR":            f"${r.arr:,.0f}",
                "Days to Renewal": r.days_to_renewal,
                "SDK Signal":     sdk_signal,
                "CSM Aware?":     "✅ Yes" if csm_aware else "⚠️ No",
            })

    if sdk_rows:
        sdk_df = pd.DataFrame(sdk_rows)
        st.dataframe(
            sdk_df,
            width="stretch",
            hide_index=True,
            column_config={
                "CSM Aware?": st.column_config.TextColumn(
                    "CSM Aware?",
                    help="Whether the LLM-generated briefing references the SDK issue. "
                         "'No' means the risk exists but hasn't been flagged in CSM notes.",
                ),
                "Days to Renewal": st.column_config.NumberColumn(
                    "Days to Renewal",
                    help="Deadline pressure: accounts renewing BEFORE April 30 face simultaneous "
                         "SDK cutoff and contract renewal in the same window.",
                ),
            },
        )
        not_aware = sum(1 for r in sdk_rows if r["CSM Aware?"] == "⚠️ No")
        if not_aware:
            st.warning(
                f"⚠️ **{not_aware} account(s)** are on a deprecated SDK with no CSM note mentioning it. "
                "These are invisible in the CRM — only surfaced by cross-referencing the engineering changelog.",
                icon=None,
            )
    else:
        st.success("No deprecated-SDK accounts in the 90-day renewal window.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🧠 How It Works":
    st.title("🧠 How It Works")
    st.caption("Architecture, agentic flow, and the reasoning behind every design decision.")

    # ── Problem statement ──────────────────────────────────────────────────────
    with st.expander("📌 The problem this solves", expanded=True):
        st.markdown("""
Every quarter, a BizOps team asks: **which accounts are about to churn, and why?**

Without this system:
- A CSM manually reads 120 accounts' notes, tickets, and usage metrics
- They miss the account that *looks* healthy on NPS but is quietly migrating to a homegrown solution
- They don't know 8 accounts are on a deprecated SDK losing security patches in 30 days — because that lives in the engineering changelog, not in Salesforce
- By the time they flag a risk, there are 2 weeks left before renewal

With this system:
- 5 data sources ingested, reconciled, and cross-referenced automatically
- Every account in the 90-day window gets a deterministic risk score
- An LLM reads the score and writes a *specific, actionable briefing* per account
- Full pipeline runs in ~60–90 seconds
        """)

    st.markdown("---")

    # ── Pipeline flow ──────────────────────────────────────────────────────────
    st.subheader("Pipeline Flow")
    st.markdown("The pipeline runs in 4 sequential stages. Blue = deterministic Python. Purple = LLM call.")

    _ARW = "<div style='text-align:center;font-size:24px;padding-top:28px;color:#94a3b8'>→</div>"
    n1, a1, n2, a2, n3, a3, n4, a4, n5 = st.columns([3, .5, 3, .5, 3, .5, 3, .5, 3])
    with n1:
        st.markdown(
            f"<div style='{_NODE_STYLE}'>"
            "📥 <b>Stage 1</b><br>Load &amp; Reconcile<br>"
            "<small style='color:#64748b;font-weight:400'>5 files → DataFrames<br>Fuzzy name matching</small>"
            "</div>", unsafe_allow_html=True)
    with a1: st.markdown(_ARW, unsafe_allow_html=True)
    with n2:
        st.markdown(
            f"<div style='{_NODE_STYLE}'>"
            "⚡ <b>Stage 2</b><br>Signal Computation<br>"
            "<small style='color:#64748b;font-weight:400'>Usage · Support · NPS<br>SDK × Changelog</small>"
            "</div>", unsafe_allow_html=True)
    with a2: st.markdown(_ARW, unsafe_allow_html=True)
    with n3:
        st.markdown(
            f"<div style='{_NODE_LLM_STYLE}'>"
            "🤖 <b>Stage 3</b><br>LLM CSM Extraction<br>"
            "<small style='color:#7c3aed;font-weight:400'>LLM parses messy<br>unstructured notes</small>"
            "</div>", unsafe_allow_html=True)
    with a3: st.markdown(_ARW, unsafe_allow_html=True)
    with n4:
        st.markdown(
            f"<div style='{_NODE_STYLE}'>"
            "📊 <b>Stage 4</b><br>Risk Scoring<br>"
            "<small style='color:#64748b;font-weight:400'>Weighted sum of signals<br>→ High / Medium / Low</small>"
            "</div>", unsafe_allow_html=True)
    with a4: st.markdown(_ARW, unsafe_allow_html=True)
    with n5:
        st.markdown(
            f"<div style='{_NODE_LLM_STYLE}'>"
            "✍️ <b>Stage 5</b><br>LLM Explanation<br>"
            "<small style='color:#7c3aed;font-weight:400'>Plain-English briefing<br>per at-risk account</small>"
            "</div>", unsafe_allow_html=True)

    st.markdown("")

    # ── Data sources ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Data Sources")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("**📋 accounts.csv**")
        st.caption("120 accounts · spine of the pipeline")
        st.markdown("Firmographics, ARR, contract end date, plan tier, CSM assignment. "
                    "Every other source joins to this.")

        st.markdown("**📈 usage_metrics.csv**")
        st.caption("720 rows · 6 months × 120 accounts")
        st.markdown("Monthly active users, API calls, SDK version. Used to compute trend signals.")
    with d2:
        st.markdown("**🎫 support_tickets.csv**")
        st.caption("271 tickets · 96 of 120 accounts")
        st.markdown("Ticket priority (P1–P4), status (open/closed/escalated), resolution time.")

        st.markdown("**⭐ nps_responses.csv**")
        st.caption("98 responses · 98 of 120 accounts")
        st.markdown("NPS score 0–10 + verbatim comment. Scanned for competitor mentions.")
    with d3:
        st.markdown("**📝 csm_notes.txt**")
        st.caption("Unstructured · messy by design")
        st.markdown("Mixed date formats, typos in account names, one note in Mandarin. "
                    "Parsed by the LLM after fuzzy account-name reconciliation.")

        st.markdown("**📋 changelog.md**")
        st.caption("The non-obvious source")
        st.markdown("Contains SDK v3.x deprecation notices. Cross-referenced with usage_metrics "
                    "to surface technical churn risk invisible to the CRM.")

    # ── Why deterministic + LLM ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Why Deterministic Scoring + LLM Explanation?")

    bad, good = st.columns(2)
    with bad:
        st.error("❌ The wrong approach — LLM assigns the tier")
        st.code(
            '# Non-reproducible. Same account on Monday vs Friday\n'
            '# can get different tiers.\n'
            'tier = llm.ask("Is NovaTech High, Medium, or Low risk?")\n'
            '# Auditable? No.  Trackable over time? No.',
            language="python",
        )
        st.caption("Can't explain a tier change. Can't audit the logic. Drifts with model updates.")

    with good:
        st.success("✅ The right approach — deterministic score, LLM explanation")
        st.code(
            '# Pure function. Same input → same output. Always.\n'
            'score = score_account(account)\n'
            '# LLM explains WHY, not what tier.\n'
            'report = generate_risk_report(account, score)',
            language="python",
        )
        st.caption("Reproducible. Auditable. Trackable. The LLM adds value without introducing instability.")

    # ── LLM call map ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Where the LLM Is Used (and Why)")

    llm1, llm2, llm3 = st.columns(3)
    with llm1:
        st.markdown("#### 🤖 CSM Note Parsing")
        st.markdown("""
**Input:** Raw, messy CSM call notes
**Output:** Structured `CsmSignal` per account
**Technique:** Chain-of-Thought reasoning + 3 few-shot examples

The model reasons through each signal field (competitor mentioned? exec on call?) before committing to a boolean. Confidence score controls how much weight the signal gets in scoring.

**Why LLM:** Rule-based regex can't handle typos, Mandarin, implicit churn threats, or the difference between "might look around" (low confidence) and "30% discount or we walk" (high confidence).
        """)
    with llm2:
        st.markdown("#### 📋 Changelog Extraction")
        st.markdown("""
**Input:** Raw `changelog.md` markdown
**Output:** Structured deprecation event registry
**Technique:** CoT scan + 1 few-shot example + severity rubric

The model scans each changelog section for deprecation markers and structures them with deadlines, affected versions, and required customer actions.

**Why LLM:** The changelog uses natural language that varies per release. A regex for "deprecated" would miss "sunset", "end-of-life", "stop receiving patches", etc.
        """)
    with llm3:
        st.markdown("#### ✍️ Risk Explanation")
        st.markdown("""
**Input:** Pre-computed score + all signals + account context
**Output:** 2-3 sentence explanation + specific recommended action
**Technique:** CoT narrative step + 2 few-shot examples with BAD vs GOOD contrast

The model first identifies which 1-2 signals tell the most coherent story (`signal_narrative` field), then writes from that narrative. The examples explicitly show what generic output looks like so the model avoids it.

**Why LLM:** "NPS score of 3 detected" is useless. "Vanguard Retail's champion called out losing faith in the roadmap after a 6-week-open ticket broke their workflows" is actionable.
        """)

    # ── Non-obvious insight explained ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("The Non-Obvious Insight Explained")
    st.markdown("""
SDK v3.x loses security patches on **April 30 2026**. REST API v2 sunsets the same day.

This information lives in the **engineering changelog** — not in Salesforce, not in tickets, not in CSM notes.
A CSM has no reason to know their account is on v3.2.0 unless an engineer tells them.

The `changelog_signals` module is the only place in the system where this join happens:
""")
    st.code(
        "# Cross-reference: usage_metrics.sdk_version × changelog.deprecation_registry\n"
        "# This join exists nowhere in the CRM.\n"
        "is_deprecated = _is_deprecated(account.usage.sdk_version)\n"
        "# → adds 2.5 weight to the risk score silently\n"
        "# → appears in the dashboard even if the CSM has zero notes about it",
        language="python",
    )
    st.info(
        "**Result:** Accounts can be scored as High risk purely from SDK + support signals, "
        "even if their CSM thinks they're healthy. The dashboard's SDK Insight Panel surfaces "
        "exactly these accounts.",
        icon="💡",
    )

    # ── Stage deep dives ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Stage Deep Dives")
    st.caption("Expand any stage to see the exact code logic, a dry-run example, and the output it produces.")

    # Stage 1
    with st.expander("📥 Stage 1 — Load & Filter: how 120 accounts become ~30"):
        st.markdown("#### What it does")
        st.markdown(
            "Reads all 5 raw files, normalises dirty data, then keeps only accounts whose "
            "contract expires within the next **90 days**."
        )

        st.markdown("#### Raw data problems this stage fixes")
        problems, fixes = st.columns(2)
        with problems:
            st.markdown(
                "<div style='background:#fef2f2;border:1px solid #fecaca;border-radius:8px;"
                "padding:12px 16px;font-size:13px'>"
                "<b style='color:#dc2626'>Raw (dirty)</b><br><br>"
                "<code>account_id: \" ACC-042 \"</code><br>"
                "<code>arr: \"$21,000\"</code><br>"
                "<code>contract_end_date: \"06/15/2026\"</code><br>"
                "<code>priority: \"p1 \"</code><br>"
                "<code>verbatim_comment: NaN</code>"
                "</div>",
                unsafe_allow_html=True,
            )
        with fixes:
            st.markdown(
                "<div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;"
                "padding:12px 16px;font-size:13px'>"
                "<b style='color:#16a34a'>After Stage 1 (clean)</b><br><br>"
                "<code>account_id: \"ACC-042\"</code><br>"
                "<code>arr: 21000.0</code><br>"
                "<code>contract_end_date: 2026-06-15 (datetime)</code><br>"
                "<code>priority: \"P1\"</code><br>"
                "<code>verbatim_comment: \"\"</code>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### 90-day window calculation")
        st.code(
            "today = date.today()\n"
            "cutoff = today + timedelta(days=90)\n\n"
            "accounts_df['days_to_renewal'] = (\n"
            "    accounts_df['contract_end_date'] - today\n"
            ").dt.days\n\n"
            "in_window = accounts_df[\n"
            "    (accounts_df['contract_end_date'] >= today) &\n"
            "    (accounts_df['contract_end_date'] <= cutoff)\n"
            "]",
            language="python",
        )
        st.markdown(
            "_Why 90 days?_ Standard enterprise CS practice — enough lead time to act. "
            "Accounts outside this window have lower urgency and would inflate the dashboard with noise."
        )

        st.markdown("#### Dry run — April 29, 2026 → cutoff July 28, 2026")
        st.code(
            "Input:  120 accounts\n\n"
            "Account               Contract end   Days left   Decision\n"
            "──────────────────────────────────────────────────────────\n"
            "BrightPath Solutions  2026-06-15     47          ✓ IN\n"
            "Pinnacle Media Group  2026-05-10     11          ✓ IN\n"
            "Zenith Publishing     2026-07-20     82          ✓ IN\n"
            "Thunderbolt Motors    2026-09-01     125         ✗ OUT  (too far)\n"
            "Meridian Health       2026-03-10     -50         ✗ OUT  (already expired)\n\n"
            "Output: ~30 accounts forwarded to Stage 2",
            language="text",
        )

        st.markdown("#### What Stage 1 produces")
        st.code(
            "BEFORE:  5 raw files on disk\n\n"
            "AFTER:\n"
            "  accounts_df  → 30 rows  (filtered to 90-day window)\n"
            "  usage_df     → ~180 rows (6 months × 30 accounts)\n"
            "  tickets_df   → variable  (not all accounts have tickets)\n"
            "  nps_df       → variable  (not all accounts have NPS)\n"
            "  csm_notes    → one big string, all notes concatenated",
            language="text",
        )

    # Stage 2
    with st.expander("⚡ Stage 2 — Deterministic Signals: pure math, zero LLM"):
        st.markdown(
            "Four independent modules run in sequence. Each reads one data source and "
            "produces a `dict[account_id → SignalObject]`. **No LLM. Same input = same output. Always.**"
        )

        st.markdown("---")
        st.markdown("#### 📈 Usage Signals  (`pipeline/signals/usage_signals.py`)")
        st.markdown("Three calculations on the last 6 months of `active_users` per account:")

        u1, u2 = st.columns(2)
        with u1:
            st.markdown("**Code**")
            st.code(
                "users = [45, 42, 38, 31, 22, 8]  # Jan→Jun\n\n"
                "# 1. Month-over-Month\n"
                "mom = (users[-1] - users[-2]) / users[-2] * 100\n"
                "# = (8-22)/22*100 = -63.6%\n"
                "usage_drop_flag = mom < -30  # True\n\n"
                "# 2. 3-month slope (catches gradual bleeds)\n"
                "last3 = users[-3:]  # [31, 22, 8]\n"
                "slope = np.polyfit(range(3), last3, 1)[0]  # -11.5\n\n"
                "# 3. Near-zero check\n"
                "near_zero = users[-1] <= 2  # False (8 > 2)",
                language="python",
            )
        with u2:
            st.markdown("**Why two trend signals?**")
            st.markdown(
                "<div style='background:#fffbeb;border:1px solid #fde68a;border-radius:8px;"
                "padding:12px 16px;font-size:13px;margin-bottom:8px'>"
                "<b>Account A (gradual bleed) — MoM misses it:</b><br>"
                "<code>50→48→46→44→42→40</code><br>"
                "MoM = -4.8% → no flag<br>"
                "Slope = -2.0 → <b>fires</b>"
                "</div>"
                "<div style='background:#fef2f2;border:1px solid #fecaca;border-radius:8px;"
                "padding:12px 16px;font-size:13px'>"
                "<b>Account B (sudden crash) — slope misses it:</b><br>"
                "<code>50→50→50→50→50→10</code><br>"
                "MoM = -80% → <b>fires</b><br>"
                "Slope over last 3 months = -20 → also fires"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("**Output for ACC-042 (BrightPath Solutions):**")
        st.code(
            "UsageSignal(\n"
            "    account_id       = 'ACC-042',\n"
            "    latest_users     = 8,\n"
            "    mom_change_pct   = -63.6,\n"
            "    three_month_slope= -11.5,\n"
            "    usage_drop_flag  = True,   # -63.6 < -30\n"
            "    near_zero_usage  = False,  # 8 > 2\n"
            "    sdk_version      = 'v3.1.0',\n"
            ")",
            language="python",
        )

        st.markdown("---")
        st.markdown("#### 🎫 Support Signals  (`pipeline/signals/support_signals.py`)")

        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**Code**")
            st.code(
                "p1 = group[group['priority'] == 'P1']\n"
                "open_t = group[group['status'] == 'open']\n"
                "escalated = group[group['status'] == 'escalated']\n\n"
                "has_unresolved_p1 = len(\n"
                "    p1[p1['status'].isin(['open', 'escalated'])]\n"
                ") > 0\n\n"
                "avg_res = group['resolution_time_hours'].mean()",
                language="python",
            )
        with s2:
            st.markdown("**Dry run — ACC-042 tickets:**")
            st.code(
                "ticket-1: P1  closed     48h\n"
                "ticket-2: P1  open        0h   ← unresolved!\n"
                "ticket-3: P2  closed     12h\n"
                "ticket-4: P1  escalated   0h   ← unresolved!\n\n"
                "p1_count          = 3\n"
                "open_tickets      = 1\n"
                "escalated_tickets = 1\n"
                "avg_resolution_h  = 15.0\n"
                "has_unresolved_p1 = True",
                language="text",
            )

        st.markdown("---")
        st.markdown("#### ⭐ NPS Signals  (`pipeline/signals/nps_signals.py`)")

        n1, n2 = st.columns(2)
        with n1:
            st.markdown("**NPS category + competitor scan:**")
            st.code(
                "# Standard NPS industry categories\n"
                "def _nps_category(score):\n"
                "    if score in range(0, 7):  return 'detractor'\n"
                "    if score in range(7, 9):  return 'passive'\n"
                "    return 'promoter'\n\n"
                "COMPETITORS = ['strapi','sanity','contentful',\n"
                "               'hygraph','kontent.ai','builder.io',\n"
                "               'wordpress','drupal','prismic']\n\n"
                "def _detect_competitor(text):\n"
                "    return any(c in text.lower() for c in COMPETITORS)",
                language="python",
            )
        with n2:
            st.markdown("**Dry run — ACC-042:**")
            st.code(
                "score:   4\n"
                "comment: 'We've been evaluating\n"
                "          Contentful as a backup option'\n\n"
                "_nps_category(4):\n"
                "  4 in range(0,7)? YES → 'detractor'\n\n"
                "_detect_competitor(comment):\n"
                "  'contentful' in text? YES → True\n\n"
                "NpsSignal(\n"
                "  score=4, category='detractor',\n"
                "  competitor_mentioned=True\n"
                ")",
                language="text",
            )

        st.markdown("---")
        st.markdown("#### 📋 Changelog Signals  (`pipeline/signals/changelog_signals.py`)")
        st.markdown(
            "Uses the `sdk_version` already captured in `UsageSignal` — no new file read. "
            "Cross-references against the engineering deprecation registry."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Code**")
            st.code(
                "def _is_deprecated(sdk_version: str) -> bool:\n"
                "    v = sdk_version.lower().strip()\n"
                "    # ^v?3[\\.-] matches v3.1.0, 3.0, v3.x\n"
                "    # does NOT match v4.3.1\n"
                "    return bool(re.match(r'^v?3[\\.-]', v))\n"
                "           or v in ('v3.x', '3.x')\n\n"
                "def _days_to_deadline() -> int:\n"
                "    deadline = date(2026, 4, 30)\n"
                "    return (deadline - date.today()).days",
                language="python",
            )
        with c2:
            st.markdown("**Dry run — ACC-042 (sdk = v3.1.0):**")
            st.code(
                "v = 'v3.1.0'\n"
                "re.match(r'^v?3[\\.-]', 'v3.1.0'):\n"
                "  v?  → matches 'v'\n"
                "  3   → matches '3'\n"
                "  [.] → matches '.'\n"
                "→ MATCH → is_deprecated = True\n\n"
                "days_to_deadline: 1 day\n\n"
                "ChangelogSignal(\n"
                "  sdk_version='v3.1.0',\n"
                "  is_deprecated=True,\n"
                "  days_to_deadline=1,\n"
                "  deadline='2026-04-30'\n"
                ")",
                language="text",
            )
        st.info(
            "**The non-obvious join:** `usage_metrics.sdk_version × changelog.deprecation_registry` "
            "exists nowhere in the CRM. A CSM has zero visibility into which SDK version their "
            "account runs — this module is the only place these two facts meet.",
            icon="💡",
        )

    # Stage 3
    with st.expander("🤖 Stage 3 — LLM CSM Extraction: turning messy notes into structured signals"):
        st.markdown(
            "**The challenge:** `csm_notes.txt` is deliberately messy — mixed date formats, "
            "misspelled account names, one note in Mandarin, implicit churn threats no regex can catch."
        )

        st.markdown("#### The raw input (excerpt)")
        st.code(
            "3/18 - britepath - budget cut 20%, CTO leading CMS eval, missed last 2 QBRs\n"
            "---\n"
            "4/5 - Zenith Publishing - Renewl conversation started. They want a 30% discount\n"
            "or they walk. Competitor POC with Kontent.ai apparently already underway.\n"
            "CRO was cc'd on the last email thread.\n"
            "---\n"
            "march 25 -- meridian health -- priya\n"
            "Good news/bad news. NPS came back at 8 but their actual usage has cratered.\n"
            "Turns out they built a custom middleware layer...\n"
            "---\n"
            "[Note in Mandarin characters]\n"
            "---",
            language="text",
        )

        st.markdown("#### Step 1 — Split on `---`")
        st.code(
            "chunks = re.split(r'\\n---+\\n', raw_notes)\n"
            "# → ['3/18 - britepath - ...', '4/5 - Zenith Publishing - ...', ...]",
            language="python",
        )

        st.markdown("#### Step 2 — 3-strategy account→chunk matching")
        match1, match2, match3 = st.columns(3)
        with match1:
            st.markdown(
                "<div style='background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;"
                "padding:12px;font-size:13px;text-align:center'>"
                "<b style='color:#0369a1'>Strategy 1</b><br>Account ID match<br>"
                "<small>'acct ACC-042' in chunk</small>"
                "</div>",
                unsafe_allow_html=True,
            )
        with match2:
            st.markdown(
                "<div style='background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;"
                "padding:12px;font-size:13px;text-align:center'>"
                "<b style='color:#0369a1'>Strategy 2</b><br>Exact name match<br>"
                "<small>'brightpath solutions' in chunk</small>"
                "</div>",
                unsafe_allow_html=True,
            )
        with match3:
            st.markdown(
                "<div style='background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;"
                "padding:12px;font-size:13px;text-align:center'>"
                "<b style='color:#0369a1'>Strategy 3</b><br>First-word fallback<br>"
                "<small>'brightpath' → finds 'britepath'</small>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### Step 3 — Chain-of-Thought LLM call + Pydantic validation")
        cot1, cot2 = st.columns(2)
        with cot1:
            st.markdown("**The prompt requires reasoning first:**")
            st.code(
                "# Pydantic schema the LLM must return\n"
                "class _CsmExtraction(BaseModel):\n"
                "    reasoning: str          # MUST come first\n"
                "    competitor_mentioned: bool\n"
                "    budget_cut_mentioned: bool\n"
                "    exec_escalation: bool\n"
                "    migration_risk: bool\n"
                "    renewal_threatened: bool\n"
                "    missed_meetings: int    # >= 0\n"
                "    positive_signal: bool\n"
                "    confidence: float       # 0.0 – 1.0",
                language="python",
            )
        with cot2:
            st.markdown("**Dry run — ACC-042 BrightPath note:**")
            st.code(
                'Note: "3/18 - britepath - budget cut 20%,\n'
                '       CTO leading CMS eval, missed last 2 QBRs"\n\n'
                'LLM reasoning field:\n'
                '  competitor_mentioned: no named platform → FALSE\n'
                '  budget_cut_mentioned: "20% budget cut" explicit → TRUE (1.0)\n'
                '  exec_escalation: "CTO leading" = C-suite → TRUE (0.9)\n'
                '  migration_risk: "CMS eval" = evaluating alt → TRUE (0.8)\n'
                '  renewal_threatened: not stated → FALSE\n'
                '  missed_meetings: "missed last 2 QBRs" → 2\n'
                '  confidence: avg of fired = 0.9\n\n'
                'Output:\n'
                '  CsmSignal(budget_cut=True, exec_escalation=True,\n'
                '            migration_risk=True, missed_meetings=2,\n'
                '            confidence=0.9)',
                language="text",
            )

        st.markdown("#### Confidence scale")
        conf_data = pd.DataFrame([
            {"Score": "1.0", "Meaning": "Explicitly stated word-for-word", "Example": '"30% discount or we walk"'},
            {"Score": "0.8", "Meaning": "Strongly implied with clear context", "Example": '"CTO asked about migration paths"'},
            {"Score": "0.6", "Meaning": "Plausible inference from indirect language", "Example": '"they mentioned looking at options"'},
            {"Score": "< 0.6", "Meaning": "Weak signal — gets 50% weight in scoring", "Example": '"might be a bit unhappy"'},
        ])
        st.dataframe(conf_data, hide_index=True, width="stretch")
        st.caption("The model is instructed to apply the **lowest** confidence that fits — err conservative.")

    # Stage 4
    with st.expander("📊 Stage 4 — Scoring + Explanation: pure math, then plain English"):
        st.markdown(
            "**Part A (Scorer):** a pure function — same input always produces the same output. "
            "No LLM, no randomness.  \n"
            "**Part B (Explainer):** the LLM receives the pre-computed score and writes the briefing."
        )

        st.markdown("#### Scoring rules summary")
        rule1, rule2 = st.columns(2)
        with rule1:
            st.markdown(
                "<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;"
                "padding:14px;font-size:13px'>"
                "<b>P1 ticket cap</b><br>"
                "Max 3 P1s count toward score (max 6.0 pts).<br>"
                "<i>Without cap: 10 P1s = 20 pts, drowning all other signals.</i><br><br>"
                "<b>Missed meetings cap</b><br>"
                "Max 2 missed meetings count (max 2.0 pts).<br><br>"
                "<b>Competitor deduplication</b><br>"
                "If NPS already fired <code>competitor_mentioned</code>, CSM won't fire it again.<br>"
                "<i>Without dedup: same fact = +6.0 pts (double-counting).</i>"
                "</div>",
                unsafe_allow_html=True,
            )
        with rule2:
            st.markdown(
                "<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;"
                "padding:14px;font-size:13px'>"
                "<b>CSM confidence discount</b><br>"
                "<code>conf_mult = 1.0 if confidence ≥ 0.6 else 0.5</code><br><br>"
                "migration_risk, conf=0.9 → 3.0 × 1.0 = <b>3.0</b><br>"
                "migration_risk, conf=0.4 → 3.0 × 0.5 = <b>1.5</b><br><br>"
                "<b>NPS double-penalty</b><br>"
                "score ≤ 6 → +2.0 (detractor)<br>"
                "score ≤ 3 → +1.0 more (stacks: total +3.0)<br>"
                "<i>Score 4 = +2.0.  Score 2 = +3.0.</i>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### Example A — HIGH risk: BrightPath Solutions (ACC-042)")
        st.code(
            "Signal                          Weight   Running total\n"
            "──────────────────────────────  ──────   ─────────────\n"
            "3 P1 tickets min(3,3)×2.0       6.0      6.0   ← top signal\n"
            "Migration risk (conf=0.9)        3.0      9.0   ← crosses HIGH (7.0)\n"
            "Competitor in NPS verbatim       3.0      12.0\n"
            "Unresolved P1 ticket             2.5      14.5\n"
            "Deprecated SDK v3.1.0            2.5      17.0\n"
            "Budget cut mentioned             2.0      19.0\n"
            "Executive escalation (CTO)       2.0      21.0\n"
            "Low NPS score (4)                2.0      23.0\n"
            "2 missed QBRs                    2.0      25.0\n"
            "Escalated ticket                 1.5      26.5\n"
            "SDK deadline in 1 day            1.5      28.0\n"
            "Declining 3-month slope          1.5      29.5\n"
            "─────────────────────────────────────────────────\n"
            "Final: 29.5 → 🔴 HIGH   top_signal: '3 P1 tickets'",
            language="text",
        )

        st.markdown("#### Example B — MEDIUM risk: Pinnacle Media Group (ACC-017)")
        st.code(
            "Signal                          Weight   Running total\n"
            "──────────────────────────────  ──────   ─────────────\n"
            "Low NPS score (5)               2.0      2.0\n"
            "Budget cut (conf=0.7)           2.0      4.0   ← crosses MEDIUM (3.5)\n"
            "Declining 3-month slope         1.5      5.5\n"
            "1 missed meeting (conf=0.7)     1.0      6.5\n"
            "─────────────────────────────────────────────────\n"
            "Final: 6.5 → 🟡 MEDIUM   top_signal: 'Low NPS score'",
            language="text",
        )

        st.markdown("#### The explanation LLM call (Part B)")
        st.markdown("For HIGH and MEDIUM accounts only — LOW accounts get a static fallback:")
        st.code(
            "# What gets sent to the LLM:\n"
            "ACCOUNT CONTEXT:\n"
            "  Name: BrightPath Solutions | Plan: Starter | ARR: $21,000\n"
            "  Days to renewal: 47 | CSM: Sarah Chen\n\n"
            "RISK SCORE:\n"
            "  Tier: High (29.5) | Top signal: 3 P1 tickets\n"
            "  All signals: [3 P1 tickets, Migration risk, Competitor in NPS, ...]\n\n"
            "SIGNAL DETAIL:\n"
            "  Usage: 8 active users | MoM: -63.6% | slope: -11.5\n"
            "  SDK: DEPRECATED v3.1.0, patches stop 2026-04-30 (1 day)\n"
            "  NPS: 4 (detractor) | verbatim: 'evaluating Contentful as backup'\n"
            "  CSM: budget cut 20%, CTO leading CMS eval, missed 2 QBRs\n\n"
            "# What the LLM returns:\n"
            'explanation: "BrightPath\'s 3 P1 tickets are all linked to their v3.1.0\n'
            "  SDK, which loses security patches tomorrow — the same week their contract\n"
            '  renews. Their CTO is personally leading a CMS eval and they\'ve missed 2 QBRs."\n\n'
            'recommended_action: "Sarah Chen should escalate to VP of CS today and request\n'
            "  a joint engineering call with BrightPath's CTO this week — bring a concrete\n"
            '  v3→v4 migration timeline. Do not let this reach renewal without a written commitment."',
            language="text",
        )

        st.markdown("#### Low-risk skip — cost and latency optimisation")
        st.code(
            "if score.tier in (RiskTier.HIGH, RiskTier.MEDIUM):\n"
            "    report = generate_risk_report(account, score)   # LLM call\n"
            "else:\n"
            "    # Static fallback — no LLM call needed\n"
            "    report = RiskReport(\n"
            "        explanation='Low risk — no significant signals detected.',\n"
            "        recommended_action='Maintain regular check-in cadence.',\n"
            "    )\n"
            "# Saves ~2-3s and one API call per low-risk account.",
            language="python",
        )

    # ── End-to-end trace ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("End-to-End Trace: BrightPath Solutions")
    st.caption(
        "One account, all 4 stages — from raw CSV rows to the final briefing on the dashboard."
    )

    with st.expander("🔍 See the complete pipeline trace for ACC-042", expanded=False):
        t1, t2 = st.columns([1, 2])
        with t1:
            st.markdown(
                "<div style='background:#f0f9ff;border:2px solid #bae6fd;border-radius:10px;"
                "padding:16px;font-size:13px'>"
                "<b style='color:#0369a1;font-size:15px'>Account snapshot</b><br><br>"
                "🏢 <b>BrightPath Solutions</b><br>"
                "🆔 ACC-042<br>"
                "💰 ARR: $21,000<br>"
                "📅 Renewal: 47 days<br>"
                "👤 CSM: Sarah Chen<br>"
                "📦 Plan: Starter<br>"
                "🏭 Industry: Travel"
                "</div>",
                unsafe_allow_html=True,
            )
        with t2:
            st.markdown(
                "<div style='background:#faf5ff;border:2px solid #e9d5ff;border-radius:10px;"
                "padding:16px;font-size:13px'>"
                "<b style='color:#7c3aed;font-size:15px'>Final report (Stage 4 output)</b><br><br>"
                "🔴 <b>HIGH RISK</b> — Score: 29.5<br>"
                "📌 Top signal: 3 P1 tickets (6.0 pts)<br><br>"
                "<i>\"BrightPath's 3 P1 tickets are all linked to their v3.1.0 SDK, which loses "
                "security patches tomorrow — the same week their contract renews. Their CTO is "
                "personally leading a CMS evaluation, and they've missed 2 QBRs this quarter.\"</i><br><br>"
                "<b>Action:</b> Sarah Chen should escalate to VP of CS today and request a joint "
                "engineering call with BrightPath's CTO this week."
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### The pipeline baton — what each stage adds")
        st.code(
            "STAGE 1 OUTPUT:\n"
            "  accounts_df row: ACC-042, BrightPath Solutions, arr=21000,\n"
            "                   contract_end=2026-06-15, days_to_renewal=47\n\n"
            "STAGE 2 OUTPUT (4 signal objects):\n"
            "  UsageSignal:     users=8, mom=-63.6%, slope=-11.5, sdk='v3.1.0'\n"
            "  SupportSignal:   p1_count=3, unresolved_p1=True, escalated=1\n"
            "  NpsSignal:       score=4, category='detractor', competitor=True\n"
            "  ChangelogSignal: deprecated=True, days_to_deadline=1\n\n"
            "STAGE 3 OUTPUT (LLM-extracted):\n"
            "  CsmSignal:       budget_cut=True, exec_escalation=True,\n"
            "                   migration_risk=True, missed_meetings=2, confidence=0.9\n\n"
            "STAGE 4 OUTPUT (final):\n"
            "  RiskScore:  raw=29.5, tier=HIGH, top_signal='3 P1 tickets'\n"
            "  RiskReport: explanation + recommended_action (LLM-written)\n"
            "              contributing_signals: [3 P1 tickets, Migration risk,\n"
            "                                     Competitor in NPS, Unresolved P1, ...]",
            language="text",
        )

    # ── Rules vs LLM summary ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Rules vs LLM — Full Breakdown")

    r1, r2 = st.columns(2)
    with r1:
        st.markdown(
            "<div style='background:#f0f9ff;border:2px solid #bae6fd;border-radius:10px;padding:16px'>"
            "<b style='color:#0369a1;font-size:15px'>⚡ Deterministic Python (rules)</b>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("""
- MoM change calculation
- 3-month linear slope (`numpy.polyfit`)
- Usage drop flag (`< -30%`)
- Near-zero usage flag (`≤ 2`)
- P1 ticket counting + cap at 3
- Unresolved P1 detection
- Escalated ticket counting
- NPS category (detractor / passive / promoter)
- Competitor keyword scan in NPS verbatim
- SDK version regex check
- Days to deadline calculation
- Weighted sum scoring
- Tier assignment (`≥ 7.0 = HIGH`)
- Top signal selection (sort by weight)
- 90-day window filter
- Competitor deduplication across sources
""")
    with r2:
        st.markdown(
            "<div style='background:#faf5ff;border:2px solid #e9d5ff;border-radius:10px;padding:16px'>"
            "<b style='color:#7c3aed;font-size:15px'>🤖 LLM (3 calls total)</b>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("""
- **Call 1 (per account):** Parse CSM notes → structured `CsmSignal`
- **Call 2 (once at startup):** Extract changelog deprecations → `DeprecationEvent` list
- **Call 3 (per HIGH/MEDIUM account):** Score + signals → plain-English briefing + action

**The rule: LLMs parse language. Rules compute numbers. Never swap them.**

Swapping causes non-reproducible tiers, silent score drift on model updates,
and untraceable audit trails — the exact problems this pipeline was built to eliminate.
""")
        st.warning(
            "If the LLM is used to assign a tier directly: same account on Monday vs Friday "
            "can get different tiers. Can't explain a change. Can't audit the logic. "
            "Drifts silently with model updates.",
            icon="⚠️",
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SIGNAL REFERENCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📚 Signal Reference":
    st.title("📚 Signal Reference")
    st.caption("Every signal, its weight, when it fires, and which data source it comes from.")

    # ── Tier thresholds ────────────────────────────────────────────────────────
    st.subheader("Risk Tier Thresholds")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown(
            "<div style='background:#fef2f2;border:2px solid #fecaca;border-radius:10px;"
            "padding:16px;text-align:center'>"
            "<div style='font-size:28px'>🔴</div>"
            "<div style='font-size:20px;font-weight:700;color:#dc2626'>High Risk</div>"
            "<div style='font-size:24px;font-weight:600;margin:4px 0'>Score ≥ 7.0</div>"
            "<div style='color:#64748b;font-size:13px'>Immediate escalation required.<br>"
            "CSM action needed this week.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            "<div style='background:#fffbeb;border:2px solid #fde68a;border-radius:10px;"
            "padding:16px;text-align:center'>"
            "<div style='font-size:28px'>🟡</div>"
            "<div style='font-size:20px;font-weight:700;color:#d97706'>Medium Risk</div>"
            "<div style='font-size:24px;font-weight:600;margin:4px 0'>Score 3.5 – 7.0</div>"
            "<div style='color:#64748b;font-size:13px'>Proactive outreach needed.<br>"
            "Monitor closely over next 2 weeks.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with t3:
        st.markdown(
            "<div style='background:#f0fdf4;border:2px solid #bbf7d0;border-radius:10px;"
            "padding:16px;text-align:center'>"
            "<div style='font-size:28px'>🟢</div>"
            "<div style='font-size:20px;font-weight:700;color:#16a34a'>Low Risk</div>"
            "<div style='font-size:24px;font-weight:600;margin:4px 0'>Score < 3.5</div>"
            "<div style='color:#64748b;font-size:13px'>Maintain regular cadence.<br>"
            "Look for expansion opportunities.</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Signal weights table ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Signal Weights")
    st.caption(
        "Each signal adds its weight to the account's raw score when it fires. "
        "CSM signals below 0.6 confidence are weighted at 50%."
    )

    SIGNAL_DETAILS = {
        "renewal_threatened":  ("🔴 CSM Notes",  "Explicit churn threat — 'we'll leave', discount ultimatum, or non-renewal stated. LLM-extracted from CSM notes."),
        "migration_risk":      ("🔴 CSM Notes",  "Account actively evaluating competitors or building a homegrown solution. LLM-extracted."),
        "competitor_mentioned":("🔴 CSM / NPS",  "A named competitor (Contentful, Strapi, Sanity, etc.) appears in NPS verbatim or CSM notes."),
        "near_zero_usage":     ("📈 Usage",       "Latest month has ≤ 2 active users — the account has effectively stopped using the product."),
        "usage_drop_flag":     ("📈 Usage",       "Month-over-month active user decline exceeds 30%."),
        "deprecated_sdk":      ("📋 Changelog",   "Account is on SDK v3.x which loses security patches on April 30 2026."),
        "unresolved_p1":       ("🎫 Support",     "At least one P1 (critical) ticket is currently open or escalated."),
        "p1_tickets":          ("🎫 Support",     "Per P1 ticket (capped at 3). Each P1 adds this weight up to a maximum of 6.0 total."),
        "exec_escalation":     ("🔴 CSM Notes",  "A C-suite or VP-level executive (CTO, CRO, CISO, CMO) appeared on a call. LLM-extracted."),
        "budget_cut":          ("🔴 CSM Notes",  "Explicit budget reduction or SaaS vendor consolidation mentioned. LLM-extracted."),
        "low_nps":             ("⭐ NPS",         "NPS score is ≤ 6 (detractor range)."),
        "escalated_tickets":   ("🎫 Support",     "At least one ticket has been escalated to engineering or management."),
        "negative_mom_trend":  ("📈 Usage",       "Linear regression over the last 3 months shows a downward slope greater than 0.5 users/month."),
        "sdk_deadline_30d":    ("📋 Changelog",   "Additional weight if the SDK deprecation deadline is within 30 days of today."),
        "detractor_nps":       ("⭐ NPS",         "Additional weight for NPS score ≤ 3 (strong detractor, not just dissatisfied)."),
        "missed_meetings":     ("🔴 CSM Notes",  "Per missed QBR or no-show (capped at 2). LLM-extracted."),
    }

    signal_rows = []
    for key, weight in sorted(SIGNAL_WEIGHTS.items(), key=lambda x: -x[1]):
        source, description = SIGNAL_DETAILS.get(key, ("—", key))
        signal_rows.append({
            "Signal": key.replace("_", " ").title(),
            "Weight": weight,
            "Source": source,
            "Fires When": description,
        })

    signal_df = pd.DataFrame(signal_rows)
    st.dataframe(
        signal_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Weight": st.column_config.NumberColumn(
                "Weight",
                format="%.1f",
                help="Points added to the raw risk score when this signal fires.",
            ),
            "Source": st.column_config.TextColumn(
                "Source",
                help="Which data file drives this signal.",
                width=120,
            ),
            "Fires When": st.column_config.TextColumn(
                "Fires When",
                help="The exact condition that triggers this signal.",
                width=450,
            ),
        },
    )

    # ── Confidence discount ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("CSM Signal Confidence Discount")
    st.markdown("""
LLM-extracted CSM signals include a confidence score (0.0 – 1.0).
Signals extracted with low confidence are penalised to prevent uncertain inferences from dominating the score.

| Confidence | Weight Applied | Example |
|------------|---------------|---------|
| ≥ 0.6 | Full weight | "30% discount or we walk" — explicitly stated |
| < 0.6 | 50% weight | "They might be looking around" — weakly implied |

The calibration scale:
- **1.0** → explicitly stated word-for-word
- **0.8** → strongly implied with clear context
- **0.6** → plausible inference from indirect language
- **0.4** → reading between the lines
    """)

    # ── Scoring example ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Scoring Walk-Through Example")
    st.markdown("**NovaTech Industries** — showing exactly how a High score is built:")
    st.code(
        "Signal fired                       Weight   Running score\n"
        "─────────────────────────────────  ──────   ─────────────\n"
        "Renewal threatened (conf=0.9)      4.0      4.0\n"
        "Migration risk (conf=0.9)          3.0      7.0   ← crosses High threshold\n"
        "Competitor mentioned (NPS)         3.0      10.0\n"
        "4 P1 tickets → min(4,3)×2.0       6.0      16.0\n"
        "Deprecated SDK v3.2.0             2.5      18.5\n"
        "Unresolved P1                      2.5      21.0\n"
        "Exec escalation (CTO on call)      2.0      23.0\n"
        "Declining 3-month usage slope      1.5      24.5\n"
        "Escalated tickets                  1.5      26.0\n"
        "─────────────────────────────────────────────────\n"
        "Final score: 26.0 → 🔴 HIGH RISK",
        language="text",
    )
