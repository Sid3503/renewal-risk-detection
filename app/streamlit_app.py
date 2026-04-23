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

# ── Pipeline runner with session-state cache ──────────────────────────────────

def _run_with_progress() -> list[RiskReport]:
    """Run pipeline, streaming stage updates into the UI as they happen."""
    progress_bar = st.progress(0.0)
    status_box   = st.empty()
    log_box      = st.empty()
    log_lines: list[str] = []

    def _stage(msg: str, pct: float) -> None:
        progress_bar.progress(pct)
        status_box.markdown(f"**{msg}**")
        log_lines.append(msg)
        log_box.markdown(
            "<div style='background:#f8fafc;border:1px solid #e2e8f0;"
            "border-radius:8px;padding:10px 14px;font-size:13px;"
            "font-family:monospace;max-height:180px;overflow-y:auto'>"
            + "".join(
                f"<div class='stage-row'>"
                f"<div class='stage-dot {'done' if i < len(log_lines)-1 else 'active'}'></div>"
                f"{line}</div>"
                for i, line in enumerate(log_lines)
            )
            + "<div class='flow-bar'></div>"
            + "</div>",
            unsafe_allow_html=True,
        )

    reports = run_pipeline(stage_callback=_stage)
    progress_bar.progress(1.0)
    status_box.success("✅ Pipeline complete — results cached for this session.")
    log_box.empty()
    return reports


def get_reports() -> list[RiskReport]:
    """Return cached reports or run pipeline on first call."""
    if "reports" not in st.session_state:
        _show_loading_splash()
        st.session_state["reports"] = _run_with_progress()
    return st.session_state["reports"]


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
      <span>🗂️ 5 data sources</span>
      <span>📋 120 accounts</span>
      <span>🔍 30 in renewal window</span>
      <span>✨ 2 LLM passes</span>
    </div>
    """, unsafe_allow_html=True)
    st.info(
        "💡 **How this works:** Deterministic signal scoring (no LLM for risk scores) "
        "+ an LLM for CSM note parsing and plain-English briefings. "
        "First run: ~60–90 s. Subsequent interactions: instant (session-cached).",
        icon=None,
    )
    st.markdown("---")
    st.markdown("**Pipeline progress**")


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
        st.session_state.pop("reports", None)
        st.rerun()
    st.caption("Scores are deterministic weighted sums.\nLLM writes explanations only.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Dashboard":
    st.title("⚠️ Renewal Risk Intelligence")
    st.caption("Contentstack BizOps · Accounts renewing in the next 90 days · LLM-powered risk briefings")

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
            use_container_width=True,
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
                "**Contributing signals**",
                help="All signals that added to this account's risk score, sorted by weight (highest first). "
                     "See Signal Reference page for weight values.",
            )
            for i, signal in enumerate(report.contributing_signals):
                weight_indicator = "🔴" if i == 0 else ("🟡" if i < 3 else "⚪")
                st.markdown(f"{weight_indicator} {signal}")

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
            use_container_width=True,
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
        use_container_width=True,
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
