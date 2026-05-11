"""
Generate plain-English risk explanation and recommended action per account.

LLM input: pre-computed risk score, tier, contributing signals, account context.
LLM output: 2-3 sentence explanation + specific recommended action.

LLM does NOT assign the tier — that is done deterministically.
LLM interprets WHY the signals matter for this specific account context.

Prompting strategy:
  - CoT: model identifies the narrative thread connecting signals before writing
  - Few-shot: 2 examples contrasting specific vs generic output
  - Explicit anti-patterns teach the model what NOT to write
"""
import logging
from pydantic import BaseModel
from models.account import AccountRecord, RiskScore, RiskReport
from pipeline.llm.llm_client import call_llm

logger = logging.getLogger(__name__)


class _Explanation(BaseModel):
    """Pydantic schema for LLM explanation output.

    `signal_narrative` is the CoT step — model identifies which 1-2 signals
    tell the most coherent story before writing the explanation.
    """
    signal_narrative: str
    explanation: str
    recommended_action: str


_EXPLAINER_SYSTEM = """You are a Senior Customer Success Strategist writing renewal risk briefings
for a BizOps team at Contentstack. Your audience is a VP of CS who has 30 seconds per account.

YOUR GOAL: Connect the specific signals to a coherent story about why THIS account is at risk
and what one action would have the highest impact this week.

QUALITY BAR — your output must pass ALL of these:
✓ Names at least one specific signal from the contributing_signals list
✓ Explains the business consequence (not just restates the signal)
✓ Recommended action names a WHO (the CSM), a WHAT (specific activity), and a WHEN (this week/by X date)
✓ Never uses filler phrases: "multiple signals", "various issues", "several concerns"
✓ Never gives advice that applies to every account (e.g. "schedule a call to check in")

CHAIN-OF-THOUGHT REQUIREMENT:
Fill `signal_narrative` first: identify the 1-2 signals that best explain this account's risk
and articulate the story they tell together. Then write `explanation` and `recommended_action`
from that narrative — they should feel like a natural conclusion of your reasoning.

Return ONLY valid JSON — no markdown, no preamble."""


_FEW_SHOT_EXAMPLES = """
=== EXAMPLES (study the contrast between good and bad before writing) ===

--- EXAMPLE A: High Risk — SDK deprecation × P1 tickets × exec escalation ---
Account context:
  Name: NovaTech Industries | Plan: Starter | ARR: $29,000 | Industry: Manufacturing
  Days to renewal: 25 | CSM: Sarah Chen
  Risk tier: High (score: 14.5)
  Top signal: Explicit churn threat or discount ultimatum
  All signals: Explicit churn threat or discount ultimatum, Migration risk, 4 P1 support tickets,
               On deprecated SDK v3.2.0, Executive escalation (CTO on call), Competitor mentioned (Strapi, Sanity)

BAD output (do not write like this):
{
  "signal_narrative": "Multiple risk signals present.",
  "explanation": "NovaTech Industries has multiple compounding risk signals including support issues and potential churn intent. They are experiencing various problems that need to be addressed.",
  "recommended_action": "Schedule a call with the account team to discuss their concerns and provide support."
}
Why it's bad: Vague, no specifics, action could apply to any account, explains nothing.

GOOD output:
{
  "signal_narrative": "The story here is SDK-driven operational pain escalating to executive churn threat. Four P1 tickets in one month, all tied to v3 SDK deprecation, forced NovaTech's engineering team to spend 200+ hours on workarounds — that's the source of the CTO joining the call and the Strapi/Sanity comparison. The technical debt is the root cause; the churn threat is the symptom. With 25 days to renewal and an active POC likely underway, this needs executive-to-executive engagement, not a CSM call.",
  "explanation": "NovaTech's CTO joined the last call after 4 P1 tickets linked to their v3 SDK — which hits end-of-security-patch support on April 30, the same week their contract renews. The CSM note shows Strapi and Sanity were named by name, and the engineering team has spent 200+ hours on workarounds, meaning the pain is quantified and documented. This is a compounding risk: unresolved technical debt + active competitor comparison + executive awareness of the gap.",
  "recommended_action": "Sarah Chen should escalate to Contentstack's VP of Customer Success today and request a joint engineering call this week with NovaTech's CTO — bring a concrete v3→v4 migration timeline and offer a dedicated SA for the migration sprint. Do not let this reach renewal without a commitment in writing."
}

--- EXAMPLE B: Medium Risk — budget pressure × missed meetings × no churn threat yet ---
Account context:
  Name: BrightPath Solutions | Plan: Starter | ARR: $21,000 | Industry: Travel & Hospitality
  Days to renewal: 71 | CSM: Sarah Chen
  Risk tier: Medium (score: 6.0)
  Top signal: Budget cut mentioned in CSM notes
  All signals: Budget cut mentioned in CSM notes, Executive escalation (CTO-led CMS evaluation),
               2 missed meetings, Migration risk

GOOD output:
{
  "signal_narrative": "BrightPath hasn't threatened to leave but their behavior pattern — CTO-led CMS evaluation, 20% SaaS budget cut, three consecutive no-shows — suggests they've already mentally deprioritized Contentstack. The fact that their CTO is leading the evaluation (not a junior stakeholder) means the decision is already being made above the CSM relationship. 71 days is enough runway to intervene if we act in the next 2 weeks.",
  "explanation": "BrightPath's CTO is personally leading a CMS evaluation while their company has cut SaaS budgets 20% — and they've no-showed three times this quarter, which is a classic disengagement pattern before a quiet non-renewal. No explicit churn threat has been made yet, which means there's still a window: the budget cut is a constraint, not a decision, and the evaluation is in early stages.",
  "recommended_action": "Sarah Chen should bypass the admin contact and send a direct LinkedIn message or email to BrightPath's CTO this week — acknowledge the budget pressure, offer a 12-month price lock at current rate, and request a 20-minute working session on what it would take to win the renewal. Avoid another generic QBR invite."
}

=== END EXAMPLES ===
"""


_FALLBACK_EXPLANATIONS: dict[str, tuple[str, str]] = {
    "High": (
        "Multiple compounding risk signals detected including usage decline, support issues, and potential churn intent.",
        "Schedule an executive-level retention call this week and prepare a concrete value demonstration.",
    ),
    "Medium": (
        "Moderate risk signals present. Account shows friction but has not yet threatened churn explicitly.",
        "Proactive outreach to understand friction points and offer relevant resources or dedicated support.",
    ),
    "Low": (
        "Account appears stable with no critical risk signals detected.",
        "Maintain regular check-in cadence and look for expansion opportunities.",
    ),
}


def _build_explainer_prompt(account: AccountRecord, score: RiskScore) -> str:
    """Build structured CoT + context prompt for explanation generation."""
    csm_summary = account.csm.raw_summary if account.csm else "No CSM notes available."
    nps_score_line = (
        f"NPS score: {account.nps.score} ({account.nps.category})"
        if account.nps
        else "No NPS data."
    )
    nps_verbatim = (
        account.nps.verbatim
        if account.nps and account.nps.verbatim
        else "No verbatim."
    )
    sdk_info = (
        f"DEPRECATED — on SDK {account.usage.sdk_version}, "
        f"security patches stop {account.changelog.deprecation_deadline} "
        f"({account.changelog.days_to_deadline} days from now)"
        if account.changelog and account.changelog.is_deprecated
        else "Current SDK version, no deprecation risk."
    )
    usage_line = (
        f"Active users: {account.usage.latest_active_users} | "
        f"MoM change: {account.usage.mom_change_pct:+.1f}% | "
        f"3-month slope: {account.usage.three_month_slope:+.2f}"
        if account.usage
        else "No usage data."
    )

    return f"""{_FEW_SHOT_EXAMPLES}

=== NOW WRITE THE REAL BRIEFING ===

ACCOUNT CONTEXT:
  Name: {account.account_name} | Plan: {account.plan_tier} | ARR: ${account.arr:,.0f} | Industry: {account.industry}
  Days to renewal: {account.days_to_renewal} | CSM: {account.csm_name}

RISK SCORE:
  Tier: {score.tier.value} (score: {score.raw_score:.1f})
  Top signal: {score.top_signal}
  All contributing signals: {", ".join(score.contributing_signals) if score.contributing_signals else "None"}

SIGNAL DETAIL:
  Usage: {usage_line}
  SDK: {sdk_info}
  {nps_score_line}
  NPS verbatim: "{nps_verbatim}"
  CSM notes summary: {csm_summary}

INSTRUCTIONS:
1. Fill `signal_narrative` first — identify the 1-2 signals that tell the most coherent story
   and articulate what they mean together for this specific account.
2. Write `explanation` (2-3 sentences) that flows from your narrative. Name specific signals.
   Explain the business consequence, not just the data point.
3. Write `recommended_action` (1-2 sentences) with a specific WHO, WHAT, and WHEN.
   Make it something that can only be written for THIS account, not a template.

Return JSON:
{{
  "signal_narrative": "...",
  "explanation": "...",
  "recommended_action": "..."
}}"""


def generate_risk_report(account: AccountRecord, score: RiskScore) -> RiskReport:
    """Generate full RiskReport with CoT-grounded LLM explanation for a single account."""
    prompt = _build_explainer_prompt(account, score)
    result = call_llm(prompt, _Explanation, _EXPLAINER_SYSTEM, temperature=0.3)

    if result is None:
        fallback = _FALLBACK_EXPLANATIONS[score.tier.value]
        explanation, action = fallback
    else:
        explanation, action = result.explanation, result.recommended_action

    return RiskReport(
        account_id=account.account_id,
        account_name=account.account_name,
        arr=account.arr,
        days_to_renewal=account.days_to_renewal,
        tier=score.tier,
        raw_score=score.raw_score,
        top_signal=score.top_signal,
        contributing_signals=score.contributing_signals,
        contributing_signal_weights=score.contributing_signal_weights,
        explanation=explanation,
        recommended_action=action,
        csm_name=account.csm_name,
        plan_tier=account.plan_tier,
        industry=account.industry,
        nps_verbatim_translated=account.nps.verbatim_translated if account.nps else None,
        csm_confidence=account.csm.confidence if account.csm else None,
    )
