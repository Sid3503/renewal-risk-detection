"""
Extract structured signals from raw csm_notes.txt using an LLM.

Notes are messy by design:
  - Mixed date formats (Mar 12, 3/15, 2026-03-20, march 25)
  - Typos in account names (BritePath, Pinacle, Thunderbolt Moters)
  - One note in Mandarin
  - Account IDs present for some entries, absent for others

Prompting strategy:
  - CoT: model reasons through each signal before committing to a boolean
  - Few-shot: 3 annotated examples anchor what "good" extraction looks like
  - Confidence calibration examples prevent over/under-confidence
"""
import re
import logging
import pandas as pd
from pydantic import BaseModel, Field
from models.signals import CsmSignal
from pipeline.llm.llm_client import call_llm

logger = logging.getLogger(__name__)


class _CsmExtraction(BaseModel):
    """Pydantic schema for LLM CSM note extraction.

    `reasoning` is a CoT scratchpad — the model thinks through each signal
    before committing to booleans, which measurably improves accuracy on
    ambiguous notes.
    """
    reasoning: str
    account_name_found: str
    competitor_mentioned: bool
    budget_cut_mentioned: bool
    exec_escalation: bool
    migration_risk: bool
    renewal_threatened: bool
    missed_meetings: int = Field(ge=0)
    positive_signal: bool
    raw_summary: str
    confidence: float = Field(ge=0.0, le=1.0)


_CSM_SYSTEM = """You are a senior Customer Success analyst at a B2B SaaS company.
Your job is to read raw CSM call notes and extract risk signals with surgical precision.

SIGNAL DEFINITIONS — apply these exactly:
- competitor_mentioned: A named competing platform appears (Contentful, Strapi, Sanity, Hygraph,
  Kontent.ai, Builder.io, WordPress, Drupal, Prismic). Vague phrases like "other options" do NOT count.
- budget_cut_mentioned: Explicit mention of budget reduction, SaaS spend cuts, vendor consolidation,
  or finance team involvement in renewal. "Tight budget" alone is NOT enough.
- exec_escalation: A C-suite or VP-level person (CTO, VP Eng, VP Product, CMO, CRO, CISO, CFO)
  was on the call or sent a direct communication. A CSM flagging internally does NOT count.
- migration_risk: The account is actively evaluating alternatives, building an in-house solution,
  or has started a competitor POC. "Mentioned they might look around" is weak — use low confidence.
- renewal_threatened: An explicit churn signal — "we'll leave", "explore other options", discount
  ultimatum, or contract non-renewal stated or strongly implied. Unhappiness alone is NOT enough.
- missed_meetings: Count of no-shows, cancelled QBRs, or unanswered outreach in the note.
- positive_signal: Clear expansion interest, enthusiastic renewal, budget approval, or deep
  product advocacy. Neutral satisfaction does NOT count.

CONFIDENCE CALIBRATION:
- 1.0 → explicitly stated word-for-word ("We are evaluating Contentful")
- 0.8 → strongly implied with clear context ("their CTO asked about migration paths")
- 0.6 → plausible inference from indirect language
- 0.4 → weak signal, reading between the lines
- Apply the LOWEST confidence that fits — err conservative.

CHAIN-OF-THOUGHT REQUIREMENT:
Always fill `reasoning` first. Walk through each signal: what phrase in the note supports
or refutes it. Only then fill the boolean fields. This thinking improves accuracy.

If the note is in a non-English language: translate internally, then extract normally.

Return ONLY valid JSON — no markdown, no preamble."""


# ---------------------------------------------------------------------------
# Few-shot examples embedded directly in the user prompt
# These anchor the model's understanding of the expected output quality
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES = """
=== EXAMPLES (study these carefully before extracting the real note) ===

--- EXAMPLE 1: High-risk note with multiple explicit signals ---
Account: Zenith Publishing
Note:
"4/5 - Renewl conversation started. They want a 30% discount or they walk. Current ARR is
high but margin is already thin. Editor performance on their 50k+ entry library is painful
and they know it. Competitor POC with Kontent.ai is apparently already underway.
CRO was cc'd on the last email thread which is never a good sign."

Expected output:
{
  "reasoning": "competitor_mentioned: 'Kontent.ai' named explicitly — TRUE. budget_cut_mentioned: No budget cut language, they want a discount but that's a negotiation tactic not a budget cut — FALSE. exec_escalation: CRO cc'd on email thread — C-suite involvement — TRUE. migration_risk: 'competitor POC with Kontent.ai is apparently already underway' — active POC in progress — TRUE. renewal_threatened: '30% discount or they walk' — explicit churn ultimatum — TRUE. missed_meetings: None mentioned — 0. positive_signal: No expansion interest, purely defensive — FALSE. confidence: All signals are explicitly stated or strongly implied — 0.9.",
  "account_name_found": "Zenith Publishing",
  "competitor_mentioned": true,
  "budget_cut_mentioned": false,
  "exec_escalation": true,
  "migration_risk": true,
  "renewal_threatened": true,
  "missed_meetings": 0,
  "positive_signal": false,
  "raw_summary": "Zenith Publishing has issued a 30% discount ultimatum with an active Kontent.ai POC underway and CRO involvement signaling executive awareness. This is the highest-severity churn signal pattern: a concurrent competitor evaluation combined with a hard commercial demand.",
  "confidence": 0.9
}

--- EXAMPLE 2: Subtle silent-churn pattern — tricky confidence ---
Account: Meridian Health
Note:
"march 25 -- meridian health -- priya
Good news/bad news. NPS came back at 8 (decent) but their actual usage has cratered.
Turns out they built a custom middleware layer and are slowly moving content to a
homegrown solution. The score reflects the *people* like our support team, not that
they plan to stay. Classic silent churn pattern. VP of Digital was on the last call
and asked about data export APIs."

Expected output:
{
  "reasoning": "competitor_mentioned: No named competitor — they're building internal, not switching vendors — FALSE. budget_cut_mentioned: No budget language — FALSE. exec_escalation: 'VP of Digital was on the last call' — VP-level exec present — TRUE. migration_risk: 'built a custom middleware layer and are slowly moving content to a homegrown solution' — active migration in progress — TRUE, high confidence. renewal_threatened: CSM says 'classic silent churn pattern' and they're actively migrating — strongly implied but not explicitly stated — TRUE at 0.7 confidence. missed_meetings: None mentioned — 0. positive_signal: NPS 8 noted but CSM explicitly says it doesn't reflect intent to stay — FALSE. confidence: Migration risk and exec escalation are explicit; churn threat is inferred from the migration pattern — 0.75.",
  "account_name_found": "meridian health",
  "competitor_mentioned": false,
  "budget_cut_mentioned": false,
  "exec_escalation": true,
  "migration_risk": true,
  "renewal_threatened": true,
  "missed_meetings": 0,
  "positive_signal": false,
  "raw_summary": "Meridian Health is executing a silent migration to a homegrown solution despite an NPS of 8 — the CSM explicitly identifies this as a silent churn pattern. VP of Digital's interest in data export APIs suggests they are already planning the offboarding.",
  "confidence": 0.75
}

--- EXAMPLE 3: Expansion signal, low risk ---
Account: Ironclad Security
Note:
"3/20 ironclad security — just closed a great QBR. They want to add 30 more seats and
upgrade to Enterprise. Already got budget approval. renewal + expansion coming in May.
champagne time."

Expected output:
{
  "reasoning": "competitor_mentioned: No competitor named — FALSE. budget_cut_mentioned: Opposite — budget was approved for expansion — FALSE. exec_escalation: No exec mentioned on the call, just a QBR — FALSE. migration_risk: No indication of evaluation — FALSE. renewal_threatened: Explicit expansion intent, opposite of churn — FALSE. missed_meetings: No missed meetings — 0. positive_signal: 'want to add 30 more seats', 'upgrade to Enterprise', 'budget approval' — strong expansion signals — TRUE. confidence: All signals are explicitly stated — 1.0.",
  "account_name_found": "ironclad security",
  "competitor_mentioned": false,
  "budget_cut_mentioned": false,
  "exec_escalation": false,
  "migration_risk": false,
  "renewal_threatened": false,
  "missed_meetings": 0,
  "positive_signal": true,
  "raw_summary": "Ironclad Security is a strong expansion candidate with budget-approved seat addition and Enterprise upgrade in May. No risk signals present.",
  "confidence": 1.0
}

=== END EXAMPLES ===
"""


def _split_notes_into_chunks(raw_notes: str) -> list[str]:
    """Split csm_notes.txt into individual note chunks by '---' separator."""
    chunks = re.split(r"\n---+\n", raw_notes)
    return [c.strip() for c in chunks if c.strip()]


def _find_best_chunk(account_name: str, account_id: str, chunks: list[str]) -> str | None:
    """Find the note chunk most likely to be about this account."""
    name_lower = account_name.lower()
    first_word = name_lower.split()[0] if name_lower else ""

    for chunk in chunks:
        chunk_lower = chunk.lower()
        if (
            f"acct {account_id}" in chunk_lower
            or f"account {account_id}" in chunk_lower
            or f"#{account_id}" in chunk_lower
        ):
            return chunk

    for chunk in chunks:
        if name_lower in chunk.lower():
            return chunk

    if len(first_word) > 4:
        for chunk in chunks:
            if first_word in chunk.lower():
                return chunk

    return None


def extract_csm_signals(
    raw_notes: str,
    accounts_df: pd.DataFrame,
) -> dict[str, CsmSignal]:
    """Extract CsmSignal for every account that has a matching note chunk."""
    chunks = _split_notes_into_chunks(raw_notes)
    signals: dict[str, CsmSignal] = {}

    for _, row in accounts_df.iterrows():
        account_id = str(row["account_id"])
        account_name = str(row["account_name"])
        chunk = _find_best_chunk(account_name, account_id, chunks)

        if not chunk:
            logger.debug("No CSM note chunk found for %s", account_name)
            continue

        prompt = f"""{_FEW_SHOT_EXAMPLES}

=== NOW EXTRACT THE REAL NOTE ===

Account we are analyzing: {account_name} (ID: {account_id})

Note:
{chunk}

Step 1 — Think through each signal in the `reasoning` field: cite the exact phrase that
supports or refutes it. Be skeptical. Weak language → lower confidence.
Step 2 — Fill the boolean fields based solely on your reasoning.
Step 3 — Write `raw_summary` as 2 sentences: first what happened, second what it means for renewal.
Step 4 — Set `confidence` as the average certainty across all signals you fired as true.

Return JSON matching exactly this schema:
{{
  "reasoning": "...",
  "account_name_found": "...",
  "competitor_mentioned": true/false,
  "budget_cut_mentioned": true/false,
  "exec_escalation": true/false,
  "migration_risk": true/false,
  "renewal_threatened": true/false,
  "missed_meetings": 0,
  "positive_signal": true/false,
  "raw_summary": "...",
  "confidence": 0.0
}}"""

        result = call_llm(prompt, _CsmExtraction, _CSM_SYSTEM)
        if result is None:
            logger.warning("CSM extraction failed for account %s", account_name)
            continue

        signals[account_id] = CsmSignal(
            account_id=account_id,
            account_name_in_notes=result.account_name_found,
            competitor_mentioned=result.competitor_mentioned,
            budget_cut_mentioned=result.budget_cut_mentioned,
            exec_escalation=result.exec_escalation,
            migration_risk=result.migration_risk,
            renewal_threatened=result.renewal_threatened,
            missed_meetings=result.missed_meetings,
            positive_signal=result.positive_signal,
            raw_summary=result.raw_summary,
            confidence=result.confidence,
        )
        logger.info(
            "CSM extracted for %s (confidence=%.2f)", account_name, result.confidence
        )

    return signals
