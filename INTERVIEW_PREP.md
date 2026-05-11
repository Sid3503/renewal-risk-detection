# Interview Prep: Renewal Risk Intelligence Engine

> Your cheat sheet for walking through this project in an interview. Covers what it is, how it works, where LLMs are used vs rules, the non-obvious insight, and answers to every likely question — with examples you can actually say out loud.

---

## The 30-Second Pitch

> Say this when they ask "walk me through your project."

"I built a renewal risk intelligence pipeline for a BizOps team. It ingests five messy data sources — usage metrics, support tickets, NPS responses, CSM call notes, and a product changelog — and automatically identifies which accounts are likely to churn at renewal and why.

The scoring is fully deterministic: a weighted sum of risk signals that always produces the same result for the same input. The LLM is used in two specific places — parsing unstructured CSM notes into structured signals, and writing plain-English briefings with specific recommended actions. Everything else is pure Python logic.

The output is a Streamlit dashboard where the BizOps team can see every account in the 90-day renewal window ranked by risk, drill into any account, and get a briefing that tells them exactly what to do and who should do it."

---

## The Problem It Solves

### Before this system existed:
- A CSM manually reads 120 accounts' notes, tickets, and usage data every quarter
- They miss the account that looks healthy on NPS but is quietly migrating to a homegrown solution
- They have no idea 8 accounts are on a deprecated SDK losing security patches in 30 days — because that lives in the engineering changelog, not Salesforce
- By the time they flag a risk, there are 2 weeks left before renewal

### After:
- All five data sources are ingested and cross-referenced automatically
- Every account in the 90-day window gets a risk score with a breakdown of exactly which signals fired
- The LLM writes a specific briefing: not "schedule a call" but "Sarah Chen should escalate to VP of CS today and bring a v3→v4 migration timeline"
- Runs in ~60–90 seconds

### Business impact:
- Turns a manual, quarterly scramble into an automated, always-on signal
- Surfaces invisible technical risk (SDK deprecation) that no CRM tool would catch
- Gives CSMs a specific action, not just a flag — reduces time-to-intervention
- Prioritizes by ARR at risk, so the team works the highest-value accounts first

---

## The Pipeline: 4 Stages

```
RAW DATA (5 files)
      ↓
STAGE 1: Ingestion — load, validate, fuzzy-match account names
      ↓
STAGE 2: Deterministic signals — usage, support, NPS, changelog (pure Python)
      ↓
STAGE 3: LLM CSM extraction — unstructured notes → structured signals
      ↓
STAGE 4: Score (deterministic) + Explain (LLM) → RiskReport per account
      ↓
STREAMLIT DASHBOARD
```

Each stage produces typed Pydantic objects. No raw dicts passed between stages — if a signal is missing, it's `None`, not a KeyError.

---

## The 5 Data Sources and What They Contribute

| Source | What it is | What the system extracts |
|--------|-----------|--------------------------|
| `accounts.csv` | 120 accounts with ARR, contract dates, CSM name | The spine — every account flows through this |
| `usage_metrics.csv` | 6 months of monthly active users + SDK version | Usage trend, MoM drop, 3-month slope, SDK version |
| `support_tickets.csv` | Support ticket history with priority + status | P1 count, unresolved P1s, escalations |
| `nps_responses.csv` | NPS scores + verbatim comments | Score category, competitor mentions in text |
| `csm_notes.txt` | Raw, messy, unstructured call notes | Competitor mentions, budget cuts, exec escalation, churn threats (via LLM) |
| `changelog.md` | Product changelog with deprecation notices | Deprecated SDK versions + deadlines (via LLM, then cross-referenced) |

---

## Where Rules Are Used vs Where LLM Is Used

This is the most important design question you'll get. Know it cold.

### Rules (deterministic Python) — used for everything that can be computed

| What | Where | Why rules, not LLM |
|------|-------|-------------------|
| Usage MoM change | `usage_signals.py` | It's math. `(users[-1] - users[-2]) / users[-2] * 100` |
| 3-month slope | `usage_signals.py` | Linear regression with numpy — always reproducible |
| P1 ticket count | `support_signals.py` | Count rows where priority == "P1" |
| NPS category | `nps_signals.py` | Score 0–6 = detractor, 7–8 = passive, 9–10 = promoter |
| Competitor in NPS verbatim | `nps_signals.py` | Keyword list check — fast, auditable |
| SDK deprecation check | `changelog_signals.py` | Regex match on version string against known deprecated list |
| Risk score | `risk_scorer.py` | Weighted sum — pure function, same input = same output |
| Tier assignment | `risk_scorer.py` | Score ≥ 7.0 = High, ≥ 3.5 = Medium, else Low |

### LLM (Gemini 2.0 Flash) — used only where rules can't work

| What | Where | Why LLM, not rules |
|------|-------|-------------------|
| Parse CSM notes | `csm_extractor.py` | Notes are free-text, multilingual, full of typos and ambiguity. Rules can't reliably extract "exec escalation" from "their CTO jumped on the call last minute" |
| Extract changelog deprecations | `changelog_extractor.py` | Changelog is prose. Parsing deprecation deadlines from natural language text is an NLP problem |
| Write risk briefings | `risk_explainer.py` | The score is deterministic. The *explanation* of why it matters for this specific account, in plain English, with a specific action — that's where LLM nuance adds value |

### The key principle:
> **LLMs parse language. Rules compute numbers. Never swap them.**

If you let the LLM assign the risk score, the same account can get a different tier on Monday vs Friday. You can't track score changes over time. You can't audit why a tier changed. The scoring is a pure function — same input, same output, always.

---

## Dry Run: One Account End to End

Let's trace **BrightPath Solutions** through the entire pipeline.

### Input data:
```
accounts.csv:
  account_id: ACC-042, account_name: BrightPath Solutions
  arr: $21,000, contract_end_date: 2026-06-15, csm: Sarah Chen

usage_metrics.csv (last 6 months):
  active_users: [28, 25, 22, 18, 14, 10]
  sdk_version: v3.1.0

csm_notes.txt (raw text):
  "3/18 - britepath - budget cut 20%, CTO leading CMS eval, missed last 2 QBRs"
  ← Note the typo: "britepath" not "BrightPath"

nps_responses.csv:
  score: 4
  verbatim: "We've been evaluating Contentful as a backup option"
```

### Stage 1: Ingestion
```
loader.py reads all CSVs, parses dates, strips whitespace from account_id

reconciler.py:
  Note says "britepath" → normalize → "britepath"
  Fuzzy match against "brightpath solutions" → score: 87 > threshold 75
  → matched to ACC-042 ✓
```

### Stage 2: Deterministic signals
```
usage_signals.py:
  MoM change: (10 - 14) / 14 × 100 = -28.6%  → usage_drop_flag = False (threshold is -30%)
  3-month slope: polyfit([18, 14, 10]) = -4.0  → negative_mom_trend fires
  SDK: v3.1.0

changelog_signals.py:
  v3.1.0 matches regex ^v?3[\.\-] → is_deprecated = True
  Days to deadline (April 30): 6 days → sdk_deadline_30d fires

nps_signals.py:
  score 4 → category = "detractor", low_nps fires
  verbatim contains "contentful" → competitor_mentioned fires
```

### Stage 3: LLM CSM extraction
```
Prompt sent to Gemini:
  "Account: BrightPath Solutions (ACC-042)
   Note: '3/18 - britepath - budget cut 20%, CTO leading CMS eval, missed last 2 QBRs'"

LLM reasons through each signal (chain-of-thought):
  budget_cut_mentioned: "20% budget cut explicitly stated" → TRUE, confidence 1.0
  exec_escalation: "CTO leading the eval" → TRUE, confidence 0.9
  migration_risk: "CMS eval" → TRUE, confidence 0.8
  missed_meetings: "missed last 2 QBRs" → 2, confidence 1.0
  renewal_threatened: not explicitly stated → FALSE

CsmSignal returned:
  budget_cut=True, exec_escalation=True, migration_risk=True,
  missed_meetings=2, confidence=0.8
```

### Stage 4: Scoring
```
risk_scorer.py — weighted sum:

  Signal                    Weight    Notes
  ─────────────────────────────────────────────────────
  negative_mom_trend         1.5      3-month slope < -0.5
  deprecated_sdk             2.5      v3.1.0 is deprecated
  sdk_deadline_30d           1.5      6 days to deadline
  low_nps                    2.0      score = 4
  competitor_mentioned (NPS) 3.0      "contentful" in verbatim
  budget_cut (CSM)           2.0 × 0.8 conf = 1.6  (conf discount)
  exec_escalation (CSM)      2.0 × 0.8 conf = 1.6
  migration_risk (CSM)       3.0 × 0.8 conf = 2.4
  missed_meetings (CSM)      2 × 1.0 × 0.8 conf = 1.6
  ─────────────────────────────────────────────────────
  Raw score: 17.2  → HIGH RISK (threshold: 7.0)

Top signal: "Competitor mentioned in NPS verbatim" (weight 3.0, highest)
```

### Stage 4: LLM explanation
```
Prompt includes: score, tier, all signals, usage data, NPS verbatim, CSM summary

LLM output:
  explanation: "BrightPath's CTO is personally leading a CMS evaluation while
                their company has cut SaaS budgets 20% — and they've no-showed
                twice this quarter. Their SDK v3.1.0 hits end-of-security-patch
                support in 6 days, the same week their contract renews."

  recommended_action: "Sarah Chen should bypass the admin contact and send a
                       direct message to BrightPath's CTO this week — acknowledge
                       the budget pressure, offer a 12-month price lock, and
                       request a 20-minute working session on what it would take
                       to win the renewal."
```

### Final output:
```json
{
  "account_name": "BrightPath Solutions",
  "tier": "High",
  "raw_score": 17.2,
  "arr": 21000,
  "days_to_renewal": 52,
  "top_signal": "Competitor mentioned in NPS verbatim",
  "explanation": "BrightPath's CTO is personally leading...",
  "recommended_action": "Sarah Chen should bypass..."
}
```

---

## The Non-Obvious Insight: SDK Deprecation

This is the answer to "what insight would a simple rule-based system miss?"

### The setup:
SDK v3.x loses security patches on **April 30, 2026**. REST API v2 sunsets the same day. This is in the engineering changelog.

### The gap:
- CSM notes don't mention it
- NPS doesn't mention it
- Support tickets don't mention it
- The only place this exists is `changelog.md` — which no CS tool reads

### What the system does:
```python
# changelog_signals.py
# This join — usage_metrics.sdk_version × changelog.deprecated_versions
# exists nowhere in the CRM

for account_id, usage in usage_signals.items():
    deprecated = _is_deprecated(usage.sdk_version)  # v3.x check
    # Account looks healthy on NPS and usage
    # But they're on v3.1.0 with 6 days until security patches stop
```

### Why it matters:
An account on v3.x with a renewal in May looks fine to a CSM. NPS is 7, no P1 tickets, usage is stable. But when April 30 hits, their integrations break and security patches stop — right in the middle of their renewal window. The system flags this before it becomes a crisis.

### How to say it in the interview:
> "The non-obvious insight is that some accounts appear healthy on every CRM signal but are sitting on a technical time bomb. SDK v3.x loses security patches on April 30. That information only exists in the engineering changelog. I built a module that cross-references each account's SDK version from usage metrics against the deprecation registry extracted from the changelog. That join doesn't exist anywhere else in the system — it's the only place where engineering data meets customer data."

---

## The Fuzzy Name Matching Problem

### Why it exists:
CSM notes are written by humans. They contain typos:

| Note says | Actual name |
|-----------|-------------|
| `BritePath Solutions (sic)` | `BrightPath Solutions` |
| `Pinacle Media` | `Pinnacle Media Group` |
| `Thunderbolt Moters` | `Thunderbolt Motors` |

A simple `.lower()` match fails on ~20% of notes.

### How it's solved:
```python
from rapidfuzz import process, fuzz

# Normalize first: lowercase, strip punctuation, remove "(sic)"
normalized = "britepath solutions"

# Fuzzy match against all known account names
result = process.extractOne(
    normalized,
    list(lookup.keys()),
    scorer=fuzz.token_sort_ratio,  # handles word reordering + typos
)
# → ("brightpath solutions", 87, index)
# 87 > 75 threshold → accepted
```

`token_sort_ratio` handles:
- Character swaps (`Moters` → `Motors`)
- Word reordering (`Solutions BrightPath` → `BrightPath Solutions`)
- Extra noise words (stripped in normalization)

If the score is below 75, the match is rejected. The account gets no CSM signal rather than a wrong one — a conservative choice that avoids false positives.

---

## The LLM Prompting Strategy

Three techniques used, all in `csm_extractor.py` and `risk_explainer.py`:

### 1. Chain-of-Thought (CoT)
The prompt includes a `reasoning` field that the model must fill *before* committing to boolean values. This forces the model to cite the exact phrase that supports or refutes each signal.

```
"reasoning": "competitor_mentioned: 'Kontent.ai' named explicitly — TRUE.
budget_cut_mentioned: They want a discount but that's a negotiation tactic,
not a budget cut — FALSE.
exec_escalation: CRO cc'd on email thread — C-suite involvement — TRUE..."
```

Without CoT, the model pattern-matches on surface language and misclassifies ambiguous notes.

### 2. Few-Shot Examples
3 annotated examples are embedded directly in the prompt:
- **Example 1**: High-risk note with explicit signals (easy case)
- **Example 2**: Subtle silent-churn — NPS looks fine but usage is cratering (hard case)
- **Example 3**: Expansion signal — model must NOT over-flag (negative case)

### 3. Confidence Calibration
The model outputs a confidence score (0.0–1.0) for each extraction. Signals below 0.6 confidence contribute only **50% of their weight** to the risk score.

```
1.0 → explicitly stated ("We are evaluating Contentful")
0.8 → strongly implied ("their CTO asked about migration paths")
0.6 → plausible inference from indirect language
0.4 → weak signal, reading between the lines
```

This means "they might be looking at Contentful" (0.4 confidence) contributes 1.5 points instead of 3.0. The score reflects the uncertainty.

---

## Likely Interview Questions + Answers

### "Why not just use the LLM to assign the risk score?"

> "LLMs are non-deterministic. The same account on Monday vs Friday can get a different tier. You can't track 'this account moved from Medium to High since last week' if the score changes based on model temperature. The scoring is a pure function — same input, same output, always. That makes it auditable, trackable over time, and explainable to a VP who asks why an account is flagged. The LLM writes the explanation, not the score."

### "What happens if the LLM fails?"

> "Every LLM call has a fallback. If CSM extraction fails, the account just gets no CSM signal — it still gets scored on usage, support, NPS, and changelog. If the explainer fails, the account gets a tier-appropriate generic explanation. The pipeline never crashes due to an LLM failure. I also built in retry with exponential backoff — 3 attempts with increasing delays before giving up."

### "How do you handle the messy CSM notes?"

> "Three layers. First, the reconciler uses rapidfuzz fuzzy matching to map typo'd account names to the correct account ID — 'BritePath' matches 'BrightPath' at 87% confidence. Second, the LLM handles the unstructured text itself — it's multilingual, handles mixed date formats, and uses chain-of-thought reasoning to avoid misclassifying ambiguous language. Third, the confidence score discounts uncertain extractions so they can't single-handedly push an account to High risk."

### "What's the non-obvious insight?"

> "SDK deprecation. Some accounts look healthy on every CRM signal — good NPS, stable usage, no P1 tickets — but they're on SDK v3.x, which loses security patches on April 30. That information only exists in the engineering changelog. I built a module that cross-references each account's SDK version from usage metrics against the deprecation registry. That join doesn't exist anywhere in the CRM. It surfaces accounts that would look fine until the deadline hits and their integrations break."

### "How would you scale this to production?"

> "A few things. First, async LLM calls — currently sequential, could run concurrently with asyncio to drop from 90s to ~15s. Second, LLM response caching keyed on a hash of the account ID and note text — re-runs on unchanged data don't re-call the API. Third, replace flat file ingestion with Salesforce and Zendesk API connectors. Fourth, store RiskReport snapshots per run so you can detect score deltas — 'this account moved from Medium to High since last week' is the feature BizOps teams actually want. Fifth, Slack alerts when a High-risk account's ARR exceeds a threshold."

### "Why Pydantic for the models?"

> "Two reasons. First, it enforces typed contracts between pipeline stages — if a signal is missing, it's `None`, not a KeyError that crashes the pipeline. Second, it's the cleanest way to validate LLM output. I pass a Pydantic schema to the LLM client, and it either returns a validated object or `None`. No raw string parsing scattered across the codebase. If the LLM returns malformed JSON, the validation catches it and triggers a retry."

### "What would you change with more time?"

> "Historical score tracking is the biggest one — storing snapshots per run so you can see direction of travel, not just current state. After that: async pipeline for speed, LLM caching for cost, and Salesforce integration to replace the flat files. I'd also add confidence intervals to the score — instead of '14.5', report '14.5 ± 2.0' to reflect uncertainty in the CSM signal extractions."

### "Is this LangGraph? Is this agentic?"

> "No LangGraph — the pipeline is a linear sequence of stages, not a graph with conditional routing or loops. It doesn't need to be agentic because the problem is well-defined: ingest → signal → score → explain. LangGraph would add complexity without benefit here. The LLM calls are stateless — each one gets a prompt and returns a structured object. If I were building a system where the LLM needed to decide which tool to call next, or loop back based on intermediate results, that's when I'd reach for LangGraph."

### "How do you prevent the LLM from over-flagging?"

> "Three ways. First, precise signal definitions in the system prompt — 'tight budget alone is NOT budget_cut_mentioned, it requires explicit mention of budget reduction or vendor consolidation.' Second, the confidence calibration scale — the model is instructed to apply the *lowest* confidence that fits, not the highest. Third, the few-shot examples include a negative case (an expansion account) so the model learns what NOT to flag. And the confidence discount means even if the model over-flags at 0.4 confidence, it only contributes half the weight."

---

## The Scoring System at a Glance

```
Signal                    Weight   Fires when
──────────────────────────────────────────────────────────────────
renewal_threatened          4.0    Explicit churn threat in CSM notes
migration_risk              3.0    Actively evaluating competitors
competitor_mentioned        3.0    Named competitor in NPS or CSM notes
near_zero_usage             3.0    ≤ 2 active users in latest month
usage_drop_flag             2.5    >30% MoM active user decline
deprecated_sdk              2.5    On SDK v3.x (deadline April 30)
unresolved_p1               2.5    Open P1 support ticket
p1_tickets                  2.0    Per P1 ticket (capped at 3)
low_nps                     2.0    NPS score ≤ 6
budget_cut                  2.0    Budget reduction in CSM notes
exec_escalation             2.0    C-suite on call
escalated_tickets           1.5    Escalated support tickets
negative_mom_trend          1.5    Declining 3-month usage slope
sdk_deadline_30d            1.5    SDK deadline within 30 days
detractor_nps               1.0    NPS ≤ 3 (strong detractor)
missed_meetings             1.0    Per missed meeting (capped at 2)
──────────────────────────────────────────────────────────────────
Tier thresholds:  High ≥ 7.0  |  Medium ≥ 3.5  |  Low < 3.5
CSM confidence < 0.6 → signals get 50% weight
P1 tickets capped at 3 contributions
Missed meetings capped at 2 contributions
```

---

## LLM Call Map (Know This Cold)

| Module | Called when | Input | Output | If it fails |
|--------|-------------|-------|--------|-------------|
| `changelog_extractor.py` | Once at startup | `changelog.md` text | Deprecation registry | Hardcoded fallback |
| `csm_extractor.py` | Once per account with a note | Note chunk + account name | `CsmSignal` with confidence | Account gets no CSM signal |
| `risk_explainer.py` | Once per High/Medium account | Score + signals + context | Explanation + action | Generic tier fallback text |

Total calls for a typical run (30 accounts, 25 notes, 20 High/Medium): ~46 LLM calls, ~60–90 seconds.

---

## Key Files to Know

| File | One-line description |
|------|---------------------|
| `pipeline/__init__.py` | `run_pipeline()` — the orchestrator, 4 stages |
| `pipeline/scoring/weight_config.py` | All weights, thresholds, config — change behavior here only |
| `pipeline/scoring/risk_scorer.py` | Pure function: `AccountRecord → RiskScore` |
| `pipeline/llm/csm_extractor.py` | Most complex LLM module — CoT + few-shot + confidence |
| `pipeline/llm/risk_explainer.py` | Generates the plain-English briefing |
| `pipeline/signals/changelog_signals.py` | The non-obvious insight — SDK × deprecation join |
| `pipeline/ingestion/reconciler.py` | Fuzzy name matching with rapidfuzz |
| `models/signals.py` | Pydantic models for all 5 signal types |
| `models/account.py` | `AccountRecord → RiskScore → RiskReport` progression |
| `app/streamlit_app.py` | Dashboard — cached pipeline run, filters, drill-down |

---

## Things to Mention Proactively

These show depth without being asked:

- **Confidence discounting**: CSM signals below 0.6 confidence get 50% weight — uncertainty is reflected in the score, not ignored
- **Deduplication**: If NPS already fired `competitor_mentioned`, CSM won't fire it again — no double-counting
- **Cap logic**: P1 tickets capped at 3, missed meetings capped at 2 — prevents one noisy signal from dominating
- **Fallback chain**: Every LLM call has a fallback — the pipeline never crashes due to an API failure
- **Pydantic everywhere**: Typed contracts between stages — missing signals are `None`, not crashes
- **Tests without API key**: All tests are deterministic — no LLM calls in the test suite
- **The Mandarin note**: The LLM handles it natively — multilingual support for free

---

## What This Is NOT

Be honest about these if asked:

- **Not LangGraph / not agentic**: Linear pipeline, not a graph. No conditional routing, no tool-calling loops. Doesn't need to be.
- **Not real-time**: Batch pipeline, runs on demand. Not streaming.
- **Not production-ready as-is**: Flat file ingestion, no auth, no historical tracking, no Salesforce connector. It's a prototype that demonstrates the architecture.
- **Not a replacement for CSMs**: It surfaces signals and suggests actions. A human still makes the call.
