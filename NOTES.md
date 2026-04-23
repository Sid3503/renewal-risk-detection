# Design Notes & Deep Dive

> This document explains the thinking behind every major decision in the Renewal Risk Intelligence Engine — the tradeoffs, the non-obvious insights, and what you'd change if you had more time. Read this if you want to understand *why* the system works the way it does, not just *what* it does.

---

## What Is This?

This is a **renewal risk intelligence pipeline** built for a BizOps team at a B2B SaaS company (Contentstack). Every quarter, the team needs to know: *which accounts are about to churn, and why?*

The old way: gut feel, scattered Salesforce notes, last-minute Slack threads.

The new way: five raw data sources → automated signal extraction → deterministic risk scoring → LLM-generated plain-English briefings → a Streamlit dashboard the team can actually use.

---

## The Problem It Solves

### Without this system:
- A CSM manually reads 120 accounts' notes, tickets, and usage data
- They miss the account that looks healthy on NPS but is quietly migrating to a homegrown solution
- They don't know that 8 accounts are on a deprecated SDK that loses security patches in 30 days — because that information lives in the engineering changelog, not in Salesforce
- By the time they flag a risk, there are 2 weeks left before renewal

### With this system:
- All five data sources are ingested, reconciled, and cross-referenced automatically
- Every account in the 90-day renewal window gets a deterministic risk score with a breakdown of exactly which signals fired
- An LLM reads the score and writes a specific, actionable briefing — not "schedule a call" but "Sarah Chen should escalate to VP of CS today and bring a v3→v4 migration timeline"
- The whole pipeline runs in ~60–90 seconds

---

## Architecture Philosophy: Deterministic Scoring + LLM Explanation

This is the most important design decision in the entire system.

### The wrong way: let the LLM assign the risk score

```
# BAD — don't do this
prompt = f"Here is account data for {account_name}. What is their churn risk? High, Medium, or Low?"
tier = llm.call(prompt)  # Non-reproducible. Auditable? No. Trackable over time? No.
```

The problem: LLMs are non-deterministic. The same account on Monday vs Friday can get different tiers. You can't track "this account moved from Medium to High since last week" if the score changes based on model temperature.

### The right way: deterministic scoring, LLM for language

```
# GOOD — what this system does
score = score_account(account)       # Pure function. Same input → same output. Always.
report = generate_risk_report(account, score)  # LLM writes the *explanation*, not the score
```

The LLM does exactly three things:
1. **Parse unstructured CSM notes** into structured signals (the notes are noise otherwise)
2. **Extract the changelog deprecation registry** into machine-readable format
3. **Generate the plain-English explanation + action** (where nuance and context actually matter)

Everything else is deterministic Python.

---

## The Five Signal Types

### 1. Usage Signals
**Source:** `usage_metrics.csv` — 6 months of monthly active users, API calls, SDK version

**What it computes:**
- Month-over-month change in active users
- 3-month linear slope (trend detection, not just point-in-time)
- Drop flag: MoM decline > 30%
- Near-zero flag: ≤ 2 active users in latest month

**Example:**
```
Account: NovaTech Industries
Month 1: 45 users → Month 2: 42 → Month 3: 38 → Month 4: 31 → Month 5: 22 → Month 6: 8

MoM change: -63.6%  ← usage_drop_flag fires (weight: 2.5)
3-month slope: -7.0  ← negative_mom_trend fires (weight: 1.5)
Latest users: 8  ← near_zero_usage does NOT fire (threshold is ≤ 2)
```

### 2. Support Signals
**Source:** `support_tickets.csv`

**What it computes:**
- P1 ticket count (capped at 3 for scoring — one bad month shouldn't dominate)
- Unresolved P1 flag
- Escalated ticket count
- Average resolution time

**Example:**
```
Account: Meridian Health
P1 tickets: 4  → contribution = min(4, 3) × 2.0 = 6.0 points
Unresolved P1: True  → +2.5 points
Escalated: 2  → +1.5 points
Total support contribution: 10.0 points
```

### 3. NPS Signals
**Source:** `nps_responses.csv`

**What it computes:**
- NPS category (detractor ≤ 6, passive 7–8, promoter 9–10)
- Strong detractor flag (score ≤ 3)
- Competitor mention in verbatim text (keyword matching against known competitor list)

**Example:**
```
Account: BrightPath Solutions
NPS score: 4  → low_nps fires (weight: 2.0)
Verbatim: "We've been evaluating Contentful as a backup option"
  → competitor_mentioned fires (weight: 3.0)
```

### 4. Changelog Signals (The Non-Obvious One)
**Source:** `changelog.md` + `usage_metrics.csv` (SDK version field)

This is the insight that a purely rule-based system would miss entirely.

**What it does:** Cross-references each account's current SDK version against the deprecation registry extracted from the changelog. SDK v3.x loses security patches on April 30, 2026 — the same date REST API v2 sunsets.

**Why it's non-obvious:** This information lives in the engineering changelog, not in Salesforce or the CRM. A CSM has no reason to know their account is on v3.2.0. The `changelog_signals` module is the only place in the entire system where this join happens.

**Example:**
```
Account: Thunderbolt Motors
SDK version (from usage_metrics): v3.1.0
Changelog says: v3.x deprecated, deadline April 30 2026

→ deprecated_sdk fires (weight: 2.5)
→ sdk_deadline_30d fires if deadline ≤ 30 days away (weight: 1.5)
→ CSM note: no mention of SDK at all ← invisible risk
```

### 5. CSM Signals (LLM-Extracted)
**Source:** `csm_notes.txt` — raw, messy, unstructured text

This is where the LLM earns its place. The notes contain:
- Mixed date formats (`Mar 12`, `3/15`, `2026-03-20`, `march 25`)
- Typos in account names (`BritePath`, `Pinacle`, `Thunderbolt Moters`)
- One note in Mandarin
- Account IDs present for some entries, absent for others

**What the LLM extracts:**
```json
{
  "competitor_mentioned": true,
  "budget_cut_mentioned": false,
  "exec_escalation": true,
  "migration_risk": true,
  "renewal_threatened": true,
  "missed_meetings": 2,
  "positive_signal": false,
  "confidence": 0.85
}
```

---

## The LLM Prompting Strategy

### Chain-of-Thought (CoT) for CSM Extraction

The CSM extractor uses a `reasoning` field as a scratchpad before committing to booleans. This measurably improves accuracy on ambiguous notes.

```
"reasoning": "competitor_mentioned: 'Kontent.ai' named explicitly — TRUE.
budget_cut_mentioned: No budget language, they want a discount but that's
a negotiation tactic not a budget cut — FALSE.
exec_escalation: CRO cc'd on email thread — C-suite involvement — TRUE..."
```

The model reasons through each signal before filling the boolean fields. Without this, the model pattern-matches on surface language and misclassifies ambiguous notes.

### Few-Shot Examples

Both the CSM extractor and the risk explainer embed 2–3 annotated examples directly in the prompt. The examples are chosen to cover:
- High-risk explicit signals (easy case)
- Subtle silent-churn patterns (hard case — NPS looks fine but usage is cratering)
- Expansion signals (negative case — model must not over-flag)

### Confidence Calibration

The CSM extractor uses a calibrated confidence scale:
```
1.0 → explicitly stated word-for-word ("We are evaluating Contentful")
0.8 → strongly implied with clear context ("their CTO asked about migration paths")
0.6 → plausible inference from indirect language
0.4 → weak signal, reading between the lines
```

Signals below 0.6 confidence contribute only **50% of their weight** to the risk score. A CSM saying "they might be looking at Contentful" is very different from an explicit churn threat — the scoring reflects that.

### Anti-Pattern Teaching for the Explainer

The risk explainer prompt explicitly shows the model what *not* to write:

```
BAD: "NovaTech has multiple compounding risk signals including support issues
      and potential churn intent. Schedule a call to discuss their concerns."

GOOD: "NovaTech's CTO joined the last call after 4 P1 tickets linked to their
       v3 SDK — which hits end-of-security-patch support on April 30, the same
       week their contract renews..."
```

The quality bar requires: a specific signal named, a business consequence explained, and a recommended action with WHO + WHAT + WHEN.

---

## Fuzzy Name Reconciliation

### The Problem

CSM notes reference accounts by name, not by ID. And the names have typos:

| Note says | Actual account name |
|-----------|---------------------|
| `BritePath Solutions (sic)` | `BrightPath Solutions` |
| `Pinacle Media` | `Pinnacle Media Group` |
| `Thunderbolt Moters` | `Thunderbolt Motors` |

A simple `.lower().strip()` match fails on ~20% of notes.

### The Solution: `rapidfuzz.token_sort_ratio`

```python
from rapidfuzz import process, fuzz

result = process.extractOne(
    "britepath solutions",
    list(lookup.keys()),
    scorer=fuzz.token_sort_ratio,
)
# Returns: ("brightpath solutions", 87, index)
# 87 > 75 threshold → match accepted
```

`token_sort_ratio` handles:
- Character transpositions (`Moters` → `Motors`)
- Word reordering (`Solutions BrightPath` → `BrightPath Solutions`)
- Extra noise words (`(sic)` stripped in normalization)

The threshold is 75. Below that, the match is rejected and the account gets no CSM signal rather than a wrong one.

---

## The Scoring System

### How Weights Work

Every signal that fires adds its weight to a running score. The weights are all in `pipeline/scoring/weight_config.py` — change behavior by editing that file, never by touching the scorer logic.

```python
SIGNAL_WEIGHTS = {
    "renewal_threatened": 4.0,   # Highest — explicit churn threat
    "migration_risk": 3.0,       # Active competitor evaluation
    "competitor_mentioned": 3.0, # Named competitor in notes/NPS
    "near_zero_usage": 3.0,      # Almost no one using the product
    "usage_drop_flag": 2.5,      # >30% MoM decline
    "deprecated_sdk": 2.5,       # On v3.x with April 30 deadline
    "unresolved_p1": 2.5,        # Open P1 ticket
    ...
}

TIER_THRESHOLDS = {
    "High": 7.0,    # score >= 7.0
    "Medium": 3.5,  # score >= 3.5
    # else: Low
}
```

### Example Score Breakdown

```
Account: NovaTech Industries

Signal                          Weight
─────────────────────────────────────
Renewal threatened               4.0
Migration risk                   3.0
Competitor mentioned (NPS)       3.0
4 P1 tickets (capped at 3×2.0)   6.0
Unresolved P1                    2.5
Usage dropped 63% MoM            2.5
On deprecated SDK v3.2.0         2.5
SDK deadline in 6 days           1.5
Executive escalation (CTO)       2.0
─────────────────────────────────────
Raw score:                      27.0  → HIGH RISK
```

### Why P1 Tickets Are Capped

Without a cap, a single account with 10 P1 tickets would score 20 points from tickets alone, drowning out every other signal. The cap at 3 means tickets can contribute at most 6.0 points — significant, but not overwhelming.

---

## The Non-Obvious Insight: SDK Deprecation × Support Ticket Correlation

This is the insight the assignment asked for — something a simple rule-based system would miss.

**The setup:** SDK v3.x loses security patches on April 30, 2026. REST API v2 sunsets the same day. This is documented in the changelog.

**The gap:** CSM notes don't mention it. NPS doesn't mention it. Support tickets don't mention it. The only place this information exists is in the engineering changelog — which no CS tool reads.

**The join that doesn't exist anywhere else:**
```python
# changelog_signals.py
for account_id, usage in usage_signals.items():
    deprecated = _is_deprecated(usage.sdk_version)  # v3.x check
    # This join: usage_metrics.sdk_version × changelog.deprecated_versions
    # exists nowhere in the CRM
```

**What it surfaces:** Accounts that look healthy on NPS and usage but are sitting on a technical time bomb. When the deadline hits, their integrations break and security patches stop — right in the middle of their renewal window.

---

## What I'd Change With More Time

### 1. Streaming / Async Pipeline
Currently processes accounts sequentially. With `asyncio` and concurrent LLM calls, the ~60–90s runtime drops to ~10–15s.

```python
# Current (sequential)
for account in accounts:
    csm_signal = extract_csm_signals(account)  # 2-3s each

# Better (concurrent)
async def process_all(accounts):
    tasks = [extract_csm_signals_async(a) for a in accounts]
    return await asyncio.gather(*tasks)
```

### 2. LLM Response Caching
Cache LLM responses keyed on `hash(account_id + note_text)`. Re-runs on unchanged data don't re-call the API. Saves cost and latency.

### 3. Historical Score Tracking
Store `RiskReport` snapshots per pipeline run. Enable delta detection:
```
"NovaTech moved from Medium (score: 5.2) to High (score: 14.5) since last week"
```
This is the feature BizOps teams actually want — not just the current state, but the direction of travel.

### 4. Confidence Intervals
Instead of a single score, report a range:
```
NovaTech: 14.5 ± 2.0 (CSM signals at 0.85 confidence)
BrightPath: 6.0 ± 3.5 (CSM signals at 0.45 confidence — wide range)
```

### 5. Salesforce Integration
Replace flat file ingestion with a Salesforce API connector. CSM notes, account data, and contract dates all live in SF — the flat files are a prototype stand-in.

### 6. Automated Alerts
Slack/email notification when a High-risk account's ARR exceeds a threshold, or when an account's score increases by more than X points between runs.

---

## Known Limitations

**CSM note matching fallback:** The `_find_best_chunk()` function uses first-word substring matching as a last resort. Two accounts sharing a first word (e.g., two "Pacific" companies) could mis-match. A production system should enforce `account_id` tagging in all CSM notes.

**Multilingual notes:** The Mandarin note works correctly because the LLM handles multilingual text natively. However, the account ID detection in `_find_best_chunk()` only matches English ID patterns — a non-English note without a matching account name in the text will be missed.

**NPS competitor detection:** Keyword-based (`in` check against a list). False negatives are possible for misspellings or indirect references ("the other major headless CMS"). A production system would use embedding similarity or a dedicated NER model.

**No deduplication on NPS:** If an account has multiple NPS responses, only the first one is used. A production system would aggregate or use the most recent.

**Flat file ingestion:** The pipeline reads from `data/raw/`. In production, these would be API calls to Salesforce, Mixpanel, Zendesk, and the internal changelog service.
