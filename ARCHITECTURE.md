# Architecture: Renewal Risk Intelligence Engine

> A complete map of the system — every file, every folder, every data flow, and every decision point. Read this to understand how the pieces fit together before touching any code.

---

## System Overview

The pipeline takes **5 raw data files** and produces **ranked risk reports** for every account renewing in the next 90 days. Each report includes a deterministic risk score, a tier (High / Medium / Low), and an LLM-generated plain-English briefing with a specific recommended action.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAW DATA SOURCES                             │
│  accounts.csv  usage_metrics.csv  support_tickets.csv               │
│  nps_responses.csv  csm_notes.txt  changelog.md                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  INGESTION  │  loader.py + reconciler.py
                    │  STAGE 1   │  Load, validate, fuzzy-match names
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │   SIGNAL COMPUTATION    │  STAGE 2 (deterministic)
              │  usage / support / nps  │
              │  changelog / csm (LLM)  │  STAGE 3 (LLM for CSM only)
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │   SCORING   │  STAGE 4 — risk_scorer.py
                    │  (pure fn)  │  Weighted sum → High/Medium/Low
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  EXPLAINER  │  STAGE 4 — risk_explainer.py
                    │    (LLM)    │  Plain-English briefing + action
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  DASHBOARD  │  app/streamlit_app.py
                    │  Streamlit  │  Ranked table + drill-down + SDK panel
                    └─────────────┘
```

---

## Project Structure

```
renewal_intelligence_takehome/
│
├── data/
│   ├── raw/                    ← All source data files (read-only inputs)
│   │   ├── accounts.csv        ← 120 accounts: firmographics + contract dates
│   │   ├── usage_metrics.csv   ← 6 months of monthly usage per account
│   │   ├── support_tickets.csv ← Support ticket history
│   │   ├── nps_responses.csv   ← NPS survey responses with verbatim comments
│   │   ├── csm_notes.txt       ← Raw, messy CSM call notes (unstructured)
│   │   └── changelog.md        ← Product changelog with deprecation notices
│   └── processed/              ← Reserved for future pipeline output snapshots
│
├── models/
│   ├── signals.py              ← Pydantic models for all 5 signal types
│   └── account.py              ← AccountRecord, RiskScore, RiskReport models
│
├── pipeline/
│   ├── __init__.py             ← run_pipeline() — the main orchestrator
│   ├── ingestion/
│   │   ├── loader.py           ← Load + validate all 5 data sources
│   │   └── reconciler.py       ← Fuzzy name matching (rapidfuzz)
│   ├── signals/
│   │   ├── usage_signals.py    ← MoM trend, slope, drop detection
│   │   ├── support_signals.py  ← P1 count, escalations, open tickets
│   │   ├── nps_signals.py      ← NPS category, competitor detection
│   │   └── changelog_signals.py← SDK version × deprecation cross-ref
│   ├── llm/
│   │   ├── llm_client.py       ← OpenAI GPT-4o-mini via LangChain (retry, validation)
│   │   ├── csm_extractor.py    ← LLM: unstructured notes → structured signals
│   │   ├── changelog_extractor.py ← LLM: changelog → deprecation registry
│   │   └── risk_explainer.py   ← LLM: score + signals → plain-English briefing
│   └── scoring/
│       ├── risk_scorer.py      ← Deterministic weighted sum scorer
│       └── weight_config.py    ← All weights, thresholds, and config constants
│
├── app/
│   └── streamlit_app.py        ← Dashboard: filters, risk table, drill-down
│
├── tests/
│   ├── test_reconciler.py      ← Fuzzy matching unit tests
│   ├── test_signals.py         ← Signal computation unit tests
│   └── test_scorer.py          ← Scoring logic unit tests
│
├── .env.example                ← Template for OPENAI_API_KEY
├── requirements.txt            ← Python dependencies
├── README.md                   ← Setup and quick start
├── NOTES.md                    ← Design decisions and tradeoffs
└── ARCHITECTURE.md             ← This file
```

---

## File-by-File Breakdown

### `models/signals.py`
Defines the five Pydantic signal models — the typed contracts between pipeline stages. Every signal module returns a `dict[str, <SignalType>]` keyed by `account_id`.

| Model | Source | Key Fields |
|-------|--------|------------|
| `UsageSignal` | usage_metrics.csv | `mom_change_pct`, `three_month_slope`, `usage_drop_flag`, `near_zero_usage`, `sdk_version` |
| `SupportSignal` | support_tickets.csv | `p1_count`, `has_unresolved_p1`, `escalated_tickets` |
| `NpsSignal` | nps_responses.csv | `score`, `category`, `competitor_mentioned` |
| `ChangelogSignal` | changelog.md × usage | `is_deprecated`, `days_to_deadline`, `affected_features` |
| `CsmSignal` | csm_notes.txt (LLM) | `renewal_threatened`, `migration_risk`, `exec_escalation`, `confidence` |

### `models/account.py`
Three models that represent the pipeline's output progression:

```
AccountRecord  →  RiskScore  →  RiskReport
(all signals     (weighted      (score +
 attached)        sum + tier)    LLM explanation)
```

- `AccountRecord`: The enriched account with all five signals attached. This is what gets passed to the scorer.
- `RiskScore`: The deterministic output — raw score, tier, contributing signals list, top signal.
- `RiskReport`: The final deliverable — everything in `RiskScore` plus the LLM explanation and recommended action.

---

### `pipeline/__init__.py` — The Orchestrator

`run_pipeline()` is the single entry point. It runs all four stages in sequence and returns a sorted list of `RiskReport`.

```
Stage 1: Load data
  load_accounts()          → DataFrame (120 rows)
  load_usage_metrics()     → DataFrame (720 rows, 6 per account)
  load_support_tickets()   → DataFrame (variable)
  load_nps_responses()     → DataFrame (variable)
  load_csm_notes()         → raw string

  Filter: keep only accounts renewing in next 90 days

Stage 2: Deterministic signals (no LLM)
  compute_usage_signals()     → dict[account_id, UsageSignal]
  compute_support_signals()   → dict[account_id, SupportSignal]
  compute_nps_signals()       → dict[account_id, NpsSignal]
  compute_changelog_signals() → dict[account_id, ChangelogSignal]

Stage 3: LLM CSM extraction
  extract_csm_signals()    → dict[account_id, CsmSignal]
  (one LLM call per account that has a matching note chunk)

Stage 4: Score + explain (per account)
  score_account(AccountRecord)         → RiskScore
  generate_risk_report(account, score) → RiskReport
  (LLM explanation only for High + Medium tiers by default)

Return: sorted list of RiskReport, highest score first
```

---

### `pipeline/ingestion/loader.py`

Loads all five raw files into typed DataFrames with basic validation:
- Strips whitespace from `account_id` (common CSV artifact)
- Parses dates with `errors="coerce"` (bad dates become NaT, not crashes)
- Fills numeric NaN with 0
- CSM notes and changelog are loaded as raw strings — the LLM handles parsing

### `pipeline/ingestion/reconciler.py`

Solves the name-matching problem: CSM notes reference accounts by name with typos.

```
"BritePath Solutions (sic)"  →  account_id: ACC-042 (BrightPath Solutions)
"Pinacle Media"              →  account_id: ACC-017 (Pinnacle Media Group)
"Thunderbolt Moters"         →  account_id: ACC-089 (Thunderbolt Motors)
```

**Strategy:**
1. Build a `normalized_name → account_id` lookup from the accounts master
2. Normalize: lowercase, strip punctuation, remove `(sic)` noise
3. Exact match first (fast path)
4. `rapidfuzz.token_sort_ratio` fuzzy match if no exact match
5. Accept if score ≥ 75; reject otherwise (returns `None`)

---

### `pipeline/signals/usage_signals.py`

Computes three risk signals from 6 months of usage data per account:

```python
# MoM change: last two months
mom_change = (users[-1] - users[-2]) / users[-2] * 100

# 3-month slope: linear regression on last 3 months
slope = np.polyfit(range(3), users[-3:], 1)[0]

# Flags
usage_drop_flag = mom_change < -30   # >30% decline
near_zero_usage = latest_users <= 2  # almost no one using it
```

The slope catches gradual decline that a single MoM comparison would miss. An account going from 50 → 45 → 40 → 35 users has a slope of -5 even though no single month crosses the 30% threshold.

### `pipeline/signals/support_signals.py`

Aggregates ticket data per account:
- P1 count (capped at 3 in the scorer, not here — the signal carries the raw count)
- Unresolved P1: any P1 with status `open` or `escalated`
- Escalated tickets: status = `escalated`
- Average resolution time in hours

### `pipeline/signals/nps_signals.py`

Maps NPS score to category and detects competitor mentions:
```python
# Category mapping
0-6  → "detractor"
7-8  → "passive"
9-10 → "promoter"

# Competitor detection (keyword list from weight_config.py)
COMPETITOR_SIGNALS = ["contentful", "strapi", "sanity", "hygraph", ...]
competitor_mentioned = any(c in verbatim.lower() for c in COMPETITOR_SIGNALS)
```

### `pipeline/signals/changelog_signals.py`

The non-obvious insight module. Cross-references each account's SDK version (from `usage_metrics.csv`) against the deprecation registry.

```python
def _is_deprecated(sdk_version: str) -> bool:
    # Matches v3.x.y, 3.x.y, v3.x using regex prefix matching
    return bool(re.match(r"^v?3[\.\-]", v)) or v in ("v3.x", "3.x")
```

This join — `usage_metrics.sdk_version × changelog.deprecated_versions` — exists nowhere in the CRM. It's the only place in the system where engineering data (changelog) meets customer data (usage metrics).

---

### `pipeline/llm/llm_client.py`

The single LLM gateway. All three LLM callers (`csm_extractor`, `changelog_extractor`, `risk_explainer`) go through this module.

**Key design choices:**
- **Structured output via LangChain**: `model.with_structured_output(schema)` handles JSON parsing and Pydantic validation natively — no manual parsing or markdown fence stripping needed
- **Retry with backoff**: 3 attempts, 2s × attempt delay

```python
def call_llm(user_prompt, output_schema, ...) -> T | None:
    # Returns validated Pydantic object or None on persistent failure
    # Callers must handle None — no exceptions propagate
```

### `pipeline/llm/csm_extractor.py`

The most complex LLM module. Extracts structured risk signals from raw CSM notes.

**Prompting strategy:**
1. **Chain-of-Thought**: `reasoning` field forces the model to think through each signal before committing to booleans
2. **Few-shot examples**: 3 annotated examples covering high-risk, subtle silent-churn, and expansion patterns
3. **Confidence calibration**: explicit 1.0/0.8/0.6/0.4 scale with examples anchoring each level
4. **Signal definitions**: precise definitions prevent over-flagging (e.g., "tight budget" alone is NOT `budget_cut_mentioned`)

**Note chunking:** The raw `csm_notes.txt` is split on `---` separators into individual note chunks. Each account gets matched to its chunk via `_find_best_chunk()` — first by account ID pattern, then by exact name, then by first-word substring.

### `pipeline/llm/changelog_extractor.py`

Parses `changelog.md` into a structured deprecation registry. Used once at pipeline initialization.

**Output:** A list of `DeprecationEvent` objects with feature name, affected versions, deadline, severity, and required customer action.

**Fallback:** If the LLM fails, hardcoded known deprecations are returned. The pipeline never crashes due to changelog extraction failure.

### `pipeline/llm/risk_explainer.py`

Generates the plain-English briefing for each High and Medium risk account.

**Quality bar enforced in the prompt:**
- Must name at least one specific signal from the contributing list
- Must explain the business consequence, not just restate the data
- Recommended action must have WHO + WHAT + WHEN
- No filler phrases ("multiple signals", "various issues")
- No generic advice ("schedule a call to check in")

**Fallback:** If the LLM fails, tier-appropriate generic fallback text is used. The pipeline always produces a complete `RiskReport`.

---

### `pipeline/scoring/weight_config.py`

The single source of truth for all scoring behavior. **Change behavior by editing this file only — never touch `risk_scorer.py` for weight adjustments.**

```python
SIGNAL_WEIGHTS = {
    "renewal_threatened": 4.0,   # Explicit churn threat — highest weight
    "migration_risk":     3.0,   # Active competitor evaluation
    "competitor_mentioned": 3.0, # Named competitor in notes or NPS
    "near_zero_usage":    3.0,   # Almost no active users
    "usage_drop_flag":    2.5,   # >30% MoM decline
    "deprecated_sdk":     2.5,   # On v3.x with April 30 deadline
    "unresolved_p1":      2.5,   # Open P1 ticket
    "p1_tickets":         2.0,   # Per P1 ticket (capped at 3)
    "low_nps":            2.0,   # NPS score ≤ 6
    "budget_cut":         2.0,   # Budget reduction mentioned
    "exec_escalation":    2.0,   # C-suite on call
    "escalated_tickets":  1.5,   # Escalated support tickets
    "negative_mom_trend": 1.5,   # Declining 3-month slope
    "sdk_deadline_30d":   1.5,   # SDK deadline within 30 days
    "detractor_nps":      1.0,   # NPS ≤ 3 (strong detractor)
    "missed_meetings":    1.0,   # Per missed meeting (capped at 2)
}

TIER_THRESHOLDS = {
    "High":   7.0,   # score >= 7.0
    "Medium": 3.5,   # score >= 3.5
    # else: Low
}
```

### `pipeline/scoring/risk_scorer.py`

A pure function: `AccountRecord → RiskScore`. Same input always produces the same output.

**Notable rules:**
- P1 tickets capped at 3 contributions (`min(p1_count, 3) × 2.0`)
- Missed meetings capped at 2 contributions
- CSM signals below `CSM_CONFIDENCE_THRESHOLD` (0.6) get 50% weight
- Competitor mention deduplication: if NPS already fired `competitor_mentioned`, CSM won't fire it again

---

### `app/streamlit_app.py`

The dashboard. Calls `run_pipeline()` once and caches the result for the session.

**Layout:**
```
Sidebar: Filter by tier / CSM / industry
─────────────────────────────────────────
Main:
  [KPI row] Accounts in window | High | Medium | Low | ARR at Risk
  ─────────────────────────────────────────────────────────────────
  [Risk table] Sortable, all filtered accounts ranked by score
  ─────────────────────────────────────────────────────────────────
  [Account detail] Select account → explanation + signals + action
  ─────────────────────────────────────────────────────────────────
  [SDK insight panel] Accounts on deprecated SDK in renewal window
```

---

## Data Flow: End to End

Here is the complete data flow for a single account, from raw CSV to final report:

```
accounts.csv row:
  account_id: ACC-042
  account_name: BrightPath Solutions
  arr: 21000
  contract_end_date: 2026-06-15
  plan_tier: Starter
  csm_name: Sarah Chen

  ↓ loader.py: parse + validate

usage_metrics.csv (6 rows for ACC-042):
  Month 1-6: active_users = [28, 25, 22, 18, 14, 10], sdk_version = v3.1.0

  ↓ usage_signals.py:
  UsageSignal(mom_change_pct=-28.6, three_month_slope=-4.0,
              usage_drop_flag=False, near_zero_usage=False, sdk_version="v3.1.0")

  ↓ changelog_signals.py:
  ChangelogSignal(is_deprecated=True, days_to_deadline=6,
                  affected_features=["REST API v2", "SDK v3 security patches"])

csm_notes.txt chunk for "BritePath Solutions (sic)":
  "3/18 - britepath - budget cut 20%, CTO leading CMS eval, missed last 2 QBRs"

  ↓ reconciler.py: "britepath" → fuzzy match → ACC-042 (score: 87)

  ↓ csm_extractor.py (LLM call):
  CsmSignal(budget_cut_mentioned=True, exec_escalation=True,
            migration_risk=True, missed_meetings=2, confidence=0.8)

nps_responses.csv:
  score: 4, verbatim: "We've been evaluating Contentful as a backup"

  ↓ nps_signals.py:
  NpsSignal(score=4, category="detractor", competitor_mentioned=True)

  ↓ score_account() — deterministic weighted sum:
  budget_cut:          2.0
  exec_escalation:     2.0
  migration_risk:      3.0
  missed_meetings:     2 × 1.0 × 0.8 conf = 1.6  (conf discount applied)
  deprecated_sdk:      2.5
  sdk_deadline_30d:    1.5
  low_nps:             2.0
  competitor_mentioned (NPS): 3.0
  ─────────────────────────
  Raw score: 17.6  → HIGH RISK

  ↓ generate_risk_report() (LLM call):
  explanation: "BrightPath's CTO is personally leading a CMS evaluation while
                their company has cut SaaS budgets 20% — and they've no-showed
                twice this quarter..."
  recommended_action: "Sarah Chen should bypass the admin contact and reach
                       BrightPath's CTO directly this week..."

  ↓ RiskReport (final output):
  {account_name: "BrightPath Solutions", tier: "High", score: 17.6,
   arr: 21000, days_to_renewal: 52, explanation: "...", action: "..."}
```

---

## LLM Call Map

| Module | When Called | Input | Output | Fallback |
|--------|-------------|-------|--------|----------|
| `changelog_extractor.py` | Once at startup | `changelog.md` text | `ChangelogDeprecations` | Hardcoded known deprecations |
| `csm_extractor.py` | Once per account with a matching note | Note chunk + account context | `CsmSignal` | Account gets no CSM signal |
| `risk_explainer.py` | Once per High/Medium account | Score + all signals + account context | `explanation` + `recommended_action` | Tier-appropriate generic fallback |

Total LLM calls per pipeline run: `1 + N_csm_notes + N_high_medium_accounts`

For a 30-account renewal window with 25 CSM notes and 20 High/Medium accounts: ~46 LLM calls, ~60–90 seconds.

---

## Tests

```
tests/
├── test_reconciler.py   ← Fuzzy matching: typos, threshold edge cases, exact matches
├── test_signals.py      ← Signal computation: MoM calc, slope, NPS categories, SDK detection
└── test_scorer.py       ← Scoring: weight application, tier thresholds, CSM confidence discount
```

Tests run without an API key — all LLM modules are excluded from the test suite. Run with:
```bash
pytest tests/ -v
```

---

## Configuration Reference

All tunable parameters live in `pipeline/scoring/weight_config.py`:

| Constant | Default | Effect |
|----------|---------|--------|
| `SIGNAL_WEIGHTS` | (see above) | Per-signal score contribution |
| `TIER_THRESHOLDS` | High: 7.0, Medium: 3.5 | Score → tier mapping |
| `CSM_CONFIDENCE_THRESHOLD` | 0.6 | Below this, CSM signals get 50% weight |
| `RENEWAL_WINDOW_DAYS` | 90 | Accounts renewing within this window are analyzed |
| `DEPRECATED_SDK_VERSIONS` | v3.0–v3.2, v3.x | SDK versions flagged as deprecated |
| `DEPRECATED_SDK_DEADLINE` | 2026-04-30 | Hard cutoff date for SDK v3.x |
| `COMPETITOR_SIGNALS` | contentful, strapi, sanity, ... | Competitor names detected in text |
