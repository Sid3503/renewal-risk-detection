# Renewal Risk Intelligence Engine

> A production-grade pipeline that ingests messy multi-source SaaS account data, computes deterministic risk signals, and uses an LLM to generate plain-English renewal risk briefings — so your BizOps team knows which accounts are about to churn before the account team has to ask.

---

## What It Does

Every quarter, a BizOps team needs to answer one question: **which accounts are at risk of churning at renewal, and why?**

This engine answers that question automatically by:

1. Ingesting five data sources — accounts, usage metrics, support tickets, NPS responses, and raw CSM notes
2. Computing deterministic risk signals (usage trends, P1 tickets, NPS scores, SDK deprecation)
3. Using an LLM to extract structured signals from messy, unstructured CSM notes
4. Scoring every account in the 90-day renewal window with a weighted risk model
5. Generating a plain-English briefing per account with a specific recommended action
6. Presenting everything in a Streamlit dashboard with filters, a ranked risk table, and account drill-downs

The scoring is fully deterministic — same input always produces the same score. The LLM is used only for parsing unstructured text and generating human-readable explanations.

---

## Quick Start

### Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys) with access to `gpt-4o-mini`

### Setup

```bash
# 1. Clone the repo and navigate into it
git clone <repo-url>
cd renewal_intelligence_takehome

# 2. Create and activate a virtual environment (recommended)
python -m venv ren_env
source ren_env/bin/activate        # macOS / Linux
# ren_env\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
cp .env.example .env
# Open .env and set: OPENAI_API_KEY=your_key_here
```

### Run the Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard opens at `http://localhost:8501`. The first load runs the full pipeline — expect **60–90 seconds** for LLM calls. Results are cached for the session, so subsequent interactions are instant.

### Run Tests (No API Key Needed)

```bash
pytest tests/ -v
```

All tests are deterministic and run without any API calls.

---

## Dashboard Walkthrough

Once the pipeline finishes, you'll see:

**KPI Row** — accounts in the 90-day window, count by tier, and total ARR at risk for High-tier accounts.

**Risk Table** — all accounts ranked by score, with tier badge, top signal, ARR, days to renewal, CSM, and plan.

**Account Detail** — select any account to see the full LLM-generated briefing: why it's at risk, which signals fired, and a specific recommended action with a named CSM, activity, and timeline.

**SDK Insight Panel** — accounts on deprecated SDK v3.x in the renewal window. This surfaces technical churn risk that lives only in the engineering changelog and would be invisible to a purely CRM-based system.

---

## How the Scoring Works

Every account gets a **raw score** — a weighted sum of all risk signals that fired. The score maps to a tier:

| Tier | Score Threshold |
|------|----------------|
| 🔴 High | ≥ 7.0 |
| 🟡 Medium | ≥ 3.5 |
| 🟢 Low | < 3.5 |

**Example signals and their weights:**

| Signal | Weight | Fires When |
|--------|--------|------------|
| Explicit churn threat | 4.0 | CSM note contains renewal ultimatum |
| Migration / competitor eval | 3.0 | Account actively evaluating alternatives |
| Competitor mentioned | 3.0 | Named competitor in NPS verbatim or CSM notes |
| Near-zero active users | 3.0 | ≤ 2 active users in latest month |
| Usage dropped >30% MoM | 2.5 | Month-over-month active user decline |
| Deprecated SDK (v3.x) | 2.5 | Account on SDK losing security patches Apr 30 |
| Unresolved P1 ticket | 2.5 | Open P1 support ticket |
| Low NPS (score ≤ 6) | 2.0 | Detractor NPS response |
| Executive escalation | 2.0 | C-suite on call per CSM notes |

All weights and thresholds are in `pipeline/scoring/weight_config.py`. Change behavior by editing that file — no other code needs to change.

---

## The Non-Obvious Insight

SDK v3.x is deprecated as of **April 30, 2026** — REST API v2 sunsets and security patches stop on the same date. Some accounts in the renewal window are on v3.x with **no CSM note flagging the technical risk**. They look healthy on NPS and usage, but they're sitting on a technical time bomb.

The `changelog_signals` module surfaces this by cross-referencing each account's SDK version (from `usage_metrics.csv`) against the deprecation registry extracted from `changelog.md`. This join exists nowhere in the CRM — it's the only place in the system where engineering data meets customer data.

---

## Project Structure

```
├── data/raw/             ← Source data files (read-only)
├── pipeline/
│   ├── __init__.py       ← run_pipeline() — main orchestrator
│   ├── ingestion/        ← loader.py, reconciler.py (fuzzy name matching)
│   ├── signals/          ← One file per signal type (deterministic)
│   ├── scoring/          ← risk_scorer.py, weight_config.py
│   └── llm/              ← llm_client.py, csm_extractor.py, risk_explainer.py
├── models/               ← Pydantic models: signals.py, account.py
├── app/                  ← streamlit_app.py
├── tests/                ← test_reconciler.py, test_signals.py, test_scorer.py
├── NOTES.md              ← Design decisions, tradeoffs, and deep dives
└── ARCHITECTURE.md       ← Full system map: files, data flow, LLM call map
```

For a full breakdown of every file and the complete data flow, see [ARCHITECTURE.md](ARCHITECTURE.md).
For design decisions, prompting strategy, and known limitations, see [NOTES.md](NOTES.md).

---

## Configuration

All scoring behavior is controlled by `pipeline/scoring/weight_config.py`:

```python
SIGNAL_WEIGHTS       # Per-signal weight contributions
TIER_THRESHOLDS      # High/Medium/Low score cutoffs
RENEWAL_WINDOW_DAYS  # Default: 90 days
CSM_CONFIDENCE_THRESHOLD  # Below 0.6, CSM signals get 50% weight
DEPRECATED_SDK_VERSIONS   # SDK versions flagged as deprecated
COMPETITOR_SIGNALS        # Competitor names detected in text
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Usage trend slope calculation |
| `pydantic` | Typed models and LLM output validation |
| `rapidfuzz` | Fuzzy account name matching |
| `langchain-openai` | OpenAI GPT-4o-mini LLM calls via LangChain |
| `langchain-core` | LangChain message types and structured output |
| `streamlit` | Dashboard UI |
| `python-dotenv` | API key loading from `.env` |
| `pytest` | Test runner |

---

## Troubleshooting

**Pipeline returns no reports**
- Check that `data/raw/` contains all six files
- Verify `OPENAI_API_KEY` is set in `.env`

**LLM calls failing / slow**
- The pipeline adds automatic retry backoff on failure
- If all LLM calls fail, the pipeline still runs: CSM signals are skipped and explanations fall back to generic tier-appropriate text

**Fuzzy matching misses an account**
- The match threshold is 75. If a CSM note has a very different name, it won't match
- Check `reconciler.py` logs for "No match for account name" warnings

**Tests failing**
- Tests don't require an API key — if they fail, it's a logic issue, not an auth issue
- Run `pytest tests/ -v` for detailed output
