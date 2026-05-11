# The 4 Pipeline Stages — Complete Guide

> Everything explained simply. Analogies first, dry runs, no jargon. Exactly how the data flows from raw CSV files to the final risk report on your screen.

---

## Before the Stages: Key Terms

You need these definitions before anything makes sense.

**ARR (Annual Recurring Revenue)**
How much money an account pays per year.
```
ARR = $21,000 means the account pays $21k/year
```

**Account**
A company that is a customer. 120 accounts in the dataset.

**CSM (Customer Success Manager)**
The human who manages the relationship with each account. They write call notes, attend QBRs, flag risks.

**Signal**
A single piece of evidence that an account might churn.
```
"Usage dropped 63% last month"  ← one signal
"CTO joined the call"           ← another signal
"NPS score is 4"                ← another signal
```

**Risk Tier**
The final verdict for an account:
```
HIGH   → score >= 7.0  → needs immediate action
MEDIUM → score >= 3.5  → needs attention soon
LOW    → score < 3.5   → looks fine for now
```

**PipelineState** — the baton passed between all 4 stages:
```
Starts with:    {5 raw CSV/text files}
After Stage 1:  + clean DataFrames, filtered to 90-day window
After Stage 2:  + 4 signal dicts (usage, support, nps, changelog)
After Stage 3:  + csm_signals dict (LLM-extracted)
After Stage 4:  + RiskScore per account + RiskReport with explanation
```

---

## The Big Picture

Think of the pipeline as a baton race. 4 stages. Each does one job, adds their result to the baton, passes it on.

```
5 raw files (CSV + text)
        |
Stage 1: Load & Filter
        Load all files into clean tables
        Keep only accounts renewing in next 90 days
        |
Stage 2: Compute Deterministic Signals
        Pure math + rules on the clean data
        No LLM. Same input = same output. Always.
        |
Stage 3: LLM CSM Extraction
        Send messy human-written notes to the LLM
        Get back structured risk flags per account
        |
Stage 4: Score + Explain
        Add up all signals into a risk score (math)
        LLM writes the plain-English briefing (language)
        |
List of RiskReports → Streamlit Dashboard
```

---

## Stage 1: The Loader & Filter

**Job: Read all 5 raw files, clean them up, keep only the accounts that matter right now.**

### The Analogy

You work at a hospital. Every morning someone drops 5 folders on your desk:
- Folder 1: Patient master list (name, DOB, insurance)
- Folder 2: Vitals readings for the last 6 months
- Folder 3: Complaint history
- Folder 4: Patient satisfaction surveys
- Folder 5: Handwritten doctor notes

Your first job: read all 5 folders, fix any obvious errors (missing dates, blank fields), and pull out only the patients whose appointments are in the next 90 days. Everyone else can wait.

That's Stage 1.

---

### What Gets Loaded

**`accounts.csv` → `load_accounts()`**

This is the spine. Every account flows through this. It has:
- `account_id` — the unique ID (e.g. `ACC-042`)
- `account_name` — company name (e.g. `BrightPath Solutions`)
- `arr` — annual revenue from this account
- `contract_end_date` — when their contract expires
- `csm_name` — which CSM owns this account

```python
df = pd.read_csv("accounts.csv")

# Clean account_id: " ACC-042 " → "ACC-042"
df["account_id"] = df["account_id"].astype(str).str.strip()

# Parse dates: "2026-06-15" → actual date object
# errors="coerce" means: if a date is broken, make it null instead of crashing
df["contract_end_date"] = pd.to_datetime(df["contract_end_date"], errors="coerce")

# Parse money: "$21,000" or "21000" → the number 21000
# fillna(0) means: if it can't be parsed, use 0 instead of crashing
df["arr"] = pd.to_numeric(df["arr"], errors="coerce").fillna(0)
```

**`usage_metrics.csv` → `load_usage_metrics()`**

6 rows per account — one per month for the last 6 months. Each row has:
- `account_id`, `month`
- `active_users` — how many people used the product that month
- `api_calls`, `content_entries_created`, `workflows_triggered`
- `sdk_version` — which version of the SDK they're running

```python
# Force numeric columns — bad values become 0
for col in ["api_calls", "active_users", ...]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
```

**`support_tickets.csv` → `load_support_tickets()`**

One row per ticket. Key columns:
- `priority` — P1 (critical), P2, P3
- `status` — open, closed, escalated
- `resolution_time_hours` — how long it took to fix

```python
# Normalize priority: "p1" or "P1 " → "P1"
df["priority"] = df["priority"].astype(str).str.upper().str.strip()

# Normalize status: "Open" or "OPEN" → "open"
df["status"] = df["status"].astype(str).str.lower().str.strip()
```

**`nps_responses.csv` → `load_nps_responses()`**

One row per account. Key columns:
- `score` — 0 to 10
- `verbatim_comment` — what the customer actually wrote

```python
# Fill blank verbatim with empty string (not null — avoids crashes later)
df["verbatim_comment"] = df["verbatim_comment"].fillna("").astype(str)
```

**`csm_notes.txt` → `load_csm_notes()`**

Loaded as a single raw string. Not a table. The LLM handles parsing.

```python
text = (RAW_DIR / "csm_notes.txt").read_text(encoding="utf-8")
# Returns one big string with all notes separated by "---"
```

---

### The 90-Day Filter

After loading accounts, the pipeline calculates how many days until each account's contract expires. Then it keeps only the ones expiring within 90 days.

```python
today = date.today()
cutoff = today + timedelta(days=90)

# Calculate days_to_renewal for every account
accounts_df["days_to_renewal"] = (
    accounts_df["contract_end_date"] - today
).dt.days

# Keep only accounts renewing in the window
in_window = accounts_df[
    (accounts_df["contract_end_date"] >= today)
    & (accounts_df["contract_end_date"] <= cutoff)
]
```

Why 90 days? Standard enterprise CS practice. 90 days gives enough lead time to actually do something. Accounts outside this window have lower urgency and would inflate the dashboard.

### Dry Run

```
accounts.csv has 120 rows.
Today: April 27, 2026
Cutoff: July 26, 2026

After filter:
  ACC-042 BrightPath Solutions  contract_end: 2026-06-15  days_to_renewal: 49  ✓ IN
  ACC-017 Pinnacle Media Group  contract_end: 2026-05-10  days_to_renewal: 13  ✓ IN
  ACC-089 Thunderbolt Motors    contract_end: 2026-09-01  days_to_renewal: 127 ✗ OUT (too far)
  ...

Result: 30 accounts in the 90-day window
```

### What Stage 1 Produces

```
BEFORE Stage 1:
  5 raw files on disk

AFTER Stage 1:
  accounts_df  → 30 rows (filtered to 90-day window)
  usage_df     → ~180 rows (6 months × 30 accounts)
  tickets_df   → variable (not all accounts have tickets)
  nps_df       → variable (not all accounts have NPS)
  csm_notes    → one big string
```

Stage 1 done. Baton passed to Stage 2.

---

## Stage 2: The Signal Computers

**Job: Run math and rules on the clean data to produce risk flags. No LLM. Pure Python.**

### The Analogy

A doctor's assistant reviews the patient folders before the doctor sees them. They don't diagnose — they just compute facts:
- "Blood pressure: 160/100 — flagged HIGH"
- "Weight change: -15kg in 3 months — flagged"
- "Missed last 2 appointments — flagged"

They don't interpret. They just measure and flag. The doctor (Stage 4) interprets later.

Stage 2 is that assistant. Four modules, each reading one data source, each producing a dict of signals keyed by `account_id`.

---

### Signal 1: Usage Signals (`usage_signals.py`)

**Source:** `usage_metrics.csv` — 6 months of monthly active users per account

**What it does:** Groups the 6 rows per account, sorts by month, then computes 3 things.

#### Computation 1: Month-over-Month (MoM) Change

```
users = [45, 42, 38, 31, 22, 8]
         Jan  Feb  Mar  Apr  May  Jun
                              ↑    ↑
                           prev  latest

MoM = (latest - prev) / prev × 100
    = (8 - 22) / 22 × 100
    = -63.6%
```

In code:
```python
users = group["active_users"].values   # [45, 42, 38, 31, 22, 8]
mom_change = (users[-1] - users[-2]) / users[-2] * 100
# users[-1] = last element = 8
# users[-2] = second to last = 22
```

**Flag:** `usage_drop_flag = mom_change < -30`
If last month dropped more than 30%, the flag fires.

```
-63.6% < -30%? YES → usage_drop_flag = True
```

#### Computation 2: 3-Month Slope

MoM only compares two points. What if the decline is gradual?

```
Account A: 50 → 49 → 48 → 47 → 46 → 45   (slow bleed, no single month > 30%)
Account B: 50 → 50 → 50 → 50 → 50 → 10   (sudden crash, MoM = -80%)
```

Account A would score 0 on MoM. But it's clearly declining. The slope catches it.

```python
last3 = users[-3:]   # last 3 months: [31, 22, 8]

# np.polyfit fits a straight line through the 3 points
# [0] gives the slope — how steeply the line goes up or down
slope = np.polyfit(range(3), last3, 1)[0]
# range(3) = [0, 1, 2] — the x-axis (month positions)
# last3    = [31, 22, 8] — the y-axis (user counts)
# slope = -11.5 means "losing ~11.5 users per month on average"
```

**Flag:** `three_month_slope < -0.5` fires in the scorer (not here — the signal just stores the number)

#### Computation 3: Near-Zero Usage

```python
near_zero_usage = latest <= 2
# If only 0, 1, or 2 people used the product last month → flag
```

#### Computation 4: SDK Version

```python
sdk = str(group.iloc[-1]["sdk_version"])
# .iloc[-1] = last row (most recent month)
# Grabs whatever SDK version they were on in the latest month
# e.g. "v3.1.0"
```

#### What Gets Stored

```python
UsageSignal(
    account_id="ACC-042",
    latest_active_users=8,
    mom_change_pct=-63.6,
    three_month_slope=-11.5,
    usage_drop_flag=True,    # -63.6 < -30
    near_zero_usage=False,   # 8 > 2
    sdk_version="v3.1.0",
)
```

---

### Signal 2: Support Signals (`support_signals.py`)

**Source:** `support_tickets.csv` — one row per ticket

**What it does:** Groups tickets by `account_id`, then counts and filters.

```python
# All tickets for this account
group = tickets for ACC-042

# Filter to P1 tickets only
p1 = group[group["priority"] == "P1"]
# group["priority"] == "P1" creates a True/False column
# group[...] keeps only the True rows

# Filter to open tickets
open_t = group[group["status"] == "open"]

# Filter to escalated tickets
escalated = group[group["status"] == "escalated"]

# Average resolution time
avg_res = group["resolution_time_hours"].mean()

# Has any P1 that is STILL open or escalated?
has_unresolved_p1 = len(p1[p1["status"].isin(["open", "escalated"])]) > 0
# .isin(["open", "escalated"]) = True if status is either of those two
```

#### Dry Run

```
Tickets for ACC-042:
  ticket-1: priority=P1, status=closed,    resolution_hours=48
  ticket-2: priority=P1, status=open,      resolution_hours=0
  ticket-3: priority=P2, status=closed,    resolution_hours=12
  ticket-4: priority=P1, status=escalated, resolution_hours=0

p1 = [ticket-1, ticket-2, ticket-4]   → p1_count = 3
open_t = [ticket-2]                   → open_tickets = 1
escalated = [ticket-4]                → escalated_tickets = 1
avg_res = (48 + 0 + 12 + 0) / 4 = 15.0 hours

has_unresolved_p1:
  P1 tickets with status open or escalated: [ticket-2, ticket-4]
  len > 0? YES → True
```

```python
SupportSignal(
    account_id="ACC-042",
    total_tickets=4,
    p1_count=3,
    open_tickets=1,
    escalated_tickets=1,
    avg_resolution_hours=15.0,
    has_unresolved_p1=True,
)
```

---

### Signal 3: NPS Signals (`nps_signals.py`)

**Source:** `nps_responses.csv` — one row per account

**What it does:** Two things — categorize the score, detect competitor names in the text.

#### NPS Category

```python
def _nps_category(score: int) -> str:
    if score in range(0, 7):    # 0,1,2,3,4,5,6
        return "detractor"
    if score in range(7, 9):    # 7,8
        return "passive"
    if score in range(9, 11):   # 9,10
        return "promoter"
```

This is the standard NPS industry definition. Not invented here.

#### Competitor Detection

```python
COMPETITOR_SIGNALS = [
    "strapi", "sanity", "contentful", "hygraph",
    "kontent.ai", "kontent", "builder.io",
    "wordpress", "drupal", "prismic",
]

def _detect_competitor(text: str) -> bool:
    text_lower = text.lower()
    # Check if ANY competitor name appears anywhere in the text
    return any(c in text_lower for c in COMPETITOR_SIGNALS)
```

#### Dry Run

```
NPS row for ACC-042:
  score: 4
  verbatim_comment: "We've been evaluating Contentful as a backup option"

_nps_category(4):
  4 in range(0,7)? YES → "detractor"

_detect_competitor("We've been evaluating Contentful as a backup option"):
  text_lower = "we've been evaluating contentful as a backup option"
  Is "strapi" in text? NO
  Is "sanity" in text? NO
  Is "contentful" in text? YES → return True
```

```python
NpsSignal(
    account_id="ACC-042",
    score=4,
    category="detractor",
    verbatim="We've been evaluating Contentful as a backup option",
    competitor_mentioned=True,
)
```

---

### Signal 4: Changelog Signals (`changelog_signals.py`)

**Source:** The `UsageSignal` objects already computed (specifically the `sdk_version` field)

This is the non-obvious insight module. It doesn't read a new CSV — it takes the SDK version already extracted from usage metrics and checks it against the deprecation rules.

#### SDK Deprecation Check

```python
def _is_deprecated(sdk_version: str) -> bool:
    v = sdk_version.lower().strip()
    # Regex: ^v?3[\.\-]
    # ^ = start of string
    # v? = optional "v" character
    # 3 = literal "3"
    # [\.\-] = followed by "." or "-"
    # Matches: v3.1.0, 3.1.0, v3.2, 3.0
    # Does NOT match: v4.3.1 (starts with 4, not 3)
    return bool(re.match(r"^v?3[\.\-]", v)) or v in ("v3.x", "3.x")
```

#### Days to Deadline

```python
def _days_to_deadline() -> int:
    deadline = datetime.strptime("2026-04-30", "%Y-%m-%d").date()
    return (deadline - date.today()).days
    # If today is April 27 2026: (April 30 - April 27) = 3 days
```

#### Dry Run

```
UsageSignal for ACC-042: sdk_version = "v3.1.0"

_is_deprecated("v3.1.0"):
  v = "v3.1.0"
  re.match(r"^v?3[\.\-]", "v3.1.0"):
    starts with "v"? yes (v? matches)
    followed by "3"? yes
    followed by "."? yes
  → MATCH → True

_days_to_deadline():
  deadline = April 30, 2026
  today    = April 27, 2026
  → 3 days
```

```python
ChangelogSignal(
    account_id="ACC-042",
    sdk_version="v3.1.0",
    is_deprecated=True,
    deprecation_deadline="2026-04-30",
    days_to_deadline=3,
    affected_features=[
        "REST API v2 (sunset April 30)",
        "SDK v3 security patches stop April 30"
    ],
)
```

Why is this non-obvious? Because this information only exists in the engineering changelog. No CSM knows their account is on v3.1.0. No Salesforce field tracks SDK version. This join — `usage_metrics.sdk_version × changelog.deprecated_versions` — exists nowhere else in the system.

### What Stage 2 Produces

```
BEFORE Stage 2:
  usage_df, tickets_df, nps_df (DataFrames)

AFTER Stage 2:
  usage_signals    = {"ACC-042": UsageSignal(...), "ACC-017": UsageSignal(...), ...}
  support_signals  = {"ACC-042": SupportSignal(...), ...}  ← only accounts with tickets
  nps_signals      = {"ACC-042": NpsSignal(...), ...}      ← only accounts with NPS
  changelog_signals = {"ACC-042": ChangelogSignal(...), ...}
```

Each is a dict: `account_id → signal object`. If an account has no tickets, it simply won't appear in `support_signals`. The scorer handles missing signals gracefully.

Stage 2 done. Baton passed to Stage 3.

---

## Stage 3: The LLM CSM Extractor

**Job: Read messy human-written call notes. Extract structured risk flags. This is where the LLM earns its place.**

### The Analogy

You have a stack of handwritten doctor notes. Some are neat. Some are scrawled. One is in Mandarin. Several have the patient's name spelled wrong. Your job: read each one and fill out a standardized form:

```
[ ] Competitor mentioned?
[ ] Budget cut mentioned?
[ ] Executive escalation?
[ ] Migration risk?
[ ] Renewal threatened?
[ ] Missed meetings: ___
[ ] Positive signal?
Confidence: ___
```

A keyword search can't do this. "They might look at Contentful" doesn't contain the word "competitor." "Their CTO jumped on the call" doesn't contain "exec_escalation." A human reads it and understands. The LLM does the same.

---

### The Raw Input

`csm_notes.txt` is one big text file. All notes for all accounts, separated by `---`:

```
3/18 - britepath - budget cut 20%, CTO leading CMS eval, missed last 2 QBRs
---
4/5 - Zenith Publishing - Renewl conversation started. They want a 30% discount
or they walk. Competitor POC with Kontent.ai apparently already underway.
CRO was cc'd on the last email thread.
---
march 25 -- meridian health -- priya
Good news/bad news. NPS came back at 8 but their actual usage has cratered.
Turns out they built a custom middleware layer...
---
[Note in Mandarin characters]
---
```

---

### Step 1: Split Into Chunks

```python
def _split_notes_into_chunks(raw_notes: str) -> list[str]:
    chunks = re.split(r"\n---+\n", raw_notes)
    # re.split splits on "---" (one or more dashes) surrounded by newlines
    # Returns a list of individual note strings
    return [c.strip() for c in chunks if c.strip()]
```

Result: a list of individual note strings, one per account.

---

### Step 2: Match Each Account to Its Chunk

For each account in the 90-day window, find which chunk belongs to it.

Three matching strategies, tried in order:

```python
def _find_best_chunk(account_name, account_id, chunks):

    # STRATEGY 1: Account ID match (most reliable)
    # Looks for "acct ACC-042" or "#ACC-042" in the chunk
    for chunk in chunks:
        if f"acct {account_id}" in chunk.lower():
            return chunk

    # STRATEGY 2: Exact name match
    # Looks for "brightpath solutions" in the chunk
    for chunk in chunks:
        if account_name.lower() in chunk.lower():
            return chunk

    # STRATEGY 3: First-word match (fallback)
    # "brightpath solutions" → first word = "brightpath"
    # Looks for "brightpath" anywhere in the chunk
    first_word = account_name.lower().split()[0]
    if len(first_word) > 4:   # only if first word is long enough to be distinctive
        for chunk in chunks:
            if first_word in chunk.lower():
                return chunk

    return None   # no match found
```

This is where the typo problem gets handled. The note says "britepath" but the account name is "BrightPath Solutions." Strategy 3 catches it: first word of "BrightPath Solutions" is "brightpath" → found in "britepath" chunk.

---

### Step 3: Send to LLM

For each account that has a matching chunk, one LLM call is made.

The prompt has three parts:

**Part 1: System prompt** — tells the LLM who it is and defines each signal precisely:
```
"competitor_mentioned: A NAMED competing platform appears.
 Vague phrases like 'other options' do NOT count."

"budget_cut_mentioned: Explicit mention of budget reduction.
 'Tight budget' alone is NOT enough."

"exec_escalation: A C-suite or VP-level person was on the call.
 A CSM flagging internally does NOT count."
```

Why so precise? Without these definitions, the LLM over-flags. "They mentioned they're watching costs" would trigger `budget_cut_mentioned`. The definitions prevent that.

**Part 2: Few-shot examples** — 3 annotated examples showing exactly what good extraction looks like:
- Example 1: High-risk note with explicit signals (easy case)
- Example 2: Subtle silent-churn — NPS looks fine but usage is cratering (hard case)
- Example 3: Expansion signal — model must NOT flag anything (negative case)

**Part 3: Chain-of-Thought requirement** — the model must fill a `reasoning` field FIRST, before any booleans:
```
"reasoning": "competitor_mentioned: 'Kontent.ai' named explicitly — TRUE.
budget_cut_mentioned: They want a discount but that's a negotiation tactic,
not a budget cut — FALSE.
exec_escalation: CRO cc'd on email thread — C-suite involvement — TRUE..."
```

Why CoT? Without it, the model pattern-matches on surface language. "They're evaluating options" → `migration_risk=True`. With CoT, the model has to cite the exact phrase. If it can't find one, it marks FALSE.

---

### Step 4: Validate the Response

The LLM returns JSON. It's validated against a Pydantic schema before being used:

```python
class _CsmExtraction(BaseModel):
    reasoning: str
    account_name_found: str
    competitor_mentioned: bool
    budget_cut_mentioned: bool
    exec_escalation: bool
    migration_risk: bool
    renewal_threatened: bool
    missed_meetings: int        # must be >= 0
    positive_signal: bool
    raw_summary: str
    confidence: float           # must be 0.0 to 1.0
```

If the JSON is malformed or missing fields, the call is retried (up to 3 times). If all retries fail, the account gets no CSM signal — it still gets scored on usage, support, NPS, and changelog.

---

### The Confidence Score

The LLM also outputs a confidence score (0.0–1.0) for the whole extraction:

```
1.0 → explicitly stated word-for-word ("We are evaluating Contentful")
0.8 → strongly implied with clear context ("their CTO asked about migration paths")
0.6 → plausible inference from indirect language
0.4 → weak signal, reading between the lines
```

The model is instructed to apply the LOWEST confidence that fits — err conservative.

This confidence number flows into Stage 4 scoring. Signals from a low-confidence extraction contribute less to the final score.

---

### Full Dry Run

```
Account: BrightPath Solutions (ACC-042)

Step 1 — Split notes into chunks:
  chunk-0: "3/18 - britepath - budget cut 20%, CTO leading CMS eval, missed last 2 QBRs"
  chunk-1: "4/5 - Zenith Publishing - ..."
  chunk-2: "march 25 -- meridian health -- ..."
  ...

Step 2 — Find chunk for BrightPath Solutions:
  Strategy 1 (ID match): "acct ACC-042" in any chunk? NO
  Strategy 2 (exact name): "brightpath solutions" in any chunk? NO
  Strategy 3 (first word): first word = "brightpath"
    Is "brightpath" in chunk-0 ("britepath")? 
    "britepath".contains("brightpath")? NO
    "brightpath".contains("britepath")? NO
    Wait — the check is: is first_word IN chunk.lower()
    first_word = "brightpath"
    chunk-0.lower() = "3/18 - britepath - budget cut..."
    "brightpath" in "3/18 - britepath - ..."? NO (it's "britepath" not "brightpath")
    
    Hmm — but "britepath" contains "bright"? Let's check:
    first_word = "brightpath"
    "brightpath" in "britepath"? NO — "britepath" is shorter
    
    Actually: the reconciler handles this BEFORE the LLM call.
    The reconciler uses rapidfuzz to match "britepath" → "brightpath solutions"
    and returns account_id ACC-042. Then _find_best_chunk uses the account_id
    to find the chunk. So Strategy 1 would find it if the note had "#ACC-042",
    or Strategy 2 finds it if the reconciler already matched the name.

Step 3 — LLM call with chunk-0:
  Prompt includes:
    - Signal definitions
    - 3 few-shot examples
    - "Account: BrightPath Solutions (ACC-042)"
    - "Note: 3/18 - britepath - budget cut 20%, CTO leading CMS eval, missed last 2 QBRs"

  LLM reasons through each signal:
    reasoning: "competitor_mentioned: No named competitor — FALSE.
    budget_cut_mentioned: '20% budget cut' explicitly stated — TRUE, confidence 1.0.
    exec_escalation: 'CTO leading CMS eval' — CTO is C-suite — TRUE, confidence 0.9.
    migration_risk: 'CMS eval' — evaluating alternatives — TRUE, confidence 0.8.
    renewal_threatened: Not explicitly stated — FALSE.
    missed_meetings: 'missed last 2 QBRs' — 2 missed meetings — 2.
    positive_signal: No expansion signals — FALSE.
    confidence: average of fired signals = (1.0 + 0.9 + 0.8) / 3 = 0.9"

  LLM returns:
    {
      "competitor_mentioned": false,
      "budget_cut_mentioned": true,
      "exec_escalation": true,
      "migration_risk": true,
      "renewal_threatened": false,
      "missed_meetings": 2,
      "positive_signal": false,
      "confidence": 0.9
    }

Step 4 — Stored as CsmSignal:
  CsmSignal(
      account_id="ACC-042",
      budget_cut_mentioned=True,
      exec_escalation=True,
      migration_risk=True,
      missed_meetings=2,
      confidence=0.9,
      raw_summary="BrightPath's CTO is leading a CMS evaluation while the company
                   has cut SaaS budgets 20% and missed 2 QBRs. This is a classic
                   pre-churn disengagement pattern."
  )
```

### What Stage 3 Produces

```
BEFORE Stage 3:
  csm_signals = {}

AFTER Stage 3:
  csm_signals = {
      "ACC-042": CsmSignal(budget_cut=True, exec_escalation=True, migration_risk=True, ...),
      "ACC-017": CsmSignal(renewal_threatened=True, competitor_mentioned=True, ...),
      "ACC-089": None  ← no matching note found
      ...
  }
```

Stage 3 done. Baton passed to Stage 4.

---

## Stage 4: The Scorer + Explainer

**Job: Combine all signals into a risk score (pure math), then ask the LLM to write the briefing (pure language).**

### The Analogy

A doctor reviews the assistant's standardized forms (all the signals). They add up the severity of each finding to get a risk level. Then they dictate a note to a medical writer who turns it into a readable patient summary.

The doctor doesn't write the summary — they provide the facts. The writer doesn't decide the risk level — they explain it.

That's Stage 4. Scorer = doctor. Explainer = medical writer.

---

### Part A: The Scorer (`risk_scorer.py`)

This is a pure function. Same input always produces the same output. No randomness. No LLM.

**The core loop:**

```python
score = 0.0
fired = []   # list of (label, weight) for every signal that fired

# Check each signal group in order
# If the signal fired → add its weight to score, record it in fired[]
```

#### Usage Signals

```python
if account.usage:
    u = account.usage

    if u.usage_drop_flag:           # MoM < -30%
        score += 2.5
        fired.append(("Usage dropped X% MoM", 2.5))

    if u.near_zero_usage:           # latest_users <= 2
        score += 3.0
        fired.append(("Near-zero active users (X)", 3.0))

    if u.three_month_slope < -0.5:  # declining trend
        score += 1.5
        fired.append(("Declining 3-month usage trend", 1.5))
```

#### Support Signals — with a cap

```python
if account.support:
    s = account.support

    if s.p1_count > 0:
        # CAP: at most 3 P1 tickets count toward the score
        # Without cap: 10 P1 tickets = 20 points, drowning everything else
        p1_contrib = min(s.p1_count, 3) * 2.0
        score += p1_contrib
        fired.append(("X P1 support tickets", p1_contrib))

    if s.has_unresolved_p1:
        score += 2.5
        fired.append(("Unresolved P1 ticket", 2.5))

    if s.escalated_tickets > 0:
        score += 1.5
        fired.append(("X escalated tickets", 1.5))
```

Why cap P1 tickets at 3? One account with 10 P1 tickets would score 20 points from tickets alone. That would make it look worse than an account with an explicit churn threat (4.0) + migration risk (3.0) + competitor (3.0) = 10.0. The cap prevents one noisy signal from dominating.

#### NPS Signals — two separate flags for score 4

```python
if account.nps:
    n = account.nps

    if n.score <= 6:        # detractor
        score += 2.0
        fired.append(("Low NPS score (X)", 2.0))

    if n.score <= 3:        # STRONG detractor — extra penalty
        score += 1.0
        fired.append(("Strong detractor NPS", 1.0))

    if n.competitor_mentioned:
        score += 3.0
        fired.append(("Competitor mentioned in NPS verbatim", 3.0))
```

A score of 4 fires `low_nps` (2.0) but NOT `detractor_nps` (1.0).
A score of 2 fires BOTH: 2.0 + 1.0 = 3.0 total from NPS score alone.

#### Changelog Signals

```python
if account.changelog and account.changelog.is_deprecated:
    score += 2.5
    fired.append(("On deprecated SDK vX.X", 2.5))

    if account.changelog.days_to_deadline <= 30:
        score += 1.5
        fired.append(("SDK deadline in X days", 1.5))
```

Two separate signals: being on a deprecated SDK (2.5), AND being within 30 days of the deadline (1.5). An account on v3.x with 60 days left gets 2.5. An account on v3.x with 6 days left gets 4.0.

#### CSM Signals — with confidence discount

```python
if account.csm:
    c = account.csm

    # If LLM was confident (>= 0.6): full weight
    # If LLM was uncertain (< 0.6): half weight
    conf_mult = 1.0 if c.confidence >= 0.6 else 0.5

    csm_checks = [
        (c.budget_cut_mentioned, "budget_cut",      2.0, "Budget cut mentioned"),
        (c.exec_escalation,      "exec_escalation",  2.0, "Executive escalation"),
        (c.migration_risk,       "migration_risk",   3.0, "Migration risk"),
        (c.renewal_threatened,   "renewal_threatened", 4.0, "Explicit churn threat"),
        (c.competitor_mentioned, "competitor_mentioned", 3.0, "Competitor in CSM notes"),
    ]

    for condition, key, weight, label in csm_checks:
        if condition:
            w = weight * conf_mult
            score += w
            fired.append((label, w))
```

Example:
```
migration_risk = True, confidence = 0.9
conf_mult = 1.0 (0.9 >= 0.6)
contribution = 3.0 × 1.0 = 3.0

migration_risk = True, confidence = 0.4
conf_mult = 0.5 (0.4 < 0.6)
contribution = 3.0 × 0.5 = 1.5
```

#### Competitor Deduplication

```python
# If NPS already fired competitor_mentioned, CSM won't fire it again
nps_competitor_already_fired = bool(account.nps and account.nps.competitor_mentioned)

# The CSM competitor check only fires if NPS didn't already fire it
(c.competitor_mentioned and not nps_competitor_already_fired, ...)
```

Why? Both NPS and CSM notes can mention a competitor. Without deduplication, the same fact (competitor mentioned) would add 3.0 + 3.0 = 6.0 points. That's double-counting.

#### Missed Meetings — also capped

```python
if c.missed_meetings > 0:
    # CAP: at most 2 missed meetings count
    miss_contrib = min(c.missed_meetings, 2) * 1.0 * conf_mult
    score += miss_contrib
    fired.append(("X missed meetings", miss_contrib))
```

#### Tier Assignment

```python
def _tier_from_score(score: float) -> RiskTier:
    if score >= 7.0:   return RiskTier.HIGH
    if score >= 3.5:   return RiskTier.MEDIUM
    return RiskTier.LOW
```

#### Top Signal

```python
# Sort all fired signals by weight, highest first
fired_sorted = sorted(fired, key=lambda x: x[1], reverse=True)

# The top signal is the one with the highest weight
top_signal = fired_sorted[0][0]   # e.g. "Explicit churn threat"

# Contributing signals is the full list, highest weight first
contributing = [label for label, _ in fired_sorted]
```

---

### Full Scoring Dry Run: BrightPath Solutions

```
Signals available:
  UsageSignal:     mom_change=-28.6%, slope=-4.0, usage_drop_flag=False, near_zero=False, sdk=v3.1.0
  SupportSignal:   p1_count=3, has_unresolved_p1=True, escalated=1
  NpsSignal:       score=4, competitor_mentioned=True
  ChangelogSignal: is_deprecated=True, days_to_deadline=3
  CsmSignal:       budget_cut=True, exec_escalation=True, migration_risk=True,
                   missed_meetings=2, confidence=0.9

Scoring:

  USAGE:
    usage_drop_flag: -28.6% < -30%? NO → skip
    near_zero_usage: 8 <= 2? NO → skip
    three_month_slope: -4.0 < -0.5? YES → +1.5
    fired: [("Declining 3-month usage trend", 1.5)]

  SUPPORT:
    p1_count=3 > 0: min(3,3) × 2.0 = 6.0 → +6.0
    has_unresolved_p1=True → +2.5
    escalated=1 > 0 → +1.5
    fired: [..., ("3 P1 tickets", 6.0), ("Unresolved P1", 2.5), ("1 escalated", 1.5)]

  NPS:
    score=4 <= 6 → +2.0
    score=4 <= 3? NO → skip
    competitor_mentioned=True → +3.0
    fired: [..., ("Low NPS (4)", 2.0), ("Competitor in NPS", 3.0)]

  CHANGELOG:
    is_deprecated=True → +2.5
    days_to_deadline=3 <= 30 → +1.5
    fired: [..., ("On deprecated SDK v3.1.0", 2.5), ("SDK deadline in 3 days", 1.5)]

  CSM (confidence=0.9, conf_mult=1.0):
    budget_cut=True → 2.0 × 1.0 = +2.0
    exec_escalation=True → 2.0 × 1.0 = +2.0
    migration_risk=True → 3.0 × 1.0 = +3.0
    renewal_threatened=False → skip
    competitor_mentioned=False (NPS already fired it) → skip
    missed_meetings=2: min(2,2) × 1.0 × 1.0 = +2.0
    fired: [..., ("Budget cut", 2.0), ("Exec escalation", 2.0), ("Migration risk", 3.0), ("2 missed meetings", 2.0)]

  TOTAL:
    1.5 + 6.0 + 2.5 + 1.5 + 2.0 + 3.0 + 2.5 + 1.5 + 2.0 + 2.0 + 3.0 + 2.0 = 29.5

  TIER: 29.5 >= 7.0 → HIGH RISK

  fired_sorted (by weight, highest first):
    ("3 P1 tickets", 6.0)
    ("Migration risk", 3.0)
    ("Competitor in NPS", 3.0)
    ("Unresolved P1", 2.5)
    ("On deprecated SDK v3.1.0", 2.5)
    ("Budget cut", 2.0)
    ("Exec escalation", 2.0)
    ("2 missed meetings", 2.0)
    ("Low NPS (4)", 2.0)
    ("1 escalated ticket", 1.5)
    ("SDK deadline in 3 days", 1.5)
    ("Declining 3-month trend", 1.5)

  top_signal = "3 P1 tickets"

RiskScore(
    account_id="ACC-042",
    raw_score=29.5,
    tier=HIGH,
    top_signal="3 P1 tickets",
    contributing_signals=["3 P1 tickets", "Migration risk", "Competitor in NPS", ...]
)
```

---

### Part B: The Explainer (`risk_explainer.py`)

The score is done. Now the LLM writes the briefing.

**What gets sent to the LLM:**

```
ACCOUNT CONTEXT:
  Name: BrightPath Solutions | Plan: Starter | ARR: $21,000 | Industry: Travel
  Days to renewal: 49 | CSM: Sarah Chen

RISK SCORE:
  Tier: High (score: 29.5)
  Top signal: 3 P1 tickets
  All signals: 3 P1 tickets, Migration risk, Competitor in NPS, ...

SIGNAL DETAIL:
  Usage: Active users: 8 | MoM: -28.6% | 3-month slope: -4.0
  SDK: DEPRECATED — on v3.1.0, security patches stop 2026-04-30 (3 days from now)
  NPS score: 4 (detractor)
  NPS verbatim: "We've been evaluating Contentful as a backup option"
  CSM notes summary: BrightPath's CTO is leading a CMS evaluation while the
                     company has cut SaaS budgets 20% and missed 2 QBRs.
```

**Quality bar enforced in the prompt:**
- Must name at least one specific signal
- Must explain the business consequence, not just restate the data
- Recommended action must have WHO + WHAT + WHEN
- No filler phrases ("multiple signals", "various issues")
- No generic advice ("schedule a call to check in")

**What the LLM returns:**

```json
{
  "signal_narrative": "The story here is compounding: P1 tickets from SDK v3.1.0
    deprecation are driving engineering pain, which is why the CTO is now leading
    a CMS evaluation. The NPS verbatim naming Contentful confirms the evaluation
    is real. With 3 days until security patches stop and 49 days to renewal,
    this is the most urgent account in the window.",

  "explanation": "BrightPath's 3 P1 tickets are all linked to their v3.1.0 SDK,
    which loses security patches in 3 days — the same week their contract renews.
    Their CTO is personally leading a CMS evaluation, their NPS verbatim names
    Contentful, and they've missed 2 QBRs this quarter.",

  "recommended_action": "Sarah Chen should escalate to VP of CS today and request
    a joint engineering call with BrightPath's CTO this week — bring a concrete
    v3→v4 migration timeline and offer a dedicated SA for the sprint. Do not let
    this reach renewal without a written commitment."
}
```

**If the LLM fails:** Tier-appropriate generic fallback text is used. The pipeline always produces a complete report.

```python
_FALLBACK_EXPLANATIONS = {
    "High":   ("Multiple compounding risk signals detected...", "Schedule executive call..."),
    "Medium": ("Moderate risk signals present...", "Proactive outreach..."),
    "Low":    ("Account appears stable...", "Maintain regular check-in..."),
}
```

**Low-tier accounts skip the LLM entirely:**

```python
if score.tier in (RiskTier.HIGH, RiskTier.MEDIUM):
    report = generate_risk_report(account, score)   # LLM call
else:
    report = RiskReport(
        explanation="Low risk — no significant signals detected.",
        recommended_action="Maintain regular check-in cadence.",
        ...
    )
```

Why? Low-risk accounts don't need a custom briefing. Skipping the LLM call saves ~2-3 seconds per account and reduces API cost.

---

### What Stage 4 Produces

```
BEFORE Stage 4:
  AccountRecord with all 5 signals attached

AFTER Stage 4:
  RiskReport(
      account_id="ACC-042",
      account_name="BrightPath Solutions",
      arr=21000.0,
      days_to_renewal=49,
      tier=HIGH,
      raw_score=29.5,
      top_signal="3 P1 tickets",
      contributing_signals=["3 P1 tickets", "Migration risk", ...],
      explanation="BrightPath's 3 P1 tickets are all linked to their v3.1.0 SDK...",
      recommended_action="Sarah Chen should escalate to VP of CS today...",
      csm_name="Sarah Chen",
      plan_tier="Starter",
      industry="Travel & Hospitality",
  )
```

This is the final output. All reports are sorted by `raw_score` descending (highest risk first) and sent to the Streamlit dashboard.

---

## The Complete Flow — One Page Summary

```
INPUT: 5 raw files (accounts.csv, usage_metrics.csv, support_tickets.csv,
                    nps_responses.csv, csm_notes.txt)

STAGE 1 (Loader & Filter)
  pd.read_csv() on all 4 CSVs → clean DataFrames
  csm_notes.txt → raw string
  Filter accounts: keep only those renewing in next 90 days
  Result: ~30 accounts in window

STAGE 2 (Deterministic Signals — no LLM)
  usage_signals.py:
    groupby account_id → sort by month → compute MoM, slope, flags, SDK version
    Result: dict[account_id → UsageSignal]

  support_signals.py:
    groupby account_id → count P1s, open, escalated, avg resolution
    Result: dict[account_id → SupportSignal]

  nps_signals.py:
    per row → map score to category → keyword search verbatim for competitors
    Result: dict[account_id → NpsSignal]

  changelog_signals.py:
    per UsageSignal → regex check sdk_version against v3.x pattern → days to deadline
    Result: dict[account_id → ChangelogSignal]

STAGE 3 (LLM CSM Extraction)
  Split csm_notes.txt on "---" → list of chunks
  For each account in window:
    Find matching chunk (ID match → exact name → first-word fallback)
    If found: send chunk to LLM with CoT + few-shot prompt
    LLM returns: {competitor, budget_cut, exec_escalation, migration_risk,
                  renewal_threatened, missed_meetings, confidence}
    Validate with Pydantic → store as CsmSignal
  Result: dict[account_id → CsmSignal]

STAGE 4 (Score + Explain — per account)
  For each account in window:
    Build AccountRecord (attach all 5 signals)
    score_account():
      Loop through all signals
      If signal fired → add weight to score
      Special rules: P1 cap at 3, missed meetings cap at 2,
                     CSM confidence discount, competitor deduplication
      Sort fired signals by weight → top_signal
      score >= 7.0 → HIGH, >= 3.5 → MEDIUM, else LOW
    If HIGH or MEDIUM:
      generate_risk_report() → LLM writes explanation + action
    Else:
      Use generic fallback text
  Sort all reports by raw_score descending

OUTPUT: list[RiskReport] → Streamlit dashboard
  Each report: account_name, tier, score, top_signal, contributing_signals,
               explanation, recommended_action, arr, days_to_renewal, csm_name
```

---

## Quick Reference: All Signal Weights

```
Signal                    Weight   Source    Fires when
──────────────────────────────────────────────────────────────────────
renewal_threatened          4.0    CSM/LLM   Explicit churn threat in notes
migration_risk              3.0    CSM/LLM   Actively evaluating competitors
competitor_mentioned        3.0    NPS/CSM   Named competitor in text
near_zero_usage             3.0    Usage     <= 2 active users latest month
usage_drop_flag             2.5    Usage     >30% MoM active user decline
deprecated_sdk              2.5    Changelog On SDK v3.x (deadline April 30)
unresolved_p1               2.5    Support   Open P1 support ticket
p1_tickets                  2.0    Support   Per P1 ticket (capped at 3)
low_nps                     2.0    NPS       NPS score <= 6
budget_cut                  2.0    CSM/LLM   Budget reduction in notes
exec_escalation             2.0    CSM/LLM   C-suite on call
escalated_tickets           1.5    Support   Escalated support tickets
negative_mom_trend          1.5    Usage     Declining 3-month slope
sdk_deadline_30d            1.5    Changelog SDK deadline within 30 days
detractor_nps               1.0    NPS       NPS <= 3 (strong detractor)
missed_meetings             1.0    CSM/LLM   Per missed meeting (capped at 2)
──────────────────────────────────────────────────────────────────────
Tier thresholds:  HIGH >= 7.0  |  MEDIUM >= 3.5  |  LOW < 3.5
CSM confidence < 0.6 → signals get 50% weight
P1 tickets capped at 3 contributions (max 6.0 points)
Missed meetings capped at 2 contributions (max 2.0 points)
Competitor mention deduplicated: NPS and CSM can't both fire it
```

---

## Quick Reference: Where Rules Are Used vs LLM

```
RULES (deterministic Python)          LLM (GPT-4o-mini)
─────────────────────────────────     ──────────────────────────────────────
MoM change calculation                Parse CSM notes → structured signals
3-month slope (numpy polyfit)         Extract changelog deprecations
Usage drop flag (< -30%)              Write plain-English risk briefing
Near-zero usage flag (<= 2)           Write recommended action
P1 ticket counting                    (that's it — 3 things only)
Unresolved P1 detection
Escalated ticket counting
NPS category (detractor/passive/promoter)
Competitor keyword search in NPS
SDK version regex check
Days to deadline calculation
Weighted sum scoring
Tier assignment (>= 7.0 = HIGH)
Top signal selection (sort by weight)
90-day window filter
```

The rule: **LLMs parse language. Rules compute numbers. Never swap them.**

---

## Quick Reference: What Each File Does

```
pipeline/__init__.py          run_pipeline() — orchestrates all 4 stages
pipeline/ingestion/loader.py  pd.read_csv() + clean for all 5 files
pipeline/ingestion/reconciler.py  rapidfuzz fuzzy name matching
pipeline/signals/usage_signals.py    MoM, slope, drop flag, SDK version
pipeline/signals/support_signals.py  P1 count, unresolved, escalated
pipeline/signals/nps_signals.py      Category, competitor keyword search
pipeline/signals/changelog_signals.py  SDK regex check, days to deadline
pipeline/llm/llm_client.py    Single LLM gateway — retry, Pydantic validation
pipeline/llm/csm_extractor.py  CoT + few-shot prompt → CsmSignal
pipeline/llm/changelog_extractor.py  Changelog prose → DeprecationEvent list
pipeline/llm/risk_explainer.py  Score + signals → explanation + action
pipeline/scoring/risk_scorer.py  Pure function: AccountRecord → RiskScore
pipeline/scoring/weight_config.py  All weights, thresholds, config constants
models/signals.py             Pydantic models for all 5 signal types
models/account.py             AccountRecord, RiskScore, RiskReport
app/streamlit_app.py          Dashboard — cached pipeline, filters, drill-down
```
