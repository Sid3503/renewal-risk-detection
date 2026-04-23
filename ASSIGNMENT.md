# Renewal Intelligence Engine — Take-Home Assignment

## Role: Applied AI Engineer, BizOps

**Time Budget:** 6–8 hours (submit within 4 days)

---

## Context

Contentstack's BizOps team sits at the intersection of Sales, CS, Finance, and Product. One recurring pain point is **renewal risk** — every quarter, the team scrambles to figure out which accounts are likely to churn or downgrade, often relying on gut feel, scattered Salesforce notes, and last-minute Slack threads.

You've been given a synthetic but realistic dataset representing one quarter's worth of account signals. Your job is to build a working prototype that helps the BizOps team **identify at-risk renewals and explain why** — before the account team has to ask.

---

## What You're Given

This ZIP contains 5 data files:

1. **`accounts.csv`** — 120 accounts with firmographic and contract details
2. **`usage_metrics.csv`** — Monthly product usage for the past 6 months
3. **`support_tickets.csv`** — Support ticket history
4. **`csm_notes.txt`** — Unstructured CSM (Customer Success Manager) call notes. These are messy by design.
5. **`nps_responses.csv`** — Recent Net Promoter Score survey responses
6. **`changelog.md`** — Contentstack product changelog for the past 2 quarters

---

## Your Task

Build a **Renewal Risk Intelligence tool** that:

1. **Ingests all the above data sources** and reconciles them (note: there are inconsistencies — account names don't always match IDs cleanly)

2. **Produces a risk-scored list** of all accounts renewing in the next 90 days, with a clear risk tier (e.g., High / Medium / Low)

3. **Generates a plain-English explanation** for each at-risk account — what signals contributed, and what action the account team should consider

4. **Surfaces at least one non-obvious insight** that a simple rule-based system would miss (this is deliberately vague — show us how you think)

---

## Requirements

- Use **Python** as the primary language
- You **must** use at least one LLM API (OpenAI, Anthropic, Mistral, open-source or any LLM of your choice) as a meaningful part of your pipeline, not just a gimmick
- Provide a **README.md** explaining your approach, tradeoffs you made, what you'd do with more time, and what you'd change for production
- Include a way to **run and demo** your solution (CLI, Jupyter notebook, Streamlit app — your call)
- Create a recording in which you clearly explain and walk through your architecture (not a presentation or PPT), and provide links to both the video and the GitHub repository as your final deliverable.

---

## What We're Explicitly NOT Telling You

- We haven't defined what "at risk" means. You decide.
- The CSM notes are messy for a reason. How you handle noise is part of the test.
- Some data is contradictory. How you resolve conflicting signals matters.
- The changelog is included for a reason — but we won't tell you what. Figure it out.
- Not all data is in English.

---

## Evaluation Criteria

We're looking for:

- **Technical skill** — Can you wrangle messy, multi-modal data?
- **AI fluency** — Do you use LLMs thoughtfully (not just as a parlor trick)?
- **Product sense** — Do you understand what's actually useful for a BizOps team?
- **Critical thinking** — Can you make judgment calls with incomplete information?
- **Communication** — Can you explain your approach clearly?
- **Learn-it-all attitude** — Are you willing to try something new and be honest about it?

Good luck. We're excited to see what you build.

- A