"""
Extract structured deprecation events from changelog.md using an LLM.

Used to populate a machine-readable deprecation registry cross-referenced
against account SDK versions during pipeline initialization.

Prompting strategy:
  - CoT: model scans each section for deprecation markers before extracting
  - Few-shot: 1 example showing a changelog snippet and expected structured output
  - Severity rubric anchors critical vs warning vs info decisions
"""
import logging
from pydantic import BaseModel, Field
from pipeline.llm.llm_client import call_llm

logger = logging.getLogger(__name__)


class DeprecationEvent(BaseModel):
    """A single deprecation or sunset event from the changelog."""
    feature: str
    affected_versions: list[str]
    deadline: str
    severity: str
    description: str
    customer_action_required: str


class ChangelogDeprecations(BaseModel):
    """All deprecation events extracted from the changelog."""
    reasoning: str
    events: list[DeprecationEvent] = Field(default_factory=list)


_CHANGELOG_SYSTEM = """You are a technical analyst reading a SaaS product changelog to identify
every deprecation notice, sunset date, and breaking change that requires customer action.

SEVERITY RUBRIC — apply this exactly:
- critical: Security risk, data loss risk, or hard cutoff with NO fallback after the deadline.
  Example: "SDK v3 security patches stop April 30" = critical.
- warning: Feature removal or API version sunset where migration is required but not security-impacting.
  Example: "Legacy editor removed in May 2026" = warning.
- info: Behavior change with backwards compatibility or soft deprecation with no deadline yet.
  Example: "Webhook payload v1 must be explicitly requested" = info.

WHAT TO EXTRACT — only include events with at least one of:
  1. An explicit deadline date
  2. A version number being retired
  3. A customer migration action required

DO NOT extract: new features, bug fixes, performance improvements, or known issues without deprecation language.

CHAIN-OF-THOUGHT REQUIREMENT:
Fill `reasoning` first: scan each changelog section and list every deprecation marker you found
(look for words: deprecated, sunset, removed, breaking, end-of-life, stop receiving, must migrate).
Then build the `events` array from that list.

Return ONLY valid JSON — no markdown, no preamble."""


_FEW_SHOT_EXAMPLE = """
=== EXAMPLE (study before extracting the real changelog) ===

Input snippet:
```
### v4.2.0 — October 15, 2025
**Breaking Changes**
- 🔴 SDK v4.2.0+ changes the response envelope format. The `entry` wrapper is now `data`.
  Applications using `response.entry` must update to `response.data`.
- Webhook payload format v2 is now the default. v1 payloads must be explicitly requested
  via header `X-Webhook-Version: 1`.

**Deprecations**
- ⚠️ REST Content Delivery API v2 will be sunset on March 31, 2026.
  All customers on SDK v3.x must migrate to v4.x. See migration guide.
```

Expected output:
{
  "reasoning": "Scanning v4.2.0 section: found 2 breaking changes and 1 deprecation. Breaking change 1: response envelope change — affects all SDK versions below v4.2.0, no deadline given but breaking immediately. Breaking change 2: webhook v1 not deprecated, just demoted to opt-in — info only. Deprecation: REST API v2 sunset March 31 2026 — explicit deadline, migration required — critical.",
  "events": [
    {
      "feature": "Content Delivery API response envelope (response.entry → response.data)",
      "affected_versions": ["v4.0.0", "v4.1.0", "below v4.2.0"],
      "deadline": "",
      "severity": "warning",
      "description": "Applications using response.entry must update to response.data — breaking change in SDK v4.2.0.",
      "customer_action_required": "Update all code referencing response.entry to use response.data before upgrading to SDK v4.2.0."
    },
    {
      "feature": "REST Content Delivery API v2",
      "affected_versions": ["v3.0", "v3.1", "v3.2", "v3.x"],
      "deadline": "2026-03-31",
      "severity": "critical",
      "description": "REST API v2 will be sunset on March 31 2026 — all SDK v3.x customers must migrate to v4.x.",
      "customer_action_required": "Migrate from SDK v3.x to v4.x before March 31 2026 using the provided migration guide."
    }
  ]
}

=== END EXAMPLE ===
"""


def extract_deprecations(changelog_text: str) -> ChangelogDeprecations:
    """
    Parse changelog.md and return all deprecation/sunset events as structured data.
    Falls back to hardcoded known deprecations if LLM fails.
    """
    prompt = f"""{_FEW_SHOT_EXAMPLE}

=== NOW EXTRACT FROM THE REAL CHANGELOG ===

{changelog_text}

Step 1 — Fill `reasoning`: list every section you scanned and every deprecation marker you found.
Step 2 — Build `events` array with one entry per distinct deprecation/breaking change.
         If a single announcement covers multiple versions, list all affected versions explicitly.
         If deadline was extended (e.g. from March 31 to April 30), use the LATEST date.

Return JSON:
{{
  "reasoning": "...",
  "events": [
    {{
      "feature": "...",
      "affected_versions": ["..."],
      "deadline": "YYYY-MM-DD or empty string",
      "severity": "critical or warning or info",
      "description": "one sentence",
      "customer_action_required": "one sentence describing what the customer must do"
    }}
  ]
}}"""

    result = call_llm(prompt, ChangelogDeprecations, _CHANGELOG_SYSTEM)
    if result is not None:
        logger.info(
            "Extracted %d deprecation events from changelog (reasoning length: %d chars)",
            len(result.events),
            len(result.reasoning),
        )
        return result

    logger.warning("Changelog LLM extraction failed — using hardcoded fallback")
    return ChangelogDeprecations(
        reasoning="LLM extraction failed — using hardcoded fallback from known changelog contents.",
        events=[
            DeprecationEvent(
                feature="REST Content Delivery API v2",
                affected_versions=["v3.0", "v3.1", "v3.2", "v3.x"],
                deadline="2026-04-30",
                severity="critical",
                description="API v2 sunset on April 30 2026 — final extension, no further delays.",
                customer_action_required="Migrate from SDK v3.x to v4.x before April 30 2026.",
            ),
            DeprecationEvent(
                feature="SDK v3.x security patches",
                affected_versions=["v3.0", "v3.1", "v3.2", "v3.x"],
                deadline="2026-04-30",
                severity="critical",
                description="SDK v3.x stops receiving security patches after April 30 2026.",
                customer_action_required="Upgrade to SDK v4.2.3 or later before April 30 2026.",
            ),
            DeprecationEvent(
                feature="Legacy Workflow Engine v1",
                affected_versions=["workflow_v1"],
                deadline="2026-02-28",
                severity="warning",
                description="Legacy workflows can no longer be edited after February 28 2026.",
                customer_action_required="Migrate existing workflows to the new Workflow Engine before Feb 28.",
            ),
            DeprecationEvent(
                feature="Legacy Editor",
                affected_versions=["pre-v4.4"],
                deadline="2026-05-01",
                severity="warning",
                description="Legacy editor fully removed in v4.4.0 (May 2026).",
                customer_action_required="Complete migration to the new editor using tools.contentstack.com/editor-migration.",
            ),
        ],
    )
