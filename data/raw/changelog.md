# Contentstack Product Changelog

## Q4 2025

### v4.3.0 — December 15, 2025
**New Features**
- Agent OS (Beta): Introduced AI-powered content agents for automated content operations. Available to Enterprise tier customers in the beta program.
- Composable Architecture: New modular stack configuration allowing customers to plug in best-of-breed services.
- Enhanced Locale Management: Complete rewrite of locale fallback logic. Fixed edge cases with nested fallback chains.

**Improvements**
- GraphQL API: 3x faster query execution for entries with 50+ content types.
- Management Console: Redesigned dashboard with real-time analytics.
- Webhooks: Added retry logic with exponential backoff (max 5 retries).

**Deprecations**
- ⚠️ REST Content Delivery API v2 endpoints will be sunset on March 31, 2026. All customers on SDK v3.x must migrate to v4.x before this date. See migration guide: docs.contentstack.com/migration/v3-to-v4
- Legacy Workflow Engine (v1) is now deprecated. Existing workflows will continue to function but cannot be edited after February 28, 2026.

---

### v4.2.3 — November 1, 2025
**Bug Fixes**
- Fixed: Locale fallback returning null for entries with circular references (affects SDK v4.0.0 and v4.1.0 only)
- Fixed: Bulk publish timeout for libraries with >10,000 entries
- Fixed: SSO SAML assertion parsing for certain IdP configurations

**Security**
- Patched XSS vulnerability in custom field renderer (CVE-2025-8834)
- Added support for FIPS 140-2 compliant encryption (Enterprise tier only)

---

### v4.2.0 — October 15, 2025
**New Features**
- Visual Builder: New WYSIWYG page builder for non-technical users. Note: Requires SDK v4.1.0 or later.
- Content Scheduling: Time-based publish/unpublish with timezone support.

**Breaking Changes**
- 🔴 SDK v4.2.0+ changes the response envelope format for the Content Delivery API. The `entry` wrapper is now `data`. Applications using `response.entry` must update to `response.data`. This is a BREAKING CHANGE for all SDK versions below v4.2.0.
- Webhook payload format v2 is now the default. v1 payloads must be explicitly requested via header `X-Webhook-Version: 1`.

---

## Q1 2026

### v4.3.1 — January 20, 2026
**Improvements**
- Agent OS (Beta): Added support for custom agent creation with natural language instructions.
- Performance: 40% reduction in publish latency for workflows with >5 steps.
- SDK v4.3.0: Added TypeScript types for all API responses.

**Bug Fixes**
- Fixed: New editor crash on Safari 17.x when editing rich text fields with embedded assets
- Fixed: Role permission inheritance not applying correctly for nested stacks
- Fixed: Audit log gaps for bulk operations performed via Management API

**Known Issues**
- Content scheduling may show incorrect times for users in UTC+13/UTC+14 timezones (fix planned for v4.3.2)

---

### v4.3.2 — March 1, 2026
**Critical Update**
- 🔴 REST Content Delivery API v2 sunset date EXTENDED to April 30, 2026 (was March 31). Final extension — no further delays.
- Fixed: Timezone bug in content scheduling (all timezones now supported)
- Fixed: GraphQL query timeout for deeply nested references (>8 levels)

**Deprecations**
- ⚠️ Legacy editor will be fully removed in v4.4.0 (expected May 2026). All customers must migrate to the new editor. Migration tool available at: tools.contentstack.com/editor-migration
- ⚠️ SDK v3.x will stop receiving security patches after April 30, 2026. Strongly recommend upgrading to v4.2.3 or later.

**Security**
- Patched privilege escalation vulnerability in Management API (CVE-2026-1102). All customers should update to v4.3.2 immediately.

---

### Upcoming — v4.4.0 (Expected May 2026)
**Preview**
- Agent OS General Availability: Full production release of AI content agents.
- Marketplace v2: New app framework with improved sandboxing.
- Legacy editor removal (see deprecation notice above)
- Enhanced audit logging with real-time streaming to external SIEM tools
