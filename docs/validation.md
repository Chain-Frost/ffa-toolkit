# Sanitized baseline validation

25 November 2026. Local Windows validation, separate isolated CPython environments:

| Check | Python 3.11.16 | Python 3.14.6 |
| --- | --- | --- |
| Ruff 0.16.6 lint and format | Passed | Passed |
| Strict Pyright 1.1.414 | 0 errors/warnings | 0 errors/warnings |
| Full applicable pytest | 680 passed | 680 passed |
| Wheel and sdist, distribution content, clean installed CLI | Passed | Passed |

The independent Codex reviewer checked the complete sanitized working tree, ran Ruff/format,
strict Pyright and the full 3.14 suite (676 tests before four additional custody cases), then independently
verified all eight synthetic adapter/custody tests. Substantive findings on identities, unadmitted
third-party data, stale CLI guidance and lost reader coverage were corrected and reviewed.
Scenario numeric observations/controls and report tables were independently checked unchanged after
neutralizing identities. Fixed-width fixture whitespace is preserved through narrow Git attributes.

Gitleaks 8.30.1 repository-wide scan found no leaks. Additional bounded patterns checked identifying
project names, station identifiers, local paths, email addresses, source-private URLs and restricted
sample-derived outputs; synthetic test addresses are deliberately non-routable. Pattern scanning and
independent review do not establish rights for excluded artifacts; unresolved third-party corpora,
textbook samples/derived fits and restricted publications are excluded entirely.

These are local results. Hosted execution must be checked separately, and private branch protection
requires an account plan supporting it. The prepared backlog is inactive until coordinated cutover.
Native qualification remains pending; mathematical checks and report-internal replay are not native
Flike equivalence. No package release, public visibility or final cutover is authorized by these results.
