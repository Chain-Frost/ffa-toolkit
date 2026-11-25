# WP09 legacy conformance qualification and release gate

**Target:** TUFLOW/BMT Flike 5.0.306.0, file format 3.10 (`tuflow-bmt-flike-5.0.306.0-file3.10`). **Current decision: BLOCKED.** This document and its ledgers are an initial, independently maintained qualification framework, **not** an assertion that any statistical method replicates Flike.

## Files and ownership

- `qualification/legacy-matrix.json`: minimum known coverage rows with **seven separate axes**: family, estimator, observations, prior mode, zero threshold, rating-error setting and output. Rows distinguish `required`, `unresolved` and independently evidenced `target_unsupported` admission. The current 40 rows are a **provisional lower bound**, not a frozen complete family-by-option cross-product. Removing baseline row IDs or required support questions is rejected by the checker; any owner-approved scope correction needs a reviewed code-and-matrix change. WP09 owns it; method agents supply version-pinned evidence and propose expansions.
- `qualification/evidence.json`: native-reference admissions, independent mathematical checks, synthetic repeatability, method documentation, independent reviews, platform/package/CLI evidence and owner decisions. The two historical staged cases are **candidates**, not benchmarks. A qualified native rerun gets a **new** ID with an exact compatibility profile, executable/file versions, executable/input/settings/output hashes, native run log and independent admission URL. The gate retains the validated run record and binds every comparison to that run's **exact reference input, settings and output SHA-256 hashes and profile/version**. It does not alter the immutable candidate manifest.
- `scripts/qualification_gate.py`: standard-library-only evidence validator and fail-closed release gate. It does not call any statistical implementation, run proprietary executables or publish packages.
- `tests/test_qualification_gate.py`: policy tests, including a constructed synthetic pass case to demonstrate the gate is reachable. Synthetic test evidence is **never** actual legacy acceptance evidence.

## Invocation

From the repository root, using the Python 3.14 tooling established by merged WP01 PR #14:

```console
python scripts/qualification_gate.py --validate-only
python scripts/qualification_gate.py --report qualification/release-readiness.generated.md
python -m pytest tests/test_qualification_gate.py
ruff check scripts/qualification_gate.py tests/test_qualification_gate.py
ruff format --check scripts/qualification_gate.py tests/test_qualification_gate.py
pyright
```

`--validate-only` exits **0** for a structurally valid *pending* matrix, **2** for broken/misrepresented evidence. The normal release gate exits **0** only when ready, **1** when valid but blocked, **2** when invalid. The generated report always names its pending rows and separate release checks. The reviewer must also check the evidence artifacts: **this script verifies ledger claims and explicit numerical comparisons, not the contents of remote URLs or the independence of reviewers**.

## Requirements before setting a row to qualified

1. Establish whether the exact family/estimator/observation/prior/zero/rating/output tuple is supported by the pinned legacy build. Expand the provisional rows into **each required supported tuple**. Unsupported combinations need objective source evidence, a recorded scope decision and `target_support` proof; merely removing the row is prohibited. Resolve all outstanding support questions.
2. For each row, attach independent mathematical evidence, a *qualified* native comparison, suitable method documentation and independent review. Bayesian and other stochastic cases additionally require repeated synthetic experiments, with a separately justified precision basis. CLI/report rows require installed replay evidence. Evidence IDs must identify the exact matrix row; reusing proof from a different combination is invalid.
3. Qualified native comparison evidence must identify a **separately admitted native-run ID**, the same compatibility profile, executable and file versions, source revision, **matching native reference input/settings/output hashes**, report-precision rationale and **one or more actual numerical comparisons**. The evidence record's `artifact_sha256` identifies the comparison artifact and is *not* the native-reference `output_sha256`. An archived candidate ID is never a qualified run: editing `pairing_status`, `status` or `flike_qualified` cannot promote it, and a native-run admission may not reuse its ID. RMC/FlowFreq candidates and mathematical vectors alone also cannot qualify a comparison. Supported comparator policies are:
   - `deterministic`: explicitly justified absolute plus relative tolerances (no universal percentage).
   - `printed_rounding`: absolute difference at most half the recorded **last printed unit**, which must reflect output-specific report precision.
   - `stochastic`: difference no greater than the explicitly justified multiplier times combined reported standard error, with at least two reference and candidate repeats. This is numerical repeatability, not hydrological parameter uncertainty.
4. Freeze the completed legacy support matrix, record the owner's scope-approval URL and pin the candidate **release source commit**. Evidence used for passing checks must come from that exact revision. No unresolved correctness defect may remain in the claimed scope.
5. Pass Ruff, format, strict Pyright, unit/integration and mathematical tests; build and inspect wheel/sdist; verify **installed** `ffa` on Windows and Linux Python 3.14 outside the checkout; replay qualified cases; verify packaged offline method documentation and acquire independent statistical review. Owner release authorisation is a separate final gate and is **not** granted by this repository tool.

Keep source artifacts, a source hash ledger, native run settings, raw output, independent calculations, scripts to reproduce comparisons and per-output numerical precision decisions as immutable evidence. Differences must be classified **observation/model/default/algorithm/rounding/stochastic/unknown**, not repaired by modifying the reference. Recording `passed` without independent evidence, distinct native admission and matching reference-run custody is not qualification. The gate validates the manifest, ledger and declared comparisons, but it cannot itself verify the contents of remote evidence links or replace independent native source inspection. The committed ledger deliberately has no admitted native runs and remains structurally valid but release-blocked (0/40 rows, 0/12 final checks); artificially promoting candidates is an invalid ledger (exit 2), not a release workaround.

## Initial release-readiness report (23 November 2026)

| Condition | Initial status |
| --- | --- |
| Frozen and owner-approved full capability/combination matrix | **BLOCKED** — 40 provisional rows; all pending; family/option support still unresolved |
| Verified native reference admission | **BLOCKED** — Case A and Case B are `candidate_settings_unverified`; no matched controlled native rerun |
| Independently verified legacy numerical outputs | **BLOCKED** — zero qualified rows; no proven reference conformance |
| Independent maths, synthetic repeats, numerical precision and independent review | **BLOCKED** — no evidence admitted yet |
| Packaging, installed CLI, Windows/Linux, documentation and source-specific CI | **BLOCKED** — WP01 PR #14 has merged and its own Windows/Linux CI passed; WP09's exact-branch checks and WP02–WP08 remain prerequisites |
| Owner acceptance of scope and release | **NOT AUTHORISED** — no publication or merge implied |

The two staged legacy candidates remain in `reference/legacy-candidates/`; they are **not** included in wheel or source archives. WMAwater future methods, full POT fitting and any GUI/TUI/browser application are outside this MVP. Completing this framework **does not close issue #12 or umbrella #2**. Track remaining package dependencies under [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md), and update this readiness report after evidence has been admitted and independently reviewed.