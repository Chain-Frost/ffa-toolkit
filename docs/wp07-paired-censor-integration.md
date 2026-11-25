# WP07 / #31 — Audited ARR-style PILF replacement as a likelihood view

**Partial mathematical/record adapter only; not a qualified Flike 5.0.306.0
likelihood or a CLI fitting mode.** Parent [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md);
input ownership [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md);
likelihood ownership [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md).
See [existing source/ARR guidance](wp07-low-flow-evidence.md).

## Two separate representations, no duplicate event opportunities

ARR Book 3 §2.8.6.4's editor operation has **two distinct record changes**:
selected original annual flows are marked excluded, and a separately stated
count-only censor group is entered with **synthetic negative years** to satisfy
the editor's non-overlap condition. Those negative years are not historical
flood opportunities. The count group has **k at-or-below, zero above**.

WP07 `propose_arr_mgb_censoring` and `apply_approved_arr_mgb_censoring`
preserve this source/editor representation. They are not fed straight to the
WP05 likelihood: its `exposure_intervals` checks require genuine positive,
bounded years and the synthetic group deliberately has `ExposureKind.UNRESOLVED`.

The **derived mathematical view** from `paired_censor_likelihood_view` is
different and explicitly named: it maps **each verified selected original
systematic year** to one `LEFT_CENSORED` likelihood constraint with
`upper_bound=h`, retaining its original `value`, year and raw
`source_flag`. The **synthetic placeholder group is never appended** to the
likelihood view. Existing source historical groups are retained only if their
real-year exposure and counts are complete and disjoint from the original
annual series. The immutable source proposal still carries the explicit
synthetic group as separate audit evidence.

For independent annual events `Q_i` and a continuous comparison kernel
whose `F(h)` is the stated observation probability, this mathematical
replacement contributes

```text
k at/below h and 0 above:     F(h)^k
one left-censor per real year: product over selected years of F(h) = F(h)^k
log contribution:            k * log F(h)
```

There is **no binomial multiplier** for `count_above=0`, because
`choose(k,0)=1`. Individual records give an auditable proof of each counted
year's identity, unlike a fabricated second period. The original selected
flows are not simultaneously evaluated by their exact densities. This
equivalence is mathematical under the stated sampling/endpoint assumptions;
it **does not assert Flike's actual generated group likelihood**, dependence,
threshold rounding or zero-atom law. Existing exclusions and missing years
remain separate and have not been recoded.

## Fail-closed source classification boundary

- Caller must pass `approve=True` **and** an explicit `approval_source`.
  The supplied MGBT p-values remain external and unqualified as native Flike
  probabilities. The proposed selection and ARR group are independently
  rederived from their original flows, rank probabilities, source cohort,
  threshold and synthetic-group identifiers before any view is constructed.
- A source `source_flag` is never treated as a Flike meaning by default:
  every selected non-null raw flag must belong to an **explicitly passed**
  `verified_exact_flags` set with a `verified_flag_source` explanation.
  A synthetic fixture can identify its own test flag `"0"`; that is not
  an assertion that raw project-file `0` means exact in all Flike builds.
- Reject changed/missing selected identities, non-exact or non-systematic
  candidates, contradictory count, threshold touching a selected flow,
  missing/overlapping/unbounded existing group exposure, unresolved existing
  group exposure or incomplete group counts. Preserve source discharge units,
  year identifiers, recorded-zero fields, prior exclusions and original
  values with no positive offset.
- The merged WP02 [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md)
  exposes `MappedAnnualSeries` with exact `Decimal` values, original source
  lines, source SHA-256 identifiers and complete exposure-period membership.
  `paired_censor_from_mapped_source` now **requires original file3.10
  bytes** and reparses those bytes using WP02's actual `parse_fld`. It
  checks the independently recomputed SHA-256, complete original source lines,
  exact original `Observation` records and full `Decimal.as_tuple()`
  representation (including exponent/precision) against the supplied mapping,
  **before** replaying `map_annual_series` to check source-period exposure.
  A changed Decimal that rounds to the same binary64 flow is therefore
  rejected. Original source bytes must come from the intended file: this
  comparison proves consistency with bytes supplied to the adapter, not
  custody of an inaccessible third-party archive. Unsupported manually
  assembled or CSV-only mappings do not bypass this file3.10 gate.
  The derived view separately retains selected `MappedAnnualObservation`
  wrappers with the original Decimal values, source lines and file hash.
- The WP02 mapper's `ready_for_likelihood=False` remains **unchanged**:
  a caller's explicit approval and flag-evidence source are required even
  for the mathematical view. In synthetic tests raw flag `0` is explicitly
  classified for that **fixture only**, never asserted as an actual legacy
  Flike source-flag meaning. Unknown nonzero flags and reported unverified
  zeroes are rejected by the real WP02 adapter.

## Current verification and integration dependency

The bounded PR provides the immutable `PairedCensorLikelihoodView`,
per-year source links, fail-closed proposal/exposure checks and tests against
the **actual merged WP05 `log_likelihood`** from [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md).
The tests establish that selected records contribute `k log F(h)` once
as annual left-censor terms while retaining their original magnitudes, that
the negative synthetic group is **not added a second time**, and that real-year
historical periods remain disjoint or raise on overlap. This uses a declared
synthetic unit-exponential comparison law, not a qualified legacy model.
The original source is retained for audit and only the derived annual view
is passed to the mathematical likelihood. Neither `contracts.py` nor WP05's
generic historical-group logic is changed to treat negative editor placeholders
as genuine years.

The source-row/Decimal bridge is implemented and tested against
**independently reparsed synthetic file3.10 bytes**, with an explicit
regression for a Decimal mutation smaller than binary64 precision and
another for altered original source bytes. Original Decimal representation,
row flags, units, year labels and source SHA-256 must agree before the
selected source-year cohort is evaluated by **actual WP05**
`log_likelihood` without any second fake-year group. Separate WP07
mathematical tests retain pre-existing excluded/missing evidence and reject
overlapping historical exposure. These tests prove a bounded mathematical
integration, **not** native Flike flag meanings or external archive custody.

The remaining **child #31 completion gates** are exact-final-head Python 3.14
Windows/Ubuntu CI and independent Codex review of this new WP02 integration,
with findings addressed. No Flike defaults are required for this bounded
work. Parent #10 still retains the native target controls, original flag
meanings, zero-threshold priors and posterior semantics.

The project remains **SUL-1.0**. No external runtime code or new dependency
has been introduced; no GUI, service, automatic Flike fit or publication is
enabled.
