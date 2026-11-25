# Initial delivery: supported-calculation matrix

Audit date: 25 November 2026. Accepted Toolkit baseline: `34cc841f45555246bbb3ac98c8166e55548678ad`.
Planning and reconciliation: [#14](https://github.com/Chain-Frost/ffa-toolkit/issues/14).
Initial software acceptance: [#1](https://github.com/Chain-Frost/ffa-toolkit/issues/1).
This human-readable crosswalk uses the existing [method register](method-registry.json),
[compatibility contract](compatibility-contract.md), [user documentation contract](user-documentation-contract.md)
and [development requirements](development.md). It is not a new runtime schema or release gate.

Toolkit is the active repository for public software development. Work-package owners continue their
bounded assignments here; consult linked issues and open PRs before editing shared files. A migration
PR is pending work until independently reviewed and accepted. Restricted primary-source acquisition,
original custody and native executable/settings evidence remain external and must never enter public
issues, comments, Git history or package resources. Original-to-new routing is maintained privately.

**Mathematical verification is separate from native qualification.** Every row remains unqualified
against the compatibility target. A green software workflow or published-source identity audit is not
a settings-matched native run. Evidence descriptions below identify existing component tests; they do
not establish end-to-end acceptance of unfinished workflows.

## Calculation workflows

Deterministic admission is complete, caller-classified, strictly positive exact systematic annual maxima
in Year,AMS format, with explicit units/year definition. Bayesian workflows require explicit same-coordinate
likelihood, supplied named joint prior, reference, proposal, seeds and controls. No rating adjustment,
prior, observation deletion or exposure is inferred. Every accepted workflow retains original file/settings
hashes, source classification, algorithm version and explicit probability/output definitions.

| Family / estimator | Observations and prior/rating | Outputs and uncertainty | Accepted component evidence | Installed path at baseline | Owner and remaining acceptance |
| --- | --- | --- | --- | --- | --- |
| GEV L H=0 and LH H=1–4, fixed H | Complete exact systematic AMS; no prior/rating | Parameters and F/AEP quantiles; raw fourth-ratio discrepancy is not standardized z | Sample subset identities, theoretical moments, inversion, integration, unit/boundary tests | Yes, `ffa-gev-fixed-h/1`, JSON/CSV/text | Preserve path; [#12](https://github.com/Chain-Frost/ffa-toolkit/issues/12) support diagnostics; [#7](https://github.com/Chain-Frost/ffa-toolkit/issues/7) plots/guidance |
| Gumbel ordinary L, raw Q | Exact AMS; no prior/rating | Deterministic parameters/quantiles | Internal estimator, independent formulas and boundary/unit checks | No | [#11](https://github.com/Chain-Frost/ffa-toolkit/issues/11) |
| LN2 ordinary L, raw Q | Exact AMS, explicit positive reference; no prior/rating | Natural-log coordinates and quantiles | Internal estimator and independent formula/unit checks | No | [#11](https://github.com/Chain-Frost/ffa-toolkit/issues/11) |
| LP3 ordinary L, ln(Q/reference) | Exact AMS, explicit positive reference; no prior/rating | Log-coordinate/skew parameters, quantiles and support admission | Exact-ratio, high-precision, false-zero and unit/boundary checks | No | [#11](https://github.com/Chain-Frost/ffa-toolkit/issues/11) |
| GP ordinary L, raw Q | Annual maxima, explicit fixed/free location; no prior/rating; not POT exposure | Parameters, quantiles, endpoint/location policy | Both internal variants, independent inversion/support tests | No | [#11](https://github.com/Chain-Frost/ffa-toolkit/issues/11) |
| GEV standardized diagnostic and optimized H | Exact AMS, candidate H=0–4 | Conditional variance/z, explicit selection/tie/no-pass rules | Fixed-H core only; standardized/selection mathematics not yet accepted | No | [#2](https://github.com/Chain-Frost/ffa-toolkit/issues/2); primary mathematical verification required |
| Declared deterministic estimator uncertainty | Admitted exact AMS/model/family/selection | Explicit bootstrap sampling intervals, attempted/failure counts; not posterior limits | Fixed-H GEV research primitive with supplied seed/count/mass and type-7 interpolation; broader uncertainty unaccepted | No bootstrap path | [#2](https://github.com/Chain-Frost/ffa-toolkit/issues/2), [#11](https://github.com/Chain-Frost/ffa-toolkit/issues/11); define supported uncertainty and reselection policy |
| LN Bayesian fixed/estimated log-SD | Explicit positive exact or admitted positive censor records, disjoint systematic years; supplied named joint Gaussian prior; no rating | Distinct posterior/predictive/expected-parameter outputs; numerical precision separately | Sampler, full-covariance proposal and weighted outputs accepted; source binding pending | No | [#3](https://github.com/Chain-Frost/ffa-toolkit/issues/3) binding → [#6](https://github.com/Chain-Frost/ffa-toolkit/issues/6) installed; [#4](https://github.com/Chain-Frost/ffa-toolkit/issues/4)/[#5](https://github.com/Chain-Frost/ffa-toolkit/issues/5) scale/precision |
| LP3, Gumbel, GEV, GP Bayesian bindings | Family-admitted exact/censored input; explicit prior support/transforms/reference; no rating | Same distinct posterior quantities and output-specific precision | Kernel, likelihood/prior and sampler components only; actual family posterior verification pending | No | [#8](https://github.com/Chain-Frost/ffa-toolkit/issues/8), [#5](https://github.com/Chain-Frost/ffa-toolkit/issues/5); GP interpretation explicit |
| Supplied regional prior | Classified records for an admitted family; named joint covariance/coordinates/Jacobian and training-gauge provenance | Posterior and numerical precision, not an inferred default | Prior-density/covariance metadata tests; end-to-end regional inference pending | No | [#8](https://github.com/Chain-Frost/ffa-toolkit/issues/8), [#6](https://github.com/Chain-Frost/ffa-toolkit/issues/6) for LN, [#1](https://github.com/Chain-Frost/ffa-toolkit/issues/1) combination disposition |
| Grouped censor and historical period | Disjoint complete counted groups/thresholds and perception exposure; year-labelled historical exceedances; never fabricated dates | Posterior/precision with exposure and multiplicities conserved | Historical/group likelihood, overlap and paired-censor component tests; source-to-posterior orchestration pending | No | [#9](https://github.com/Chain-Frost/ffa-toolkit/issues/9) |
| Shared rating uncertainty | Admitted records/epochs/units; explicit joint weighted scenarios, measured y=true q exp(b), supplied prior | Whole-series marginal likelihood and posterior/precision | Shared versus independently marginalized rating component tests; actual posterior path pending | No | [#10](https://github.com/Chain-Frost/ffa-toolkit/issues/10); no native rating-law claim |
| Low-flow screen and adopted censor/exclusion treatment | Systematic records preserving original years, ties, thresholds, counts and true-zero distinctions | Screening audit and actual adopted record treatment; never automatic row deletion | USGS comparator, rank/p-value/sweep and source-preserving integration checks; no native variant claim | No screening-to-fit path | [#9](https://github.com/Chain-Frost/ffa-toolkit/issues/9), [#7](https://github.com/Chain-Frost/ffa-toolkit/issues/7) |
| LN/LP3 zero threshold, native no-prior/default modes | Observation/prior law must first be identified | Threshold-logit coordinates are not a complete inference law | Coordinate primitives only; actual unknown-law inference unsupported | Reject | Native contract remains external; [#1](https://github.com/Chain-Frost/ffa-toolkit/issues/1) must disposition any independently specified mathematical counterpart |
| Unknown-year historical magnitudes and unmapped controls | Preserve unknown chronology/controls; never invent dates/settings | No inference until an explicit supported model exists | Source-safe reader and defined counted/year-labelled likelihood terms only | No | [#9](https://github.com/Chain-Frost/ffa-toolkit/issues/9), external native evidence where required |
| POT, new production MCMC, regional fitting, nonstationary/WMAwater additions | Separate future scope | Not initial delivery | No completion claim | No | Future work; the GP distribution/annual-maxima fit above is not a POT workflow |

Signed model quantiles, undefined/infinite mathematical quantities and numerical failures must be
reported honestly. Bayesian expected-parameter quantiles, mean/median design quantiles, predictive
inverse, expected F/direct AEP at a flow, reciprocal expected AEP, pointwise posterior limits and
log10-discharge moments are distinct. Each executed posterior path needs model/estimand tests and
[#5](https://github.com/Chain-Frost/ffa-toolkit/issues/5) precision acceptance, not merely a formatter.

## Shared delivery and evidence

| Requirement | Accepted baseline | Remaining work and owner |
| --- | --- | --- |
| Packaging and quality | Python 3.11–3.14, `ffa-toolkit` distribution, `ffa_methods` namespace, `ffa` CLI, lint/strict typing/build/resources/installed-smoke contract | Preserve [development CI policy](development.md); PyPI publishing remains disabled |
| Source parsing/chronology | Explicit annual records, opaque legacy controls, original numeric precision, source hashes and conservative adapters | Workflow-specific admission in [#6](https://github.com/Chain-Frost/ffa-toolkit/issues/6)/[#9](https://github.com/Chain-Frost/ffa-toolkit/issues/9); no competing reader skeleton |
| Five kernels and coordinates | Independent formula/invariant/boundary tests; admitted sanitized report-internal LP3 replay | [#8](https://github.com/Chain-Frost/ffa-toolkit/issues/8) family binding; report replay is not a native posterior rerun |
| Likelihood/prior/rating | Mathematical density, Jacobian, exposure and joint-rating components | Actual posterior/installed orchestration [#8](https://github.com/Chain-Frost/ffa-toolkit/issues/8)/[#9](https://github.com/Chain-Frost/ffa-toolkit/issues/9)/[#10](https://github.com/Chain-Frost/ffa-toolkit/issues/10) |
| Importance execution | Real independent Gaussian-proposal draws, weighted outputs, local mode/full inverse-curvature builder | Explicit gamma-squared covariance scaling [#4](https://github.com/Chain-Frost/ffa-toolkit/issues/4), output-specific precision [#5](https://github.com/Chain-Frost/ffa-toolkit/issues/5); mean MCSE/global ESS are not rare-tail certificates |
| Primary literature | Public scientific citations and independently written explanations | Mathematical review before standardized diagnostics/scaling; restricted full texts, transcriptions and acquisition evidence remain external |
| External validation | Read-only pinned external-input audit and deterministic cohort comparator; synthetic ordinary-CI fixtures only | [#13](https://github.com/Chain-Frost/ffa-toolkit/issues/13): archive/publication identity and settings-matched posterior comparator; never commit raw archives/records |
| Reporting and examples | Typed formatter/output bridge, fixed-H installed numerical example, six bundled technical pages | [#7](https://github.com/Chain-Frost/ffa-toolkit/issues/7): static scientific plots, original explanations and real offline worked examples for every installed path |
| Native qualification | Fail-closed ledger/matrix machinery with synthetic admission/rejection tests | Real native evidence and final supported-combination acceptance remain external and unresolved; no new gate |

## Owner dispositions still required

These decisions do not authorize dropping requirements or delay already specified independent work.

- [#2](https://github.com/Chain-Frost/ffa-toolkit/issues/2): define ordinary-family bootstrap and
  optimized-H selection/uncertainty choices, failure/tie/reselection rules and independent tests.
  Unknown native bootstrap details remain separate.
- [#1](https://github.com/Chain-Frost/ffa-toolkit/issues/1), [#8](https://github.com/Chain-Frost/ffa-toolkit/issues/8),
  [#9](https://github.com/Chain-Frost/ffa-toolkit/issues/9), [#10](https://github.com/Chain-Frost/ffa-toolkit/issues/10):
  approve family × exposure × supplied/regional prior × shared-rating combinations, including GP
  annual-maxima interpretation and rating/exposure overlap. Any excluded initial requirement needs an
  explicit owner disposition and linked follow-up, not silent scope reduction.
- [#1](https://github.com/Chain-Frost/ffa-toolkit/issues/1): disposition whether an independently
  specified no-prior/zero-threshold counterpart is required before initial acceptance. Unknown native
  laws remain unsupported; unknown-year magnitudes require a defined model, not synthetic dates.
- [#7](https://github.com/Chain-Frost/ffa-toolkit/issues/7): define and prove supported historical/censored
  plotting-position rules, preserving exposure; explicitly disposition mathematically unspecified plots.

## Final installed-workflow acceptance

Use the existing register, shared contracts, quality workflow, distribution verifier and qualification
machinery. This checklist informs [#1](https://github.com/Chain-Frost/ffa-toolkit/issues/1), not a replacement gate.

- [ ] Every required row has an accepted explicit model/configuration and owner/PR, or an explicit
  owner-approved scope/evidence disposition. No missing algorithm disappears into a native-only label.
- [ ] Verify coordinates, priors/Jacobians, rating densities, exposure/chronology/count conservation,
  source identity, units, support and transforms against the actual mathematical sources.
- [ ] At a coherent final head, build/verify wheel and sdist, install outside checkout, and execute every
  admitted file/settings-to-result row with hashes, method versions, JSON/CSV/text and applicable plots.
  Supplied-particle formatting examples do not substitute for executed fitting.
- [ ] Independent analytic/high-precision or admitted source-pinned tests cover estimators, actual
  posteriors, output quantities, failure/admission, bootstrap definitions and quantity-specific numerical
  error, tails, indeterminate results and reproducibility. Preserve unit invariance and support reporting.
- [ ] Reject ambiguous/missing/excluded/zero records, overlapping exposure, invalid priors/units/settings,
  support/numerical failures and existing output directories; never report partial misleading success.
- [ ] Installed help/explain/references and rerunnable safe offline examples cover every selectable
  calculation, distinguish illustrative/executed/verified/native-qualified evidence, and preserve notices.
- [ ] Record actual Ruff/format, strict Pyright, pytest, builds/resources/installed results and exact-head
  independent review under current CI policy. Local and hosted results remain separate.
- [ ] Keep native readiness explicit. Synthetic gate enforcement and initial software completion do not
  establish native equivalence or authorize a package release. Publishing remains separately gated.
