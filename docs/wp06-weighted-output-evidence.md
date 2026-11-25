# WP06 — Bayesian weighted-output foundation

**Status: implemented weighted-output foundation, not native-Flike-qualified.** This module works on supplied particles and named, original-coordinate density terms. The existing supplied-particle output module does **not** generate draws. A separate [explicit Gaussian importance-execution component](wp06-importance-execution.md) now generates actual independently weighted particles from a fully caller-supplied proposal, without constructing a native-mode/Hessian approximation or enabling a complete fitting engine/CLI. The separate [#47 local curvature builder](methods/importance-sampling-reference.md) supplies a proposal for a bounded smooth interior-mode target. Broader source/CLI integration and tail precision remain active in #9/#40. Exact vendor defaults/controls and native comparisons are separately deferred to #39/#12; their absence does not block independently specified mathematical execution. See [development stages](development-stages.md).

## Explicit equations and output meanings

For parameter draws \(\theta_i\) in the **original source coordinates**, with log likelihood \(\ell_i\), log prior density \(\pi_i\), and log proposal density \(g_i\), compute \(a_i=\ell_i+\pi_i-g_i\) and \(w_i=\exp(a_i-\max a)/\sum_j\exp(a_j-\max a)\). The upstream likelihood/prior owner must provide densities and **all Jacobians in the same coordinates**. Raw density terms and draw order are preserved; no proposal or prior is fabricated.

Posterior moments are \(E[\theta]=\sum_iw_i\theta_i\) and population covariance \(C=\sum_iw_i(\theta_i-E[\theta])(\theta_i-E[\theta])^T\). Reports retain correlations, \(ESS=1/\sum_iw_i^2\), entropy effective size, maximum weight, and zero-weight counts. Undefined correlations for constant parameters remain undefined. User-specified ESS/max-weight gates reject poorly concentrated weights; **there is no qualified default threshold**, and global ESS cannot certify rare-tail accuracy.

For non-exceedance probability \(F=1-\mathrm{AEP}\) and fixed flow \(q\), the following are **distinct**:

| Output | Equation | Status |
| --- | --- | --- |
| Expected-parameter quantile | \(q_F(E[\theta])\) | Independently tested; average **loge SD** coordinate, not converted SD |
| Posterior mean and median quantile | \(E[q_F(\theta)]\); weighted median of per-draw quantiles | Independently tested; not substitutes for expected-parameter quantile |
| Quantile 90% limits | Weighted 5% and 95% empirical quantiles | Interval mass configurable; FLIKE interpolation rule unverified |
| Quantile base-10 moments | Weighted mean/SD of \(\log_{10}q_F(\theta_i)\) | Undefined if any positive-weight quantile is nonpositive; no invented floor |
| Expected non-exceedance | \(E[F(q\mid\theta)]\) | Independently tested; source **Expected probability** matches this convention numerically |
| Expected AEP | \(E[S(q\mid\theta)]\) | Computed directly with stable survival evaluation |
| Return period | \(1/E[S(q\mid\theta)]\) | **Not** \(E[1/S(q\mid\theta)]\) |
| Return-period limits | Weighted per-draw \(1/S_i(q)\) limits | Mathematically defined but legacy interval algorithm unverified |
| Predictive quantile | \(\inf\{q:E[F(q\mid\theta)]\ge F\}\) | Generalised inverse handles probability jumps and a zero atom |

The weighted inverse-ECDF uses the **smallest sample value whose cumulative weight reaches the requested fraction**. This choice is explicit and reproducible, but not known to be the FLIKE empirical interpolation rule. Tail-limit ESS values are weight-resolution diagnostics, **not** Monte Carlo standard errors. Infinite return periods are not substituted with the large scientific-notation numbers in the sample reports; their nature as limits/sentinels is unknown.

The synthetic test atom is **not** evidence that Flike has an independent literal-zero mixture. The Case B sample prints finite mean log10(q) while the lower quantile limit is displayed as 0.00. Printed rounding cannot establish whether the underlying limit is exactly zero; the legacy rule for these log moments remains unknown. This module exposes a mathematical undefined case when a positive-weight *quantile draw* is nonpositive instead of silently excluding particles or truncating flows.

## Candidate-report checks (not algorithm conformance)

Candidate source reports and original/staged hashes are in `reference/legacy-candidates/`, `docs/sample-data-audit.md` and the reference manifest. Both reports identify program version **5.0.306.0**, file **3.10**. They are `candidate_settings_unverified`, **not** qualified matched runs.

- **Case A:** expected probability `0.01599` at discharge `1.92` corresponds to return period `1.02` (because `1/(1-0.01599)=1.01625`). At its **expected-parameter** 50-year flow `3647.523`, the second table has expected probability `0.9761832`, return period `41.98716` years—not 50. This establishes why the estimands must remain separate.
- **Case B (later):** two zero-flow rows give probability `0.12580` and `1.14` years; `1/(1-0.12580)=1.1439`. The extra logit parameter and five zero years need the qualified WP07 threshold model.
- Natural-log posterior coordinates are separate from base-10 reported quantile moments. A corrupted fourth parameter label with embedded NULs and the unexplained 20,000-row parameter-sample files must **not** be interpreted as verified weights.

These are **semantic identities checked at printed precision**. There is no source/project/settings-matched native rerun, no repeatability study and no verified Monte Carlo tolerance.

## Initial mathematical/software work — issues #9 and #40

1. Inspect and document the primary-source importance-sampling algorithm and explicit supported configuration before implementing it. Specify parameter initialisation, proposal support/density, actual random draws, posterior evaluation, weight handling, precision/stopping controls and failure behaviour. Weighted supplied-particle summaries and deterministic verification quadrature do not implement this execution path.
2. Integrate the reviewed WP02 data/provenance, WP03 kernels, WP04 initializers, explicit WP05 densities/rating model and supported WP07 observation treatment. An unavailable native default is not needed for a fully specified mathematical calculation; an actually unspecified statistical law remains unsupported.
3. Verify the complete declared target/proposal and output estimands using tractable independent posteriors, meaningful synthetic integrations, numerical extremes and repeated-run precision checks. Report sampling error separately from flood uncertainty. Preserve named coordinates and distinct prior/likelihood/proposal contributions. Production MCMC remains outside this scope.
4. Integrate the tested execution with the CLI/report path, document configuration/reproducibility and limitations, pass required CI/build/installed checks, and obtain independent final-head review. Issue #9 remains open for this real implementation work; it does not wait for native comparison once its defined initial scope is complete.

## Deferred native evidence — issue #39

Identify the exact vendor proposal/initialiser, parameter-file meanings, sample-count/RNG/stopping defaults and interval/interpolation/report rules. Acquire build-pinned settings-matched native runs for quantities, tails, zeros and failures; establish native repeatability, printed precision and comparison tolerances under #12. Preserve unknown extreme bounds and zero/log-moment rules rather than guessing them. These requirements govern the later native claim, not the independent implementation gate.

**Prior validation checkpoint:** PR #20 head 9bde24a761d7148d586b4b55514f781fd945b6d1 passed [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md) on Ubuntu and Windows, Python 3.14: Ruff lint/format, strict Pyright, pytest (36 passed), builds, distribution checks and installed-wheel CLI smoke tests. These were **prior-head** checks, not proof for this revised implementation. Refer to PR #20 for final-head CI and independent review status. Neither checkpoint qualifies native Flike fitting or a release.

Source basis: ARR Book 3 §2.6.3.6–2.6.3.7 and the repository's `docs/statistical-methods.md` and `docs/architecture-and-validation.md`; exact vendor algorithm evidence remains outstanding.


## Bounded independent mathematical audit (23 November 2026)

This is a source-level mathematical audit of the partial output component, not a verified vendor run or an independent Codex review. Supplied proposal draws and source-coordinate likelihood/prior/proposal terms remain intact. No production MCMC, quadrature, new proposal or inferred legacy default has been introduced.

| Finding | Reproduction and correction |
| --- | --- |
| **F1 — predictive inversion across extreme scales** | With exponential scales 1 and exp(700), weights 0.99/0.01 and F=0.5, fixed 180-step real-space bisection returned roughly 4.587e249, whose mixture CDF is approximately 0.99. Correct median is ln(99/49) = 0.7032995520239632. The replacement uses the monotone binary64 ordering of finite discharges to identify the first representable value at or above the CDF target, in at most 64 interior CDF evaluations after exponentially widening the component-PPF bracket if necessary. The analytic median, adjacent-float inequality and discrete jumps at positive/negative locations are tested. |
| **F2 — extreme but valid weighted-quantile input** | Two unnormalised weights of 1e308 have a well-defined 50/50 empirical law but overflow when summed directly. The implementation now scales by the largest positive weight before summing. Negative/NaN/zero-total weights remain rejected. |
| **F3 — finite ratios with overflowing intermediates** | The density ratio (-1e308) + (-1e308) - (-1e308) is representable, but ordinary floating addition and even a single poorly ordered math.fsum call overflow in intermediate arithmetic. Evaluate the three possible three-term orderings with math.fsum and use the first representable result; reject genuinely unrepresentable ratios. Original density terms are not transformed or replaced. |
| **F4 — invalid standalone tail resolution arguments** | The weight-only tail ESS helper now explicitly rejects empty or unequal-length inputs, negative/nonfinite weights and NaN/negative-infinite quantiles or bounds rather than returning an empty tail or an incidental exception. |
| **F5 — independent Codex review: component PPF rounding (P2)** | A single scale-1 exponential PPF at F=0.25 is 0.2876820724517809, whose evaluated CDF rounds down to 0.24999999999999997; the previous equal-endpoint bracket falsely raised an exception. Expand the binary64 lower/upper bracketing keys geometrically until the evaluated CDF straddles F, then locate its minimal representable generalized inverse. New tests cover one and two identical particles, the adjacent-float inequality, and downstream quantile reporting. No arbitrary vendor tolerance or alternate distribution is introduced. |


**Reviewed without a demonstrated mathematical defect:** original-coordinate weighted population mean, covariance, SD and correlation (undefined for a constant coordinate); stable normalisation and concentrated-weight ESS/entropy diagnostics; step-ECDF empirical limits and nonpositive-quantile log10 warnings; and distinct expected-coordinate, posterior-mean/median quantile, posterior-predictive quantile, expected F, direct expected survival/AEP, reciprocal expected AEP and per-draw period-limit estimands. The synthetic tests also retain explicit failures for invalid CDF/SF/PPF components. These proofs do not establish legacy parameter proposals or interval interpolation.

**Deterministic conjugate Normal verification:** The independent high-precision benchmark in PR #22, reference/public-candidates/numerics-vectors/likelihood-bayes-vectors.json, specifies observations [1,2,3], Normal(0,1) prior on the unknown mean, known observation SD 2, posterior mean 6/7, variance 4/7, SD 2/sqrt(7) and equal-tail 95% limits [-0.6244506519246364, 2.338736366210351]. The WP06 **test-only** adapter evaluates separate actual Normal likelihood, prior and Normal(0, SD=2) proposal log densities at equally spaced probability-midpoint nodes; these are **deterministic quadrature, not iid particles**. The test checks moment errors and empirical-step limit errors at 256, 1024 and 4096 nodes. At 4096 nodes the independent off-repository calculation gave mean/variance errors near machine precision and maximum limit error below 0.0011. This is a step-quantile discretisation tolerance for this *particular analytic test*, not a Flike report/stochastic acceptance tolerance. ESS for deterministic nodes is not an effective iid Monte Carlo sample size.

The PR #22 reference file is intentionally not copied onto this branch: the test uses the analytic expression and cited high-precision limits; upstream evidence remains in its own authoritative PR until merged or explicitly integrated. RMC ModeCurve remains a comparison-side estimator only. Legacy report literal estimands, threshold rules, matched native runs and actual precision remain unresolved.
