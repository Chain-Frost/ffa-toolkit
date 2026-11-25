# Fixed-H GEV ordinary L and LH mathematical fitting

**Method ID:** `lh-moments-gev`. **Implementation:** independently verified **fixed H=0,1,2,3,4** GEV mathematical estimation only. **Native Flike compatibility:** not established; exact vendor defaults, optimized H, unsupported record treatments and native comparison are deferred to [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md).

## Purpose and required inputs

The installed `ffa fit` command fits one GEV distribution to a *complete, strictly positive, exact, equally weighted* annual-maximum series. It uses only the documented `Year,AMS` CSV layout already parsed by WP02 and an **explicit** `ffa-gev-fixed-h/1` JSON configuration. The caller must affirm that every row is a valid annual maximum for the declared `year_definition` and `discharge_unit`. That declaration is recorded, not externally authenticated.

Missing cells, reported zeroes, unknown native flags, excluded observations, historical/interval/threshold censoring and grouped exposure are **not** silently discarded or fitted. A file3.10 `.fld` project is not a valid input to this bounded CLI path. The choice of `lh_shift` and desired non-exceedance probabilities is required; no vendor/default estimator, H optimization, prior or bootstrap selection is inferred.

## Mathematical coordinates and estimator

For sorted exact annual peaks `q(1) <= ... <= q(n)`, define sample LH moments from unbiased combinations of order statistics on subsets of `H+r` data (`r=1,2,3,4`; `H=0` yields ordinary L moments). The production estimator accumulates adjacent data spacings with exact combinatorial counts, preserving meaningful zero and one-ULP dispersion. It fits the first three sample moments to the corresponding GEV population LH moments (ARR Book 3, Table 3.2.4; equations 3.2.64–3.2.77; Wang 1997).

The explicitly documented `k` convention is

```text
F = non-exceedance probability,  AEP = 1 - F
Q(F) = u + alpha/k * [1 - (-ln F)**k],  alpha > 0,  k > -1
k -> 0: Q(F) = u - alpha * ln(-ln F)
```

Parameters `u`, `alpha` and `k` are **deterministic point estimates**, not posterior means or Monte Carlo summaries. The fixed `H` is 0–4. No transform into unverified vendor shape sign, log-scale or zero atom is applied.

## Reproducible installed example

The package includes original, synthetic `resources/examples/fixed-gev-ams.csv` and `resources/examples/fixed-gev-settings.json` data. Copy those two files to a writable directory before running:

```console
ffa fit --ams fixed-gev-ams.csv --settings fixed-gev-settings.json --output fixed-gev-report
ffa explain lh-moments-gev --format markdown
```

The output directory must not exist and its parent must already exist. It contains `results.json`, `results.csv` and `report.txt`. These files include input/settings SHA-256 digests; explicit units, year definition, fixed H and probability choices; named point estimates; fitted quantiles; the actual sample LH moments and observation count; and, when available, the **unstandardized** fourth-to-second LH ratio residual. That raw residual is **not** Wang (1998)'s standardized optimized-H z statistic.

The sample is a synthetic mathematical example, **not** a native Flike run or flood-design recommendation. SHA-256 records the actual file bytes but does not authenticate the caller's source classification.

## Uncertainty and limitations

No confidence/probability intervals, bootstrap uncertainty or posterior uncertainty are calculated by this command. The tested fixed-H bootstrap primitive in WP04 is separate, not silently enabled. Unbounded GEV mathematical predictions can be negative; an output warning identifies a nonphysical discharge rather than truncating the model. Physical support and engineering suitability require separate appraisal.

**Scientific foundations:** ARR 2019 Book 3 Chapter 2 §§2.6.4.5, 2.8.10.3, equations 3.2.64–3.2.77 and Table 3.2.4 (`ARR3`); Wang (1997), *LH moments for statistical analysis of extreme events* (`WANG1997`); Wang (1998) conditional variance for future standardized diagnostics (`WANG1998`). See `ffa references ARR3` and `ffa references WANG1997`.

**Deferred:** optimized-H selection, complete historical/censored/zero workflows, all family/estimator combinations, actual importance sampling, production Bayesian fits, verified Flike control mappings and native settings-matched numerical comparisons. These remain tracked by WP04 #7, WP06 #9, WP08 #11 and native-only #39. No GUI, public Python API or release claim is made.
