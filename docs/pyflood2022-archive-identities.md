# pyflood2022 archived-result identity audit

[Toolkit #13](https://github.com/Chain-Frost/ffa-toolkit/issues/13) tracks this read-only
archived-result audit and the separately outstanding settings-matched posterior comparison.
The existing deterministic stress matrix remains separate. This audit does not perform a new
statistical fit or a native comparison.

## Source and publication identity

Lerat and Vaze (2025), [How February 2022 redefines extreme floods in
Australia](https://doi.org/10.1038/s43247-025-02307-z), Methods, equations
(1)–(4), uses posterior mean 1% AEP quantiles from the full-data fit and
a leave-one-year-out fit. Equation (4) is the difference of their
logarithms. It is not a ratio of the observed peak to its design flood.
The [published supplement, Note 3](https://static-content.springer.com/esm/art:10.1038%2Fs43247-025-02307-z/MediaObjects/43247_2025_2307_MOESM2_ESM.pdf)
describes the LP3 sensitivity case and the lower-20th-percentile
left-censored GEV case. Exact threshold interpolation, ties, whether the
threshold is recomputed for each omitted year, prior bounds and fitting
code/settings remain unverified.

The paper's Results text reports 43,946 events; its Figure 3 caption
reports 42,946. The pinned v5 archives contain 42,946 unique station/year
keys at 1,094 sites, matching the caption. This is an internal publication
inconsistency, not evidence that the pinned files have lost 1,000 rows.
The earliest archive year label is 1885, despite the Results description
of records starting from 1950. Test date/year consistency and document
the actual values rather than relabel or trim them.

The [GitHub source](https://github.com/jlerat/pyflood2022/tree/d603936ae642feef94228b39c33b7307dc236b8e)
is fixed to the same revision and blob sizes/hashes as
[the source audit](pyflood2022-validation.md). The separately archived
[Zenodo 1.0 release](https://zenodo.org/records/14271026) dates from
4 December 2024, predating the final 15 May 2025 article. Do not assume
its data and the GitHub v5 files are byte-identical or establish final
fitting-run identity from the repository README alone.

## Audit contract

Run `python -m scripts.audit_pyflood2022_results PINNED_ASSET_DIRECTORY`
after obtaining the three exact pinned assets. The existing isolated
source-audit workflow runs this audit independently of preliminary fit
results and the stress matrix. Verify source blobs again; require identical
unique raw peak keys; compare timestamp, specific peak and runoff field
tokens across archives. No external fitting code is executed or imported.

Inspect the source's comment metadata without identifying path/email strings. Compare the actual peak timestamps
with the retained WATERYEAR labels, enumerating each possible starting
month and start/end-year label convention. Agreement counts are descriptive
evidence, not an inferred source processing contract. Preserve missing and
unparseable timestamps and pre-1950 counts.

Audit eight blocks: uncensored/censored archive, peak flow (m3/s)/ten-day
runoff (mm), GEV/LOGPEARSON3. Check all source keys, not a filtered regional
event subset. Treat each ALL/JACKKNIFE/SURPRISE/CI5/CI95 cell as its original
decimal token; separately count missing, invalid, nonfinite and nonpositive
values as applicable. A finite archived value does not prove successful
MCMC convergence.

For a positive pair A=Q100_ALL and J=Q100_JACKKNIFE, compare the archived
S to ln(A)-ln(J). First check the central displayed values using a binary64
roundoff allowance proportional to the absolute logarithms and S (32 times
the unit-in-last-place at 1). If not central, test the **conditional**
hypothesis that every displayed decimal token was independently rounded
to its last displayed decimal place: form each interval by adding and
subtracting half of that place, expand conversion to binary64 outwards,
and compare the natural-log ratio bounds with the S interval. If the
intervals overlap, report compatible display rounding; if disjoint, report
inconsistent under that rounding hypothesis. If bounds cannot be evaluated,
report unresolved. This is not a declared exporter precision or proof of
the original unrounded results. Neither a broad relative tolerance nor an
inferred change of logarithm base is used to make cases pass.

For CI5/CI95, distinguish ordered/reversed and unavailable bounds, and
report the position of the ALL mean without treating a mean outside a
central credible interval as a defect by itself. Check repeat consistency
of each site-level ALL/CI field across its annual rows; retain partial
availability and conflicting values separately. These are archive-internal
identities, not coverage or calibration tests.

The receipt contains aggregate counts, maximum absolute central residual,
deterministic SHA-256 fingerprints of sorted JSON key sets without
example keys or station identifiers. It contains no copied flood series or
individual quantile values. Source files are discarded. Job success means
the experiment and its classifications completed; it does not mean every
archived value is scientifically qualified. Original data and results are
not admitted for redistribution or package resources.

## Remaining

After this audit, retain any inconsistent output blocks and publication
provenance gaps for [Toolkit #13](https://github.com/Chain-Frost/ffa-toolkit/issues/13).
A settings-matched independent posterior comparator still needs equivalent
priors/coordinates, actual fitting-data subsets, censoring and jackknife
rules, uncertainty definitions and numerical precision evidence. Source
licence and data-provider redistribution conditions require separate
admission. No production estimator changes, native Flike claim or release.

## Executed sanitized audit — 25 November 2026

The pinned three input assets were downloaded into a temporary directory outside checkout, byte/blob
identities were verified, and the new audit was executed locally with CPython 3.14.6. It exited 0 after
classifying all eight blocks. All 42,946 raw station/year keys paired uniquely; peak timestamps, specific
peak-flow and runoff tokens had zero differences across the archive pair. The external temporary
input directory was removed after execution. No archive, original row or raw quantile is committed.

| Archive | Variable | Family | Compatible display rounding | Inconsistent under rounding | Missing identity fields |
| --- | --- | --- | ---: | ---: | ---: |
| Uncensored | Peak flow | GEV | 10,981 | 0 | 31,965 |
| Uncensored | Peak flow | LP3 | 10,597 | 0 | 32,349 |
| Uncensored | Ten-day runoff | GEV | 5,223 | 65 | 37,658 |
| Uncensored | Ten-day runoff | LP3 | 5,092 | 69 | 37,785 |
| Censored | Peak flow | GEV | 10,981 | 0 | 31,965 |
| Censored | Peak flow | LP3 | 10,477 | 0 | 32,469 |
| Censored | Ten-day runoff | GEV | 5,213 | 76 | 37,657 |
| Censored | Ten-day runoff | LP3 | 5,037 | 73 | 37,836 |

No block had reversed available CI endpoints in this execution. Missing and conflicting repeated
site-level fields, mean-versus-interval positions and water-year hypotheses remain independently
reported diagnostics; they are not silently excluded from the identity counts. The runoff identity
inconsistencies are retained as unresolved archive/publication findings under Toolkit #13. Conditional
rounding compatibility does not prove the unrounded identity or original exporter precision. Audit
exit zero means the classifications completed, not that every archived result passed scientifically.
This run does not refit the data, replay MCMC, establish a matched posterior comparator or qualify Flike.
