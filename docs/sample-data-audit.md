# Admitted scenario data

The owner authorized reuse of the two supplied scenarios on 25 November 2026 subject to removal of
identifying project names. This repository retains only the identity-sanitized six-file subset as
[case-a and case-b](../reference/legacy-candidates/README.md). All original material and the exact
original-to-sanitized hash mapping remain in private custody.

Case A has 19 positive observations and 21 LP3 quantile rows. Case B has 40 observations, including
five at/below zero threshold, and 19 LP3 quantile rows. Both reports identify Flike 5.0.306.0 / file3.10.
Names, titles and filenames have neutral case identifiers; paths remain redacted. Numerical tables,
observations, controls, line endings and embedded NULs are preserved. The admitted sanitized bundle's
member hashes are pinned in the manifest; its `original` fields refer to that admitted sanitized bundle,
not to the restricted original archives. Archive identifiers and original mapping are not published.

These are candidate pairs with unverified settings, not controlled native goldens. Matching values and
years do not establish priors, run options, flag meanings or native repeatability. Report-internal
mathematical tests preserve printed-rounding acceptance and do not prove inference compatibility.
No other archive contents, research publications, parameter dumps or employer/client material is admitted.
The entire reference tree is excluded from wheel and sdist. See [external evidence](external-evidence.md).
