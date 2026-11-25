# Agent instructions for FFA Toolkit

Develop an independent Python CLI for flood frequency analysis. Read the compatibility contract,
method register, development requirements, owning issue and current branch before editing. Record the
actual base commit, inspect existing work and preserve unrelated changes. Keep focused branches and PRs;
coordinate shared files and do not duplicate active assignments during cutover.

Preserve admitted parameter coordinates, priors including Jacobians, estimator modes, censor/zero and
rating-error rules, and output estimands. Do not substitute preferred mathematics for unresolved native
behaviour and call it replication. Verify mathematics independently using ARR and primary literature;
qualify compatibility only using independently admitted settings-matched native evidence.

CLI only. Static plots/reports and offline documentation are permitted; GUI, TUI, plugins, browser apps,
services and a separately supported public Python API are outside scope. Internal callable functions
are appropriate. New production algorithms and WMAwater additions are future scope.

Python 3.11–3.14, preferring 3.14; modern built-in annotations compatible with the declared minimum.
Use pinned tools in pyproject.toml, Ruff lint/format with 120 columns and strict Pyright across first-party
source, tests and development scripts. Run the complete commands in docs/development.md, including
build, distribution verification and a clean installed-wheel/CLI smoke outside checkout for packaging.
Do not invent tests that mirror prose, weaken numerical tolerances or overwrite reference outputs.
Report checks actually run and distinguish local from hosted results.

## Licence and third-party material

Retain Sustainable Use License 1.0 and package declaration. Preserve applicable notices; independently
establish data and fixture reuse rights. Do not introduce GPL/AGPL implementation code or dependencies.
Public availability alone does not establish redistribution rights. Keep unadmitted data, publications,
archives, native binaries and research artifacts outside this repository and all distributions.
Never execute archive-supplied binaries/scripts/macros. Treat fixtures as immutable static inputs.
Record transformations and provenance; never infer run correspondence from filenames.

Keep missing/excluded/censored/zero observations and exposure distinct. Preserve natural-log, log-SD,
skew, logit and shape conventions, and distinguish expected-parameter, posterior-quantile and predictive
outputs. Sampling precision is separate from flood uncertainty; use quantity-specific justified criteria.

Update substantive numerical method documentation, registry and validation records. PRs must state the
problem, behaviour, validation, limitations and source evidence. Use Closes only for completed issues;
partial work uses Refs. Independent review and qualification are separate states. No merge, release,
package publication or public visibility without the corresponding explicit owner authorization.
