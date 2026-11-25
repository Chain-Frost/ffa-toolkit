# Development and remaining requirements

This repository is a sanitized accepted-software baseline. Development cutover is pending synchronization
and owner approval. Existing implementation assignments remain with their current owners until explicitly
transferred. The package/import names remain `ffa-methods` / `ffa_methods`; the command is `ffa`.

## Remaining software requirements

- Expose the implemented ordinary L-moment families through the installed CLI.
- Report fitted GEV support and endpoint observations explicitly.
- Complete standardized Wang diagnostics, optimized H selection and declared uncertainty.
- Integrate explicitly configured lognormal posterior inference, proposal covariance scaling and
  output-specific Monte Carlo precision/stopping criteria.
- Extend explicit posterior bindings to other families, historical/grouped exposure and shared rating error.
- Deliver scientific reports, static plots, accessible method explanations and offline worked examples.
- Complete independent whole-cohort external comparisons, archive identity review and a settings-matched
  posterior comparator without committing external raw archives or records.
- Reconcile the supported-calculation matrix and finish installed end-to-end acceptance.

Native executable/settings custody, unknown vendor defaults, controlled reference runs, repeatability
and final compatibility admission remain separately managed restricted qualification work. Native gaps
must not be misrepresented as missing mathematical implementation or as passing native acceptance.
New production methods and WMAwater additions remain future scope.

## Setup and validation

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pyright
python -m pytest
python -m build
python scripts/verify_distribution.py dist
python scripts/smoke_installed.py dist
git diff --check
```

Activate the environment or invoke its Python explicitly. Run both Python 3.11 and 3.14 qualification.
Pinned development tool versions and strict typing configuration are in `pyproject.toml`; Ruff uses
120 columns. Run focused tests during development and the complete applicable contract before review.
Routine ready PR validation uses one Ubuntu Python 3.11 runner; main uses Ubuntu 3.11/3.14 and Windows
3.14. Manual pre-release qualification covers both OS/Python endpoints. Draft PRs skip automatic jobs.
A build or green workflow does not authorize a release or publication.

[External evidence](external-evidence.md) is optional and separate from ordinary public tests.

## Prepared backlog

These are inactive preparation issues pending coordinated cutover; existing assignments continue.

- [Complete independent mathematical CLI delivery](https://github.com/Chain-Frost/ffa-toolkit/issues/1)
- [Standardized LH diagnostics, optimized H and declared uncertainty](https://github.com/Chain-Frost/ffa-toolkit/issues/2)
- [Bind an explicit lognormal source model to executed posterior inference](https://github.com/Chain-Frost/ffa-toolkit/issues/3)
- [Expose explicit proposal covariance scaling](https://github.com/Chain-Frost/ffa-toolkit/issues/4)
- [Verify output-specific Monte Carlo precision and acceptance](https://github.com/Chain-Frost/ffa-toolkit/issues/5)
- [Install an explicitly configured lognormal Bayesian CLI workflow](https://github.com/Chain-Frost/ffa-toolkit/issues/6)
- [Deliver scientific static reports, plots and offline examples](https://github.com/Chain-Frost/ffa-toolkit/issues/7)
- [Add explicit posterior family bindings and installed paths](https://github.com/Chain-Frost/ffa-toolkit/issues/8)
- [Integrate historical and grouped exposure into posterior inference](https://github.com/Chain-Frost/ffa-toolkit/issues/9)
- [Integrate explicitly configured shared-rating posterior inference](https://github.com/Chain-Frost/ffa-toolkit/issues/10)
- [Expose ordinary L-moment families through the installed CLI](https://github.com/Chain-Frost/ffa-toolkit/issues/11)
- [Report fitted GEV support and endpoint observations](https://github.com/Chain-Frost/ffa-toolkit/issues/12)
- [Complete read-only independent external validation](https://github.com/Chain-Frost/ffa-toolkit/issues/13)
- [Reconcile supported calculations and final installed acceptance](https://github.com/Chain-Frost/ffa-toolkit/issues/14)
- [Verify hosted CI and branch-rule enforcement before cutover](https://github.com/Chain-Frost/ffa-toolkit/issues/15)
