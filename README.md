# FFA Toolkit

Independent flood frequency analysis software, delivered as the `ffa` command-line tool.
Development-stage software under Sustainable Use License 1.0; not OSI-approved open source.
FFA Toolkit is not affiliated with or endorsed by TUFLOW, BMT or WMAwater.

## Install and run

Requires Python 3.11–3.14 (3.14 preferred).

```powershell
python -m pip install .
ffa --help
ffa explain lh-moments-gev
ffa explain l-moments --format markdown
ffa fit --ams docs/examples/fixed-gev-ams.csv --settings docs/examples/fixed-gev-settings.json --output example-result
ffa fit --ams docs/examples/ordinary-ams.csv --settings docs/examples/ordinary-settings.json --output ordinary-result
ffa fit --ams docs/examples/bayesian-fixed-ams.csv --settings docs/examples/bayesian-fixed-settings.json --output bayesian-result
```

The example uses generated annual maxima. The installed calculation path supports explicit-settings,
complete positive annual maxima, fixed-H GEV L/LH fitting and ordinary Gumbel, LN2, LP3 and fixed/free-location GP L-moment fits. See [ordinary L-moment guidance](docs/methods/l-moments.md). Other mathematical components include five
distribution kernels, ordinary L-moment estimators, likelihoods, priors, shared rating errors, low-flow
screening, importance proposals and sampling, and weighted Bayesian outputs. Internal components are
not all available through the CLI. The explicit lognormal Bayesian workflow supports supplied Gaussian priors and classified systematic annual records, subject to the [workflow scope and acceptance](docs/wp08-bayesian-cli.md). Unsupported workflows fail explicitly. See the
[supported-calculation matrix](docs/initial-delivery-matrix.md) for the installed boundary,
remaining workflow requirements and independent acceptance criteria.

Mathematical verification and native Flike qualification are separate. No native equivalence is claimed.
The reference compatibility profile is Flike 5.0.306.0 / file3.10; it is not a certification.
CLI only: static reports, plots and offline documentation are permitted. No GUI, TUI, plugin, service,
browser application or separately supported public Python API is in scope.

See [development](docs/development.md), [architecture](docs/architecture-and-validation.md),
[method register](docs/method-registry.json), [scientific references](docs/references.md),
[compatibility contract](docs/compatibility-contract.md), [fixture admission](docs/sample-data-audit.md)
and [third-party notices](THIRD_PARTY_NOTICES.md).

The distribution name is `ffa-toolkit`; the internal import namespace remains `ffa_methods`.
PyPI publishing is [prepared but disabled](docs/publishing.md). No package release has been published.
