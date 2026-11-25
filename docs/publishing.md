# PyPI preparation — publishing disabled

The distribution is `ffa-toolkit`, the CLI is `ffa`, and the internal namespace is `ffa_methods`.
The package remains development version `0.0.0.dev0` under Sustainable Use License 1.0.
Renaming or making the repository public does not publish a Python package or establish native equivalence.

The owner-configured pending PyPI trusted publisher must match these exact values:

| Field | Value |
| --- | --- |
| PyPI project | `ffa-toolkit` |
| GitHub owner | `Chain-Frost` |
| Repository | `ffa-toolkit` |
| Workflow filename | `pypi-publish.yml` |
| GitHub environment | `pypi` |

The workflow has only a manual dispatch trigger. Its `publish` input defaults to false and the
repository variable `PYPI_PUBLISH_ENABLED` is false. Pushes, tags and GitHub releases never trigger
this workflow. A manual preparation run builds, tests and smoke-installs on Python 3.11 and 3.14
without uploading to PyPI or TestPyPI. Artifacts stay in GitHub for seven days.

The publishing job requires both a deliberately selected publish input and the separately enabled
repository gate, plus successful validation. Only that job receives OIDC `id-token: write`; it consumes
validated artifacts from the same run, uses the protected `pypi` environment and requires no PyPI token.
Publishing also requires a non-development stable version and an existing matching `vVERSION` tag at
that exact main commit. Environment deployment is restricted to main with owner review required.

## Future owner-approved release

This setup does not authorize a release. Before enabling the publishing gate: complete the supported
scope, scientific/user documentation and release acceptance, obtain independent review and successful
current hosted qualification across the supported platforms, choose the version and approved tag,
confirm PyPI pending-publisher details, and obtain explicit owner release approval. Keep build/test
jobs without OIDC permissions. Do not create a dummy upload to activate the pending PyPI project.

After that separate approval, an owner may enable the repository variable, dispatch an explicitly
approved publish run on main with its matching tag, approve the environment deployment, and restore
the gate to false. Pending registration remains pending until the first actual upload; authentication
cannot be certified end-to-end without that future authorized publishing operation.

References: [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
and [publisher configuration](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
