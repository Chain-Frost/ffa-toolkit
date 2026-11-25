# Optional external evidence

Ordinary public tests run without external private files or credentials. Owner-authorized sanitized
scenarios are committed for parser and report-internal mathematical tests, with their limits recorded
in the fixture manifest. They are not native qualification evidence.

For private/native qualification, prepare a separate workspace outside checkout with `docs/`,
`qualification/` and `reference/legacy-candidates/` matching the public gate schemas. Supply a complete
method register, matrix, evidence ledger and candidate manifest there. From an installed development
environment run:

```powershell
python scripts/qualification_gate.py --root EXTERNAL_WORKSPACE --validate-only
python scripts/qualification_gate.py --root EXTERNAL_WORKSPACE --report EXTERNAL_WORKSPACE/readiness.md
```

The first checks ledger integrity; the second also requires all native release gates. A missing native
proof blocks qualification. Never commit the external workspace or its report. `local-evidence/`,
`private-evidence/` and `native-runs/` are ignored safeguards, not permission to publish their contents.
Use external paths by default. The public pending matrix deliberately cannot pass native qualification.

The read-only pyflood2022 comparators accept an external directory of pinned upstream files:

```powershell
python scripts/audit_pyflood2022.py EXTERNAL_INPUT_DIRECTORY
python -m scripts.validate_pyflood2022_fits EXTERNAL_INPUT_DIRECTORY
```

They verify source identities and compare independently specified mathematics. The manually dispatched
external audit downloads temporary inputs outside checkout, then removes them; raw archives, records
and datasets are not committed or uploaded. This optional comparison is separate from public routine CI
and does not establish Flike compatibility or redistribution rights.
