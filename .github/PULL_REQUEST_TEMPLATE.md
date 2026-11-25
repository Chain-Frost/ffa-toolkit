## Owning issue

Closes #<!-- actual completed work-package issue; use Refs #N for partial changes -->
<!-- Reference the current programme issue, without closing it unless all criteria are complete. -->

## Scope and behaviour

- [ ] State supported CLI behaviour and the exact compatibility profile touched.
- [ ] Confirm no GUI/public API or unqualified numerical method was introduced.
- [ ] Record files shared with WP02/WP03 and coordinate integration.

## Checks run (record actual outcomes, not assumptions)

- [ ] `python -m ruff check .`
- [ ] `python -m ruff format --check .`
- [ ] `python -m pyright`
- [ ] `python -m pytest`
- [ ] `python -m build` and archive-content verification
- [ ] Installed-wheel smoke outside checkout on intended OS
- [ ] Independent review, if required for mathematics or lifecycle

## Evidence and remaining limits

Describe observed/native reference version, input hashes, case status, numerical tolerance/precision, limitations and deferred checks as applicable. A candidate pair or successful build is not qualified inference evidence. State explicitly if any check was unavailable, and do not merge/publish without authorisation.
