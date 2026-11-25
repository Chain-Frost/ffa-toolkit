# Third-party notices and data admission

FFA Toolkit remains under Sustainable Use License 1.0. Public visibility does not make it OSI-approved
open-source software. Dependency licences remain separately applicable.

- SciPy and NumPy runtime dependencies use their own BSD licences; their distributions retain notices.
- USACE-RMC Numerics 2.2.0 at `7e8e8d1c5f26e045a35ec9fc09367de95ed05b02` uses 0BSD.
  Retained JSON vectors are independently calculated mathematics with pinned source references, not
  translated implementation code. The licence is retained under `reference/notices/Numerics-0BSD.txt`.
- USGS Bulletin 17C / Cohn MGBT published numerical examples and the USGS MGBT comparator vectors
  in WP07 tests are attributed in those tests. Federal USGS scientific information and numerical data
  are public domain, subject to separately credited material. No third-party implementation is copied.
- Owner-authorized case A/B scenario derivatives are admitted with identities removed; see
  `docs/sample-data-audit.md`. Authorization is specific to these data and does not relicense software.
- ARR and primary literature are scientific citations. No full-text publications or Wang coefficient
  transcription tables are included. Source access is not asserted to confer artifact reuse rights.

External RMC/ARR and FlowFreq gauge corpora, the textbook 31-observation sample and derived vectors
are excluded where underlying data rights are unresolved. Their original private custody is preserved.
The optional pyflood2022 validation reads external files without redistributing its archives/datasets.
No GPL/AGPL implementation code or runtime dependency is admitted. External comparator attribution
is not incorporation of its code. Public accessibility alone is never the admission criterion.
