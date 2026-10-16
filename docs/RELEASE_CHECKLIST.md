# SunCET processing-pipeline release checklist

This checklist separates a software release from the Level 0.5 processing
schema. For the first formal release, the repository `version` is `1.0.0` and
`structure.version_pipeline` is `2.0.2`.

## Automated gates

- Run `pytest -v` from the repository root. Test discovery must stay within
  `suncet_processing_pipeline/tests` and `benchmarks`; ignored scratch trees such
  as `tmp/` are not product code.
- Build a wheel with the Python 3.14 production baseline and install it into a
  clean environment; CI must also pass on every supported branch, currently
  Python 3.11–3.14. Add Python 3.15 only after the complete scientific stack
  resolves and the pipeline passes there.
- Verify `python -c "import suncet_processing_pipeline as s; print(s.__version__)"`
  prints the repository release version.
- Verify `suncet-process-new-data --help` and `./process_new_data --help` both work.
- Confirm the wheel contains required configuration and SatNOGS public decoder
  resources, and does not contain tests, scratch files, CTDB files, credentials,
  raw telemetry, or generated dashboards.

## Representative Level 0.5 regression

Before tagging, re-run one reviewed, known-good capture from each physical input
route without overwriting the preserved reference products:

- X-band: `2026-06-03_fm_xband_test_vacuum_ambient_temperature`. The checked-in
  [`level0_5_xband_golden_v1.json`](assets/level0_5_xband_golden_v1.json)
  records its two raw-input hashes, frame/checksum totals, and six unambiguous
  golden image-array hashes.
- UHF/Hydra: `2026-09-17_uhf_19200_baud_image_4x4bin_shift4_internal_led`.
  [`level0_5_uhf_golden_v1.json`](assets/level0_5_uhf_golden_v1.json) records
  playback reassembly, packet/checksum totals, DuckDB rows, and the exact
  reconstructed image-array hash.
- debug hardline/DSPS: `2026-09-17_dsps_light_leak_check`.
  [`level0_5_hardline_golden_v1.json`](assets/level0_5_hardline_golden_v1.json)
  records packet/checksum totals, DuckDB rows, stable output hashes, and the
  explicitly provisional APID 35 strict-checksum result.

A release passes only when differences from these preserved baselines are
explained and reviewed.

## Known limitations to confirm in release notes

- APID 35 DSPS output is provisional until its complete 124-byte packet and
  engineering-unit definitions are present in an authoritative CTDB.
- Level 1 production corrections await first-light solar calibration data.
- Level 3 recentering and fine-rotation corrections await real solar data needed
  to measure and validate the residual geometry.
- Level 4 CME tracking is a provisional known-window product, not continuous
  autonomous event discovery.

## Release actions

- Review the complete diff and remove editor swap files, caches, local paths,
  generated private data, and secrets.
- Ensure `CHANGELOG.md` describes the shipped behavior and known limitations.
- Commit from a clean worktree, push, wait for required CI checks, create signed
  tag `v1.0.0`, and publish matching GitHub release notes.
