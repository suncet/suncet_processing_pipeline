# SunCET Data Processing Pipeline Code Base Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/) and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

## [1.0.0] - 2026-10-16

### Added

* Lossless, APID 46-based Level 1 solar-north orientation planning with
  explicit roll-in-progress and insufficient-attitude-coverage outcomes.
* Portable runtime and development environments for ARM64/x86-64 Linux and
  Apple/Intel macOS, with supported Python 3.11–3.14 validation, Python 3.14 as
  the production baseline, and wheel-based CI installation checks. Python 3.15
  support is deferred until the complete scientific stack resolves there.
* Atomic processing provenance manifests with input/output hashes, environment
  identity, metadata-definition snapshots, and public/private privacy profiles.
* Pipeline-versioned prelaunch test-phase per-APID DuckDB telemetry storage and
  a high-level reader. The database records CTDB versions and rejects mixing a
  changed bus, CSIE, or DSPS definition under an unchanged pipeline version.
* Auditable read-only AWS ingest, version-aware replication monitoring,
  checksum-verified LASP SFTP publication, guarded one-way rclone copies,
  storage preflight, CTDB snapshot verification, and a manual SOC runbook.
* Provisional Level 2 PSF-deconvolution and Level 4 CME-tracking products,
  including inspectable diagnostics and Jetson performance/power benchmarks.
* Public SatNOGS APID 1 schema, specification, synthetic test vector, and
  provisional Kaitai decoder artifacts.
* Incremental Level 0 processing across the standard Leafspace X-band and
  Fairbanks, LASP, and SatNOGS UHF delivery routes, available as both
  `./process_new_data` and the installed `suncet-process-new-data` command.

### Changed

* Level 1 DN/s normalization now applies separate inner/outer integration
  times only when CSIE metadata explicitly identifies a composite image.
* The slowly built Level 0.5 decoder is now the canonical implementation and
  validates Bluefin transfer-frame checksums.
* Public `suncet_data` and private `suncet_ctdb` are mandatory, non-overlapping
  host-managed roots.
* SunCET metadata uses UTC; `CALPSF` identifies a content-addressed Level 2
  calibration-set manifest; and `DATAMIN`/`DATAMAX` accept floating-point data.
* The Python package/release version is independent of the Level 0.5
  output-schema `version_pipeline` value.

### Deprecated

* None.

### Removed

* Obsolete Level 0.5, engineering-debug, quicklook, and simulator-derived
  utilities whose useful behavior is preserved in maintained modules or Git
  history.

### Fixed

* JPEG-LS reconstruction for reordered CSIE packet streams and CCSDS fine-time
  interpretation as integer milliseconds.
* Compatibility decoding for the documented portion of the current APID 35
  DSPS packet, with its unknown tail retained explicitly rather than
  misreported as a checksum field.
* CSIE CTDB v1.1.9 now defines the complete 444-byte APID 537 packet, so the
  obsolete v1.1.8 wire-length workaround has been removed.
