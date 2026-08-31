# SunCET Data Processing Pipeline Code Base Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/) and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

### Added

* Portable runtime and development environments for ARM64/x86-64 Linux and
  Apple/Intel macOS, including Python 3.12 and 3.14 validation and wheel-based
  CI installation checks.
* Atomic processing provenance manifests with input/output hashes, environment
  identity, metadata-definition snapshots, and public/private privacy profiles.
* Mission-length per-APID DuckDB telemetry storage and a high-level reader.
* Auditable read-only AWS ingest, version-aware replication monitoring,
  checksum-verified LASP SFTP publication, guarded one-way rclone copies,
  storage preflight, CTDB snapshot verification, and a manual SOC runbook.
* Provisional Level 2 PSF-deconvolution and Level 4 CME-tracking products,
  including inspectable diagnostics and Jetson performance/power benchmarks.
* Public SatNOGS APID 1 schema, specification, synthetic test vector, and
  provisional Kaitai decoder artifacts.

### Changed

* The slowly built Level 0.5 decoder is now the canonical implementation and
  validates Bluefin transfer-frame checksums.
* Public `suncet_data` and private `suncet_ctdb` are mandatory, non-overlapping
  host-managed roots.
* SunCET metadata uses UTC; `CALPSF` identifies a content-addressed Level 2
  calibration-set manifest; and `DATAMIN`/`DATAMAX` accept floating-point data.

### Deprecated

* None.

### Removed

* Obsolete Level 0.5, engineering-debug, quicklook, and simulator-derived
  utilities whose useful behavior is preserved in maintained modules or Git
  history.

### Fixed

* JPEG-LS reconstruction for reordered CSIE packet streams and CCSDS fine-time
  interpretation as integer milliseconds.
