# SunCET Level 2 development handoff contract

This document describes the temporary Level 2 interface fixture supplied to
Level 3 developers. It is an integration artifact, not an accepted
science-calibrated SunCET product.

## Product boundary

SunCET Level 2 is a PSF-deconvolved image. In the production pipeline it will
consume a Level 1 image after radiometric and detector corrections. Level 3
will then apply fine geometric corrections such as precise Sun centering and
rotation to put solar north up.

The distributable fixture regenerated on 2026-08-31 used a corrected-timestamp
synthetic Level 0.5 image directly because the end-to-end Level 1 writer and
mission-approved calibration set were not complete. That historical fixture is
explicitly marked `SYNTHET = T`, `L1BYPASS = T`, and
`PROCSTAT = 'PROVISIONAL'`.

A stricter representative fixture was produced on 2026-09-23 from synthetic
`config_default` frame 300. Its Level 0.5 array is correctly treated as stored
DN, then divided pixelwise by the simulator-configured effective exposure:
0.07 seconds inside the short-exposure circle and 11.25 seconds outside. The
resulting checksum-valid Level 1 FITS declares `LEVEL = 1`, `BUNIT = 'DN/s'`,
and satisfies metadata contract `v1.0.4dev`. Level 2 consumes it through the
normal strict boundary and therefore records `L1BYPASS = F`. This is a better
Level 3 interface fixture, but remains provisional because it applies exposure
normalization only rather than the complete future Level 1 calibration chain.

## FITS interface

- One two-dimensional image in the primary HDU.
- The accepted detector array shape is exactly `(750, 1000)` in NumPy
  `(row, column)` order. The writer rejects an input of any other shape and
  rejects a deconvolution result that changes shape, because retaining the
  upstream WCS after a shape change would be incorrect.
- Floating-point pixel values; negative values are permitted after inverse
  filtering and must not be clipped by Level 3.
- The source pixel grid and helioprojective WCS are retained. No Level 3
  rotation, fine Sun centering, or resampling has been applied.
- `LEVEL = 2`, `TITLE = 'SunCET Level 2 Image'`, `DECONV = T`, and
  `DECONVCF` identify the processing stage.
- Every keyword introduced at metadata Minimum Level 0.5, 1, or 2 in the
  selected versioned metadata definition is required. The current normalized
  fixture uses `suncet_metadata_definition_v1.0.4dev-FITS.csv`. The writer
  validates this cumulative contract, declared scalar types, and FITS checksums
  before atomically replacing an output product.
- `DATE-BEG`, `DATE-OBS`, `DATE-END`, `TIMESYS`, `BUNIT`, the two-dimensional
  WCS cards, `RSUN_OBS`, `RSUN_REF`, and `DSUN_OBS` are the principal
  downstream science metadata. SunCET uses UTC: the writer requires
  `TIMESYS = 'UTC'`, preserves the UTC timestamp values, and normalizes the
  corresponding FITS comments to UTC.
- `CALPSF` names an adjacent, machine-readable calibration-set manifest, and
  `CALID` records the content-derived calibration-set identifier in its name.
  `DIFPSF`, `SCATPSF`, `SPECFIL`, and `RESPFIL` are provisional convenience
  cards naming its components. The calibration manifest records each component
  filename, size, and SHA-256 hash plus the correction factor; the processing
  provenance manifest records the resolved source paths.
- `DATASAT` and `DSATVAL` retain the upstream detector-saturation semantics.
  They describe saturation before PSF deconvolution and are not recomputed from
  deconvolution ringing or overshoot.
- For the historical synthetic bypass only, `EXP_MASK` is
  `NOT_APPLIED_SYNTHETIC_BYPASS`; it does not name a nonexistent Level 1 mask.
  The normalized frame-300 fixture instead records
  `SIMULATOR_CONFIG_CIRCLE` and carries its circle geometry and effective
  inner/outer exposures in the header and provenance sidecar.
- FITS `CHECKSUM` and `DATASUM` are written and must validate before ingestion.
  The corresponding cards on the upstream FITS product must be present and
  valid before Level 2 processing begins.
- `DECONRAT` records the summed output/input flux ratio and `DECONTOL` records
  the provisional acceptance tolerance. The development writer currently
  requires this ratio to be within 5% of unity; a science-approved tolerance
  remains calibration work.
- `DATASIG` and `DATAP01` through `DATAP99` are computed from non-zero finite
  output pixels, matching their current metadata definitions. `DATAZER`
  records the number of exact zeros.

Downstream code should use the FITS header rather than infer product state from
the filename. It should preserve unknown header cards and should not assume
that future calibrated Level 2 products remain provisional or bypass Level 1.
The authoritative CSV is a required-minimum interface rather than an exclusive
allowlist: `DECONV`, `DECONVCF`, `DIFPSF`, `SCATPSF`, `SPECFIL`, `RESPFIL`,
`PROCSTAT`, `SYNTHET`, and `L1BYPASS` are documented provisional extensions.
`CALID`, `DECONRAT`, and `DECONTOL` are likewise provisional extensions pending
the next coordinated metadata-sheet revision. A normal Level 1 input preserves
an inherited `SYNTHET` flag; `L1BYPASS` independently records whether Level 1
was skipped.

The writer refuses to replace an existing product unless overwrite is selected
explicitly. This prevents a changed code revision or calibration set from
silently replacing a product with the same pipeline-version filename. The
calibration manifest and FITS product are each written through unique temporary
files; the FITS product is checksum- and metadata-validated before atomic
replacement.

Processing manifests distributed with public fixtures use the public privacy
profile. Host identity and user-bearing absolute paths are omitted or replaced
by portable aliases. Richer internal operational records must remain outside
the public data tree. The current distributable was regenerated from clean
release checkpoint `4fbd7b9aeda539ffd7a057d3a49d063a4f503edb`; its checksum
list covers only the current FITS image, calibration manifest, README, and
sanitized provenance manifest.

## Known limitations

The diffraction/scatter inputs and spectral weighting are historical
development assets, not a versioned mission calibration release. The current
inverse filter is unregularized and contains a provisional scatter-core
correction factor. Level 1 detector corrections are not represented in this
fixture. Consequently, the file is suitable for testing Level 3 I/O, WCS,
metadata preservation, floating-point handling, and geometric corrections,
but not for radiometry or quantitative science validation.

The normalized frame-300 source also contains 3,883 pixels clipped at the
simulator's 65,535-DN storage ceiling. Their true signal is unrecoverable;
`DATASAT`, `DSATVAL`, and processing history preserve that limitation, and PSF
deconvolution can spread their error spatially.

The versioned development CSV and live development sheet were corrected on
2026-08-31 so `TIMESYS` and all DATE descriptions specify UTC, `DATAMIN` and
`DATAMAX` accept floating-point values, and `CALPSF` is defined as the
content-addressed calibration-set manifest rather than one PSF file. Future CSV
exports must preserve these corrections. The provisional extension cards still
require a coordinated metadata-sheet revision before the production schema is
frozen.

For the synthetic bypass, the writer replaces known misleading placeholders
with `DOI = 'NOT_ASSIGNED'`, `OBSTYPE = 'SYNTHETIC DEVELOPMENT FIXTURE'`, and
`FILE_RAW = 'NOT_APPLICABLE_SYNTHETIC'`. Other observer position, attitude,
quality, and program fields are inherited from the simulator. They remain
explicit metadata debt and must not be treated as flight truth merely because
they satisfy the current presence-and-type validator.
