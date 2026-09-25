# SunCET end-to-end Jetson benchmark trial summary

Trials: **3**. Values are median [minimum–maximum].

## Per-phase measurements

| Phase | Gross energy (J) | Duration (s) | Sampled peak power (W) | Peak temperature (°C) | J/frame | J/MP |
|---|---:|---:|---:|---:|---:|---:|
| level0_5_skip_images | 1110.968 [1098.954–1126.305] | 127.509 [126.318–127.796] | 9.819 [9.817–10.013] | 47.3 [46.3–47.7] | — | — |
| level0_5_full | 1249.875 [1246.055–1263.233] | 142.721 [142.420–142.727] | 9.817 [9.817–10.215] | 47.6 [47.1–47.8] | — | 13.909 [13.866–14.057] |
| level0_5_duckdb | 242.479 [241.455–246.806] | 27.457 [27.308–27.694] | 10.312 [10.312–10.315] | 47.7 [47.2–47.9] | — | — |
| level1 | 596.828 [593.904–598.337] | 63.603 [63.594–63.767] | 9.712 [9.517–9.812] | 47.5 [47.2–47.7] | 2.476 [2.464–2.483] | 3.302 [3.286–3.310] |
| level2 | 1122.050 [1121.506–1126.767] | 117.463 [117.331–117.738] | 10.815 [10.416–10.818] | 48.0 [47.4–48.0] | 4.656 [4.654–4.675] | 6.208 [6.205–6.234] |
| level3 | 468.252 [465.669–474.570] | 50.851 [50.844–51.501] | 9.414 [9.414–10.117] | 47.5 [47.3–48.0] | 1.943 [1.932–1.969] | 2.591 [2.576–2.626] |
| level4 | 529.772 [526.942–530.716] | 58.980 [58.781–59.503] | 10.118 [10.118–10.517] | 47.9 [47.5–47.9] | 2.198 [2.186–2.202] | 2.931 [2.915–2.936] |

`MP` means one million processed image pixels. A dash means the phase did not declare the corresponding workload unit.

## Hybrid totals and paired projection

| Model | Total | Gross energy (J) | Duration (s) | Average power (W) | J/target frame | J/target MP |
|---|---|---:|---:|---:|---:|---:|
| hybrid_241_frame_pipeline | Exact hybrid phase sum | 4208.286 [4207.537–4229.394] | 461.001 [460.626–462.655] | 9.136 [9.127–9.142] | 17.462 [17.459–17.549] | 23.282 [23.278–23.399] |
| hybrid_241_frame_pipeline | Projected target workload | 4357.059 [4348.024–4367.880] | 476.911 [476.386–477.756] | 9.136 [9.127–9.142] | 18.079 [18.042–18.124] | 24.105 [24.055–24.165] |

## Scope and caveats

- **Level 3:** Benchmark-only Level 3 pass-through; no applicable rotation, geometry, or special dark correction is available for this source
  The Level 3 measurement covers the present pass-through, validation, and file-I/O work only; it is not an estimate of future Level 3 rotation, geometry, or special dark-correction algorithms.
- **Plan note:** Real X-band telemetry is processed exactly once through Level 0.5. Its 124 images were test captures, so their apparent timestamp span is not used for workload normalization. The reviewed 241-frame, 15-second synthetic sequence is selected by path for Levels 1-4; no measured copy or conversion is performed at the handoff. Those synthetic arrays are 1000x750 pixels with NBIN1=NBIN2=2 (0.75 megapixel per measured frame; 2000x1500 is the unbinned detector equivalent). At the binned 9.6 arcsec/pixel scale the approximately 30 arcsec 80%-encircled-energy telescope PSF still spans about three pixels, so this default onboard binning is expected to preserve Level 4 front-tracking information while reducing pixel count by four; an eventual unbinned A/B test remains required to verify that expectation. Level 3 is a benchmark-only pass-through because no applicable geometric/dark-correction source data exist. Only the paired marginal CSIE image cost is scaled from the raw file's observed image pixels to the 241-frame synthetic pixel workload.
- **Projection `hybrid_241_frame_pipeline`:** Keep exact-file fixed Level 0.5 work and scale only the marginal CSIE image assembly from 89,864,000 observed pixels to the 241-frame synthetic workload of 180,750,000 pixels; then add DuckDB and measured Levels 1-4.
  - The projection assumes variable image-processing cost is linear in image_pixels; repeat the pair in reversed order to bound cache and thermal-order bias.
  - Scope: Phase-only model; excludes paired diagnostic duplication, interphase gaps, substitution/staging, supervisor work, network transfer, boot, and shutdown.
