# Jetson hybrid Level 0.5–4 energy characterization

Date: 2026-09-25

## Result

Three measured runs in stock `MODE_30W` completed with full 100 ms
`tegrastats` coverage and no measurement-quality warnings. The median modeled
241-frame workload consumed **4,357.059 J** in **476.911 s**, or **18.079 J**
and **1.979 s** per 2×2-binned frame. Values below are medians with observed
three-trial minimum–maximum ranges.

| Processing level | Gross energy (J) | Duration (s) | Energy per target frame (J) | Observed peak power (W) | Peak temperature (°C) |
|---|---:|---:|---:|---:|---:|
| Level 0.5, modeled 241-frame image workload including DuckDB | 1,637.307 [1,631.818–1,648.525] | 185.522 [185.413–186.162] | 6.794 | — | — |
| Level 1 | 596.828 [593.904–598.337] | 63.603 [63.594–63.767] | 2.476 | 9.712 [9.517–9.812] | 47.5 [47.2–47.7] |
| Level 2, CuPy PSF deconvolution | 1,122.050 [1,121.506–1,126.767] | 117.463 [117.331–117.738] | 4.656 | 10.815 [10.416–10.818] | 48.0 [47.4–48.0] |
| Level 3, provisional pass-through | 468.252 [465.669–474.570] | 50.851 [50.844–51.501] | 1.943 | 9.414 [9.414–10.117] | 47.5 [47.3–48.0] |
| Level 4, headless CME tracking | 529.772 [526.942–530.716] | 58.980 [58.781–59.503] | 2.198 | 10.118 [10.118–10.517] | 47.9 [47.5–47.9] |
| **Modeled Level 0.5–4 total** | **4,357.059 [4,348.024–4,367.880]** | **476.911 [476.386–477.756]** | **18.079** | **10.815 [10.416–10.818] observed maximum** | **48.0 [47.5–48.0] observed maximum** |

The Level 0.5 modeled row has no independent modeled peak. Its corresponding
exact-file Level 0.5 plus DuckDB measurements peaked at a median 10.312 W and
47.7 °C. Peak power and temperature in the total row are maxima observed
across the real measured phases; they are not scaled by the image-count model.

For comparison, the direct exact-hybrid phase sum—one full pass through the
unaltered X-band file, DuckDB, and measured Levels 1–4—was **4,208.286 J** in
**461.001 s**, or **17.462 J/frame** when amortized across the 241 target
frames. The measured Level 1–4 portion alone was about 2,716.902 J in
290.898 s, or 11.273 J and 1.207 s per target frame.

At 15-second generation cadence, the modeled batch has 7.58× throughput
headroom; at 10 seconds it has 5.05×. A 241-frame sequence acquired over one
hour needs about 7.95 minutes of sequential Level 0.5–4 processing under this
benchmark. If the module were genuinely powered off for the rest of that hour,
4,357 J corresponds to a 1.210 W time-averaged *covered-module-rail* processing
burden. That is not a spacecraft-bus power-cycle result: boot, shutdown,
carrier-board/NVMe/fan losses, conversion efficiency, and off-state leakage
remain unmeasured.

## Workload definition

This is an explicit hybrid benchmark because no single representative raw file
contains the desired synthetic CME sequence:

1. The exact 1,309,312,200-byte flight-like X-band playback is processed
   through Level 0.5. It produces 124 FITS images: 118 at 1000×752 and six at
   500×376, totaling 89,864,000 pixels. Those images were test captures; their
   apparent timestamp span is not used to normalize the benchmark.
2. The already staged, reviewed synthetic sequence is selected by path after
   Level 0.5. The handoff performs no measured copy or transformation.
3. Levels 1–4 process all 241 synthetic frames at 1000×750 pixels and
   15-second cadence. `NBIN1=NBIN2=2`, so every measured frame is 0.75 MP and
   represents a 2000×1500 unbinned detector frame.
4. Level 0.5 runs twice on hard links to the same raw inode: once with CSIE
   image assembly disabled and once with it enabled. The paired energy delta is
   scaled by the exact pixel ratio `180,750,000 / 89,864,000 = 2.011372741`.
   Fixed transfer-frame parsing, packet recovery, and non-image telemetry work
   are retained once rather than doubled.
5. DuckDB ingestion is measured separately, then included in the Level 0.5 and
   total values.

The projected Level 0.5 value is a marginal image-assembly/FITS-write model. It
does **not** directly observe a raw downlink containing 241 synthetic images,
and it does not model any additional packet parsing that such a file might
require. The exact hybrid total and modeled total are therefore both retained.

The 2×2-binned 9.6 arcsec/pixel sampling is a reasonable onboard baseline:
the approximately 30 arcsec 80%-encircled-energy telescope PSF still spans
about three binned pixels, so the images remain optics-limited rather than
strongly pixel-limited for coherent CME-front tracking. This is an engineering
expectation, not a replacement for a future binned/unbinned science A/B test.
Do not extrapolate the present energy by simply multiplying by four; FFT and
tracking costs are not guaranteed to scale linearly with pixel count.

## Measurement protocol

- Jetson AGX Orin developer kit, L4T R39.2.1, Ubuntu 24.04.4, ARM64.
- Stock `MODE_30W` (mode ID 2), ordinary dynamic clocks; no clock-pinning
  command was applied. The non-root `jetson_clocks --show` diagnostic was not
  authorized, so its read-back is recorded as unavailable rather than inferred.
- Python 3.14.7 GPU environment with NumPy 2.5.2, SciPy 1.18.0, Astropy 8.0.1,
  SunPy 8.0.0, and CuPy 14.2.0/CUDA 13.2.
- One complete unmeasured preflight followed by three measured runs. Trial 2
  reversed the full/no-image Level 0.5 order to expose cache or thermal-order
  bias. The paired image-energy delta remained positive and similar: 147.10 J,
  136.93 J, and 138.91 J across the trials.
- Thirty-second pre-run settle, excluded from phase totals; required 100 ms
  telemetry; no idle subtraction.
- Gross module power is the simultaneous sum of `VDD_GPU_SOC`, `VDD_CPU_CV`,
  and `VIN_SYS_5V0`. `VDDQ_VDD2_1V8AO` is not added again.
- Each phase closes its child logs and executes a blocking Linux `sync(2)`
  barrier before its end marker. Deferred FITS/CSV/database writes therefore
  remain charged to the phase that created them.
- Level 0.5 writes core FITS and metadata only; PNG and JPEG2000 are disabled.
  Level 4 writes `track.ecsv`, `front_samples.ecsv`, `summary.json`, and
  `COMPLETE.json`; plots, overlay, and movie are disabled.
- Level 3 is intentionally a pass-through because this synthetic source does
  not require the planned fine rotation or special dark correction. Its energy
  covers FITS materialization, provenance, checksum/schema validation, and I/O,
  not the future Level 3 science algorithm.
- Three trials are enough for this engineering baseline and showed narrow
  spread, but they do not satisfy the plan's five-run minimum for a final
  flight acceptance characterization.

The benchmark used Git base commit
`9f266d09cb13a39c62468fe575c372cddcaad0dd` plus a dirty experimental snapshot.
The measurement records preserve the tracked-diff SHA-256 and SHA-256 for every
untracked source file, including the benchmark plan and runner. This avoids
pretending that the uncommitted code was a clean release.

## Output validation and artifacts

Every measured run passed the same post-run audit:

- 124 Level 0.5 FITS images and exactly 241 FITS files at each of Levels 1, 2,
  and 3;
- valid representative FITS `CHECKSUM` and `DATASUM` at each level;
- 30 DuckDB ingestions totaling 213,195 telemetry rows;
- zero PNG, JPEG2000, or MP4 artifacts; and
- successful Level 4 CME detection across all 241 frames.

The authoritative full measurement records remain on `suncet-soc` under:

```text
/srv/suncet/data/benchmarks/end_to_end_runs/20260925_trial_01_skip_full
/srv/suncet/data/benchmarks/end_to_end_runs/20260925_trial_02_full_skip
/srv/suncet/data/benchmarks/end_to_end_runs/20260925_trial_03_skip_full
```

The compact generated aggregate is preserved in
[`benchmarks/end_to_end/results/20260925`](../benchmarks/end_to_end/results/20260925/trial_summary.md).
Those JSON, CSV, and Markdown files are derived summaries; each Jetson
`measurement.json` remains authoritative.

## Interpretation and next tests

- Level 0.5 is the largest modeled contributor (about 37.6%), followed by
  Level 2 (25.8%), Level 1 (13.7%), Level 4 (12.2%), and the provisional Level 3
  pass-through (10.7%).
- The current sequential batch is comfortably faster than both 15-second and
  10-second image generation. This establishes throughput feasibility for the
  tested batch, not bounded latency for a streaming implementation.
- Repeat with an actual raw 241-image flight-like downlink when one exists;
  that will replace the Level 0.5 projection with a direct measurement.
- Repeat with implemented Level 3 corrections, approved calibration products,
  and representative flight data.
- Compare 2×2-binned and unbinned science fidelity and compute directly.
- Measure the complete power-on/boot/process/persist/shutdown cycle at the
  external spacecraft-bus input on flight-like hardware before using these
  covered-rail joules in a flight power budget.
