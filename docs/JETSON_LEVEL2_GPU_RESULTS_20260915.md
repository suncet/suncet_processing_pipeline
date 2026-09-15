# Jetson Level 2 FP64 CPU/GPU benchmark — 2026-09-15

## Result

The opt-in CuPy FP64 backend at clean pipeline commit `b18817e` passed the
Level 2 numerical-equivalence gate and reduced warm prepared-deconvolution
time and gross covered-rail energy per frame by about half. NumPy remains the
default production-facing backend.

| Backend | Calibration preparation[^prep] | First application | Warm median per frame | Throughput | Gross energy per frame[^energy] | Average covered-rail power | Sampled peak power | Peak `tj` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NumPy FP64 | 1.178659 s | 0.311535 s | 0.310753 s | 3.218 frame/s | 2.864537 J | 9.216642 W | 9.217 W | 44.2655 °C |
| CuPy FP64 | 1.878554 s | 0.170766 s | 0.154353 s | 6.479 frame/s | 1.551054 J | 10.047208 W | 10.117 W | 44.0780 °C |

Compared with NumPy, CuPy was `2.0133×` faster, reduced warm-frame latency by
`50.33%`, and reduced gross covered-rail energy per frame by `45.85%`, while
average covered-rail power increased by `9.01%`. Both paths comfortably remain
ahead of the commandable 10–15 second image cadence for this
deconvolution-core scope.

[^prep]: Calibration preparation used a warm filesystem page cache. Each
    preparation interval was shorter than the 10-second telemetry-quality
    threshold, so these values are useful for engineering guidance but are
    not a cold-start power characterization.

[^energy]: Gross energy is measured batch energy divided by processed frames.
    It is the `tegrastats` on-module estimate `VDD_GPU_SOC + VDD_CPU_CV +
    VIN_SYS_5V0`, with no idle subtraction. It excludes Level 1, FITS product
    construction and writing, boot and shutdown, carrier conversion losses,
    and energy while the Jetson is off.

## Numerical acceptance

Both reverse-order trials produced the same numerical comparison: all
`750 × 1000` pixels were finite, maximum absolute error was
`7.275957614e-11`, RMS error was `2.834360783e-12`, and relative L2 error was
`7.711638740e-16`. This passes the engineering limits of `1e-8` absolute and
`1e-12` relative error. Each backend was deterministic across its two trials.

## Method and provenance

- Hardware and system software: Jetson AGX Orin, L4T R39.2.1 / JetPack 7.2.1,
  Ubuntu 24.04, CUDA 13.2, CuPy 14.2.0, and Python 3.14.7.
- Power policy: `MODE_30W` (`nvpmodel` mode 2), dynamic clocks, and about
  6.714 W pre-run idle power on the covered rails.
- Code: clean detached worktree at
  `b18817ef96e0ed96d1f7a7503951daeb83153549`.
- Input: one provisional pre-Level-2 synthetic frame and the exact staged
  provisional calibration assets.
- Trial design: two reverse-order NumPy/CuPy pairs, with a 20-second idle
  settle, at least 20 seconds of measured work, and 100 ms telemetry sampling.
- CuPy scope: input host-to-device transfer, both inverse-filter stages, and
  one final device-to-host result transfer.

The CUDA smoke test also passed a representative `1500 × 2000` FP64 FFT/IFFT
pair with a median duration of `0.116302 s`, maximum absolute round-trip error
of `2.22e-15`, and relative L2 error of `3.97e-16`.

## Interpretation

For a continuously running prepared pipeline, CuPy is the clear current
choice: it doubles throughput while using materially less gross energy per
frame. Process-local preparation was slower on the GPU. Using the warm-cache
timing and provisional preparation-energy estimates (`10.605 J` for NumPy and
`16.760 J` for CuPy), both time and energy break even at roughly five frames
per process. Treat that batch-size result as preliminary until cold boot,
page-cache, end-to-end Level 2/3/4 processing, output persistence, and shutdown
are measured with external DC-input instrumentation.

These results characterize the deconvolution core, not a complete powered
processing cycle or spacecraft-bus energy. The provisional calibration assets
also remain engineering inputs and are not mission-approved science
calibrations.

## End-to-end FITS acceptance

The CuPy backend also passed a separate run through the real `make_level2`
command-line path at the same clean commit. The resulting `750 x 1000`
binary64 FITS product has valid `CHECKSUM` and `DATASUM` values, contains only
finite pixels, passes the `v1.0.2dev` Level 2 metadata contract, references its
content-addressed calibration manifest, and records
`Deconvolution array backend: cupy (FP64)` in `HISTORY`. Its comparison with
the frozen CPU FITS reproduced the core-test errors exactly: maximum absolute
error `7.275957614e-11`, RMS error `2.834360783e-12`, and relative L2 error
`7.711638740e-16`.

This acceptance artifact deliberately uses `v1.0.2dev`. Activating the newer
`v1.0.3dev` metadata contract remains gated on teaching the upstream product
writer to derive the newly required `SOLAR_R` keyword; that schema task is
independent of the CuPy backend.

## Artifacts

- Controlled trial pair:
  `/srv/suncet/data/benchmarks/level2/controlled_30w_b18817e_20260915_v1/`
- CuPy runtime smoke result:
  `/srv/suncet/data/benchmarks/level2/cupy_fp64_smoke_20260915_v2/`
- End-to-end CuPy FITS product and validation record:
  `/srv/suncet/data/benchmarks/level2/product_smoke_gpu_b18817e_20260915/`
