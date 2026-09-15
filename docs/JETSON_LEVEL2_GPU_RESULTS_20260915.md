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

## End-to-end command power

A second comparison measured fresh `make_level2` child processes rather than
only the prepared deconvolution core. This scope includes interpreter/import
startup, input and calibration reads and hashes, calibration preparation,
deconvolution, provenance, FITS construction, checksum and schema validation,
and disk write. Values are medians of two reverse-order trials; gross energy
has no idle subtraction.

| Scope and backend | Command time | Gross command energy | Average covered-rail power | Sampled peak power | Peak `tj` | Time per product | Gross energy per product |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| One product, NumPy | 7.846999 s | 66.631626 J | 8.491305 W | 10.1665 W | 44.3435 °C | 7.846999 s | 66.631626 J |
| One product, CuPy | 8.601874 s | 73.471368 J | 8.544994 W | 9.9660 W | 44.2965 °C | 8.601874 s | 73.471368 J |
| 20 products, NumPy | 19.011558 s | 169.363299 J | 8.906345 W | 10.1670 W | 44.4840 °C | 0.950578 s | 8.468165 J |
| 20 products, CuPy | 17.012089 s | 153.172411 J | 9.004665 W | 10.1665 W | 44.2340 °C | 0.850604 s | 7.658621 J |

For one product, CuPy was `9.62%` slower and used `10.27%` more gross energy:
its one-time setup cost outweighed the faster frame operation. For a 20-product
single-process batch, CuPy was `10.52%` faster and used `9.56%` less gross
energy. Every command interval had complete bracketing telemetry. The
single-product intervals were shorter than the conservative 100-sample quality
threshold and therefore retain a warning; both 20-product trials exceeded it
without warnings.

The batch fixture uses 20 hard links to the same frozen frame. This is valid
for the data-independent FFT and FITS I/O timing question, but it is not a
science sequence. All 84 generated products passed an independent FITS
checksum, dimensions, finite-pixel, backend-history, clean-commit provenance,
and within-trial determinism audit. CPU/CuPy pixel errors reproduced the core
comparison exactly.

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
choice: it doubles core throughput while using materially less gross core
energy per frame. A fresh process handling only one product should currently
use NumPy. A two-point interpolation between the one- and 20-product command
measurements puts the provisional end-to-end crossover at about seven products
for both time and gross energy. At the nominal 10–15 second image cadence, that
corresponds to buffering roughly 70–105 seconds of images. This is engineering
guidance, not yet an onboard scheduling requirement: the threshold should be
remeasured with distinct representative images and cold boot/page-cache state,
then with complete Level 1-to-Level 4 processing and output persistence.

These results characterize the deconvolution core and a provisional Level 2
command, not a complete powered Level 1-to-Level 4 cycle or spacecraft-bus
energy. The provisional calibration assets also remain engineering inputs and
are not mission-approved science calibrations.

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
- Fresh-process single-product command-power trials:
  `/srv/suncet/data/benchmarks/level2/end_to_end_30w_b18817e_20260915_v1/`
- Fresh-process 20-product command-power trials:
  `/srv/suncet/data/benchmarks/level2/end_to_end_batch20_30w_b18817e_20260915_v1/`
