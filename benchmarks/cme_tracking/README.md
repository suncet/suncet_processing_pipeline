# CME tracking benchmarks

This directory contains portable benchmark manifests and, later, measured
accuracy/performance results. Large FITS inputs remain below `suncet_data` and
are referenced by relative path and SHA-256 rather than committed to Git.

The historical scenario is development data, not a final quantitative
science benchmark. Its FITS timestamps are frozen, but its relative cadence is
verified at 30 seconds from the confirmed one-hour source run, 10-second native
model step, and every-third source indices. Its 120 sample times span 0 through
3570 seconds; source index 360 is the omitted 3600-second endpoint. Absolute UTC
remains unavailable. No authoritative front-truth file exists for this run;
the legacy `Model Height Time.sav` material is explicitly excluded from
validation. Project-lead visual review accepts the current track as the frozen
engineering reference for compute and power comparisons only.

## Timestamp-correct particle-snow scenario

`config_default_no_particle_filter_20230114` is a separate 241-frame simulator
run at 15-second cadence. Its FITS `DATE-OBS` values, `DATE-BEG` values, and
filenames agree exactly from `2023-01-14T17:00:00` through
`2023-01-14T18:00:00` UTC; the final image ends at `18:00:15`. The reviewed
manifest therefore uses the FITS headers as an authoritative absolute time
axis instead of supplying an assumed cadence:

```text
benchmarks/cme_tracking/manifests/
  config_default_no_particle_filter_20230114.json
```

Particle filtering was disabled, and this run intentionally contains
worst-case particle snow. The unrestricted reference tracker associated
isolated particle hits across most position angles and prematurely latched
field-of-view contact. Its coverage/confidence gates then withheld kinematics,
which is the correct fail-safe outcome rather than a plausible-looking false
track.

For this known event only, project-lead review identified the CME sector as
solar position angles 220--330 degrees. The historical manual configuration
restricts front association to that reviewed, wrap-aware window:

```text
suncet_processing_pipeline/config_files/
  cme_tracking_config_default_no_particle_filter_20230114_v0.json
```

That result remains a visually reviewed comparator, not automatic discovery.
The recommended automatic known-window configuration is:

```text
suncet_processing_pipeline/config_files/
  cme_tracking_config_default_no_particle_filter_20230114_autonomous_v1.json
```

It searches the full circle for same-frame, radially continuous angular
fragments and links them using temporal PA overlap, radial shape, bounded gaps,
and outward motion. A robust displacement gate requires both endpoint-window
progress and a central radial span, rejecting the persistent inner-limb arc
that otherwise wins through duration. The retained component spans 236--332
degrees; eight degrees of padding defines a data-derived sector for detailed
per-angle sparse-path refinement. No reviewed PA values are supplied.

On this scenario the automatic result reports an uncensored 1.578--4.321 solar
radii, a provisional median projected speed of 757.9 km/s, and coherent FOV
contact at frame 181. The manual 220--330-degree result reports 1.570--4.321
solar radii and 715.8 km/s. This agreement is an engineering result only: the
scenario has no authoritative contour truth, and known-window sector discovery
is not continuous event discovery. Reproduce the automatic result with:

```sh
python -m suncet_processing_pipeline.make_level4 cme-track \
  --manifest \
    benchmarks/cme_tracking/manifests/config_default_no_particle_filter_20230114.json \
  --data-root "$suncet_data" \
  --config \
    suncet_processing_pipeline/config_files/cme_tracking_config_default_no_particle_filter_20230114_autonomous_v1.json \
  --event-id config-default-no-particle-filter-20230114-autonomous-v1-event-001 \
  --diagnostic-movie \
  --movie-fps 10
```

The manual configuration's seven-frame scalar-height screen rejects frames
159, 167, and 168; the automatic sector excludes the first excursion and the
same screen rejects 167 and 168. Raw front and headline-height measurements are
never replaced. Inputs remain below `$suncet_data/synthetic/level0/fits`; the
command does not copy them. Association limits are still expressed per frame,
so their physical spans differ from the historical 30-second run.

The isolated particle-noise experiment is implemented as a disabled-by-default,
NaN-aware three-frame temporal median on the polar intensity cube before
spatial smoothing and likelihood construction. The first and last global
frames are unsupported instead of padded; the retained rows keep their
original timestamps and online use has one-frame latency. Its configuration is:

```text
suncet_processing_pipeline/config_files/
  cme_tracking_config_default_no_particle_filter_20230114_temporal_median3_v1.json
```

The reproducible A/B runner is
`benchmarks/cme_tracking/compare_temporal_median.py`. On the 241-frame
particle-snow case, median-3 removes 86.7% of temporally isolated radial-peak
candidates and 37.2% of all radial-peak candidates. It retains 239 measured
frames versus 240 for raw (the difference is the explicit endpoints), moves
coherent FOV contact from frame 181 to 195, and removes the raw scalar-height
outliers at frames 167--168. Median projected speed changes from 757.9 to
748.0 km/s. Before raw FOV contact, the median absolute headline-height shift
is 0.030 solar radii; across the 79 frames where both kinematic results are
valid it is 0.035 solar radii. All predeclared real-event engineering gates
pass.

This no-truth event establishes particle rejection and track stability, not
accuracy. An analytic broad front is preserved within one radial bin median
bias, while a deliberately fast, thin five-pixel-per-frame shell is erased.
Median-3 therefore remains opt-in pending broader cadence/morphology tests and
must not silently become the production default. Saved comparison artifacts
belong below:

```text
$suncet_data/level4/cme_tracking/temporal-median3-ab-20260827/
```

Historical-manifest regeneration intentionally requires a new reviewer
identity and timestamp; it must not silently transfer the existing review
record to changed files:

```sh
PYTHONPATH=. python benchmarks/cme_tracking/create_bright_fast_manifest.py \
  --reviewed-by "<reviewer>" \
  --reviewed-at-utc "<ISO-8601 UTC timestamp>"
```

The CPU reference benchmark keeps the manifest's 30-second scientific cadence
separate from the 15-second nominal and 10-second sizing arrival cadences:

```sh
python -m suncet_processing_pipeline.benchmark_cme_tracking \
  --manifest benchmarks/cme_tracking/manifests/bright_fast_no_jitter_historical.json \
  --config suncet_processing_pipeline/config_files/cme_tracking_reference_v0.json \
  --data-root "$suncet_data" \
  --output-root "$suncet_data/benchmarks/cme_tracking" \
  --scope compute \
  --warmups 1 \
  --repetitions 5 \
  --minimum-measurement-seconds 30 \
  --idle-baseline-seconds 30 \
  --tegrastats-interval-ms 200 \
  --deadline-cadences 15,10 \
  --allow-inconsistent-synthetic-geometry \
  --expected-science-reference-npz \
    "$suncet_data/benchmarks/cme_tracking/references/bright_fast_no_jitter_reference_v0/science_reference.npz" \
  --expected-science-reference-json \
    "$suncet_data/benchmarks/cme_tracking/references/bright_fast_no_jitter_reference_v0/science_reference.json" \
  --require-reference-equivalence
```

Each completed run is an immutable directory with system and source
provenance, the frozen array-level science signature, repetition-level timing,
RSS and deadline metrics, raw and parsed `tegrastats`, energy summaries, and a
checksummed `COMPLETE.json`. The current full-window algorithm is noncausal;
average-rate feasibility is not evidence of a streaming implementation.

The first authoritative Jetson result is the stock 30 W operational-DVFS run:

```text
$suncet_data/benchmarks/cme_tracking/jetson/
  20260826T201945.506826Z_suncet-soc_a1858558/
```

It records a median 18.435 s per 120-frame event, 150.236 J gross and
26.410 J above idle per event on the covered onboard rails, and numerical
equivalence to the reviewed Mac reference. Earlier development runs that used
periodic RSS polling are superseded for timing and energy conclusions. The
authoritative harness uses the kernel-maintained process RSS high-water mark
without a sampling thread. The corresponding clean profile is stored at
`$suncet_data/benchmarks/cme_tracking/jetson/profiles/` and identifies
likelihood percentile normalization and per-angle candidate/path extraction as
the first optimization targets.

The first retained CPU A/B result is:

```text
$suncet_data/benchmarks/cme_tracking/jetson/
  20260826T203106.637354Z_suncet-soc_740e0719/
```

It skips peak finding when a score row's global maximum is below threshold,
eliminating 13.1% of peak-search calls on this event. Against the baseline it
is 0.93% faster, uses 1.54% less gross covered-rail energy, and uses 3.62% less
incremental energy above idle. Exact discrete science outputs are unchanged,
and all continuous outputs remain within the frozen cross-platform reference
tolerances.

The matching stock 15 W run is:

```text
$suncet_data/benchmarks/cme_tracking/jetson/
  20260827T152517.802796Z_suncet-soc_71d4fc6a/
```

It completes the event in 28.249 s versus 18.264 s at 30 W. Its lower load and
idle power reduce incremental event energy by 21.9%, but the longer runtime
raises gross compute-window energy by 30.8%. Compare modes over a common
scheduling horizon as well as over the compute window; neither metric alone
defines the flight-optimal policy.

The matching stock 50 W run is
`20260827T153948.149435Z_suncet-soc_150e766b`. It is dominated by 30 W for this
largely serial CPU workload: 13.9% slower, 12.6% higher gross event energy,
0.3% higher incremental event energy, and 0.8% higher projected energy over a
fixed 1,200 s horizon. All science outputs remain equivalent to the frozen
reference.

The matching MAXN upper-bound run is
`20260827T160253.877457Z_suncet-soc_e4d81622`. Relative to 30 W, it is 19.5%
faster and uses 6.8% less gross compute-window energy, while using 47.9% more
incremental event energy and 1.5% more projected energy over the fixed 1,200 s
horizon. It reached 10.609 W peak on the covered rails and 46.3 degrees C for
this CPU-only workload. Use 30 W as the normal ground-SOC mode; treat 15 W as
the leading always-on fixed-horizon candidate. For the intended flight concept
of powering on, processing an accumulated batch, and powering off, MAXN has the
lowest compute-only gross energy and is therefore the provisional candidate,
with 30 W as the bounded fallback. Neither is a flight selection until full
spacecraft-bus power-on-to-power-off cycles—including boot, I/O, combined
Level 2/3/4, product persistence, and shutdown—are measured on flight-like
hardware.


The reference pair is a preserved array-level export of the project-lead-
reviewed Mac run, stored below `suncet_data` rather than Git. Cross-platform CPU
comparisons require exact discrete front/path outputs. Floating outputs use the
versioned tolerance policy recorded in each comparison: `1e-12` absolute and
relative by default, with `5e-7` absolute and `1e-7` relative for the
intermediate likelihood cube. The exception covers observed float32-level
NumPy/SciPy build variation; it does not permit a changed selected front.
