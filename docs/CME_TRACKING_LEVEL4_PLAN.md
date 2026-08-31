# SunCET Level 4 Automated CME Tracking Plan

Last updated: 2026-08-31

Status: first known-window vertical slice, full-circle automatic angular-sector
discovery, and stock-power-mode Jetson characterization implemented. The
two-stage tracker recovers the timestamp-correct worst-case particle-snow case
without a supplied position-angle sector. Scientific validation, continuous
event discovery, and additional baselines remain in progress.

## Purpose

This document tracks the development, validation, and Jetson characterization
of the first SunCET Level 4 science product: automated coronal mass ejection
(CME) detection and tracking.

The intended science outputs are:

- apparent plane-of-sky height versus time;
- apparent plane-of-sky speed versus time;
- apparent plane-of-sky acceleration versus time;
- CME position angle, its evolution, and angular width;
- a mission CME event catalog; and
- plots and an image movie with the inferred front overlaid.

This is one Level 4 product family, not the definition of all Level 4
processing. Other Level 4 products must be able to use the same dispatcher and
provenance framework without depending on CME-specific code.

The project has a second, flight-oriented purpose: characterize accuracy,
latency, memory use, and energy use on `suncet-soc`, then assess whether an
industrial Jetson AGX Orin could support a future onboard implementation. The
ground SOC does not need energy optimization for this workload. The compelling
future use case is to process full-resolution, full-cadence images onboard and
downlink a compact height-time product when image downlink is constrained,
especially for a deep-space SunCET concept.

The Jetson AGX Orin developer kit is an algorithm-development and screening
platform. It cannot by itself establish flight qualification or the
spacecraft-bus power of a future industrial module.

## Executive recommendation

Do not begin with a generic object detector and do not define the CME as the
brightest pixel or outermost threshold crossing in each independent image.
Those are useful baselines, but they discard the strongest information in this
problem: a CME front is spatially broad and moves coherently outward through
time.

The common internal representation should be a polar front-likelihood cube:

```text
L(time, position_angle, heliocentric_radius)
```

All detector variants should produce this representation. A temporal tracker
then extracts a coherent front `r(time, position_angle)` while enforcing only
well-documented physical constraints and allowing data gaps. Apex height,
central position angle, and angular width are derived from the retained front
instead of being selected independently in each frame.

The recommended development ladder is:

1. A simple SEEDS-like threshold/connected-region tracker to establish the
   lowest-complexity baseline.
2. A CACTus-like linear Hough detector as an interpretable constant-speed
   temporal baseline.
3. A CIISCO-like outbound-motion and parabolic-Hough detector as a particularly
   relevant inner-corona EUV and constant-acceleration baseline.
4. The primary candidate: CORIMP-inspired multiscale front evidence followed by
   a constrained temporal path or graph optimizer. This permits nonconstant
   acceleration and incomplete fronts.
5. Only after the classical pipeline is measured, test a learned segmenter as
   another front-likelihood generator. It must use the same tracker, product
   schema, and evaluation harness.

CACTus is a good starting reference, but not the intended final algorithm. Its
linear Hough transform finds straight ridges in height-time space and therefore
assumes constant apparent speed. SunCET is specifically intended to measure CME
acceleration through the inner and middle corona. CIISCO is closer to the
mission problem because it uses EUV images and a parabolic Hough transform, but
its uniform-acceleration and fixed-start assumptions are still too restrictive
for the final science product.

First solve tracking inside a supplied single-event time window. Continuous
event discovery, separation of overlapping CMEs, and mission-catalog identity
are a later layer. Keeping those two problems separate makes the first
validation scientifically meaningful without pretending that a clean synthetic
sequence proves autonomous catalog performance.

Keep one scientifically authoritative reference method and product schema, but
allow a later onboard candidate to use different approximations. A bifurcation
is acceptable if regression tests quantify its science differences; the ground
and onboard methods must not silently drift into products with the same name
and different meaning.

## Is the task hard?

There are several different tasks hiding inside the phrase “find the bright
front.”

| Scope | Assessment | Reason |
| --- | --- | --- |
| Track one prominent synthetic CME in a known window | Moderate and very feasible | Polar remapping plus a simple temporal tracker should yield a useful first result quickly. |
| Produce stable height and speed for faint or artifact-contaminated events | Moderately hard | The CME is diffuse, changes shape, and competes with stationary and moving coronal structures. |
| Produce defensible acceleration with uncertainty | Harder than front detection | Differentiation magnifies small height errors; a second derivative magnifies them again. |
| Discover and separate events autonomously in continuous mission data | Hard | Event identity, overlapping eruptions, gaps, false fronts, and catalog policy become part of the problem. |
| Demonstrate robustness on real SunCET data | Hard and cannot be completed pre-launch | Synthetic-to-real mismatch and previously unseen instrument behavior must be measured in flight. |
| Compare compute and energy on the Jetson | Straightforward if controlled | The tools exist, but power mode, thermal state, timing scope, and rail coverage must be recorded consistently. |

The difficult parts are not merely computational:

- A CME is not a rigid object with a crisp, permanent boundary.
- An EUV front may represent different plasma structure than a white-light CME
  front. A single-view image supplies projected plane-of-sky motion, not the
  true three-dimensional trajectory.
- Running differences can make faint motion visible, but they also create
  paired bright/dark edges and amplify jitter, changing exposure, composite
  boundaries, and other artifacts.
- The current synthetic images visibly contain strong short/long-exposure
  boundaries, scattered-light structure, noise, and occasional image seams.
  Some of those gradients can be stronger than the CME front.
- The “leading edge” can vary with position angle. Reducing the entire front to
  one outermost pixel can cause the reported height and angle to jump between
  unrelated structures.
- Speed and especially acceleration are model-derived quantities. Raw finite
  differences are unsuitable as the production estimate.

The project is therefore achievable, but a clean demonstration and a trusted
autonomous Level 4 science product are different milestones.

## Science definitions and conventions

The product contract must distinguish measurements from derived summaries.

### Front representation

The primary measurement is a set of front samples:

```text
r(time, position_angle)
```

Each sample has a likelihood or confidence, positional uncertainty, and quality
flags. A missing or unreliable front at an angle remains missing; the tracker
must not fabricate a complete arc solely to make a visually smooth movie.

The exact definition of the target front still needs a science decision. In
different events, “front” could mean the outer EUV wave or shock-like feature,
the outer boundary of the erupting flux rope, or another coherent leading
structure. That definition should be written before final ground-truth labels
are generated.

### Height

- `height_raw` is the measured projected heliocentric radius of the selected
  coherent front summary before kinematic smoothing.
- `height_fit` is the value of the accepted smooth kinematic model.
- Heights are reported in solar radii and projected kilometers.
- The product must explicitly label these as plane-of-sky quantities.
- Synthetic three-dimensional MHD height is comparison truth, not the same
  observable. Both may be retained in validation results but must not share a
  field name.

The precise event-level height summary—apex, height at a central position angle,
or another robust statistic—will be selected against synthetic truth. The full
angularly resolved front is retained so this choice can be changed without
rerunning front detection.

### Position angle

Use a canonical solar position angle of 0 degrees at solar north, increasing
counterclockwise in the north-up science image. This matches the convention
used by the directly relevant CACTus and CIISCO literature.

Solar position angle is the only required angular coordinate. No separate clock
angle will be produced.

### Speed and acceleration

Speed and acceleration are derivatives of a fitted height-time relationship,
not independent frame-to-frame measurements. The product retains the raw
height samples, the fitted height, the fit method and parameters, residuals,
and propagated uncertainty.

The initial method should use actual observation times with either a weighted
smoothing spline or a local polynomial fit. It must support irregular cadence
and gaps. Residual-resampling or measurement Monte Carlo will propagate height
uncertainty into speed and acceleration uncertainty. Ordinary Savitzky-Golay
filtering may be a comparator for regularly sampled sequences, but must not
silently assume equal spacing.

Kinematic values near either end of the fitted interval are usually less
constrained and must carry a quality flag or be omitted where their uncertainty
is not useful.

## End-to-end architecture

```mermaid
flowchart LR
    L2[Level 2 PSF-deconvolved images] --> L3[Level 3 fine geometric corrections]
    L3 --> QC[Production input and WCS validation]
    SYN[Synthetic development adapter] --> QC
    QC --> P[Polar remap and normalization]
    P --> F[Front-likelihood cube]
    F --> D[Event discovery]
    F --> T[Temporal and angular tracker]
    D --> T
    T --> R[Front r(t, angle) with uncertainty]
    R --> K[Kinematic fit]
    K --> O[Level 4 tables, catalog, plots, and movie]
    R --> O
```

For the first single-event prototype, an operator-supplied time window replaces
the event-discovery block.

### 1. Input validation

- Accept Level 3 FITS images in production. Level 2 adds PSF deconvolution;
  Level 3 adds fine geometric corrections such as making solar north point
  precisely up.
- For initial algorithm development, explicitly allow reviewed synthetic FITS
  images to bypass unfinished Level 2 and Level 3 processing.
- Validate timestamps, WCS, solar center, pixel scale, `RSUN_REF`, data units,
  quality flags, and monotonic ordering.
- Preserve actual cadence. A reviewed synthetic manifest or explicit CLI option
  may assign a fixed cadence when timestamps are frozen. Record its review
  status and evidence in every product.
- Reject or flag duplicate times, physically inconsistent WCS, missing solar
  center, and sequences too short for the requested kinematic fit.

### 2. Preprocessing

- Normalize exposure and units using Level 3 metadata.
- Mask unusable pixels and known composite/calibration boundaries where
  required.
- Remap the useful field of view into
  `(position_angle, heliocentric_radius)` coordinates.
- Generate several evidence channels without committing the product to one:
  radial normalization, base difference, running difference, temporal
  normalization, radial gradient, and multiscale oriented edges.
- Optionally apply a Fourier `k-omega` filter that suppresses static and inbound
  structure, as in CIISCO. Its ringing and speed cutoff must be measured rather
  than assumed harmless.

### 3. Front likelihood

Combine the evidence channels into a normalized likelihood or score cube. The
first implementation is deterministic and interpretable. Each detector variant
implements the same interface:

```python
likelihood = detector.score(sequence)
```

The score is not itself a calibrated probability until calibration is
demonstrated.

### 4. Tracking

Extract coherent outward paths using dynamic programming, Viterbi-style
optimization, or a graph formulation. Candidate transitions can use:

- nondecreasing radius during the outward phase;
- configurable apparent-speed and apparent-acceleration limits;
- penalties for abrupt changes rather than a forced constant velocity;
- continuity across neighboring position angles;
- explicit missing-frame and missing-angle states; and
- event-level coherence so isolated noise cannot define the apex.

Physics limits remain configuration and provenance data. They must not be
hidden constants tuned only to the first bright-fast simulation.

The known-window prototype may optionally apply a reviewed, wrap-aware
position-angle window before path and angular-component selection. This is an
explicit operator prior for difficult development cases, not autonomous event
or sector discovery. Production discovery must instead identify a coherent
time--position-angle component from the evidence and reject sparse particle
tracks before field-of-view and kinematic decisions are made.

The implemented automatic known-window method now uses two distinct stages.
It first constructs same-frame, radially continuous angular fragments over the
full circle and links them through time using angular overlap, radial-shape,
gap, and outward-motion constraints. Robust displacement requires both
endpoint-window progress and a central 10th-to-90th-percentile radial span, so
a persistent inner arc cannot qualify because of one late excursion. The
selected coherent component defines a padded angular sector; the established
per-position-angle sparse tracker then recovers the detailed front inside that
data-derived sector. This is automatic angular-sector discovery inside a
supplied single-event time window, not continuous event discovery.

For the timestamp-correct particle-snow development case, the event-specific
configuration also applies a centered seven-frame median screen to scalar
headline heights before kinematic fitting. Otherwise eligible measurements
with residuals greater than 0.2 solar radii are flagged and omitted from the
spline; raw heights and angular front samples remain unchanged. This is a
transparent provisional safeguard, not a substitute for rejecting particle
tracks during autonomous angular-component discovery.

A three-frame temporal median on the polar intensity cube, before spatial
smoothing and all likelihood channels, is now an opt-in A/B-tested experiment
for particle-contaminated inputs. A plain running difference maps a one-frame
saturated hit into two large signed impulses rather than removing it; a
centered median rejects uncorrelated hits before they affect spatial spread or
robust normalization. It remains optional while broader faint/fast/cadence
tests are outstanding.
The centered form needs a three-frame ring buffer and one-frame latency. The
first and last likelihood frames are explicitly marked unsupported rather than
filled by duplicated or two-sample edge values.

### 5. Event association

For continuous operation, group coherent paths across time and angle into
events. This layer supplies event start/end time, angular span, and event
identity. It must support:

- no-event intervals;
- successive events in the same angular sector;
- simultaneous events in different sectors;
- partially overlapping fronts; and
- a later event crossing a previous event's disturbed corona.

### 6. Kinematics and product generation

Fit height against actual time, estimate uncertainty, derive speed and
acceleration analytically from the accepted fit, and write the tables and
diagnostics. The movie overlay is a verification product generated from the
same retained front samples, not a separate detection path.

## Algorithm baselines

| Candidate | Core idea | SunCET role | Main limitation |
| --- | --- | --- | --- |
| Framewise threshold / SEEDS-like | Detect bright regions in polar running-difference images and select an outer boundary | Minimum-complexity baseline and a clear test of the “just find the bright front” idea | Threshold sensitivity, false fronts, and weak temporal reasoning |
| CACTus-like | Detect straight ridges in time-height slices with a modified Hough transform, then cluster across angle | Interpretable temporal and constant-speed benchmark | Straight ridges suppress the acceleration that SunCET is intended to measure |
| CIISCO-like | Use radial normalization, outbound Fourier motion filtering, and a parabolic Hough transform | Highly relevant EUV inner-corona and constant-acceleration baseline | Uniform acceleration, fixed start assumptions, Fourier ringing, and reported false positives |
| CORIMP-inspired plus path tracker | Dynamic background separation, multiscale oriented edges, and temporal/angular chaining | Recommended classical production candidate | More implementation and tuning work |
| Learned segmentation | Generate a front likelihood or mask with a compact neural model | Later Jetson GPU comparator and possible robustness improvement | Requires labels, invites synthetic-to-real domain error, and does not solve temporal association by itself |

Optical flow can be evaluated as an evidence channel, including a later
Jetson-accelerated experiment, but should not define the science measurement.
EUV morphology evolves, so apparent pixel flow is not automatically the motion
of one physical front.

## Current synthetic-data audit

The present data are useful for algorithm exploration, but they are not yet a
trusted quantitative benchmark.

As of 2026-08-27:

- `$suncet_data/synthetic` is about 5.7 GB.
- `synthetic/level0` is about 4.1 GB and contains 1,196 FITS files.
- The main holdings include bright-fast no-jitter, bright-fast with jitter,
  original-Fano, bright-fast with scattered light, and dimmest variants.
- The standard rendered CME images are 1000 by 750 pixels with nominal
  9.6 arcsec/pixel WCS. At the recorded apparent solar radius, the horizontal
  field reaches roughly 4.9 solar radii from disk center.
- The standard sequences use suffixes from 000 through 357 in steps of three
  and normally contain about 120 images. The dimmest sequence is missing one
  suffix, and the historical original-Fano set contains duplicate/debug files.
- There is one synthetic Level 0.5 FITS file, three Level 1 FITS files, 92
  Level 2 FITS files, including one explicitly provisional Level 3 interface
  handoff fixture, and no synthetic Level 3 collection.
- `make_level3.py` remains a scaffold; `make_level4.py` is now a functional
  thin dispatcher for provisional CME tracking.

### Time and provenance problem

Every inspected standard synthetic FITS file has the same
`DATE-OBS=2023-02-14T17:00:00.000`, while `DATE-BEG` and `DATE-END` are `N/A`.
The underlying MHD files contain distinct times 10 seconds apart, and the
current simulator config declares a 10-second model time step. The every-third
suffixes therefore appear to be a nominal 30-second sequence, but tracking code
must not silently infer that.

Some files also contain inconsistent distance/solar-radius template metadata,
which indicates that the holdings mix simulator generations. The current
instrument simulator uses a fixed header template in paths that should instead
preserve the model observation time and generated WCS.

Before comparing speed or acceleration, the simulator must emit correct times
and a scenario manifest, or a separately reviewed manifest must map each
existing file to model time.

The source run duration has now been confirmed as one hour. Together with the
10-second native model step and retained source indices `0, 3, ..., 357`, this
establishes a verified 30-second cadence for the 120-frame curated sequence.
Samples occur from 0 through 3570 seconds; source index 360 is the omitted
3600-second endpoint. The frozen FITS headers still do not provide valid
absolute UTC. Heights do not depend on cadence; this correction doubles speeds
and quadruples accelerations relative to the former 60-second development
assumption. Retaining raw front positions makes such corrections trivial.
Tracker continuity penalties should therefore continue to use frame and pixel
units where practical rather than tight physical-speed thresholds.

### Existing truth status

There is no authoritative CME-front truth file for the current historical
sequence. The simulator provider does not independently track or label the CME,
so the legacy `$suncet_data/mhd/Model Height Time.sav` material must not be used
as ground truth for this benchmark. Quantitative front accuracy, bias, and
uncertainty coverage therefore cannot be inferred from this scenario.

Project-lead visual review found that the retained outer front follows the
intended bright leading edge throughout this first run. That review accepts the
current angular/time association and 90th-percentile scalar reduction as an
engineering baseline for compute and power characterization. It is not a
substitute for quantitative validation, which remains contingent on future
human-labeled frames, simulator-exported contours, or suitable real-data
comparisons.

### Timestamp-correct particle-snow run

The newer `config_default_no_particle_filter` sequence contains 241 frames at
15-second cadence. `DATE-OBS`, `DATE-BEG`, and each filename agree exactly from
`2023-01-14T17:00:00` through `18:00:00` UTC, with a final `DATE-END` of
`18:00:15`. Its reviewed manifest therefore uses strict FITS-header timing and
valid absolute UTC rather than a fixed-cadence override. The files declare
Level 0.5 and explicitly bypass Level 2 PSF deconvolution and Level 3 geometric
correction for this development exercise; retained simulator-header
inconsistencies are recorded in the manifest.

Particle filtering was disabled and the sequence represents worst-case
particle snow. The original unrestricted independent-angle tracker associated
bright hits across nearly the full angular domain, latched field-of-view
contact prematurely, and correctly withheld kinematics through its
coverage/confidence gates. Project-lead review then identified the actual CME
in the 220--330-degree solar-position-angle sector; that manual-sector result
remains a diagnostic comparison.

The automatic two-stage configuration now searches the full circle. Coherent
fragments discover support from 236--332 degrees and an eight-degree padding
produces the refinement sector without using the reviewed 220--330-degree
prior. Sparse per-angle recovery inside that inferred sector follows the same
visually accepted outer front across the hour. It reports 1.578--4.321 solar
radii before FOV censorship, a provisional median projected speed of about
758 km/s, and first spatially coherent boundary contact at frame 181. For
comparison, the manual-sector result reports 1.570--4.321 solar radii and about
716 km/s. The close agreement is an engineering success, not quantitative
science validation: no authoritative truth contour exists, and the two
sectors are not identical.

The automatic sector excludes one of the manual result's three otherwise
fit-eligible headline-height excursions. The provisional seven-frame,
0.2-solar-radius screen flags frames 167 and 168 and omits them only from the
spline; raw measurements and angular front samples remain unchanged. The plot
and movie distinguish retained, FOV-censored, and kinematic-outlier samples.
Speed uncertainty remains broad and acceleration is not yet scientifically
constrained. Per-frame association limits also need cadence-aware
interpretation before configurations are shared between the 15-second and
historical 30-second sequences.

The centered temporal-median-3 A/B removes 86.7% of temporally isolated
radial-peak candidates and 37.2% of all radial-peak candidates. It retains 239
measured frames versus 240 for raw, with the one-frame onset/end changes fully
explained by the unsupported centered-filter endpoints. FOV contact moves from
frame 181 to 195 and the scalar-height outlier screen rejects no filtered rows.
Median speed is 748.0 km/s versus 757.9 km/s raw. The median absolute
headline-height shift is 0.030 solar radii before raw FOV contact and 0.035
solar radii over the 79 frames where both kinematic fits are valid. All
predeclared engineering gates pass, and the synchronized A/B movie shows a
substantially smoother main rise.

There is still no truth contour, so smoother is not synonymous with more
accurate. The analytic broad-front fixture survives within one radial bin
median bias, but a deliberately fast, thin shell moving five pixels per frame
is erased. Keep median-3 opt-in until additional morphologies and cadences are
tested; a motion-tolerant two-of-three score consensus is the fallback if fast
front loss proves operationally important.

## Synthetic scenario and truth contract

The instrument simulator repository owns the generation of flight-like images.
This pipeline owns science-product generation and evaluation. Each delivered
scenario should include a machine-readable manifest with:

- stable scenario ID and short description;
- source MHD model and source revision;
- file list, checksums, model indices, UTC or elapsed time, and cadence;
- simulator Git commit, complete configuration, dependency/environment
  identity, and random seed;
- WCS, solar center, apparent solar radius, physical solar radius, and observer
  location;
- enabled instrument effects and their parameters, including exposure
  compositing, PSF, scattered light, jitter, noise, particle hits, compression,
  missing images, and intentional corruption;
- true 2-D projected front contour, mask, or
  `r_true(time, position_angle)`;
- true projected apex, central position angle, and angular width;
- true 3-D MHD quantities in separately named fields; and
- any judgment-dependent or unavailable truth explicitly marked as such.

Meng Jin confirmed that the present simulations do not include a tracked CME
front truth product. If future simulator/MHD tooling can expose it without a
large manual effort, the most valuable optional addition would be a projected
outer contour tied directly to every model time and view. Until then, use
scenario-level holdouts, analytic fixtures, and a limited frozen set of human
front annotations; do not treat visual agreement as quantitative truth.

Scenario coverage should grow to include:

- bright and faint events;
- fast, slow, accelerating, decelerating, and nonuniformly accelerating fronts;
- narrow and broad angular widths;
- different launch position angles and nonradial deflection;
- partial and fragmented fronts;
- multiple and overlapping CMEs;
- streamers and other moving non-CME structure;
- scattered light, noise, particle hits, bad pixels, and compression;
- pointing errors and residual jitter after Level 3 correction;
- cadence changes and missing frames; and
- no-CME negative sequences.

Training, validation, and test partitions are divided by entire physical
scenario, not by neighboring frames from the same simulation. Adjacent frames
are too correlated to provide an honest holdout test.

## Provisional Level 4 product contract

The research product should be easy to inspect without custom software. Use
Astropy ECSV for the provisional authoritative track and catalog tables because
it is text-readable while retaining units and metadata. An archive-facing FITS
binary-table export can be added from the same schema if required by the NASA
archive agreement; it should not become an independent source of truth.

An event directory is expected to contain:

```text
level4/cme_tracking/<event_id>/
├── COMPLETE.json
├── track.ecsv
├── front_samples.ecsv
├── summary.json
├── height_time.png
├── speed_time.png
├── acceleration_time.png                # complete 1-sigma range
├── acceleration_time_detail.png         # fitted variation; clipped 1-sigma marked
├── front_overlay.png
└── front_tracking.mp4       # optional synchronized all-frame diagnostic
```

`front_tracking.mp4` uses every input image in sequence order. It overlays the
exact retained front samples and places a moving time cursor on height, speed,
and acceleration panels. The movie uses the readable acceleration detail scale
with boundary markers wherever the 1-sigma interval continues outside the
panel; the separate full-range PNG preserves the complete uncertainty view.
Its playback frame rate is presentation metadata and does not replace or
rescale the scientific elapsed-time coordinate.

Algorithm diagnostics, likelihood cubes, and temporary masks remain under the
processing run unless they are intentionally promoted into a documented
science product.

### `track.ecsv`

One row per accepted image time, with at least:

| Field | Meaning |
| --- | --- |
| `event_id` | Stable event identifier within the catalog policy |
| `time_utc`, `elapsed_s` | Observation time and event-relative time |
| `height_raw_rsun`, `height_raw_km` | Raw projected height summary |
| `height_fit_rsun`, `height_fit_km` | Smoothed/fitted projected height |
| `height_raw_sigma_rsun` | Estimated front-location measurement uncertainty |
| `height_fit_sigma_rsun` | Propagated uncertainty of the fitted height curve |
| `speed_fit_km_s`, `speed_sigma_km_s` | Derived projected speed and uncertainty |
| `acceleration_fit_m_s2`, `acceleration_sigma_m_s2` | Derived projected acceleration and uncertainty |
| `position_angle_deg` | Canonical central solar position angle |
| `angular_width_deg` | Accepted angular span |
| `front_coverage_fraction` | Fraction of expected angular support with accepted front samples |
| `confidence` | Documented event/track score |
| `quality_flags` | Machine-readable reasons to distrust or omit values |

### `front_samples.ecsv`

Long-form rows keyed by event ID, time, and position angle, containing radius,
positional uncertainty, likelihood/score, tracker state, and quality flags.
This preserves the measured front from which event summaries are derived.

### `summary.json` and mission catalog

The summary records event start/end, median and peak fitted speed, acceleration
summaries, angular properties, quality, input/product checksums, algorithm and
configuration version, Git/source-tree identity, environment, WCS conventions,
and processing provenance. `COMPLETE.json` is written last inside a staging
directory; only a complete set is published into the final event path.

An aggregate `level4/cme_catalog.ecsv` contains one row per event and points to
the event products. Its exact archive fields are finalized with the NASA SDAC
product definition.

## Software architecture

Keep `make_level4.py` as a thin dispatcher. Place shared Level 4 mechanics and
CME-specific code in separate packages:

```text
suncet_processing_pipeline/
├── make_level4.py
└── level4/
    ├── registry.py
    ├── common/
    │   ├── product_io.py
    │   ├── quality.py
    │   └── schemas.py
    └── cme_tracking/
        ├── config.py
        ├── input.py
        ├── polar.py
        ├── preprocessing.py
        ├── likelihood.py
        ├── detectors/
        ├── tracking.py
        ├── kinematics.py
        ├── evaluation.py
        └── products.py

benchmarks/
└── cme_tracking/

suncet_processing_pipeline/tests/
└── level4/
    └── cme_tracking/
```

Design rules:

- Begin with a deterministic NumPy/SciPy reference implementation.
- Keep detector scoring, front tracking, kinematic fitting, and product writing
  separately testable.
- Pass arrays and explicit metadata between stages; do not rely on global paths.
- Use existing processing-run provenance for all science and benchmark runs.
- Keep an explicit array-backend boundary so selected kernels can later use
  CuPy, PyTorch, TensorRT, or another Jetson-supported backend without changing
  the science interface.
- Keep optional GPU dependencies out of the portable base environment until a
  measured candidate is selected.
- Cache geometry that is constant across a sequence, such as the Cartesian to
  polar mapping.
- Store working arrays as `float32` where error testing shows that it is
  scientifically equivalent; retain product precision appropriate to the
  uncertainty.
- Optimize only after a profiler identifies a material bottleneck.

The temporary synthetic adapter may accept reviewed simulator FITS files plus a
scenario manifest. Production CME tracking must still consume the Level 3
contract; it must not absorb dark, flat, PSF, pointing, or compositing
corrections that belong upstream.

PSF sensitivity is a planned paired test: run identical tracker configurations
on corresponding pre- and post-Level-2 images with geometry held fixed or
co-registered, then compare front and kinematic products. This will isolate
whether PSF deconvolution materially changes the inferred height-time profile
without confusing it with Level 3 geometric corrections.

## Validation strategy

### Two separate evaluations

1. **Known-window tracking:** Given a time interval containing one event, how
   accurately is its front and kinematics recovered?
2. **Continuous discovery:** Given an uninterrupted stream containing zero or
   more events, how accurately are events found, separated, and cataloged?

Known-window performance is not reported as event-detection performance.

### Scientific metrics

| Category | Metrics |
| --- | --- |
| Front | Radial error versus angle/time, bias, percentile error, angular coverage, contour distance or IoU when masks exist |
| Event geometry | Apex-height error, central-position-angle error, angular-width error, event duration error |
| Kinematics | Height, speed, and acceleration bias/RMSE; peak timing/value error; fit residuals |
| Uncertainty | Coverage of nominal confidence intervals and calibration of the reported confidence score |
| Discovery | Precision, recall, event F-score, false events per observing day, detection latency, split/merge rate |
| Robustness | Accuracy versus brightness/SNR, speed, width, angle, gap length, jitter, scatter, compression, and overlap |
| Reproducibility | Identical inputs/configuration produce products within documented CPU/GPU numerical tolerances |

Do not invent acceptance thresholds before the first benchmark establishes
achievable error. Thresholds will be selected from the science requirement and
synthetic/real comparison, then recorded here.

### Real-data bridge

Before launch, run the same method on relevant wide-field EUV observations such
as SWAP and other suitable public sequences with human-reviewed fronts. Those
data will not exactly reproduce SunCET, but they expose real coronal morphology
and artifacts that a synthetic-only test cannot.

After launch:

- build a blinded, human-reviewed set of early SunCET events and no-event
  intervals;
- tune only on the development subset;
- preserve a holdout set for the production acceptance decision; and
- compare automated products with independent manual measurements and, where
  appropriate, overlapping coronagraph observations.

## Jetson compute and power characterization

Benchmark frozen-reference science fidelity and resource use together. A faster
result that changes the selected front is not the same implementation. The
current scenario has no physical truth, so these comparisons measure fidelity,
not accuracy. These measurements inform a potential future onboard/deep-space
design; energy is not a limiting requirement for ground SOC processing.

### Onboard scheduling and energy constraint

For a future onboard implementation, minimize energy subject to the hard
constraint that Level 4 analysis does not fall behind sustained image
generation. The default composited-image cadence is 15 s and is commandable on
orbit; use 10 s as the shortest credible cadence for sizing and sustained-load
tests unless the instrument assumptions change.

For a batch of `N` images acquired at cadence `C`, basic feasibility requires:

```text
T_batch(N) < N * C
```

Passing that inequality without headroom is insufficient. The benchmark must
determine the required margin for Level 2/Level 3 work, other spacecraft
workloads, scheduling jitter, and sustained thermal behavior; no fixed
per-image timing target is approved in advance. Report whether the queue stays
bounded at both 15 s and 10 s cadence and whether it recovers after injected
delays or workload bursts.

The current reference method is partly batch-oriented and noncausal: its
full-window temporal background, sparse temporal path optimization, and final
smoothing spline use future frames or completed-window context. A streaming
onboard candidate therefore needs explicitly tested rolling, fixed-lag, or
deferred-finalization approximations, with their science differences measured
against the authoritative full-window result.

The initial onboard architecture candidate is hybrid:

- perform image-local work continuously as each image arrives, including polar
  remapping, filtering/differencing, and front-evidence generation;
- buffer compact polar evidence or radial/front candidates rather than another
  full-resolution image sequence where practical;
- update lightweight tracking state continuously or in microbatches; and
- finalize the path, spline, uncertainty, and event product when sufficient
  temporal context is available.

Benchmark the following scheduling policies under identical inputs, power
modes, and thermal conditions:

- one image at a time;
- microbatches of 4, 8, and 16 images;
- the full buffered sequence; and
- an adaptive policy that batches while ahead of cadence and drains work
  immediately when queue depth rises.

Energy optimization may favor a higher-power burst followed by a deeper idle
period, but this must be established by measurement rather than assumed. A
policy is feasible only if it preserves the science contract and maintains a
bounded backlog at the required cadence under sustained thermal conditions.

### Test matrix

At minimum, vary:

- algorithm/detector and configuration revision;
- CPU reference versus GPU candidate;
- image resolution or binning;
- sequence length and cadence;
- per-image, 4/8/16-image microbatch, full-batch, and adaptive scheduling, plus
  CPU thread count;
- every relevant stock `nvpmodel` mode exposed by the installed device; and
- burst/pass processing versus sustained repeated processing.

Discover mode IDs and limits from the Jetson's
`/etc/nvpmodel.conf` and `sudo nvpmodel -q --verbose`. Do not hard-code mode
names or assume that a mode's watt label equals measured consumption. Treat
MAXN as a brief upper-bound experiment, not the default operational result.

### Two performance regimes

- **Operational DVFS:** select an `nvpmodel` mode, retain normal dynamic
  frequency behavior, and record the fan profile.
- **Controlled ceiling:** select `nvpmodel` first, then use `jetson_clocks` to
  pin the permitted clocks for a repeatable performance ceiling.

Never mix these regimes in one comparison. Some mode or clock changes require a
reboot.

### Run protocol

For each measured run:

1. Record UTC, Jetson model, OS/L4T, `nvpmodel` details, clocks, fan profile,
   temperatures, pipeline Git commit, algorithm/configuration hash, Python
   environment, input manifest, dimensions, cadence, and CPU/GPU backend.
2. Start from a standardized idle/thermal condition and use a documented fan
   profile and ambient temperature.
3. Warm up the workload before timing.
4. Log `tegrastats` at an initial 100–200 ms interval around explicit monotonic
   start/end markers.
5. If the workload is only a few seconds, repeat the identical work so the
   measurement window is approximately 30–60 seconds.
6. Synchronize CUDA before stopping a GPU timer.
7. Collect at least five measured repetitions and report a distribution rather
   than the best run.
8. Inspect frequency, temperature, thermal-throttle, and overcurrent behavior
   before accepting the comparison.

Measure two timing scopes:

- **Compute-only:** input arrays are resident in memory and output remains in
  memory.
- **Production end-to-end:** NVMe FITS reads, validation, preprocessing,
  tracking, product writing, and required diagnostics.

### Reported resource metrics

- seconds per event or pass;
- frames per second;
- median, 95th-percentile, 99th-percentile, and maximum per-image and
  final-product latency;
- maximum and time-resolved queue depth, evidence that backlog remains bounded,
  and recovery time after an injected delay or burst;
- peak RAM, accelerator memory where available, and buffer memory/storage;
- CPU, GPU, and memory-controller utilization and clocks;
- peak power, maximum temperatures, steady-state temperature, and any
  thermal-throttle or overcurrent events;
- gross joules per event/pass and per image;
- incremental joules above a matched idle baseline;
- science differences from rolling, fixed-lag, microbatch, adaptive, or other
  streaming approximations relative to the full-window reference; and
- fidelity-versus-latency and fidelity-versus-energy Pareto plots until
  suitable quantitative truth exists.

On AGX Orin, derive covered onboard power from the current-power samples for:

```text
VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0
```

Do not add `VDDQ_VDD2_1V8AO` again; NVIDIA documents it as already included in
`VIN_SYS_5V0`. Integrate timestamped power samples over the exact workload
window. Keep both gross and idle-subtracted energy because they answer different
questions.

```text
gross compute energy       = integral(Pload dt) over the compute window
incremental compute energy = integral((Pload - Pidle_same_mode) dt)
power-cycle energy         = integral(Pspacecraft_bus dt) from power-on through power-off
```

Incremental energy is the marginal cost of processing only when the Jetson's
idle draw is unavoidable. If the flight architecture powers the Jetson off
between accumulated-data batches, do not subtract idle: every joule consumed
while booting, reading, processing, writing, and shutting down is attributable
to that cycle. The compute-only gross result is then directionally relevant,
but the authoritative metric must be full power-cycle energy at the spacecraft
bus, including boot/shutdown, NVMe I/O, Level 2/3/4, carrier conversion losses,
and off-state leakage.

The onboard rails are useful for comparative optimization but are not identical
to power at the developer kit DC input. Validate final candidates with an inline
DC power analyzer so carrier-board, NVMe, fan, and conversion losses are
included. A future flight assessment must repeat measurements on the industrial
SOM, flight-like carrier, power system, cooling arrangement, and thermal
environment.

### First Jetson baseline: 30 W operational DVFS

The first authoritative CPU baseline completed on 2026-08-26 in stock
`MODE_30W` (mode ID 2), with normal dynamic frequency scaling and no pinned
`jetson_clocks`. It used the 120-frame `bright_fast/no jitter` hour-long event,
resident image arrays, one warmup, five measured repetitions, at least 30 s of
identical work per repetition, a 30 s idle baseline, and 200 ms `tegrastats`.
The Jetson result is numerically equivalent to the independently preserved,
project-lead-reviewed Mac reference: every discrete path/front/FOV output and
headline height is exact; the permitted floating differences are only
cross-build roundoff within the recorded tolerance policy. This remains an
engineering-fidelity result, not validation against physical CME truth.

Authoritative run ID:

```text
20260826T201945.506826Z_suncet-soc_a1858558
```

It is stored on the Jetson under `/srv/suncet/data/benchmarks/cme_tracking/`
and in the portable data tree under
`$suncet_data/benchmarks/cme_tracking/jetson/`. `COMPLETE.json` verifies all
eight run artifacts; the Mac copy was independently checked against those
hashes.

| Metric | Result |
| --- | ---: |
| Median compute time per 120-frame event | 18.435 s |
| Timing range across five repetitions | 18.427--18.462 s |
| Median throughput | 6.509 frames/s |
| Mean covered-rail load power, median repetition | 8.149 W |
| Matched covered-rail idle power | 6.717 W |
| Median gross energy per event / frame | 150.236 J / 1.252 J |
| Median incremental energy above idle per event / frame | 26.410 J / 0.220 J |
| Process-lifetime peak RSS | 1.441 GB |
| Peak observed temperature | 46.3 degrees C |
| Peak GR3D utilization | 0% |

All workload windows had effectively complete timestamp-interpolated power
coverage. The rails are `VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0`; the values
are comparative onboard-rail measurements, not developer-kit wall/DC-input
power.

At the full-window average rate, the 18.435 s computation consumes 1.024% of
the 1,800 s acquisition horizon at 15 s cadence and 1.536% of the 1,200 s
horizon at 10 s cadence. The corresponding average-rate margins are 97.64x and
65.09x. This proves comfortable batch throughput for this Level 4 CPU
reference alone; it does not prove causal streaming, per-image latency,
combined Level 2/3/4 scheduling, or sustained queue recovery.

A clean `cProfile` run identifies two dominant regions. Front-likelihood
construction accounts for about 53% of profiled compute time, chiefly repeated
NaN-aware radial percentile/quantile normalization. Per-position-angle front
extraction accounts for about 44%, chiefly 360 independent tracks, repeated
candidate/peak construction, and uncertainty quantiles. Profiling inflates
absolute duration, so use these shares only to rank targets. The first
optimization experiment should eliminate repeated quantile/Python-loop work
on the CPU while enforcing the frozen science-equivalence gate. A GPU
likelihood backend is plausible after that; the branch-heavy path tracker is a
less obvious first GPU target.

The first retained CPU optimization adds a threshold fast-reject before
`find_peaks`: a row whose global score maximum is below threshold cannot contain
a qualifying local or endpoint peak. On the accepted likelihood cube this
skips 2,838 of 21,600 peak searches (13.1%) without changing candidate
semantics. Exact-reference validation passes on the Mac, and the Jetson result
remains numerically equivalent to the independent reference with exact
discrete science outputs. The five-repetition A/B run is:

```text
20260826T203106.637354Z_suncet-soc_740e0719
```

Relative to the unoptimized 30 W baseline, median event time improves from
18.435 to 18.264 s (0.93%), median gross covered-rail energy from 150.236 to
147.923 J (1.54%), and median incremental energy above idle from 26.410 to
25.453 J (3.62%). The change is retained because it is simple, reversible,
science-preserving, and reduces both runtime and energy. The next CPU target is
the substantially larger repeated radial percentile/quantile workload.

### Operational-DVFS power-mode matrix

The same optimized CPU implementation was then measured in stock 15 W mode.
The 15 W configuration exposes four CPU cores capped at 1.1136 GHz, versus
eight at 1.728 GHz in 30 W mode. Both runs passed the independent science-
equivalence gate and had effectively complete power coverage.

| Metric | 15 W | 30 W | 50 W | MAXN |
| --- | ---: | ---: | ---: | ---: |
| Run ID suffix | `71d4fc6a` | `740e0719` | `150e766b` | `e4d81622` |
| Online cores / CPU cap | 4 / 1.1136 GHz | 8 / 1.728 GHz | 12 / 1.4976 GHz | 12 / 2.2016 GHz |
| Median compute time/event | 28.249 s | 18.264 s | 20.804 s | 14.694 s |
| Median throughput | 4.248 frames/s | 6.570 frames/s | 5.768 frames/s | 8.167 frames/s |
| Median covered-rail load power | 6.851 W | 8.112 W | 7.998 W | 9.368 W |
| Matched covered-rail idle power | 6.148 W | 6.716 W | 6.772 W | 6.807 W |
| Peak covered-rail power | 7.818 W | 8.920 W | 8.919 W | 10.609 W |
| Median gross energy/event | 193.549 J | 147.923 J | 166.576 J | 137.806 J |
| Median incremental energy/event | 19.884 J | 25.453 J | 25.537 J | 37.643 J |
| Projected energy over fixed 1,200 s | 7.397 kJ | 8.085 kJ | 8.152 kJ | 8.206 kJ |
| Maximum observed temperature | 44.4 degrees C | 46.1 degrees C | 45.1 degrees C | 46.3 degrees C |
| 10 s cadence average-rate margin | 42.48x | 65.70x | 57.68x | 81.67x |
| 15 s cadence average-rate margin | 63.72x | 98.56x | 86.52x | 122.50x |

Relative to 30 W, 15 W is 54.7% slower and consumes 30.8% more gross energy
while actively computing, but it consumes 21.9% less incremental energy above
its own idle baseline. Which quantity governs flight energy depends on the
scheduler and power state after completion. If the Jetson remains in the same
mode for a fixed 1,200 s acquisition horizon, the covered-rail projection is
`idle_power * horizon + incremental_event_energy`: about 7.397 kJ at 15 W and
8.085 kJ at 30 W, an 8.5% advantage for 15 W. If early completion permits a
meaningfully deeper sleep state, the gross race-to-idle result may instead
favor 30 W. Treat this fixed-horizon calculation as a scheduling projection,
not a substitute for sustained measurement or whole-kit DC-input power.

The stock 50 W result is dominated by 30 W: it is 13.9% slower, consumes 12.6%
more gross event energy, consumes 0.3% more incremental event energy, and uses
0.8% more projected energy over the fixed 1,200 s horizon. The extra cores do
not compensate for the lower per-core clock in this mostly serial CPU
implementation. Its complete run ID is
`20260827T153948.149435Z_suncet-soc_150e766b`.

MAXN is the upper-bound race-to-idle result. Relative to 30 W, it is 19.5%
faster and consumes 6.8% less gross compute-window energy, but consumes 47.9%
more incremental energy above idle and 1.5% more projected energy over the
fixed 1,200 s horizon. Its peak covered-rail power was only 10.609 W and its
maximum observed temperature was 46.3 degrees C for this CPU-only workload;
this does not establish MAXN safety for GPU-heavy or sustained combined
Level 2/3/4 operation. Its complete run ID is
`20260827T160253.877457Z_suncet-soc_e4d81622`.

For the ground SOC, restore stock 30 W as the normal default: it is the fastest
bounded standard mode, has lower gross energy than 15 W and 50 W, and avoids
using MAXN as an operational setting. The intended flight concept is now a
power-cycled batch architecture: remain off while data accumulate, power on,
process the backlog, publish/persist the products, and power off. Under that
concept, the current compute-only result makes MAXN—not 15 W—the provisional
energy leader because MAXN has the lowest gross compute energy. Stock 30 W is
the bounded fallback if MAXN's instantaneous power, supply, thermal, or
qualification implications are unacceptable. The earlier fixed-horizon result
for 15 W remains relevant only to an always-on architecture.

Do not select the flight mode from this compute-only test. Measure complete
power-on-to-power-off cycles at the spacecraft bus, vary accumulated batch
size to amortize boot cost, include sustained combined Level 2/3/4 and storage
I/O, and verify peak-current and thermal constraints on flight-like industrial
hardware.

Earlier development runs that periodically polled RSS are superseded for
performance and energy conclusions. The authoritative harness now uses the
kernel-maintained `ru_maxrss` process high-water counter, with no sampling
thread. The clean profile is archived as
`$suncet_data/benchmarks/cme_tracking/jetson/profiles/mode30w_no_polling_harness_20260826.prof`.

## Current implementation checkpoint

As of 2026-08-31, the repository contains a functional known-window vertical
slice rather than only scaffolding:

- reviewed JSON manifests with a manifest SHA-256, per-frame SHA-256 values,
  explicit frame numbers, cadence status, and Level 2/Level 3 correction state;
- lazy synthetic-bypass and strict production-Level-3 FITS adapters;
- load-time and per-read hash verification with explicit flags for skipped or
  direct-directory inputs;
- full-WCS pixel scale, solar-north/east geometry, and reflected-WCS rejection;
- polar remapping; robust background, running-difference, and leading-edge
  evidence; sparse outward path optimization; angular association; and
  explicit missing states;
- a provisional robust-apex height summary, field-of-view truncation flags,
  irregular/gapped-time smoothing spline, analytic derivatives, Monte Carlo
  uncertainties, endpoint flags, provisional confidence/coverage gates, and
  unsupported-gap exclusion;
- ECSV track/front tables, JSON configuration/Git/input provenance, static
  height/speed plots, full-range and detail acceleration plots, a raw-image
  front-overlay mosaic, an optional synchronized all-frame H.264 diagnostic
  movie, a reproducible numbered method-explanation series, and
  staged/transactional product publication; and
- analytic no-event, moving-front, missing-frame, cadence-scaling, WCS,
  manifest, product, and end-to-end tests.

The input path now supports strict FITS-header timing for reviewed synthetic
sequences and scaled integer FITS images. A reviewed manifest captures the
241-frame, 15-second `config_default_no_particle_filter` run without copying
its inputs. Its worst-case particle snow exposed a real failure mode in the
unrestricted independent-angle tracker. The new full-circle coherent-fragment
stage rejects isolated hits and a robust distributed-motion gate rejects a
long-lived inner-limb arc. Its selected component automatically defines an
eight-degree-padded sector for detailed sparse-path recovery. This recovers the
known event without the reviewed 220--330-degree prior; it remains known-window
sector discovery, not continuous autonomous event discovery.

The complete repository test suite passes after the automatic coherent-sector
path, particle/FOV regressions, temporal outlier screen, temporal-median A/B,
benchmark instrumentation, and first CPU fast path were added.
The focused suite passes under Python 3.12, and the complete
120-frame workflow also runs under Python 3.14.7 with numerically identical
headline results. A fully hash-verified smoke run on `bright_fast/no jitter`
follows a smooth outward feature to an uncensored height of 4.24 solar radii,
flags first field-of-view contact at frame 95, and retains later measurements
only as censored observations. Confidence/coverage gating leaves frames 14--93
in the fitted interval. With the verified 30 s cadence, the provisional median
projected speed is about 703 km/s. This timing correction leaves tracked front
locations and heights unchanged, doubles speed, and quadruples acceleration
relative to the former 60 s development run.

That run is a successful engineering result, not a quantitatively validated
science benchmark. Project-lead visual review found that the retained outer
front follows the intended bright leading edge, so additional coherent-front
refinement is not required before Jetson benchmarking. The broad angular
support and persistent inner-corona branches remain useful diagnostics, and
position angle, width, and the scalar-height convention remain provisional.
There is no authoritative truth file for this scenario. Even after quality
gating, acceleration uncertainty is large enough that no acceleration claim is
currently supported.

The 15 W, 30 W, 50 W, and MAXN Jetson characterization matrix, bottleneck
profile, first retained CPU fast path, and Mac temporal-median particle A/B are
complete. Meng Jin's additional simulations are expected next; the next tracker
work is broader temporal-filter scenario testing on those independently
generated cases, paired Jetson cost measurement if the option is retained,
and bounded multi-hypothesis fragment association. Continuous event discovery
remains later. Further CPU/GPU optimization should follow only behind frozen science
signatures. Future science validation remains a separate workstream: obtain
human labels or exported contours, test additional simulations and real EUV
sequences, and compare height definitions and kinematic fits when quantitative
reference data exist.

## Milestones and status

### 0. Research and initial data audit — complete

- [x] Inspect the current Level 4 and Level 3 scaffolds.
- [x] Inventory the existing synthetic holdings.
- [x] Identify the frozen-time and mixed-provenance limitations.
- [x] Review CACTus, SEEDS, CIISCO, CORIMP, and later learned approaches.
- [x] Define the recommended algorithm ladder and Jetson measurement approach.

### 1. Freeze the science and dataset contracts — in progress

- [ ] Decide which physical/morphological CME front the product follows.
- [ ] Approve projected-height, width, and event-ID
  conventions.
- [x] Use solar position angle only: 0 degrees at solar north, increasing
  counterclockwise.
- [x] Permit an explicit, provenance-recorded fixed cadence for the current
  synthetic development set.
- [x] Confirm the historical bright-fast source duration and establish its
  30-second retained cadence.
- [x] Define Level 2 as PSF-deconvolved and Level 3 as fine
  geometrically corrected, including precise solar-north alignment; permit an
  explicit synthetic bypass during development.
- [x] Define the provisional Level 3 input and Level 4 output schemas.
- [x] Add a synthetic scenario manifest schema and validator.
- [x] Fix simulator timestamps/provenance for newly generated scenarios.
- [x] Validate and manifest the first 241-frame corrected-timestamp simulator
  sequence using strict FITS-header UTC at 15-second cadence.
- [x] Curate the current bright-fast data instead of treating every historical
  file as a benchmark.
- [x] Determine that no authoritative truth file exists for this scenario and
  exclude the legacy `Model Height Time.sav` material from validation.
- [ ] Export contour or `r_true(time, position_angle)` truth from the simulator
  when practical; otherwise freeze human-reviewed front annotations for a
  limited validation subset.

### 2. Build the common evaluation spine

- [x] Scaffold the modular Level 4 package and thin dispatcher.
- [x] Implement reviewed synthetic and production Level 3 input adapters.
- [x] Implement polar geometry, product schemas, and quality flags.
- [ ] Implement an evaluation runner and reproducible dataset split manifest.
- [x] Write tiny analytic fixtures with exactly known fronts and kinematics.

### 3. Establish classical baselines

- [x] Implement and test an initial deterministic leading-edge/path baseline.
- [ ] Implement and test the SEEDS-like framewise baseline.
- [ ] Implement and test the CACTus-like linear Hough baseline.
- [ ] Implement and test the CIISCO-like parabolic Hough baseline.
- [ ] Record accuracy, failure cases, runtime, and memory on the Mac and Jetson.

### 4. Build the primary tracker

- [ ] Implement multiscale front likelihood.
- [x] Implement temporal path optimization with gap states.
- [x] Associate paths across position angle and retain
  `r(time, position_angle)`.
- [x] Visually review and accept the current angular/time association as the
  engineering baseline for Jetson characterization.
- [x] Add provisional event-confidence and quality logic; calibration remains
  pending.
- [x] Add an explicit reviewed position-angle window for known-event recovery
  and record that it is a manual prior rather than discovery.
- [x] Discover a coherent time--position-angle component automatically, reject
  sparse particle trajectories and quasi-static inner structure, infer a
  padded event sector, and refine the detailed front without a supplied PA
  prior.
- [x] A/B test a disabled-by-default three-frame temporal median on the polar
  intensity cube before spatial smoothing and likelihood construction. Measure
  particle rejection, front bias, onset latency, runtime, and failure on fast
  thin fronts; explicitly mark unsupported first/last frames. The first Mac
  science A/B passes its engineering gates; paired Jetson timing/energy remains
  optional follow-up rather than part of the science decision.
- [ ] Replace the one-best-history fragment dynamic program with a bounded
  multi-hypothesis/Pareto state so start radius, running maximum, and robust
  final ranking cannot discard a valid alternative history.
- [ ] Compare against every baseline without changing the held-out set.

### 5. Derive uncertainty-aware kinematics

- [ ] Select and validate the height-time fit.
- [x] Implement the provisional irregular/gapped-time smoothing spline.
- [x] Propagate height uncertainty into speed and acceleration.
- [x] Quantify endpoint behavior and flag invalid derivative regions.
- [x] Generate provisional Level 4 tables, plots, summary, and static overlay.
- [x] Add the synchronized all-frame front and kinematics diagnostic movie.
- [x] Preserve a full-range acceleration uncertainty plot while adding a
  readable detail view with explicit uncertainty-clipping markers.

### 6. Add continuous discovery and difficult events

- [x] Add known-window no-event screening.
- [ ] Add continuous-stream no-event screening and discovery.
- [ ] Add event association, split/merge accounting, and catalog identity.
- [ ] Test simultaneous, successive, partial, and overlapping events.
- [ ] Define the operator-review path for low-confidence events.

### 7. Characterize and optimize on the Jetson

- [x] Implement the immutable benchmark harness, resident-memory compute
  boundary, frozen science signature, and tested `tegrastats` parser.
- [x] Establish the stock 30 W CPU fidelity/performance/energy baseline.
- [x] Establish CPU fidelity/performance/energy results across stock power
  modes.
- [x] Profile the pipeline and identify actual bottlenecks.
- [x] Retain the first science-equivalent CPU optimization after a controlled
  timing/energy A/B comparison.
- [ ] Implement GPU versions only for bottlenecks with plausible benefit.
- [ ] Compare CPU/GPU science equivalence, latency, and energy.
- [ ] Validate final whole-kit power with an external DC measurement.

### 8. Real-data and production integration

- [ ] Validate on human-reviewed public EUV sequences before launch.
- [ ] Integrate with operational Level 3, processing manifests, and the SOC
  scheduler.
- [ ] Build and preserve the early-flight SunCET validation/holdout sets.
- [ ] Set production acceptance thresholds from requirements and measured
  performance.
- [ ] Finalize NASA archive formats and catalog fields.
- [ ] Automate after-pass Level 4 processing only after the manual path is
  reliable and observable.

### 9. Future flight-oriented streamlining

- [ ] Identify the smallest scientifically adequate algorithm/configuration.
- [ ] Evaluate quantization, reduced precision, resolution, and cadence
  tradeoffs.
- [ ] Repeat characterization on an industrial module and flight-like
  electrical/thermal hardware.
- [ ] Maintain a streamlined implementation only if shared interfaces and
  regression products prevent science divergence.

## Decisions still needed

These do not block the dataset manifest or first baseline, but they must be
resolved before production:

1. Which visible structure is the canonical SunCET CME front?
2. Is the headline scalar height the apex, the height at the central position
   angle, or another robust front statistic?
3. What minimum event duration/front coverage permits speed and acceleration
   publication?
4. How are low-confidence and partially observed events represented rather than
   silently discarded?
5. How should events spanning multiple downlink passes retain one identity?
6. Which public real-EUV events will form the pre-launch non-synthetic
   validation set?
7. What latency and energy targets should represent the potential
   onboard/deep-space implementation?
8. Which ECSV/FITS products and plots will NASA SDAC treat as archive products
   versus processing diagnostics?

## Primary references

- [SunCET instrument concept and science requirements](https://doi.org/10.1051/swsc/2021004)
- [CACTus: automated CME recognition with polar time-height maps and a modified Hough transform](https://www.sidc.be/cactus/publi/aa1302.pdf)
- [SEEDS: automatic detection and tracking in coronagraph time series](https://www.sidc.be/users/evarob/Literature/Papers/Solar%20Physics/2008%20Olmedo%20SEEDS%20paper%20SoPhys.pdf)
- [CIISCO: accelerating eruptions in inner-corona EUV data with a parabolic Hough transform](https://arxiv.org/abs/2010.14786)
- [CORIMP II: multiscale detection and temporal tracking](https://arxiv.org/abs/1207.6125)
- [CORIMP kinematic fitting and comparison](https://arxiv.org/abs/1506.04046)
- [CME kinematic smoothing and uncertainty study](https://arxiv.org/abs/1307.8155)
- [NVIDIA Jetson Linux r39.2.1 power and performance management](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
- [NVIDIA `tegrastats` documentation](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html)
- [NVIDIA Jetson Linux r39.2.1 validation plan](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/TestPlanValidation.html)
