# Jetson Level 2 / CME-tracking paired experiment — 2026-09-22

## Result

PSF deconvolution did **not** materially change the compute required by the
CME tracker itself, but it did materially change which angular parts of the
front the current automatic tracker retained. Where both arms retained the
same time/position-angle sample, the inferred radial edge was usually nearly
unchanged. The large difference in scalar headline height came primarily from
tracking a different and substantially narrower subset of the front, not from
moving every local edge by the same amount.

This is a sensitivity/stability experiment, not an accuracy comparison. The
synthetic event has no authoritative front contour, so the result cannot say
whether the undeconvolved or deconvolved track is closer to truth.

## Controlled comparison

- Input: all 241 timestamp-correct
  `config_default_no_particle_filter_OBS_*` simulator frames at 15-second
  cadence (`750 x 1000`, `float32` at the tracker boundary).
- Scenario: intentionally difficult no-particle-filter run with worst-case
  particle snow.
- PSF: the exact `suncet_diffraction_patterns_2k_20250224.fits` model used by
  that simulator run, plus the matching mirror-scatter, response, and spectrum
  inputs at correction factor `0.4`.
- Deconvolution: the Level 2 FP64 inverse-filter implementation at clean commit
  `b18817ef96e0ed96d1f7a7503951daeb83153549`, using CuPy 14.2.0 / CUDA 13.2.
- Tracker: identical automatic-sector configuration in both arms,
  `cme_tracking_config_default_no_particle_filter_20230114_autonomous_v1.json`.
  Its temporal-median window is one frame, so no temporal median was applied.
- Measurement: one warm-up plus three alternating repetitions per arm with
  100-ms `tegrastats`; every workload window had complete interpolated power
  coverage.
- Jetson mode: stock `MODE_30W` (mode ID 2), dynamic clocks.

The compact prepared-PSF artifact contains spatial kernels reconstructed from
the exact NumPy-prepared transfer functions. It records SHA-256 hashes of the
1.31-GB diffraction cube and all other source calibration inputs. The Jetson
then rebuilt the FP64 Fourier-domain representation locally using CuPy. This
avoided transferring the full calibration cube without changing the inverse
filter being tested.

An independent reconstruction check verified that claim. All four source-file
hashes matched the archive metadata. Rebuilt diffraction and scatter transfer
functions agreed with the full-calibration preparation to approximately
`1.2e-15` maximum relative error. On the middle sequence frame, full versus
compact FP64 deconvolution differed by `3.64e-11` maximum absolute and
`2.17e-12` RMS; after the experiment's `float32` boundary, all 750,000 pixels
were bit-for-bit identical.

## Science sensitivity

| Metric | Undeconvolved | PSF deconvolved | Paired interpretation |
| --- | ---: | ---: | --- |
| Event detected | yes | yes | Detection survives in both arms. |
| Frames with a headline height | 240 | 226 | Deconvolution loses 14 measured frames. |
| Accepted angular front samples | 5,466 | 2,838 | Deconvolution retains 48.1% fewer samples. |
| Kinematically valid frames | 79 | 55 | The derivative product has less support after deconvolution. |
| First FOV-limited frame | 181 | 199 | Deconvolution delays declared FOV contact by 18 frames = 270 s. |
| Median projected speed | 757.9 km/s | 742.0 km/s | Event-level medians differ by 15.9 km/s (2.1%), but pointwise fits differ more strongly. |

The accepted-support Jaccard fraction is `0.394`. Deconvolution loses 3,119
raw-arm samples and gains 491 different samples. It yields fewer accepted
angles in 239 of 241 frames. Mean angular width changes by `-49.2 degrees`,
with a median absolute change of `50.0 degrees`; central position angle has a
median absolute change of `22.0 degrees`.

At the 2,347 time/angle samples retained by both arms:

- median absolute radial difference: `0 px`;
- mean absolute radial difference: `1.72 px` (`0.0172 R_sun`);
- 90th-percentile absolute difference: `6 px` (`0.0600 R_sun`);
- 95th-percentile absolute difference: `9 px` (`0.0900 R_sun`); and
- RMS difference: `3.80 px` (`0.0380 R_sun`).

After subtracting each frame's median bulk radial offset, the common-sample
shape residual remains similar: `1.68 px` mean absolute and `3.70 px` RMS.
Thus, PSF deconvolution usually leaves an edge in the same place when the
tracker selects the same angle; it changes the automatically selected angular
support much more strongly.

Because the headline height is the 90th percentile over the retained angular
samples, the support change propagates into a large scalar-height change. Over
the 180 common frames before either arm reaches the FOV boundary, the
deconvolved-minus-raw headline height has:

- bias: `-0.3746 R_sun`;
- median absolute difference: `0.4902 R_sun`;
- RMS difference: `0.5628 R_sun`; and
- maximum absolute difference: `1.5746 R_sun`.

The front maps explain the apparently contradictory result: common points are
locally close, while deconvolution removes broad angular support and changes
the automatically inferred event sector. Some removed points resemble the
known quasi-static inner branch; others may be legitimate broad-front
structure. Without truth contours, neither interpretation can be promoted to
an accuracy conclusion.

### Overlay-display correction

The initially generated `front_overlay_comparison.png` is scientifically
misleading and is superseded. It computed common pair percentiles and a common
asinh transform, but omitted explicit shared `vmin`/`vmax` values when calling
Matplotlib. Each panel was therefore auto-normalized independently. The
deconvolved images contain negative ringing (about 1% negative pixels, with a
minimum near `-9,400` DN/s versus the raw floor of zero), so their independent
display range mapped the ordinary background to mid-gray and made them appear
dramatically lower contrast.

The corrected true-shared-scale overlays are stored under
`display_diagnostic/front_overlay_shared_scale.png`. They show that the bulk
contrast is very similar: across the inspected frames, raw and deconvolved
medians differ by less than 1%, and total flux is conserved to about 0.2%.
There is nevertheless a smaller real effect. This unregularized inverse filter
amplifies high-frequency structure and produces positive/negative ringing.
That is especially relevant here because particle filtering was disabled and
the simulator adds detector particle hits after the optical PSF; those impulses
should not physically be PSF-deconvolved. This confounds the science-support
change and strengthens the requirement to repeat the A/B on particle-filtered
simulations.

## Compute and energy

### Tracker-only comparison

| Tracker input | Median time | Range | Median gross energy | Median covered-rail power | Sampled peak power |
| --- | ---: | ---: | ---: | ---: | ---: |
| Undeconvolved `float32` cube | 41.875 s | 41.699–42.022 s | 371.450 J | 8.866 W | 10.118 W |
| Deconvolved `float32` cube | 41.715 s | 41.598–42.107 s | 369.386 J | 8.856 W | 10.018 W |

Deconvolved tracking is `0.38%` faster and uses `0.56%` less gross energy in
the median. Those differences are smaller than the repetition ranges and
should be treated as no meaningful tracker-compute change. The tracker sees
the same shape and dtype in both arms; changing pixel content alone does not
alter its workload materially.

### Added Level 2 work

| Level 2 scope | Time | Gross energy | Mean covered-rail power | Sampled peak power | Peak temperature |
| --- | ---: | ---: | ---: | ---: | ---: |
| One-time prepared-kernel setup | 0.930 s | 7.346 J | 7.895 W | 8.788 W | 43.718 °C |
| CuPy deconvolution of 241 frames | 38.210 s | 383.681 J | 10.041 W | 10.116 W | 44.031 °C |
| Per deconvolved frame | 0.1585 s | 1.592 J | — | — | — |

Including one kernel preparation, deconvolution, and deconvolved tracking, the
sequence requires `80.855 s` and `760.413 J`. Relative to tracking the
undeconvolved cube alone, that is `1.93x` the elapsed compute time and `2.05x`
the gross covered-rail energy. It still averages only `0.336 s` of compute per
input frame, leaving roughly `44x` timing headroom against the 15-second image
cadence. The compute budget is therefore not the immediate blocker; input
sensitivity is.

These are on-module `tegrastats` rails, not developer-kit wall/DC-input power,
and do not include boot, shutdown, carrier-board losses, or a complete Level
0-to-Level 4 operational cycle.

## Decision and next work

1. Do not claim that Level 2 improves or degrades CME accuracy from this event.
   There is no truth contour.
2. Treat the current automatic tracker configuration as preprocessing
   sensitive. Its thresholds and event-sector discovery were developed on the
   undeconvolved particle-snow sequence and are not invariant to the Level 2
   intensity redistribution.
3. Keep production Level 4 development aligned with its intended Level 3
   input, which includes Level 2 deconvolution upstream. Retune or normalize
   the front-evidence stage on deconvolved inputs rather than assuming the raw
   configuration transfers unchanged.
4. Repeat the paired test on the additional independent simulations, including
   particle-filtered cases. Use exported truth contours if they become
   available; otherwise freeze limited human-reviewed contours before choosing
   a preferred arm.
5. Preserve the option for a streamlined onboard path that skips Level 2 only
   as a separately validated product implementation. The measured Level 2 cost
   is modest enough that compute alone does not yet justify that bifurcation.

## Artifacts

Jetson working set and persisted deconvolved cube:

```text
/srv/suncet/data/benchmarks/cme_tracking/psf_deconvolution_ab_20260922/
```

Durable local copy of the reports, figures, exact executed runner, and compact
prepared-PSF artifact (the redundant 723-MB deconvolved cube is not copied):

```text
$suncet_data/benchmarks/cme_tracking/jetson/psf_deconvolution_ab_20260922/
```

`COMPLETE.json` matches all six report-artifact hashes. The exact runner used
for the measurement has SHA-256
`296e2c5624eed4f81668ae5eabd8e89e3c92a664da6d9f4b5d377cc9efda9dc8`;
the prepared-PSF artifact has SHA-256
`9d52b56d03fff0ede2ea2c12af82b20fa0b8c626844e339b7139cef3cf639862`.
