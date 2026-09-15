# Level 2 Jetson benchmark harness

`benchmark_deconvolution.py` compares the exact prepared Level 2 inverse filter
with either the `numpy` or explicitly selected `cupy` backend. It records:

- the Git commit, branch, dirty paths, and tracked-diff digest;
- Python/environment/package identity and SHA-256 for every input;
- read-only `nvpmodel`, `jetson_clocks --show`, and `nvidia-smi` output;
- separate import, input-load, calibration-preparation, first-apply, warm-up,
  and measured prepared-apply durations;
- raw 100 ms `tegrastats` samples, exact-boundary gross energy, simultaneous
  sampled peak power, and current `tj` peak temperature; and
- a deterministic digest and numerical summary of the final host output.

The harness reuses `suncet_processing_pipeline.level4.jetson_metrics`, so Level
2 and Level 4 use one definition of the AGX Orin power rails. Jetson Linux R39.2
does not emit a literal `VDD_IN` field on this unit. The reported module-input
estimate is the simultaneous sum of the first/current values of
`VDD_GPU_SOC`, `VDD_CPU_CV`, and `VIN_SYS_5V0`. `VDDQ_VDD2_1V8AO` is retained
in raw telemetry but not added because it is already included in
`VIN_SYS_5V0`. No idle subtraction is applied.

## Controlled comparison

Use the same clean commit, isolated GPU environment, input files, power mode,
and dynamic/fixed clock policy for both backends. Run no other workload on the
Jetson. A 30-second pre-run interval is useful for a first controlled pair; the
temperature at the start still needs inspection, and alternating backend order
across repeated pairs is preferable to treating one CPU-then-GPU pair as a
final characterization.

From the repository root on `suncet-soc`, define paths without changing global
shell configuration:

```sh
benchmark_root=/srv/suncet/data/benchmarks/level2
tool="$PWD/benchmarks/level2/benchmark_deconvolution.py"
python=/srv/suncet/envs/suncet-level2-gpu-dev/bin/python
input=/srv/suncet/data/synthetic/level0/fits/config_default_no_particle_filter_OBS_2023-01-14T17:30:00.000_120.fits
diffraction=/srv/suncet/data/filter_mesh_diffraction/suncet_diffraction_patterns_2k.fits
scatter=/srv/suncet/data/mirror_scatter/suncet_mirror_scatter_psf_baffled.fits
spectrum=/srv/suncet/data/filter_mesh_diffraction/suncet_sample_spec.genx
response=/srv/suncet/data/calibration/suncet_spectral_resp.genx
run_dir="$benchmark_root/controlled_30w_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$run_dir"
```

Run CPU and GPU trials with required telemetry and save the last output from
each for a separate numerical comparison:

```sh
"$python" "$tool" \
  --backend numpy \
  --input-file "$input" \
  --diffraction-psf-file "$diffraction" \
  --scatter-psf-file "$scatter" \
  --spectrum-file "$spectrum" \
  --spectral-response-file "$response" \
  --warmups 2 \
  --repetitions 10 \
  --minimum-measured-seconds 10 \
  --pre-run-idle-seconds 30 \
  --telemetry required \
  --output-json "$run_dir/numpy.json" \
  --save-last-output "$run_dir/numpy_last.npy"

"$python" "$tool" \
  --backend cupy \
  --input-file "$input" \
  --diffraction-psf-file "$diffraction" \
  --scatter-psf-file "$scatter" \
  --spectrum-file "$spectrum" \
  --spectral-response-file "$response" \
  --warmups 2 \
  --repetitions 10 \
  --minimum-measured-seconds 10 \
  --pre-run-idle-seconds 30 \
  --telemetry required \
  --output-json "$run_dir/cupy.json" \
  --save-last-output "$run_dir/cupy_last.npy"
```

The measured batch automatically continues until both its minimum repetition
count and minimum duration are satisfied. Gross energy per prepared application
is emitted only when pre- and post-workload samples fully bracket the batch.
Power results from a sub-10-second batch or fewer than three internal samples
carry explicit quality warnings.

Gate the saved GPU array against the CPU array with the repository's explicit
FP64 engineering tolerances:

```sh
"$python" "$PWD/benchmarks/level2/compare_outputs.py" \
  --reference "$run_dir/numpy_last.npy" \
  --candidate "$run_dir/cupy_last.npy" \
  --output-json "$run_dir/numerical_comparison.json"
```

This is a deconvolution-core benchmark. Its CuPy scope includes input H2D, both
inverse-filter stages, and final D2H, but excludes Level 1 calibration, FITS
product construction/writing, boot, and shutdown. A future powered-cycle test
needs external spacecraft-bus instrumentation because `tegrastats` cannot
measure energy while the Jetson is off or booting.

## End-to-end command measurement

`measure_command.py` measures an external command from immediately before
child spawn through child exit. For `make_level2`, that includes interpreter
and import startup, calibration reads and hashes, PSF preparation,
deconvolution, provenance, FITS construction, checksum/schema validation, and
disk write. It excludes telemetry priming, the optional pre-run settle, result
JSON writing, system boot, and shutdown.

Run it from the repository root with an explicit `--` separator and a fresh
output directory. For example:

```sh
export suncet_data=/srv/suncet/data
export suncet_ctdb=/srv/suncet/ctdb
backend=cupy
metadata="$suncet_data/metadata/suncet_metadata_definition_v1.0.2dev-FITS.csv"
trial="$benchmark_root/end_to_end_${backend}_$(date -u +%Y%m%dT%H%M%SZ)"

"$python" benchmarks/level2/measure_command.py \
  --output-json "$trial/measurement.json" \
  --command-cwd "$PWD" \
  --pre-run-idle-seconds 20 \
  --telemetry required \
  --telemetry-interval-ms 100 \
  -- \
  "$python" -m suncet_processing_pipeline.make_level2 \
    --input-path "$input" \
    --output-path "$trial/product" \
    --diffraction-psf-file "$diffraction" \
    --scatter-psf-file "$scatter" \
    --spec-file "$spectrum" \
    --resp-file "$response" \
    --metadata-definition-file "$metadata" \
    --input-kind synthetic_level0_5_bypass \
    --product-status PROVISIONAL \
    --deconvolution-backend "$backend"
```

The provisional fixture above deliberately uses the `v1.0.2dev` contract.
Switch it to the current production candidate only after the input writer
provides every newly required keyword, including `SOLAR_R` in `v1.0.3dev`.

The wrapper uses an argument vector rather than a shell, captures the child's
standard streams, preserves its exit status, writes JSON atomically without
overwriting by default, and reports gross energy only when telemetry fully
brackets the command. Repeat in reverse backend order with fresh output paths.
Point `--input-path` at a directory to characterize a buffered multi-product
run in one process; record whether the inputs are distinct science frames or a
timing-only repeated fixture.
