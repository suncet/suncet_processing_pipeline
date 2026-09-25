# End-to-end Jetson power supervisor

`run_pipeline_benchmark.py` executes an ordered set of pipeline commands under
one continuous `tegrastats` trace. Each command is an ordinary argv array—not a
shell string—and gets an exact monotonic start/end boundary. After the child
exits, the supervisor closes its stdout/stderr logs and performs a blocking
`os.sync()` before recording the phase end. Large FITS, CSV, and database writes
therefore remain charged to the stage that created them instead of leaking into
the next stage through delayed Linux writeback. The result keeps both the
directly measured total and independently reduced Level 0.5, 1, 2, 3, and 4
intervals.

The flush is system-wide and its duration and status are retained under each
phase's `filesystem_flush` record. Keep the Jetson otherwise quiescent during a
benchmark: unrelated filesystem writes completed by the barrier would otherwise
inflate that phase. Flush energy and time are intentional parts of each stage's
gross operational cost.

The supervisor never copies or transforms science inputs. Prepare the X-band
file, synthetic Level 0.5 sequence, calibration files, and CTDB assets on the
NVMe before starting it. List them under `pre_staged_paths`; those assertions
are performed before telemetry collection. The hybrid real-telemetry/synthetic-
image substitution therefore contributes no measured phase energy.

## Plan format

```json
{
  "schema": "suncet.end_to_end_plan",
  "schema_version": 1,
  "label": "SunCET hybrid 241-frame pipeline",
  "pre_staged_paths": [
    {"name": "xband", "path": "/srv/suncet/bench/input.dat", "kind": "file"},
    {"name": "synthetic_l0_5", "path": "/srv/suncet/bench/synthetic", "kind": "directory"}
  ],
  "phases": [
    {
      "name": "level0_5_skip_images",
      "description": "Exact real X-band file, CSIE assembly disabled",
      "command": ["/path/to/python", "-m", "package.command", "--skip-csie-images", "--output", "${RUN_DIR}/level0_5_skip"],
      "cwd": "${REPO_ROOT}",
      "env": {"suncet_data": "/srv/suncet/data"},
      "required_paths": [{"path": "/srv/suncet/bench/input.dat", "kind": "file"}],
      "expected_outputs": [{"path": "${RUN_DIR}/level0_5_skip", "kind": "directory"}],
      "workload_units": {"input_files": 1}
    },
    {
      "name": "level0_5_full",
      "description": "Exact real X-band file with CSIE assembly",
      "command": ["/path/to/python", "-m", "package.command", "--output", "${RUN_DIR}/level0_5_full"],
      "workload_units": {"image_pixels": 89864000, "input_files": 1}
    },
    {
      "name": "level1",
      "command": ["/path/to/python", "-m", "package.command", "--input", "/srv/suncet/bench/synthetic", "--output", "${RUN_DIR}/level1"],
      "workload_units": {"frames": 241}
    },
    {
      "name": "level2",
      "command": ["/path/to/python", "-m", "package.level2", "--input", "${RUN_DIR}/level1", "--output", "${RUN_DIR}/level2"],
      "workload_units": {"frames": 241}
    },
    {
      "name": "level3",
      "command": ["/path/to/python", "-m", "package.level3", "--input", "${RUN_DIR}/level2", "--output", "${RUN_DIR}/level3"],
      "workload_units": {"frames": 241}
    },
    {
      "name": "level4",
      "command": ["/path/to/python", "-m", "package.level4", "--input", "${RUN_DIR}/level3", "--output", "${RUN_DIR}/level4"],
      "workload_units": {"frames": 241}
    }
  ],
  "derived_metrics": [
    {
      "name": "hybrid_241_frame_pipeline",
      "type": "paired_delta_projection",
      "baseline_phase": "level0_5_skip_images",
      "full_phase": "level0_5_full",
      "unit": "image_pixels",
      "observed_units": 89864000,
      "target_units": 180750000,
      "target_frame_count": 241,
      "additional_phases": ["level1", "level2", "level3", "level4"]
    }
  ]
}
```

`${RUN_DIR}`, `${PLAN_DIR}`, and `${REPO_ROOT}` are expanded without invoking a
shell. Relative paths in path assertions and `cwd` are relative to the plan.
`required_paths` are checked immediately before a phase; `expected_outputs`
are checked immediately after it. Generated handoffs can therefore pass by
path without an unmeasured internal copy.

The real X-band file contains 124 images captured for a ground test, including
118 at 1000×752 and six at 500×376. Their spacecraft timestamps and apparent
capture span are not used to normalize the workload. Multiplying the *whole*
Level 0.5 energy by 241/124 would incorrectly multiply fixed transfer-frame
parsing and non-image telemetry cost.

The paired-delta model instead measures the exact file twice: once without
CSIE image assembly (`E_skip`) and once with it (`E_full`). It projects only the
image-dependent delta:

```text
E_target_L0.5 = E_skip + (180,750,000 / 89,864,000) × (E_full - E_skip)
```

Here 89,864,000 is the exact observed image-pixel count and 180,750,000 is 241
synthetic frames × 1000 × 750 pixels. The paired projection retains both
exact-file measurements, the measured delta, the projected Level 0.5 value,
and the phase-only sum after adding Levels 1–4. Its
`exact_hybrid_phase_sum` is the directly measured `E_full + Levels 1–4`; its
`projected_total` is the modeled hybrid 241-frame result. Both exclude substitution,
supervisor gaps, transfer, boot, and shutdown.
With `target_frame_count` present, both totals also report their amortized
joules and seconds per target frame.

When both paired Level 0.5 phases run in one plan, `pipeline_total` is the
energy of the *benchmark suite*: it includes the diagnostic skip-images pass,
the full pass, all science phases, and interphase gaps. It must not be reported
as operational all-level energy. Reverse the skip/full order in repeated trials
to quantify filesystem-cache and thermal-order bias.

## Run

Validate expansion and all globally pre-staged inputs first:

```sh
python benchmarks/end_to_end/run_pipeline_benchmark.py \
  --plan /srv/suncet/bench/plan.json \
  --run-dir /srv/suncet/bench/runs/trial-01 \
  --dry-run
```

Then measure from the repository root:

```sh
python benchmarks/end_to_end/run_pipeline_benchmark.py \
  --plan /srv/suncet/bench/plan.json \
  --run-dir /srv/suncet/bench/runs/trial-01 \
  --pre-run-idle-seconds 20 \
  --telemetry required \
  --telemetry-interval-ms 100
```

Each phase receives separate stdout/stderr logs. Outputs are:

- `measurement.json`: full provenance, exact boundaries, raw telemetry,
  per-phase/total reductions, paired-delta projections, SHA-256 provenance for
  untracked repository files, and energy reconciliation;
- `phase_summary.csv`: total, phase, and supervisor-gap rows; and
- `telemetry_samples.csv`: the continuous raw trace in tabular form.

Use a fresh run directory for every repetition. `--overwrite-results` replaces
only the supervisor's reports and logs; it never deletes or overwrites pipeline
science products on the supervisor's behalf. `--phase NAME` can measure one or
more selected stages when all of their inputs are already staged. Selected
stages run in JSON plan order by default, regardless of the order of repeated
`--phase` options.

For a paired trial that reverses the Level 0.5 cache/thermal order without
duplicating the plan, list every phase to be executed and opt into the supplied
order explicitly:

```sh
python benchmarks/end_to_end/run_pipeline_benchmark.py \
  --plan /srv/suncet/bench/plan.json \
  --run-dir /srv/suncet/bench/runs/trial-full-then-skip \
  --phase level0_5_full \
  --phase level0_5_skip_images \
  --phase level0_5_duckdb \
  --phase level1 \
  --phase level2 \
  --phase level3 \
  --phase level4 \
  --phase-order requested \
  --telemetry required
```

`--phase` remains a selection mechanism: phases omitted from the command are not
run. Derived metrics continue to resolve measurements by phase name, so reversing
execution order does not change their formulas.

The power values are the gross sum of the R39 module rails used by the existing
Level 2/Level 4 harness. They do not include carrier-board conversion losses,
boot, shutdown, or energy drawn while the Jetson is off.

## Aggregate repeated trials

After the repeated runs finish, aggregate their authoritative
`measurement.json` records with:

```sh
python benchmarks/end_to_end/aggregate_trials.py \
  /srv/suncet/bench/runs/trial-01 \
  /srv/suncet/bench/runs/trial-02/measurement.json \
  /srv/suncet/bench/runs/trial-03 \
  --output-directory /srv/suncet/bench/runs/summary
```

A run directory and a direct path to `measurement.json` are both accepted.
The aggregator rejects failed runs and incompatible normalized phase commands,
environments, path/input declarations, workloads, plan notes,
paired-projection definitions, or telemetry sampling intervals. Phase order
may differ between otherwise compatible trials, which permits alternating the
paired baseline/full order.

It writes `trial_summary.json`, `trial_summary.csv`, and `trial_summary.md`.
Per-phase tables report median [minimum–maximum] gross energy, duration,
sampled peak power, peak temperature, joules per frame, and joules per
megapixel where the phase declares those workload units. The directly measured
`exact_hybrid_phase_sum` and modeled `projected_total` remain separate rows;
the diagnostic baseline is never folded into either. Plan notes and the Level
3 pass-through caveat are retained prominently so its present file-I/O cost is
not mistaken for future Level 3 science-correction cost. Use `--overwrite` only
when intentionally replacing an existing set of summary files.
