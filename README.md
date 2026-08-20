![Python 3.12 | 3.14](https://img.shields.io/badge/python-3.12_|_3.14-blue)
[![Tests](https://github.com/suncet/suncet_processing_pipeline/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/suncet/suncet_processing_pipeline/actions/workflows/unit-tests.yml)

# SunCET Data Processing Pipeline

The tracked build and operations roadmap for the Jetson processing node is in
[the SunCET SOC plan](docs/SOC_BUILD_PLAN.md).

## Portable data roots

Every processing host must define two absolute paths:

```sh
export suncet_data="/absolute/path/to/public-suncet-data"
export suncet_ctdb="/absolute/path/to/private-suncet-ctdb"
```

`suncet_data` is the root of the standard public/synchronized data tree. All
managed inputs, products, calibration data, trends, manifests, and named
processing runs use the same relative structure below it. `suncet_ctdb` is a
separate, private, host-managed CTDB root. The pipeline rejects configurations
where the two roots overlap so CTDB content cannot accidentally enter the
publicly synchronized data tree.

The checked-in configurations use `${suncet_ctdb}` for `paths.ctdb_base` and
select bus and CSIE versions below that root. A specialized configuration may
provide an explicit absolute CTDB root instead.

The stable top-level convention is:

```text
$suncet_data/                  public or synchronized
├── calibration/
├── metadata/
├── level0/ … level3/
├── processing_runs/
├── synthetic/
├── test_data/
└── trends/

$suncet_ctdb/                  private and host-managed
├── suncet_v<bus-version>/
└── suncet_csie_v<csie-version>/
```

Additional science/reference directories may live below `suncet_data`, but code
must address them relative to that root.

## Instructions
The SunCET Data Processing Pipeline supports Python 3.12 and the current Python 3.14 release on Intel and ARM systems. The environment specifications are architecture-neutral: Conda/Mamba or pip selects native packages for Intel/ARM macOS and Linux automatically.

The recommended production/SOC environment contains only pipeline runtime dependencies:

To use the `environment.yml` route, run the following line. The `conda` command can be replaced with `mamba` or `micromamba` as needed.

```
mamba env create -f environment.yml
mamba activate suncet
python -m pip install --no-deps -e .
```

For an exactly reproducible install, `conda-lock.yml` freezes the tested package builds for `linux-aarch64`, `linux-64`, `osx-arm64`, and `osx-64`:

```
conda-lock install --name suncet conda-lock.yml
conda activate suncet
python -m pip install --no-deps -e .
```

For development, testing, and notebooks, use the development environment instead:

```
mamba env create -f environment-dev.yml
mamba activate suncet-dev
python -m pip install --no-deps -e .
```

GNU Radio is not imported by the processing pipeline. Its large dependency stack is therefore isolated in `environment-radio.yml` and can be installed only where radio tooling is needed.

To use pip inside an existing Python 3.12+ virtual environment:

```
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

Developers using pip should install `requirements-dev.txt` instead. CUDA and JetPack are system-level Jetson dependencies and are intentionally not included in these portable Python environments.

## Processing provenance

Processing commands automatically write a unique JSON manifest under `<data folder>/processing_manifests/`. A manifest is written atomically when the run starts and finalized whether processing succeeds or raises an exception. Each manifest records:

- UTC start/end times, duration, status, command line, and parsed arguments
- Git commit, branch, dirty-tree state, and changed file names
- configuration contents and SHA-256 checksum, with sensitive-looking values redacted
- resolved data paths, private-root-relative CTDB labels, and pipeline, bus, and CSIE versions
- hostname, operating system, CPU architecture, Python, Conda, and installed package versions
- input file sizes, timestamps, and SHA-256 checksums
- created, modified, or deleted outputs and SHA-256 checksums
- exception type, message, and traceback for failed runs

The manifests are deliberately stored with the processed data rather than in the Git repository. Re-running the pipeline creates a new manifest instead of overwriting previous provenance.

## Running the Code
The code uses a lightweight run management system. First, a new run is created which makes a new directory for the run. Then, the user copies the input data (binary packet telemetary) to the input sub-directory for the run. When that is done, one or more commands are executed to perform the processing, which will leave data in output sub-directories.

```sh
python make_run.py --run-name MYRUN  # creates $suncet_data/processing_runs/MYRUN

cp TELEMETRY_PATH/*.bin "$suncet_data/processing_runs/MYRUN/input/"

# Set paths.data_to_process_path = processing_runs/MYRUN/input in this config,
# then run it explicitly.
python suncet_processor.py \
  --config "$suncet_data/processing_runs/MYRUN/config.ini"

cp "$suncet_data"/processing_runs/MYRUN/level3/* /ftp/public/level3/
```

To delete a run from disk, all thats needed is to delete its directory:

```sh
rm -r "$suncet_data/processing_runs/MYRUN"
```

## Running the tests
To run the tests, run the following command from the top-level directory:

```
$ pytest -v
```

## Discussion: Input
* To get to Level 0b: Raw binary files downlinked from the spacecraft
* To get to Level 0c, 1, and 2: the output of the preceding level

## Discussion: Output
| Level 0.5 | Level 1 | Level 2 | Level 3 | Level 4 |
| --- | --- | --- | --- | --- |
| Raw images with metadata in header | Images with basic corrections like dark and flat field | Images with point spread function deconvolution | Images with detailed corrections like fine rotation and sun-centering | Coronal Mass Ejection (CME) catalog: average speed and acceleration, height-time, speed-time, and acceleration-time plots, movies |
