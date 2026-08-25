![Python 3.12 | 3.14](https://img.shields.io/badge/python-3.12_|_3.14-blue)
[![Tests](https://github.com/suncet/suncet_processing_pipeline/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/suncet/suncet_processing_pipeline/actions/workflows/unit-tests.yml)

# SunCET Data Processing Pipeline

The tracked build and operations roadmap for the Jetson processing node is in
[the SunCET SOC plan](docs/SOC_BUILD_PLAN.md).

Pre-launch registration, APID 1 beacon decoding, dashboard setup, and
post-launch identification through SatNOGS are tracked in the
[SunCET SatNOGS onboarding plan](docs/SATNOGS_ONBOARDING_PLAN.md).
The repository also hosts the canonical
[SunCET public beacon specification](docs/SUNCET_PUBLIC_BEACON_SPEC.md).
The offline spacecraft/transmitter form entries are maintained in the
[SatNOGS DB submission draft](docs/SATNOGS_DB_SUBMISSION_DRAFT.md).

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
├── level0_5/ … level4/
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

For a new host, `local_system_setup.py` combines those setup commands into one
repeatable helper: it creates or updates the Mamba environment, installs this
checkout in editable mode, initializes the data tree, and prints the two shell
profile exports. It never edits the shell profile itself.

```sh
python local_system_setup.py \
  --data-root /absolute/public/data/path \
  --ctdb-root /absolute/private/ctdb/path
```

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

`make_run.py` also copies the current versioned FITS and NetCDF/Zarr metadata
definitions into the run directory and records their SHA-256 checksums in
`metadata_snapshot.json`. Historical runs therefore retain the definitions used
to create them even after the live development Sheet changes.

## Mission telemetry

Mission-length scalar telemetry is stored in DuckDB, with one table per APID.
This preserves each APID's natural sample times while still allowing SQL and
ASOF joins for cross-system plots. Level 0.5 decoded CSVs can be ingested with:

```sh
python -m suncet_processing_pipeline.make_telemetry_file \
  "$suncet_data/level0_5/PASS_OUTPUT/decoded"
```

Images are not stored in this database; they remain separate FITS products.

## Publishing finalized products

The SOC can publish reviewed public products through a host-local OpenSSH SFTP
alias without storing LASP credentials in this repository. The publisher is a
dry run unless `--execute` is explicitly supplied:

```sh
python -m suncet_processing_pipeline.publish_sftp \
  --local-base "$suncet_data/level1" \
  "$suncet_data/level1/2027"
```

It uploads through a temporary remote name, downloads the staged file for full
SHA-256 verification, atomically finalizes it, and writes a checksum sidecar
and JSON transfer log. It skips matching content idempotently and refuses to
overwrite a differing remote file. See the
[LASP SFTP publication procedure](docs/LASP_SFTP_PUBLICATION.md) before using
`--execute`.

## Running the Code
The code uses a lightweight run management system. First, a new run is created which makes a new directory for the run. Then, the user copies the input data (binary packet telemetary) to the input sub-directory for the run. When that is done, one or more commands are executed to perform the processing, which will leave data in output sub-directories.

The canonical Level 0.5 implementation is
`suncet_processing_pipeline.make_level0_5`. It supports X-band, hardline CCSDS,
UHF/Hydra, and combined-source ingest while retaining inspectable intermediate
binaries. For example:

```sh
python -m suncet_processing_pipeline.make_level0_5 \
  --config "$suncet_data/processing_runs/MYRUN/config.ini" \
  --folder "$suncet_data/processing_runs/MYRUN/input" \
  --input-mode combined
```

`suncet_processor.py` delegates Level 0.5 work to this same implementation.

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
