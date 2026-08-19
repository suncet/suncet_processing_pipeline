![Python 3.12 | 3.14](https://img.shields.io/badge/python-3.12_|_3.14-blue)
[![Tests](https://github.com/suncet/suncet_processing_pipeline/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/suncet/suncet_processing_pipeline/actions/workflows/unit-tests.yml)

# SunCET Data Processing Pipeline

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

## Running the Code
The code uses a lightweight run management system. First, a new run is created which makes a new directory for the run. Then, the user copies the input data (binary packet telemetary) to the input sub-directory for the run. When that is done, one or more commands are executed to perform the processing, which will leave data in output sub-directories.

```
$ python make_run.py --run-name MYRUN                   # this makes a directory processing_runs/MYRUN

$ cp TELEMETRY_PATH/*.bin procesing_runs/MYRUN/input  # add input files

$ python suncet_processor.py --run-name MYRUN          # begins processing the run and writes output

$ cp processing_runs/MYRUN/level3/* /ftp/public/level3  # copy output data to export directory
```

To delete a run from disk, all thats needed is to delete its directory:

```
$ rm -r processing_runs/MYRUN`
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
