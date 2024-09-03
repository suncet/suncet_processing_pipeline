# SunCET Data Processing Pipeline

## Instructions
The SunCET Data Processing Pipeline is written in Python. To use the software, the first step is to clone the repository and install the dependencies. The dependencies may be installed through one of two routes. First, one may use the `environment.yml` file with [Anaconda](https://www.anaconda.com/) or a compatible replacement (such as [Miniconda](https://docs.anaconda.com/miniconda/), [Mamba](https://mamba.readthedocs.io/en/latest/) or [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)). The `environment.yml` route only uses the free channels (conda-forge), which do not require a paid account. Alternatively, one can use the `requirements.txt` file directly with `pip`.

To use the `environment.yml` route, run the following line. The `conda` command can be replaced with `mamba` or `micromamba` as needed.

```
$ conda env create -f environment.yml`
$ conda activate suncet
```
	
To use the `requirements.txt` route, run the following line. This may be done inside your own environment, such as with virtualenv.
```
$ pip install -r requirements.txt
```

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
| Level 0b | Level 0c | Level 1 | Level 2 |
| --- | --- | --- | --- |
| Raw images with metadata in header | Added ancillary data needed to produce L1 | Images with dark and flat field corrections | Coronal Mass Ejection (CME) catalog: average speed and acceleration, height-time, speed-time, and acceleration-time plots, movies |
