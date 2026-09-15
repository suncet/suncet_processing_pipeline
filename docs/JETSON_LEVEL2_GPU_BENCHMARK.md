# Jetson Level 2 GPU setup and benchmark protocol

This procedure adds the smallest robust system CUDA runtime and JIT headers
needed for CuPy FFT development on `suncet-soc`. It deliberately leaves the reviewed
production environment unchanged and puts the mutable GPU-development
environment on the NVMe.

The validated platform for these commands is:

- Jetson AGX Orin (`aarch64`)
- Jetson Linux/L4T R39.2.1 and JetPack 7.2.1
- Ubuntu 24.04
- NVIDIA driver compatible with CUDA 13.2
- Python 3.14 from the locked SunCET environment

The Level 2 PSF files and frozen synthetic frame used here are provisional
engineering inputs. Passing this protocol demonstrates implementation parity
and measures compute performance; it does not approve the calibration set or
the resulting image as a mission science product.

## 1. Verify the platform and proposed packages

Do not substitute packages from another CUDA or L4T repository. Confirm the
platform and inspect the exact transaction before installing anything:

```sh
uname -m
cat /etc/nv_tegra_release
nvidia-smi
apt-cache policy cuda-libraries-13-2 cuda-cudart-dev-13-2
sudo apt-get --simulate install --no-install-recommends \
  cuda-libraries-13-2=13.2.2-1 \
  cuda-cudart-dev-13-2=13.2.86-1
```

The simulation should use the configured NVIDIA `r39.2` repository and should
not replace the BSP, kernel, or driver. Stop and review the repository state if
`13.2.2-1` is no longer available or the transaction proposes unrelated
upgrades.

Install the runtime libraries:

```sh
sudo apt update
sudo apt install --no-install-recommends \
  cuda-libraries-13-2=13.2.2-1 \
  cuda-cudart-dev-13-2=13.2.86-1
```

`cuda-libraries-13-2` supplies CUDA Runtime, NVRTC, cuFFT, nvJitLink, and the
other standard CUDA runtime libraries. `cuda-cudart-dev-13-2` adds the small
set of CUDA runtime/CCCL/driver headers that CuPy's NVRTC JIT needs for array
operations. A live smoke test established that the runtime meta-package alone
can discover the driver and cuFFT but fails on its first JIT operation with
`Failed to find CUDA headers`; the additional four packages occupy about
23 MiB. This pair still does not install `nvcc`, the full development-library
set, cuDNN, TensorRT, or the complete JetPack SDK. Add the compiler and broader
development packages later only if a reviewed implementation actually requires
custom CUDA C++.

Do not install either the full `nvidia-jetpack`/`cuda-toolkit-13-2` stack or
the `cupy-cuda13x[ctk]` extra for this task. The latter would place a second set
of CUDA component libraries in the Python environment alongside the
L4T-matched system libraries.

Record the installed runtime before benchmarking:

```sh
dpkg-query -W -f='${binary:Package}\t${Version}\n' \
  'cuda-*-13-2' 'libcu*-13-2' 'libnv*-13-2' 2>/dev/null | sort
ldconfig -p | grep -E 'lib(cudart|nvrtc|cufft|nvJitLink)\.so'
ls -ld /usr/local/cuda*
```

Do not run the unrelated `apt autoremove` suggested by Ubuntu on this Jetson;
its candidate set has included boot-support packages.

## 2. Create an isolated environment on NVMe

The locked production runtime remains on eMMC at
`/home/james/.local/share/mamba/envs/suncet-release-3da31c5`. Do not add CuPy to
that environment. Create an NVMe directory once, then clone the locked runtime
as the unprivileged `james` user:

```sh
sudo install -d -o james -g james -m 0755 /srv/suncet/envs

/home/james/miniforge3/bin/mamba create --yes \
  --prefix /srv/suncet/envs/suncet-level2-gpu-dev \
  --clone /home/james/.local/share/mamba/envs/suncet-release-3da31c5 \
  --always-copy

/srv/suncet/envs/suncet-level2-gpu-dev/bin/python -m pip install \
  --no-cache-dir \
  --requirement requirements-gpu-jetson.txt
```

`--always-copy` is intentional because the source and target are on different
filesystems. `--no-cache-dir` prevents the large wheel from accumulating in the
eMMC user cache. A live dependency dry run on this platform selected the
CPython 3.14 `manylinux2014_aarch64` CuPy wheel and only the pinned
`cuda-pathfinder` dependency.

For active backend development, the isolated environment may install the
checkout in editable mode:

```sh
/srv/suncet/envs/suncet-level2-gpu-dev/bin/python -m pip install \
  --no-deps --editable .
```

An editable, dirty checkout is suitable for development comparisons, not a
final characterization result. Once the implementation is accepted, build a
wheel from a named clean commit and create a commit-named environment directly
from `conda-lock.yml`; install that wheel and the pinned GPU overlay there.
Record the wheel hash and complete package inventory.

## 3. Validate CUDA and cuFFT

Run the standalone smoke test from the repository root:

```sh
/srv/suncet/envs/suncet-level2-gpu-dev/bin/python \
  benchmarks/level2/validate_cupy.py
```

The default `1500 x 2000` array matches the padded diffraction FFT dimensions
in the current Level 2 algorithm. The program:

- imports CuPy only after parsing its arguments;
- identifies the Python, CuPy, CUDA runtime, driver, and selected GPU;
- performs warm-up passes followed by synchronized FP64 FFT/IFFT pairs;
- reports every timed duration plus numerical round-trip errors; and
- returns nonzero for an import/runtime failure, non-finite output, or an error
  exceeding the stated limits.

It does not read or write mission data. CuPy's process cache and the CUDA driver
cache are disabled for this smoke process; production benchmarking should
measure and report its chosen cache policy explicitly.

Changing array dimensions can verify memory behavior, but it is not a valid
replacement for the representative Level 2 shape:

```sh
/srv/suncet/envs/suncet-level2-gpu-dev/bin/python \
  benchmarks/level2/validate_cupy.py \
  --rows 1500 --columns 2000 --warmups 2 --repetitions 10
```

If component discovery fails, inspect `cupy.show_config()` in the program's
output and the `/usr/local/cuda` alternatives first. Do not permanently add a
global `LD_LIBRARY_PATH` until the missing component and its intended package
have been identified.

## 4. Level 2 equivalence gate

CUDA availability alone is not acceptance. Establish the cached CPU reference
first, then apply the following order:

1. Reproduce the frozen Level 2 handoff with the cached CPU implementation.
2. Run the GPU implementation in FP64 with the same padded diffraction and
   circular scatter boundary models.
3. Compare every output pixel with the cached CPU result and the frozen Mac
   result; record maximum absolute error, RMS error, relative L2 error, finite
   pixel count, FITS checksums, and header/schema checks.
4. Treat any crop, regularization change, real-FFT conversion, or FP32 path as
   a separate science-facing experiment with its own tolerance and provenance.

The PSFs must not be spatially cropped for deconvolution. Their final arrays are
nonzero throughout, and inverse filtering has global support. The safe first
optimization is to prepare the fixed calibration data once and cache the two
PSF Fourier denominators for the run.

## 5. Timing and power scopes

Report these scopes separately so I/O and one-time work are not mistaken for
per-image GPU cost:

1. **Cold start:** interpreter/import, calibration reads and hashes, PSF merge,
   rebinning, normalization, and the initial PSF FFTs.
2. **Warm compute:** already-prepared PSF denominators and resident arrays;
   synchronize the CUDA stream immediately before and after every timed region.
3. **Per-frame transfer:** host-to-device input and the single final
   device-to-host result copy.
4. **End to end:** FITS input, validation, deconvolution, output construction,
   checksums, and provenance.
5. **Powered processing cycle:** boot/startup, the accumulated image batch,
   output persistence, shutdown, and total gross energy when evaluating a
   future spacecraft duty cycle.

For every result, record the Git commit and dirty state, environment prefix and
package inventory, input/calibration hashes, L4T and CUDA versions, `nvpmodel`
mode, clocks policy, batch size, warm-up count, peak temperature, peak power,
elapsed time, and gross energy. The Jetson uses unified physical memory, but
CuPy array creation and explicit host/device conversions still have measurable
cost and must remain inside the appropriate scope.

Begin with FP64 as the numerical reference. Real FFTs and FP32 may improve
throughput or energy, but neither should become the default until frozen-output
and representative science comparisons establish acceptable error bounds.
