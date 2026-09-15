"""Implements Image Deconvolution for SunCET

Based on prototype notebook originally developed by Dan Seaton.

The main capability provided by this module (as a library context) is 
apply_deconv()

This module can also be run as a script to experiment with deconvolution;
run the module with -h for options.
"""

import argparse
import importlib
from pathlib import Path

import astropy.units as u
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
import sunpy.io.special.genx


_DIFFRACTION_REBIN_SHAPE = (1000, 1000)
_DIFFRACTION_CROP_ROWS = slice(125, -125)
_SCATTER_REBIN_SHAPE = (750, 1000)
_SCATTER_CORE_INDEX = (375, 500)
_DEFAULT_BACKEND = "numpy"
_SUPPORTED_BACKENDS = (_DEFAULT_BACKEND, "cupy")


def _normalize_backend(backend):
    """Return a canonical deconvolution backend name."""
    if not isinstance(backend, str):
        raise ValueError(
            f"backend must be one of {_SUPPORTED_BACKENDS}, got {backend!r}"
        )
    normalized = backend.strip().lower()
    if normalized not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"backend must be one of {_SUPPORTED_BACKENDS}, got {backend!r}"
        )
    return normalized


def _array_module_for_backend(backend):
    """Load and return the requested array module.

    CuPy is deliberately imported only after the GPU backend has been selected,
    so importing and running the normal CPU pipeline has no CuPy dependency.
    """
    backend = _normalize_backend(backend)
    if backend == _DEFAULT_BACKEND:
        return np
    try:
        return importlib.import_module("cupy")
    except ImportError as exc:
        raise RuntimeError(
            "The CuPy deconvolution backend was requested, but CuPy could not be "
            "imported. Install the pinned requirements-gpu-jetson.txt overlay in "
            "the isolated Jetson GPU environment."
        ) from exc


def _calibration_signature(
    diffraction_psf_file,
    scatter_psf_file,
    resp_file,
    spec_file,
    correction_factor,
):
    calibration_paths = (
        diffraction_psf_file,
        scatter_psf_file,
        resp_file,
        spec_file,
    )
    resolved_paths = tuple(
        str(Path(path).expanduser().resolve())
        for path in calibration_paths
    )
    return (*resolved_paths, float(correction_factor))


class PreparedDeconvolver:
    """Reusable Fourier-domain representation of the Level 2 inverse filter.

    Calibration preparation is independent of an input image and is expensive:
    the diffraction model contains many full-detector planes, and both PSF FFTs
    are invariant across a processing run.  Instances of this class retain those
    two FFT denominators while preserving the historical padding, shifting, and
    cropping conventions for each image.
    """

    def __init__(
        self,
        diffraction_fpsf,
        scatter_fpsf,
        image_shape,
        *,
        backend=_DEFAULT_BACKEND,
    ):
        self.backend = _normalize_backend(backend)
        self._array_module = _array_module_for_backend(self.backend)
        self.image_shape = tuple(image_shape)
        if len(self.image_shape) != 2 or any(size <= 0 for size in self.image_shape):
            raise ValueError(
                f"image_shape must contain two positive dimensions, got {image_shape!r}"
            )

        transfer_dtype = (
            self._array_module.complex128 if self.backend == "cupy" else None
        )
        self._diffraction_fpsf = self._array_module.asarray(
            diffraction_fpsf,
            dtype=transfer_dtype,
        )
        self._scatter_fpsf = self._array_module.asarray(
            scatter_fpsf,
            dtype=transfer_dtype,
        )
        expected_diffraction_shape = tuple(size * 2 for size in self.image_shape)
        if self._diffraction_fpsf.shape != expected_diffraction_shape:
            raise ValueError(
                "Padded diffraction transfer function must have shape "
                f"{expected_diffraction_shape}, got {self._diffraction_fpsf.shape}"
            )
        if self._scatter_fpsf.shape != self.image_shape:
            raise ValueError(
                "Scatter transfer function must have shape "
                f"{self.image_shape}, got {self._scatter_fpsf.shape}"
            )
        for label, fpsf in (
            ("diffraction", self._diffraction_fpsf),
            ("scatter", self._scatter_fpsf),
        ):
            if not bool(
                self._array_module.all(
                    self._array_module.isfinite(fpsf)
                ).item()
            ):
                raise ValueError(f"{label.capitalize()} transfer function is non-finite")
            if bool(self._array_module.any(fpsf == 0).item()):
                raise ValueError(f"{label.capitalize()} transfer function contains zeros")

    def apply(self, image):
        """Apply the prepared diffraction and scatter inverse filters."""
        host_image = np.asarray(image)
        if host_image.shape != self.image_shape:
            raise ValueError(
                f"Prepared deconvolver requires image shape {self.image_shape}, "
                f"got {host_image.shape}"
            )
        if not np.all(np.isfinite(host_image)):
            raise ValueError("Image contains non-finite values")

        image = (
            self._array_module.asarray(host_image, dtype=self._array_module.float64)
            if self.backend == "cupy"
            else host_image
        )

        decon_diff = _deconvolve_scatter_prepared(
            image,
            self._diffraction_fpsf,
            self.image_shape,
            array_module=self._array_module,
        )
        decon_scatt = _deconvolve_scatter_nopad_prepared(
            decon_diff,
            self._scatter_fpsf,
            self.image_shape,
            array_module=self._array_module,
        )

        # Level 2's established interface returns a NumPy host array. Keep all
        # intermediate arrays on the device and perform exactly one full-array
        # device-to-host transfer at this boundary.
        if self.backend == "cupy":
            decon_scatt = self._array_module.asnumpy(decon_scatt)

        # Check some results to confirm the resulting image is well behaved. If the
        # ratio isn't nearly one, something is wrong with the normalization of the PSF.
        print(
            "Ratio of deconvolved to raw L1 image:",
            np.sum(decon_scatt) / np.sum(host_image),
        )
        print(
            "Value of a random group of pixels that should be pretty dark in "
            "deconvolved:",
            np.mean(decon_scatt[80:100, 80:100]),
        )
        print(
            "Value of a same pixels in L1 data:",
            np.mean(host_image[80:100, 80:100]),
        )

        return decon_scatt

    def synchronize(self):
        """Wait for pending backend work; useful at benchmark boundaries."""
        if self.backend == "cupy":
            self._array_module.cuda.get_current_stream().synchronize()


class DeconvolutionPlan:
    """Run-local, lazily prepared Level 2 deconvolution plan.

    Laziness keeps input-contract failures cheap: calibration data are not loaded
    until the first valid image reaches the deconvolution boundary.  The prepared
    transfer functions are then reused for every subsequent image in the run.
    """

    def __init__(
        self,
        diffraction_psf_file,
        scatter_psf_file,
        resp_file,
        spec_file,
        correction_factor=0.4,
        *,
        backend=_DEFAULT_BACKEND,
    ):
        self._calibration_arguments = (
            diffraction_psf_file,
            scatter_psf_file,
            resp_file,
            spec_file,
        )
        self._correction_factor = float(correction_factor)
        if not np.isfinite(self._correction_factor):
            raise ValueError("correction_factor must be finite")
        self._calibration_signature = _calibration_signature(
            *self._calibration_arguments,
            self._correction_factor,
        )
        self.backend = _normalize_backend(backend)
        self._prepared = None

    def validate_calibration(
        self,
        diffraction_psf_file,
        scatter_psf_file,
        resp_file,
        spec_file,
        correction_factor,
    ):
        signature = _calibration_signature(
            diffraction_psf_file,
            scatter_psf_file,
            resp_file,
            spec_file,
            correction_factor,
        )
        if signature != self._calibration_signature:
            raise ValueError(
                "Deconvolution plan cannot be reused with a different calibration set"
            )

    def apply(self, image):
        if self._prepared is None:
            self._prepared = prepare_deconv(
                *self._calibration_arguments,
                correction_factor=self._correction_factor,
                backend=self.backend,
            )
        return self._prepared.apply(image)

    def synchronize(self):
        """Wait for pending work if calibration preparation has occurred."""
        if self._prepared is not None:
            self._prepared.synchronize()


def _main():
    """Main method of the program to test the module's apply_deconv() function.

    Run the module with -h to see options.
    Code will display a plot on the caller's computer of the original and deconvolved
    images, side by side.
    """
    # Parse commadn line arguments
    args = _get_parser().parse_args()

    # Apply a crude calibration to Level 0 data, writte originally because no Level 1
    # data is available.
    l1_data = _crude_calibration_level0(args.data_file)

    # Call apply_deconv function
    decon_scatt = apply_deconv(
        l1_data,
        args.diffraction_psf_file,
        args.scatter_psf_file,
        args.resp_file,
        args.spec_file,
        correction_factor=args.correction_factor,
        backend=args.backend,
    )

    # Make a plot showing before/after side by side
    _make_plot(l1_data, decon_scatt, args.savefig)


def _make_plot(l1_data, decon_scatt, savefig):
    """Make a side-by-side plot of the L1 Data and the deconvolved version.

    Args
      l1_data: Level 1 image array
      decon_scatt: Deconvolved data image array
      savefig: Set to true to save to a .png on disk instead of calling plt.show()
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))

    ax[0].imshow(
        np.nan_to_num(np.log10(l1_data), nan=0.0),
        cmap="Greys_r",
        vmin=1.0,
        vmax=6,
        origin="lower",
        interpolation="none",
    )
    ax[0].set_title("Unprocessed")
    ax[1].imshow(
        np.nan_to_num(np.log10(decon_scatt), nan=0.0),
        cmap="Greys_r",
        vmin=1.0,
        vmax=6,
        origin="lower",
        interpolation="none",
    )
    ax[1].set_title("Deconvolved")

    if savefig:
        out_name = "suncet_deconv.png"
        plt.savefig(out_name)
        print(f"Saved to {out_name}")
    else:
        plt.show()


def _crude_calibration_level0(data_file):
    """Apply a crude calibration to Level 0 data, writte originally because no Level 1
    data is available.

    Args
      data_file: Path to SunCET Level 0 data (FITS)
    Returns
      l1_data: Semi-calirated Level 1 data
    """
    data_hdul = fits.open(data_file)

    # Set up some config values to calibrate the image
    naxis1 = int(data_hdul[0].header["NAXIS1"] - 1)
    naxis2 = int(data_hdul[0].header["NAXIS2"] - 1)

    detector_temp = -10.0 * u.deg_C
    dark_current_mean = 20 * 2 ** ((detector_temp.value - 20) / 5.5) * u.DN / u.s

    exp_time_short = 0.035 * u.s
    exp_time_long = 15 * u.s

    inner_fov_radius = 1.33  # rsun
    inner_fov_radius_px = (
        inner_fov_radius
        * data_hdul[0].header["RSUN"]
        / data_hdul[0].header["CDELT1"]
        * u.pix
    )
    solar_disk_center = (
        data_hdul[0].header["CRPIX1"] - 1,
        data_hdul[0].header["CRPIX2"] - 1,
    )

    y_grid, x_grid = np.mgrid[:naxis1, :naxis2]
    disk_mask = (
        np.sqrt(
            (
                (x_grid - solar_disk_center[0]) ** 2
                + (y_grid - solar_disk_center[1]) ** 2
            )
        )
        <= inner_fov_radius_px.value
    )

    # Apply calibration
    print("Dark current:", dark_current_mean)

    dark_frame = np.zeros((naxis2 + 1, naxis1 + 1)) + dark_current_mean.value

    time_normalize_frame = np.zeros((naxis2 + 1, naxis1 + 1)) + exp_time_long.value
    time_normalize_frame[np.where(disk_mask)] = exp_time_short.value

    l1_data = data_hdul[0].data / time_normalize_frame - dark_frame

    return l1_data


def _rebin_interpolate(array, shape):
    """Interpolates an array to a new shape.

    Uses bilinear interpolation in scipy

    Args
      array: image data
      shape: new shape to rebin to
    Returns
      Array of passed shape with values interpolated
    """
    zoom_factors = (shape[0] / array.shape[0], shape[1] / array.shape[1])
    array_rebin = zoom(
        array, zoom_factors, order=1
    )  # order=1 for bilinear interpolation

    return array_rebin


def _deconvolve_scatter(image, psf):
    psf_shape = psf.shape
    fPSF = _padded_psf_fft(psf)

    return _deconvolve_scatter_prepared(image, fPSF, psf_shape)


def _padded_psf_fft(psf, *, backend=_DEFAULT_BACKEND):
    backend = _normalize_backend(backend)
    array_module = _array_module_for_backend(backend)
    psf_shape = psf.shape
    psf_dtype = array_module.float64 if backend == "cupy" else None
    psf = array_module.asarray(psf, dtype=psf_dtype)
    psf_padded = array_module.zeros(
        (psf_shape[0] * 2, psf_shape[1] * 2),
        dtype=array_module.float64,
    )
    psf_padded[
        psf_shape[0] // 2 : psf_shape[0] // 2 + psf_shape[0],
        psf_shape[1] // 2 : psf_shape[1] // 2 + psf_shape[1],
    ] = psf
    return array_module.fft.fft2(psf_padded)


def _deconvolve_scatter_prepared(
    image,
    fPSF,
    psf_shape,
    *,
    array_module=np,
):
    """Apply the historical padded inverse using a prepared PSF FFT."""

    image_shape = image.shape
    image_padded = array_module.zeros(
        (image_shape[0] * 2, image_shape[1] * 2),
        dtype=array_module.float64,
    )
    image_padded[
        image_shape[0] // 2 : image_shape[0] // 2 + image_shape[0],
        image_shape[1] // 2 : image_shape[1] // 2 + image_shape[1],
    ] = image
    fImage = array_module.fft.fft2(image_padded)

    decon = array_module.real(array_module.fft.ifft2(fImage / fPSF))
    decon_shift = array_module.roll(
        decon,
        shift=(psf_shape[0], psf_shape[1]),
        axis=(0, 1),
    )
    decon_cropped = decon_shift[
        psf_shape[0] // 2 : psf_shape[0] // 2 + psf_shape[0],
        psf_shape[1] // 2 : psf_shape[1] // 2 + psf_shape[1],
    ]
    return decon_cropped


def _deconvolve_scatter_nopad(image, psf, alpha=1, epsilon=0.01):
    # Padded version of the deconvolution introduces some weird ringing at the
    # edge of the image. Not sure why. If we do a no-padding version and get a
    # better result.
    psf_shape = psf.shape
    fPSF = np.fft.fft2(psf)

    return _deconvolve_scatter_nopad_prepared(image, fPSF, psf_shape)


def _deconvolve_scatter_nopad_prepared(
    image,
    fPSF,
    psf_shape,
    *,
    array_module=np,
):
    """Apply the historical circular inverse using a prepared PSF FFT."""
    fImage = array_module.fft.fft2(image)

    decon = array_module.real(array_module.fft.ifft2(fImage / fPSF))
    decon_shift = array_module.roll(
        decon, shift=(psf_shape[0] // 2, psf_shape[1] // 2), axis=(0, 1)
    )

    return decon_shift


def prepare_deconv(
    diffraction_psf_file,
    scatter_psf_file,
    resp_file,
    spec_file,
    correction_factor=0.4,
    *,
    backend=_DEFAULT_BACKEND,
):
    """Load calibration inputs and prepare reusable PSF FFT denominators.

    Calibration reads, spectral weighting, and interpolation remain on the CPU.
    Selecting ``backend="cupy"`` explicitly moves the final FP64 PSFs to the
    GPU and prepares their complex128 Fourier denominators there.
    """
    correction_factor = float(correction_factor)
    if not np.isfinite(correction_factor):
        raise ValueError("correction_factor must be finite")
    # Resolve the optional backend before opening the large calibration assets,
    # so a missing CuPy installation fails immediately rather than after the
    # 1.3 GB diffraction cube has been read and merged.
    backend = _normalize_backend(backend)
    array_module = _array_module_for_backend(backend)

    with fits.open(diffraction_psf_file) as diffraction_psf, fits.open(
        scatter_psf_file
    ) as scatter_psf:
        if not diffraction_psf or diffraction_psf[0].data is None:
            raise ValueError("Diffraction PSF FITS contains no image data")
        if not scatter_psf or scatter_psf[0].data is None:
            raise ValueError("Scatter PSF FITS contains no image data")

        # Load standard solar spectrum and the SunCET response so we can use our
        # spectrally dependent diffraction PSF to generate a single appropriately
        # averaged PSF.
        genx_data = sunpy.io.special.genx.read_genx(spec_file)
        spec_wave = genx_data["LAMBDA"] * u.Angstrom
        spec_spec = (
            genx_data["SPECTRUM"]
            * u.ph
            * u.cm ** (-2)
            * u.sr ** (-1)
            * u.s ** (-1)
            * u.Angstrom ** (-1)
        )

        resp_data = sunpy.io.special.genx.read_genx(resp_file)
        resp_wave = resp_data["SAVEGEN0"] * u.Angstrom
        resp_resp = resp_data["SAVEGEN1"] * u.cm**2 * u.DN / u.ph * u.sr / u.pix

        interp_func = interp1d(
            resp_wave,
            resp_resp,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        interpolated_resp = (
            interp_func(spec_wave) * u.cm**2 * u.DN / u.ph * u.sr / u.pix
        )
        modulated_spec = interpolated_resp * spec_spec
        modulated_values = np.asarray(modulated_spec.value)
        if modulated_values.ndim != 1:
            raise ValueError(
                "Modulated spectrum must be one-dimensional, got "
                f"{modulated_values.shape}"
            )
        if len(diffraction_psf) != modulated_values.size:
            raise ValueError(
                "Diffraction PSF plane count must match the modulated spectrum: "
                f"{len(diffraction_psf)} planes versus {modulated_values.size} bins"
            )
        if not np.all(np.isfinite(modulated_values)):
            raise ValueError("Modulated spectrum contains non-finite weights")
        modulated_sum = np.sum(modulated_spec.value)
        if not np.isfinite(modulated_sum) or modulated_sum <= 0:
            raise ValueError("Modulated spectrum must have a finite, positive sum")

        merged_diffraction_psf_array = np.copy(diffraction_psf[0].data) * 0.0
        if merged_diffraction_psf_array.ndim != 2:
            raise ValueError(
                "Diffraction PSF planes must be two-dimensional, got "
                f"{merged_diffraction_psf_array.shape}"
            )
        diffraction_shape = merged_diffraction_psf_array.shape
        for n in range(len(diffraction_psf)):
            plane = diffraction_psf[n].data
            if plane is None or plane.shape != diffraction_shape:
                plane_shape = None if plane is None else plane.shape
                raise ValueError(
                    "All diffraction PSF planes must have shape "
                    f"{diffraction_shape}, got {plane_shape} at plane {n}"
                )
            merged_diffraction_psf_array += modulated_spec[n].value * plane
        merged_diffraction_psf_array /= modulated_sum
        if not np.all(np.isfinite(merged_diffraction_psf_array)):
            raise ValueError("Merged diffraction PSF contains non-finite values")

        diff_psf_rebinned = _rebin_interpolate(
            merged_diffraction_psf_array,
            _DIFFRACTION_REBIN_SHAPE,
        )
        # The factor of four and asymmetric padding/cropping behavior are inherited
        # science-algorithm conventions. Preserve them exactly in this cache refactor.
        diff_psf_rebinned_cropped = (
            diff_psf_rebinned[_DIFFRACTION_CROP_ROWS, :] * 4.0
        )

        scatter_data = scatter_psf[0].data
        if scatter_data.ndim != 2:
            raise ValueError(
                f"Scatter PSF must be two-dimensional, got {scatter_data.shape}"
            )
        scatter_psf_rebinned = _rebin_interpolate(
            scatter_data,
            _SCATTER_REBIN_SHAPE,
        )
        TIS = np.sum(scatter_psf_rebinned)
        if not np.isfinite(TIS):
            raise ValueError("Rebinned scatter PSF has a non-finite sum")
        scatter_psf_core = (1 - TIS) * correction_factor
        scatter_psf_rebinned[_SCATTER_CORE_INDEX] = scatter_psf_core
        scatter_sum = np.sum(scatter_psf_rebinned)
        if not np.isfinite(scatter_sum) or scatter_sum == 0:
            raise ValueError("Scatter PSF must have a finite, non-zero sum")
        scatter_psf_rebinned /= scatter_sum
        if not np.all(np.isfinite(scatter_psf_rebinned)):
            raise ValueError("Rebinned scatter PSF contains non-finite values")

    if diff_psf_rebinned_cropped.shape != _SCATTER_REBIN_SHAPE:
        raise ValueError(
            "Prepared diffraction PSF must have detector shape "
            f"{_SCATTER_REBIN_SHAPE}, got {diff_psf_rebinned_cropped.shape}"
        )
    if scatter_psf_rebinned.shape != diff_psf_rebinned_cropped.shape:
        raise ValueError(
            "Prepared diffraction and scatter PSFs must have the same shape, got "
            f"{diff_psf_rebinned_cropped.shape} and {scatter_psf_rebinned.shape}"
        )
    diffraction_fpsf = _padded_psf_fft(
        diff_psf_rebinned_cropped,
        backend=backend,
    )
    scatter_array = array_module.asarray(
        scatter_psf_rebinned,
        dtype=array_module.float64 if backend == "cupy" else None,
    )
    scatter_fpsf = array_module.fft.fft2(scatter_array)
    return PreparedDeconvolver(
        diffraction_fpsf,
        scatter_fpsf,
        diff_psf_rebinned_cropped.shape,
        backend=backend,
    )


def apply_deconv(
    l1_data,
    diffraction_psf_file,
    scatter_psf_file,
    resp_file,
    spec_file,
    correction_factor=0.4,
    *,
    deconvolver=None,
    backend=None,
):
    """Apply SunCET deconvolution algorithm and return image

    Arguments
      l1_data: image array of L1 data
      diffraction_psf_file: Path to diffraction PSF file
      scatter_psf_file: Path to scatter PSF file
      resp_file: Path to SunCET spectral response function
      spec_file: Path to spectrally dependent diffraction PSF file
      correction_factor: scalar paramter used in deconvolution process
      deconvolver: optional prepared or run-local deconvolver to reuse
      backend: optional explicit ``"numpy"`` or ``"cupy"`` backend selector.
        With a supplied deconvolver, omission uses that object's backend.
    Returns
      decon_scatt: Deconvolved image using provided arguments
    """
    if deconvolver is None:
        backend = _DEFAULT_BACKEND if backend is None else _normalize_backend(backend)
        deconvolver = prepare_deconv(
            diffraction_psf_file,
            scatter_psf_file,
            resp_file,
            spec_file,
            correction_factor=correction_factor,
            backend=backend,
        )
    else:
        if isinstance(deconvolver, DeconvolutionPlan):
            deconvolver.validate_calibration(
                diffraction_psf_file,
                scatter_psf_file,
                resp_file,
                spec_file,
                correction_factor,
            )
        if backend is not None:
            requested_backend = _normalize_backend(backend)
            prepared_backend = getattr(deconvolver, "backend", None)
            if prepared_backend != requested_backend:
                raise ValueError(
                    "Requested deconvolution backend does not match the supplied "
                    f"deconvolver: {requested_backend!r} versus "
                    f"{prepared_backend!r}"
                )
    return deconvolver.apply(l1_data)


def _get_parser():
    """Define command line parser for when running module

    Returns
      instance of argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffraction-psf-file", required=True)
    parser.add_argument("--scatter-psf-file", required=True)
    parser.add_argument("--spec-file", required=True)
    parser.add_argument("--resp-file", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--correction-factor", type=float, default=0.4)
    parser.add_argument(
        "--backend",
        choices=_SUPPORTED_BACKENDS,
        default=_DEFAULT_BACKEND,
        help="FFT backend; CuPy is an explicit optional GPU path",
    )
    parser.add_argument(
        "--savefig",
        action="store_true",
        help="Save figure to disk instead of calling plt.show()",
    )

    return parser


if __name__ == "__main__":
    _main()
