"""Implements Image Deconvolution for SunCET

Based on prototype notebook originally developed by Dan Seaton.

The main capability provided by this module (as a library context) is 
apply_deconv()

This module can also be run as a script to experiment with deconvolution;
run the module with -h for options.
"""

import argparse

import astropy.units as u
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
import sunpy.io.special.genx


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
    psf_padded = np.zeros((psf_shape[0] * 2, psf_shape[1] * 2))
    psf_padded[
        psf_shape[0] // 2 : psf_shape[0] // 2 + psf_shape[0],
        psf_shape[1] // 2 : psf_shape[1] // 2 + psf_shape[1],
    ] = psf
    fPSF = np.fft.fft2(psf_padded)

    image_shape = image.shape
    image_padded = np.zeros((image_shape[0] * 2, image_shape[1] * 2))
    image_padded[
        image_shape[0] // 2 : image_shape[0] // 2 + image_shape[0],
        image_shape[1] // 2 : image_shape[1] // 2 + image_shape[1],
    ] = image
    fImage = np.fft.fft2(image_padded)

    decon = np.real(np.fft.ifft2(fImage / fPSF))
    decon_shift = np.roll(decon, shift=(psf_shape[0], psf_shape[1]), axis=(0, 1))
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

    image_shape = image.shape
    fImage = np.fft.fft2(image)

    decon = np.real(np.fft.ifft2(fImage / fPSF))
    decon_shift = np.roll(
        decon, shift=(psf_shape[0] // 2, psf_shape[1] // 2), axis=(0, 1)
    )

    return decon_shift


def apply_deconv(
    l1_data,
    diffraction_psf_file,
    scatter_psf_file,
    resp_file,
    spec_file,
    correction_factor=0.4,
):
    """Apply SunCET deconvolution algorithm and return image

    Arguments
      l1_data: image array of L1 data
      diffraction_psf_file: Path to diffraction PSF file
      scatter_psf_file: Path to scatter PSF file
      resp_file: Path to SunCET spectral response function
      spec_file: Path to spectrally dependent diffraction PSF file
      correction_factor: scalar paramter used in deconvolution process
    Returns
      decon_scatt: Deconvolved image using provided arguments
    """
    # Load Diffraction and Scatter PSF files
    diffraction_psf = fits.open(diffraction_psf_file)
    scatter_psf = fits.open(scatter_psf_file)

    # Load standard solar spectrum and the SunCET response so we can use our
    # spectrally dependent diffraction PSF to generate a single appropriately
    # averaged PSF
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

    # Load SunCET spectral response function
    resp_data = sunpy.io.special.genx.read_genx(resp_file)
    resp_wave = resp_data["SAVEGEN0"] * u.Angstrom
    resp_resp = resp_data["SAVEGEN1"] * u.cm**2 * u.DN / u.ph * u.sr / u.pix

    # Interpolate the response function onto the spectral bins
    interp_func = interp1d(
        resp_wave, resp_resp, kind="linear", bounds_error=False, fill_value=0.0
    )
    interpolated_resp = interp_func(spec_wave) * u.cm**2 * u.DN / u.ph * u.sr / u.pix

    # Modulate the spectrum by the response function
    modulated_spec = interpolated_resp * spec_spec

    # Make PSF
    num_diff_hdus = len(diffraction_psf)
    merged_diffraction_psf_array = np.copy(diffraction_psf[0].data) * 0.0

    for n in range(num_diff_hdus):
        merged_diffraction_psf_array += (
            modulated_spec[n].value * diffraction_psf[n].data
        )

    merged_diffraction_psf_array /= np.sum(modulated_spec.value)

    # Do the deconvolution -------------------------------------------------------------
    new_shape = (1000, 1000)
    diff_psf_rebinned = _rebin_interpolate(merged_diffraction_psf_array, new_shape)

    # factor of 4 accounts for rebinning effects to preserve normalization
    diff_psf_rebinned_cropped = diff_psf_rebinned[125:-125, :] * 4.0
    decon_diff = _deconvolve_scatter(l1_data, diff_psf_rebinned_cropped)

    # There's a weird ad hoc correction to the core size from what I would
    # expect to get this deconvolution to work. I suspect it has to do with
    # the rebinning, because the perfect number (for getting values close to
    # zero in far field pixels) is about 0.25 -- and the rebin factor is 4.
    # But This value seems to overdo the deconvolution and leaves a lot of
    # noise. Aesthetically, I like a slightly larger number that leaves more
    # signal out in the wings even if it's not photomtrically perfect.
    new_shape = (750, 1000)
    scatter_psf_rebinned = _rebin_interpolate(scatter_psf[0].data, new_shape)

    TIS = np.sum(scatter_psf_rebinned)
    scatter_psf_core = (1 - TIS) * correction_factor
    scatter_psf_rebinned[375, 500] = scatter_psf_core
    scatter_psf_rebinned /= np.sum(scatter_psf_rebinned)
    decon_scatt = _deconvolve_scatter_nopad(decon_diff, scatter_psf_rebinned)

    # Check some results to confirm the resulting image is well behaved. If the ratio isn't nearly one,
    # something is wrong with the normalization of the PSF.
    print(
        "Ratio of deconvolved to raw L1 image:", np.sum(decon_scatt) / np.sum(l1_data)
    )
    print(
        "Value of a random group of pixels that should be pretty dark in deconvolved:",
        np.mean(decon_scatt[80:100, 80:100]),
    )
    print("Value of a same pixels in L1 data:", np.mean(l1_data[80:100, 80:100]))

    return decon_scatt


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
        "--savefig",
        action="store_true",
        help="Save figure to disk instead of calling plt.show()",
    )

    return parser


if __name__ == "__main__":
    _main()
