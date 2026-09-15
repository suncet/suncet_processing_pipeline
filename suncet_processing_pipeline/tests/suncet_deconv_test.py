from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from .. import suncet_deconv


def _legacy_two_stage_inverse(image, diffraction_psf, scatter_psf):
    """Independent copy of the pre-cache array operations for regression."""
    psf_shape = diffraction_psf.shape
    psf_padded = np.zeros((psf_shape[0] * 2, psf_shape[1] * 2))
    psf_padded[
        psf_shape[0] // 2 : psf_shape[0] // 2 + psf_shape[0],
        psf_shape[1] // 2 : psf_shape[1] // 2 + psf_shape[1],
    ] = diffraction_psf
    image_padded = np.zeros((image.shape[0] * 2, image.shape[1] * 2))
    image_padded[
        image.shape[0] // 2 : image.shape[0] // 2 + image.shape[0],
        image.shape[1] // 2 : image.shape[1] // 2 + image.shape[1],
    ] = image
    diffraction_result = np.real(
        np.fft.ifft2(np.fft.fft2(image_padded) / np.fft.fft2(psf_padded))
    )
    diffraction_result = np.roll(
        diffraction_result,
        shift=psf_shape,
        axis=(0, 1),
    )[
        psf_shape[0] // 2 : psf_shape[0] // 2 + psf_shape[0],
        psf_shape[1] // 2 : psf_shape[1] // 2 + psf_shape[1],
    ]

    scatter_result = np.real(
        np.fft.ifft2(
            np.fft.fft2(diffraction_result) / np.fft.fft2(scatter_psf)
        )
    )
    return np.roll(
        scatter_result,
        shift=(scatter_psf.shape[0] // 2, scatter_psf.shape[1] // 2),
        axis=(0, 1),
    )


def _write_small_calibrations(tmp_path):
    rng = np.random.default_rng(20260915)
    diffraction_planes = []
    for scale in (1.0, 1.5):
        plane = rng.uniform(1e-6, 2e-6, size=(7, 9))
        plane[3, 4] += scale
        diffraction_planes.append(plane)
    diffraction_path = tmp_path / "diffraction.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(diffraction_planes[0]),
            fits.ImageHDU(diffraction_planes[1]),
        ]
    ).writeto(diffraction_path)

    scatter = rng.uniform(1e-8, 2e-8, size=(7, 9))
    scatter_path = tmp_path / "scatter.fits"
    fits.PrimaryHDU(scatter).writeto(scatter_path)

    spectrum_path = tmp_path / "spectrum.genx"
    response_path = tmp_path / "response.genx"
    spectrum_path.touch()
    response_path.touch()
    return diffraction_path, scatter_path, response_path, spectrum_path


def _patch_small_calibration_geometry(monkeypatch):
    monkeypatch.setattr(suncet_deconv, "_DIFFRACTION_REBIN_SHAPE", (102, 120))
    monkeypatch.setattr(
        suncet_deconv,
        "_DIFFRACTION_CROP_ROWS",
        slice(1, -1),
    )
    monkeypatch.setattr(suncet_deconv, "_SCATTER_REBIN_SHAPE", (100, 120))
    monkeypatch.setattr(suncet_deconv, "_SCATTER_CORE_INDEX", (50, 60))


def _patch_genx(monkeypatch, spectrum_bins=2):
    wavelengths = np.arange(1, spectrum_bins + 1, dtype=np.float64) * 10

    def fake_read_genx(path):
        if Path(path).name == "spectrum.genx":
            return {
                "LAMBDA": wavelengths,
                "SPECTRUM": np.arange(
                    1,
                    spectrum_bins + 1,
                    dtype=np.float64,
                ),
            }
        return {
            "SAVEGEN0": wavelengths,
            "SAVEGEN1": np.ones(spectrum_bins, dtype=np.float64),
        }

    monkeypatch.setattr(
        suncet_deconv.sunpy.io.special.genx,
        "read_genx",
        fake_read_genx,
    )


def test_prepared_deconvolver_matches_existing_two_stage_inverse():
    shape = (100, 120)
    rng = np.random.default_rng(42)
    image = rng.uniform(1, 100, size=shape)

    diffraction_psf = rng.uniform(1e-8, 2e-8, size=shape)
    diffraction_psf[50, 60] += 1
    scatter_psf = rng.uniform(1e-8, 2e-8, size=shape)
    scatter_psf[50, 60] += 1
    scatter_psf /= np.sum(scatter_psf)

    expected = _legacy_two_stage_inverse(
        image,
        diffraction_psf,
        scatter_psf,
    )
    prepared = suncet_deconv.PreparedDeconvolver(
        suncet_deconv._padded_psf_fft(diffraction_psf),
        np.fft.fft2(scatter_psf),
        shape,
    )

    actual = prepared.apply(image)

    np.testing.assert_array_equal(actual, expected)


def test_deconvolution_plan_prepares_calibration_once(
    tmp_path,
    monkeypatch,
):
    calibration_paths = _write_small_calibrations(tmp_path)
    _patch_small_calibration_geometry(monkeypatch)
    _patch_genx(monkeypatch)
    real_prepare = suncet_deconv.prepare_deconv
    preparation_calls = []

    def counted_prepare(*args, **kwargs):
        preparation_calls.append((args, kwargs))
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(suncet_deconv, "prepare_deconv", counted_prepare)
    plan = suncet_deconv.DeconvolutionPlan(
        *calibration_paths,
        correction_factor=0.4,
    )
    image = np.arange(100 * 120, dtype=np.float64).reshape(100, 120) + 1

    first = suncet_deconv.apply_deconv(
        image,
        *calibration_paths,
        correction_factor=0.4,
        deconvolver=plan,
    )
    second = suncet_deconv.apply_deconv(
        image,
        *calibration_paths,
        correction_factor=0.4,
        deconvolver=plan,
    )

    assert len(preparation_calls) == 1
    np.testing.assert_array_equal(second, first)


def test_deconvolution_plan_rejects_a_different_calibration_set(tmp_path):
    calibration_paths = _write_small_calibrations(tmp_path)
    plan = suncet_deconv.DeconvolutionPlan(*calibration_paths)
    other_spectrum = tmp_path / "other_spectrum.genx"
    other_spectrum.touch()

    with pytest.raises(ValueError, match="different calibration set"):
        suncet_deconv.apply_deconv(
            np.ones((100, 120)),
            *calibration_paths[:-1],
            other_spectrum,
            deconvolver=plan,
        )


def test_prepare_deconv_rejects_psf_spectrum_count_mismatch(
    tmp_path,
    monkeypatch,
):
    calibration_paths = _write_small_calibrations(tmp_path)
    _patch_genx(monkeypatch, spectrum_bins=3)

    with pytest.raises(ValueError, match="plane count must match"):
        suncet_deconv.prepare_deconv(*calibration_paths)


def test_prepared_deconvolver_validates_transfer_and_image_shapes():
    with pytest.raises(ValueError, match="Padded diffraction transfer"):
        suncet_deconv.PreparedDeconvolver(
            np.ones((4, 4), dtype=np.complex128),
            np.ones((2, 3), dtype=np.complex128),
            (2, 3),
        )

    prepared = suncet_deconv.PreparedDeconvolver(
        np.ones((4, 6), dtype=np.complex128),
        np.ones((2, 3), dtype=np.complex128),
        (2, 3),
    )
    with pytest.raises(ValueError, match="requires image shape"):
        prepared.apply(np.ones((3, 2)))


def test_prepared_deconvolver_rejects_zero_transfer_bins():
    diffraction_fpsf = np.ones((4, 6), dtype=np.complex128)
    diffraction_fpsf[0, 0] = 0

    with pytest.raises(ValueError, match="contains zeros"):
        suncet_deconv.PreparedDeconvolver(
            diffraction_fpsf,
            np.ones((2, 3), dtype=np.complex128),
            (2, 3),
        )


class _FakeStream:
    def __init__(self):
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1


class _FakeCuda:
    def __init__(self):
        self.stream = _FakeStream()

    def get_current_stream(self):
        return self.stream


class _NumpyBackedFakeCupy:
    """Small CuPy surface backed by NumPy for CPU-only CI coverage."""

    float64 = np.float64
    complex128 = np.complex128
    fft = np.fft

    def __init__(self):
        self.cuda = _FakeCuda()
        self.host_copies = []

    @staticmethod
    def all(array):
        return np.all(array)

    @staticmethod
    def any(array):
        return np.any(array)

    @staticmethod
    def asarray(array, dtype=None):
        return np.asarray(array, dtype=dtype)

    @staticmethod
    def isfinite(array):
        return np.isfinite(array)

    @staticmethod
    def real(array):
        return np.real(array)

    @staticmethod
    def roll(array, shift, axis):
        return np.roll(array, shift=shift, axis=axis)

    @staticmethod
    def zeros(shape, dtype=None):
        return np.zeros(shape, dtype=dtype)

    def asnumpy(self, array):
        self.host_copies.append(array.shape)
        return np.asarray(array)


def test_numpy_backend_does_not_import_cupy(monkeypatch):
    def reject_import(name):
        if name == "cupy":
            raise AssertionError("CPU backend attempted to import CuPy")
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(suncet_deconv.importlib, "import_module", reject_import)

    prepared = suncet_deconv.PreparedDeconvolver(
        np.ones((4, 6), dtype=np.complex128),
        np.ones((2, 3), dtype=np.complex128),
        (2, 3),
    )

    assert prepared.backend == "numpy"


def test_cupy_backend_reports_missing_optional_dependency(monkeypatch):
    def missing_cupy(name):
        assert name == "cupy"
        raise ModuleNotFoundError("No module named 'cupy'")

    monkeypatch.setattr(suncet_deconv.importlib, "import_module", missing_cupy)

    with pytest.raises(RuntimeError, match="CuPy.*requested.*could not be imported"):
        suncet_deconv._array_module_for_backend("cupy")


def test_prepare_cupy_fails_before_opening_large_calibration_files(monkeypatch):
    def missing_cupy(name):
        assert name == "cupy"
        raise ModuleNotFoundError("No module named 'cupy'")

    monkeypatch.setattr(suncet_deconv.importlib, "import_module", missing_cupy)
    monkeypatch.setattr(
        suncet_deconv.fits,
        "open",
        lambda *_args, **_kwargs: pytest.fail(
            "calibration data must not be opened before backend validation"
        ),
    )

    with pytest.raises(RuntimeError, match="CuPy.*requested.*could not be imported"):
        suncet_deconv.prepare_deconv(
            "diffraction.fits",
            "scatter.fits",
            "response.genx",
            "spectrum.genx",
            backend="cupy",
        )


@pytest.mark.parametrize("backend", [None, "cuda", "gpu", ""])
def test_backend_selector_rejects_noncanonical_names(backend):
    with pytest.raises(ValueError, match="backend must be one of"):
        suncet_deconv.DeconvolutionPlan(
            "diffraction.fits",
            "scatter.fits",
            "response.genx",
            "spectrum.genx",
            backend=backend,
        )


def test_cupy_backend_preserves_inverse_and_returns_one_host_array(monkeypatch):
    fake_cupy = _NumpyBackedFakeCupy()
    monkeypatch.setattr(
        suncet_deconv,
        "_array_module_for_backend",
        lambda backend: fake_cupy if backend == "cupy" else np,
    )
    shape = (100, 120)
    rng = np.random.default_rng(87)
    image = rng.uniform(1, 100, size=shape).astype(np.float32)
    diffraction_psf = rng.uniform(1e-8, 2e-8, size=shape)
    diffraction_psf[50, 60] += 1
    scatter_psf = rng.uniform(1e-8, 2e-8, size=shape)
    scatter_psf[50, 60] += 1
    scatter_psf /= np.sum(scatter_psf)

    expected = _legacy_two_stage_inverse(
        image.astype(np.float64),
        diffraction_psf,
        scatter_psf,
    )
    prepared = suncet_deconv.PreparedDeconvolver(
        suncet_deconv._padded_psf_fft(diffraction_psf),
        np.fft.fft2(scatter_psf),
        shape,
        backend="cupy",
    )

    actual = prepared.apply(image)
    prepared.synchronize()

    assert prepared.backend == "cupy"
    assert isinstance(actual, np.ndarray)
    assert actual.dtype == np.float64
    assert fake_cupy.host_copies == [shape]
    assert fake_cupy.cuda.stream.synchronize_calls == 1
    np.testing.assert_array_equal(actual, expected)


def test_prepare_deconv_builds_and_applies_cupy_transfer_functions(
    tmp_path,
    monkeypatch,
):
    calibration_paths = _write_small_calibrations(tmp_path)
    _patch_small_calibration_geometry(monkeypatch)
    _patch_genx(monkeypatch)
    expected_deconvolver = suncet_deconv.prepare_deconv(*calibration_paths)
    fake_cupy = _NumpyBackedFakeCupy()
    real_array_module = suncet_deconv._array_module_for_backend

    def fake_array_module(backend):
        if backend == "cupy":
            return fake_cupy
        return real_array_module(backend)

    monkeypatch.setattr(
        suncet_deconv,
        "_array_module_for_backend",
        fake_array_module,
    )
    cupy_deconvolver = suncet_deconv.prepare_deconv(
        *calibration_paths,
        backend="cupy",
    )
    image = np.arange(100 * 120, dtype=np.float64).reshape(100, 120) + 1

    expected = expected_deconvolver.apply(image)
    actual = cupy_deconvolver.apply(image)

    assert cupy_deconvolver.backend == "cupy"
    assert cupy_deconvolver._diffraction_fpsf.dtype == np.complex128
    assert cupy_deconvolver._scatter_fpsf.dtype == np.complex128
    assert fake_cupy.host_copies == [(100, 120)]
    np.testing.assert_array_equal(actual, expected)


def test_cupy_plan_is_lazy_and_forwards_backend(tmp_path, monkeypatch):
    calibration_paths = _write_small_calibrations(tmp_path)
    preparation_calls = []

    class FakePrepared:
        backend = "cupy"

        @staticmethod
        def apply(image):
            return np.asarray(image, dtype=np.float64)

        @staticmethod
        def synchronize():
            return None

    def fake_prepare(*args, **kwargs):
        preparation_calls.append((args, kwargs))
        return FakePrepared()

    monkeypatch.setattr(suncet_deconv, "prepare_deconv", fake_prepare)
    plan = suncet_deconv.DeconvolutionPlan(
        *calibration_paths,
        backend="CuPy",
    )
    assert preparation_calls == []

    image = np.ones((2, 3))
    actual = plan.apply(image)

    assert plan.backend == "cupy"
    assert preparation_calls == [
        (
            calibration_paths,
            {"correction_factor": 0.4, "backend": "cupy"},
        )
    ]
    np.testing.assert_array_equal(actual, image)


def test_apply_deconv_rejects_backend_mismatch(tmp_path):
    calibration_paths = _write_small_calibrations(tmp_path)
    plan = suncet_deconv.DeconvolutionPlan(
        *calibration_paths,
        backend="cupy",
    )

    with pytest.raises(ValueError, match="does not match the supplied deconvolver"):
        suncet_deconv.apply_deconv(
            np.ones((2, 3)),
            *calibration_paths,
            deconvolver=plan,
            backend="numpy",
        )
