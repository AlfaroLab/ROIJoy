import numpy as np
import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
SAMPLE_HDR = DATA_DIR / "RO_004_5_2025-04-17_01-08-18_.hdr"


def test_extract_spectrum_from_polygon():
    from roijoy.envi_io import load_envi
    from roijoy.roi import extract_spectrum

    cube, wavelengths = load_envi(str(SAMPLE_HDR))
    vertices = [(100, 100), (200, 100), (200, 200), (100, 200)]

    result = extract_spectrum(cube, wavelengths, vertices)
    assert "mean" in result
    assert "std" in result
    assert "wavelengths" in result
    assert "n_pixels" in result
    assert len(result["mean"]) == 110
    assert len(result["std"]) == 110
    assert result["n_pixels"] > 0


def test_extract_subsample():
    from roijoy.envi_io import load_envi
    from roijoy.roi import extract_subsample

    cube, wavelengths = load_envi(str(SAMPLE_HDR))
    vertices = [(100, 100), (200, 100), (200, 200), (100, 200)]

    coords, spectra = extract_subsample(cube, vertices, n_samples=100)
    assert spectra.shape[0] <= 100
    assert spectra.shape[1] == 110
    assert coords.shape[0] == spectra.shape[0]
    assert coords.shape[1] == 2


def test_roi_to_normalized_coords():
    from roijoy.roi import normalize_vertices

    vertices = [(100, 200), (300, 400)]
    image_shape = (570, 570)
    normalized = normalize_vertices(vertices, image_shape)

    assert normalized[0] == pytest.approx((100/570, 200/570))
    assert normalized[1] == pytest.approx((300/570, 400/570))


def test_export_spectrum_csv(tmp_path):
    from roijoy.roi import export_spectrum_csv

    wavelengths = np.linspace(370, 806, 110)
    mean = np.random.rand(110)
    std = np.random.rand(110) * 0.1

    outpath = tmp_path / "spectrum.csv"
    export_spectrum_csv(str(outpath), wavelengths, mean, std)

    import csv
    with open(outpath) as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["Wavelength (nm)", "Mean Reflectance", "Std Dev"]
        rows = list(reader)
        assert len(rows) == 110
