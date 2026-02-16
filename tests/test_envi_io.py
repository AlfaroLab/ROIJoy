import numpy as np
import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
SAMPLE_HDR = DATA_DIR / "RO_004_5_2025-04-17_01-08-18_.hdr"


def test_load_envi_returns_cube_and_wavelengths():
    from roijoy.envi_io import load_envi
    cube, wavelengths = load_envi(str(SAMPLE_HDR))
    assert cube.shape == (570, 570, 110)
    assert len(wavelengths) == 110
    assert wavelengths[0] == pytest.approx(370.0)
    assert wavelengths[-1] == pytest.approx(806.0)


def test_render_rgb_resampled():
    from roijoy.envi_io import load_envi, render_rgb
    cube, wavelengths = load_envi(str(SAMPLE_HDR))
    rgb = render_rgb(cube, wavelengths, mode="resample")
    assert rgb.shape == (570, 570, 3)
    assert rgb.dtype == np.float32
    assert 0.0 <= rgb.min()
    assert rgb.max() <= 1.0


def test_render_rgb_bands():
    from roijoy.envi_io import load_envi, render_rgb
    cube, wavelengths = load_envi(str(SAMPLE_HDR))
    rgb = render_rgb(cube, wavelengths, mode="bands", band_indices=[54, 32, 22])
    assert rgb.shape == (570, 570, 3)


def test_apply_contrast():
    from roijoy.envi_io import apply_contrast
    rgb = np.random.rand(100, 100, 3).astype(np.float32)
    adjusted = apply_contrast(rgb, low_pct=1, high_pct=99, gain=1.0, offset=0.0)
    assert adjusted.shape == rgb.shape
    assert 0.0 <= adjusted.min()
    assert adjusted.max() <= 1.0


def test_rgb_to_base64_png():
    from roijoy.envi_io import rgb_to_base64
    rgb = np.random.rand(100, 100, 3).astype(np.float32)
    result = rgb_to_base64(rgb)
    assert result.startswith("data:image/png;base64,")
