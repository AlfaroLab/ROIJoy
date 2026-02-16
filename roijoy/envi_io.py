"""ENVI file I/O and RGB rendering for hyperspectral image cubes."""
import numpy as np
from spectral import envi
import io
import base64
from PIL import Image


def load_envi(hdr_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load an ENVI file and return the data cube and wavelengths.

    Parses actual wavelengths from the .hdr file instead of using
    a hardcoded linspace range.

    Args:
        hdr_path: Path to the .hdr file. The .bin file is expected
                  to be in the same directory with matching name.

    Returns:
        Tuple of (cube, wavelengths) where cube is shape (rows, cols, bands)
        and wavelengths is a 1D array of wavelength values in nm.
    """
    bin_path = hdr_path.rsplit('.hdr', 1)[0] + '.bin'
    data = envi.open(hdr_path, image=bin_path)
    cube = data.load()

    metadata = data.metadata
    if 'wavelength' in metadata:
        wavelengths = np.array([float(w) for w in metadata['wavelength']])
    else:
        wavelengths = np.linspace(350, 1000, cube.shape[2])

    return np.array(cube), wavelengths


def render_rgb(cube: np.ndarray, wavelengths: np.ndarray,
               mode: str = "resample",
               band_indices: list[int] | None = None) -> np.ndarray:
    """Render a 3-channel RGB image from a hyperspectral cube.

    Args:
        cube: Hyperspectral data cube (rows, cols, bands)
        wavelengths: Array of wavelength values in nm
        mode: "resample" to average wavelength ranges, "bands" for specific indices
        band_indices: Required if mode="bands". List of [R, G, B] band indices.

    Returns:
        RGB image as float32 array in [0, 1] range, shape (rows, cols, 3)
    """
    if mode == "resample":
        red_mask = (wavelengths >= 620) & (wavelengths <= 750)
        green_mask = (wavelengths >= 495) & (wavelengths <= 570)
        blue_mask = (wavelengths >= 450) & (wavelengths <= 495)

        red = np.mean(cube[:, :, red_mask], axis=2) if red_mask.any() else cube[:, :, -1]
        green = np.mean(cube[:, :, green_mask], axis=2) if green_mask.any() else cube[:, :, cube.shape[2] // 2]
        blue = np.mean(cube[:, :, blue_mask], axis=2) if blue_mask.any() else cube[:, :, 0]

        rgb = np.stack([red, green, blue], axis=2).astype(np.float32)
    elif mode == "bands":
        if band_indices is None or len(band_indices) != 3:
            raise ValueError("band_indices must be a list of 3 integers for mode='bands'")
        rgb = cube[:, :, band_indices].astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for i in range(3):
        band = rgb[:, :, i]
        lo = np.percentile(band, 1)
        hi = np.percentile(band, 99)
        if hi > lo:
            rgb[:, :, i] = np.clip((band - lo) / (hi - lo), 0, 1)
        else:
            rgb[:, :, i] = 0.0

    return rgb


def apply_contrast(rgb: np.ndarray, low_pct: float = 1, high_pct: float = 99,
                   gain: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """Apply contrast adjustments to an RGB image."""
    lo, hi = np.percentile(rgb, (low_pct, high_pct))
    if hi > lo:
        adjusted = (rgb - lo) / (hi - lo)
    else:
        adjusted = np.zeros_like(rgb)
    adjusted = np.clip(gain * adjusted + offset, 0, 1)
    return adjusted.astype(np.float32)


def rgb_to_base64(rgb: np.ndarray) -> str:
    """Convert an RGB float32 array to a base64-encoded PNG data URI."""
    img_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64}"
