# ROIJoy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a browser-based hyperspectral ROI selector with multi-image comparison and feature-based polygon matching.

**Architecture:** Plotly Dash app with Python backend. ENVI cubes loaded server-side via `spectral` library. RGB previews sent to browser as base64 images. Polygon drawing via Plotly's `drawclosedpath` mode. ORB feature matching via OpenCV for cross-image ROI propagation.

**Tech Stack:** Python 3.14, Dash 2.x, Plotly 5.x, NumPy, spectral, scikit-image, OpenCV, pandas

---

### Task 1: Environment Setup

**Files:**
- Modify: `requirements.txt`
- Create: `roijoy/app.py` (minimal placeholder)

**Step 1: Create virtual environment and install dependencies**

```bash
cd /Users/alfarolab/Dropbox/git/ROIJoy
python3 -m venv .venv
source .venv/bin/activate
pip install dash plotly numpy spectral scikit-image opencv-python pandas
```

If `spectral` fails on Python 3.14, try: `pip install spectral --no-build-isolation`
If `opencv-python` fails, try: `pip install opencv-python-headless`

**Step 2: Freeze actual installed versions**

```bash
pip freeze > requirements.txt
```

**Step 3: Create minimal app.py to verify Dash works**

```python
# roijoy/app.py
import dash
from dash import html

app = dash.Dash(__name__)
app.layout = html.Div("ROIJoy is running!")

if __name__ == "__main__":
    app.run(debug=True, port=8050)
```

**Step 4: Run to verify**

```bash
python roijoy/app.py
```

Expected: Browser opens at http://localhost:8050 showing "ROIJoy is running!"

**Step 5: Commit**

```bash
git add .gitignore requirements.txt roijoy/app.py
git commit -m "feat: environment setup with Dash app skeleton"
```

Note: Ensure `.venv/` is in `.gitignore` (already is).

---

### Task 2: ENVI I/O Module

**Files:**
- Create: `roijoy/envi_io.py`
- Create: `tests/test_envi_io.py`

This module replaces the original script's hardcoded `np.linspace(350, 1000, cube.shape[2])` with actual wavelength parsing from .hdr files.

**Step 1: Write the failing test**

```python
# tests/test_envi_io.py
import numpy as np
import pytest
from pathlib import Path

# Test data path - symlinked ENVI files
DATA_DIR = Path(__file__).parent.parent / "data"
SAMPLE_HDR = DATA_DIR / "RO_004_5_2025-04-17_01-08-18_.hdr"
SAMPLE_BIN = DATA_DIR / "RO_004_5_2025-04-17_01-08-18_.bin"


def test_load_envi_returns_cube_and_wavelengths():
    """Loading an ENVI file should return the data cube and parsed wavelengths."""
    from roijoy.envi_io import load_envi
    cube, wavelengths = load_envi(str(SAMPLE_HDR))
    assert cube.shape == (570, 570, 110)
    assert len(wavelengths) == 110
    assert wavelengths[0] == pytest.approx(370.0)
    assert wavelengths[-1] == pytest.approx(806.0)


def test_render_rgb_resampled():
    """Rendering RGB via band resampling should produce a 3-channel float image."""
    from roijoy.envi_io import load_envi, render_rgb
    cube, wavelengths = load_envi(str(SAMPLE_HDR))
    rgb = render_rgb(cube, wavelengths, mode="resample")
    assert rgb.shape == (570, 570, 3)
    assert rgb.dtype == np.float32
    assert 0.0 <= rgb.min()
    assert rgb.max() <= 1.0


def test_render_rgb_bands():
    """Rendering RGB via specific band indices should work."""
    from roijoy.envi_io import load_envi, render_rgb
    cube, wavelengths = load_envi(str(SAMPLE_HDR))
    rgb = render_rgb(cube, wavelengths, mode="bands", band_indices=[54, 32, 22])
    assert rgb.shape == (570, 570, 3)


def test_apply_contrast():
    """Contrast adjustment should clip and scale correctly."""
    from roijoy.envi_io import apply_contrast
    rgb = np.random.rand(100, 100, 3).astype(np.float32)
    adjusted = apply_contrast(rgb, low_pct=1, high_pct=99, gain=1.0, offset=0.0)
    assert adjusted.shape == rgb.shape
    assert 0.0 <= adjusted.min()
    assert adjusted.max() <= 1.0


def test_rgb_to_base64_png():
    """Converting RGB array to base64 PNG should return a data URI string."""
    from roijoy.envi_io import rgb_to_base64
    rgb = np.random.rand(100, 100, 3).astype(np.float32)
    result = rgb_to_base64(rgb)
    assert result.startswith("data:image/png;base64,")
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/alfarolab/Dropbox/git/ROIJoy
source .venv/bin/activate
python -m pytest tests/test_envi_io.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'roijoy.envi_io'`

**Step 3: Write the implementation**

```python
# roijoy/envi_io.py
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
    # Derive .bin path from .hdr path
    bin_path = hdr_path.rsplit('.hdr', 1)[0] + '.bin'

    data = envi.open(hdr_path, image=bin_path)
    cube = data.load()

    # Parse wavelengths from header metadata
    metadata = data.metadata
    if 'wavelength' in metadata:
        wavelengths = np.array([float(w) for w in metadata['wavelength']])
    else:
        # Fallback to linspace if no wavelength info in header
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
        green = np.mean(cube[:, :, green_mask], axis=2) if green_mask.any() else cube[:, :, cube.shape[2]//2]
        blue = np.mean(cube[:, :, blue_mask], axis=2) if blue_mask.any() else cube[:, :, 0]

        rgb = np.stack([red, green, blue], axis=2).astype(np.float32)
    elif mode == "bands":
        if band_indices is None or len(band_indices) != 3:
            raise ValueError("band_indices must be a list of 3 integers for mode='bands'")
        rgb = cube[:, :, band_indices].astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Normalize each channel independently using percentile clipping
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
    """Apply contrast adjustments to an RGB image.

    Args:
        rgb: Input RGB image (float32, [0, 1] range)
        low_pct: Low percentile for clipping (0-50)
        high_pct: High percentile for clipping (50-100)
        gain: Multiplicative gain (0.5-2.0)
        offset: Additive offset (-0.5 to 0.5)

    Returns:
        Adjusted RGB image clipped to [0, 1]
    """
    lo, hi = np.percentile(rgb, (low_pct, high_pct))
    if hi > lo:
        adjusted = (rgb - lo) / (hi - lo)
    else:
        adjusted = np.zeros_like(rgb)
    adjusted = np.clip(gain * adjusted + offset, 0, 1)
    return adjusted.astype(np.float32)


def rgb_to_base64(rgb: np.ndarray) -> str:
    """Convert an RGB float32 array to a base64-encoded PNG data URI.

    Args:
        rgb: RGB image as float32 in [0, 1] range

    Returns:
        Data URI string like "data:image/png;base64,..."
    """
    img_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64}"
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_envi_io.py -v
```

Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add roijoy/envi_io.py tests/test_envi_io.py
git commit -m "feat: ENVI I/O module with actual wavelength parsing"
```

---

### Task 3: ROI Data Model & Spectrum Extraction

**Files:**
- Create: `roijoy/roi.py`
- Create: `tests/test_roi.py`

**Step 1: Write the failing test**

```python
# tests/test_roi.py
import numpy as np
import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
SAMPLE_HDR = DATA_DIR / "RO_004_5_2025-04-17_01-08-18_.hdr"


def test_extract_spectrum_from_polygon():
    """Extracting spectrum from a polygon ROI should return mean and std."""
    from roijoy.envi_io import load_envi
    from roijoy.roi import extract_spectrum

    cube, wavelengths = load_envi(str(SAMPLE_HDR))
    # Simple rectangle as polygon vertices (x, y pixel coords)
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
    """Extracting a subsample should return up to 100 pixel spectra with coords."""
    from roijoy.envi_io import load_envi
    from roijoy.roi import extract_subsample

    cube, wavelengths = load_envi(str(SAMPLE_HDR))
    vertices = [(100, 100), (200, 100), (200, 200), (100, 200)]

    coords, spectra = extract_subsample(cube, vertices, n_samples=100)
    assert spectra.shape[0] <= 100
    assert spectra.shape[1] == 110
    assert coords.shape[0] == spectra.shape[0]
    assert coords.shape[1] == 2  # x, y


def test_roi_to_normalized_coords():
    """ROI vertices should normalize to [0, 1] range based on image dimensions."""
    from roijoy.roi import normalize_vertices

    vertices = [(100, 200), (300, 400)]
    image_shape = (570, 570)  # rows, cols
    normalized = normalize_vertices(vertices, image_shape)

    assert normalized[0] == pytest.approx((100/570, 200/570))
    assert normalized[1] == pytest.approx((300/570, 400/570))


def test_export_spectrum_csv(tmp_path):
    """Exporting spectrum data should create a valid CSV file."""
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
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_roi.py -v
```

Expected: FAIL with import error

**Step 3: Write the implementation**

```python
# roijoy/roi.py
"""ROI data model, spectrum extraction, and CSV export."""
import numpy as np
from skimage.draw import polygon
import csv


def extract_spectrum(cube: np.ndarray, wavelengths: np.ndarray,
                     vertices: list[tuple[float, float]]) -> dict:
    """Extract mean and std spectrum from a polygon ROI.

    Args:
        cube: Hyperspectral data cube (rows, cols, bands)
        wavelengths: Wavelength array
        vertices: List of (x, y) pixel coordinates forming the polygon

    Returns:
        Dict with keys: mean, std, wavelengths, n_pixels
    """
    r = np.array([v[1] for v in vertices])
    c = np.array([v[0] for v in vertices])
    rr, cc = polygon(r, c, cube.shape[:2])

    spectra = cube[rr, cc, :]
    return {
        "mean": spectra.mean(axis=0),
        "std": spectra.std(axis=0),
        "wavelengths": wavelengths,
        "n_pixels": len(rr),
    }


def extract_subsample(cube: np.ndarray, vertices: list[tuple[float, float]],
                      n_samples: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Extract a random subsample of pixel spectra from a polygon ROI.

    Args:
        cube: Hyperspectral data cube
        vertices: Polygon vertices as (x, y) pixel coords
        n_samples: Max number of pixels to sample

    Returns:
        Tuple of (coords, spectra) where coords is (N, 2) and spectra is (N, bands)
    """
    r = np.array([v[1] for v in vertices])
    c = np.array([v[0] for v in vertices])
    rr, cc = polygon(r, c, cube.shape[:2])

    spectra = cube[rr, cc, :]
    coords = np.column_stack((cc, rr))  # x, y format

    if len(spectra) > n_samples:
        idx = np.random.choice(len(spectra), n_samples, replace=False)
        return coords[idx], spectra[idx]
    return coords, spectra


def normalize_vertices(vertices: list[tuple[float, float]],
                       image_shape: tuple[int, int]) -> list[tuple[float, float]]:
    """Normalize pixel vertices to [0, 1] range.

    Args:
        vertices: List of (x, y) pixel coordinates
        image_shape: (rows, cols) of the image

    Returns:
        List of (x_norm, y_norm) in [0, 1] range
    """
    rows, cols = image_shape
    return [(x / cols, y / rows) for x, y in vertices]


def export_spectrum_csv(path: str, wavelengths: np.ndarray,
                        mean: np.ndarray, std: np.ndarray) -> None:
    """Export spectrum summary to CSV.

    Args:
        path: Output file path
        wavelengths: Wavelength array
        mean: Mean reflectance per band
        std: Standard deviation per band
    """
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Wavelength (nm)", "Mean Reflectance", "Std Dev"])
        for w, m, s in zip(wavelengths, mean, std):
            writer.writerow([f"{w:.1f}", f"{m:.6e}", f"{s:.6e}"])
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_roi.py -v
```

Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add roijoy/roi.py tests/test_roi.py
git commit -m "feat: ROI data model with spectrum extraction and CSV export"
```

---

### Task 4: Feature-Based ROI Matching

**Files:**
- Create: `roijoy/matching.py`
- Create: `tests/test_matching.py`

**Step 1: Write the failing test**

```python
# tests/test_matching.py
import numpy as np
import pytest


def test_match_identical_images_returns_same_vertices():
    """Matching an ROI on identical images should return nearly the same polygon."""
    from roijoy.matching import match_roi

    # Create a simple test image with distinct features
    img = np.random.RandomState(42).randint(0, 255, (200, 200, 3), dtype=np.uint8)

    vertices = [(50, 50), (100, 50), (100, 100), (50, 100)]
    result = match_roi(img, img, vertices)

    assert result["success"] is True
    assert len(result["vertices"]) == 4
    # On identical images, vertices should be very close to original
    for orig, matched in zip(vertices, result["vertices"]):
        assert abs(orig[0] - matched[0]) < 5
        assert abs(orig[1] - matched[1]) < 5


def test_match_featureless_images_falls_back():
    """Matching on blank images should fall back to copy-coordinates."""
    from roijoy.matching import match_roi

    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    vertices = [(50, 50), (100, 50), (100, 100), (50, 100)]

    result = match_roi(blank, blank, vertices)

    assert result["success"] is True
    assert result["method"] == "copy"  # Fell back to coordinate copy


def test_copy_roi_returns_exact_vertices():
    """Copy mode should return exactly the same vertices."""
    from roijoy.matching import copy_roi

    vertices = [(50.5, 60.7), (100.2, 50.3)]
    result = copy_roi(vertices)
    assert result["vertices"] == vertices
    assert result["method"] == "copy"
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_matching.py -v
```

**Step 3: Write the implementation**

```python
# roijoy/matching.py
"""Feature-based ROI matching across images using ORB."""
import numpy as np
import cv2


def match_roi(source_rgb: np.ndarray, target_rgb: np.ndarray,
              vertices: list[tuple[float, float]],
              min_inliers: int = 10) -> dict:
    """Match an ROI from source image to target image using ORB features.

    Args:
        source_rgb: Source image (uint8, H x W x 3)
        target_rgb: Target image (uint8, H x W x 3)
        vertices: Polygon vertices in source image (x, y) pixel coords
        min_inliers: Minimum RANSAC inliers to consider match valid

    Returns:
        Dict with: success (bool), vertices (transformed), method (str),
                   n_inliers (int), confidence (float)
    """
    # Convert to grayscale for feature detection
    src_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    tgt_gray = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY)

    # Detect ORB features
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(src_gray, None)
    kp2, des2 = orb.detectAndCompute(tgt_gray, None)

    # If either image has no features, fall back to copy
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return copy_roi(vertices)

    # Match features using BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Ratio test to filter good matches
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    if len(good_matches) < min_inliers:
        return copy_roi(vertices)

    # Compute homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        return copy_roi(vertices)

    n_inliers = int(mask.sum()) if mask is not None else 0
    if n_inliers < min_inliers:
        return copy_roi(vertices)

    # Transform polygon vertices
    pts = np.float32(vertices).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    return {
        "success": True,
        "vertices": [(float(p[0]), float(p[1])) for p in transformed],
        "method": "feature",
        "n_inliers": n_inliers,
        "confidence": n_inliers / len(good_matches),
    }


def copy_roi(vertices: list[tuple[float, float]]) -> dict:
    """Fallback: copy polygon vertices as-is to target image.

    Args:
        vertices: Polygon vertices (x, y) pixel coords

    Returns:
        Dict with success, vertices, method
    """
    return {
        "success": True,
        "vertices": list(vertices),
        "method": "copy",
        "n_inliers": 0,
        "confidence": 0.0,
    }
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_matching.py -v
```

Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add roijoy/matching.py tests/test_matching.py
git commit -m "feat: ORB feature-based ROI matching with copy fallback"
```

---

### Task 5: Dash Layout — Multi-Panel Image Grid

**Files:**
- Create: `roijoy/layout.py`
- Modify: `roijoy/app.py`
- Create: `roijoy/assets/style.css`

This is the core UI task. No unit tests for layout — we verify visually.

**Step 1: Create the CSS**

```css
/* roijoy/assets/style.css */
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0;
    background: #0f1117;
    color: #e0e0e0;
}

.sidebar {
    position: fixed;
    left: 0;
    top: 0;
    width: 280px;
    height: 100vh;
    background: #1a1d26;
    padding: 20px;
    overflow-y: auto;
    border-right: 1px solid #2a2d36;
    box-sizing: border-box;
    z-index: 100;
}

.sidebar h1 {
    font-size: 1.4em;
    margin: 0 0 20px 0;
    color: #4ecdc4;
}

.sidebar h3 {
    font-size: 0.85em;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #888;
    margin: 20px 0 8px 0;
}

.main-content {
    margin-left: 280px;
    padding: 16px;
}

.image-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
}

.image-panel {
    background: #1a1d26;
    border-radius: 8px;
    border: 1px solid #2a2d36;
    overflow: hidden;
}

.image-panel .panel-header {
    padding: 8px 12px;
    background: #22252e;
    font-size: 0.85em;
    color: #aaa;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.spectrum-panel {
    background: #1a1d26;
    border-radius: 8px;
    border: 1px solid #2a2d36;
    padding: 12px;
    margin-bottom: 16px;
}

.roi-table {
    background: #1a1d26;
    border-radius: 8px;
    border: 1px solid #2a2d36;
    padding: 12px;
}

/* Dash component overrides */
.Select-control {
    background: #22252e !important;
    border-color: #2a2d36 !important;
}

.rc-slider-track {
    background: #4ecdc4 !important;
}

button.dash-button {
    background: #4ecdc4;
    color: #0f1117;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    cursor: pointer;
    font-weight: 600;
}

button.dash-button:hover {
    background: #45b7b0;
}
```

**Step 2: Create the layout module**

```python
# roijoy/layout.py
"""Dash layout for ROIJoy multi-panel interface."""
import dash
from dash import html, dcc
import plotly.graph_objects as go


def make_empty_figure():
    """Create an empty Plotly figure with dark theme for image display."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1d26",
        plot_bgcolor="#1a1d26",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1], scaleanchor="x"),
        dragmode="drawclosedpath",
        newshape=dict(
            line=dict(color="#4ecdc4", width=2),
            fillcolor="rgba(78, 205, 196, 0.1)",
        ),
    )
    return fig


def make_spectrum_figure():
    """Create an empty spectrum comparison figure."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1d26",
        plot_bgcolor="#1a1d26",
        margin=dict(l=60, r=20, t=40, b=40),
        xaxis_title="Wavelength (nm)",
        yaxis_title="Reflectance",
        title="Spectrum Comparison",
        height=300,
    )
    return fig


MAX_PANELS = 6


def create_layout():
    """Build the full Dash layout."""
    # Sidebar
    sidebar = html.Div(className="sidebar", children=[
        html.H1("ROIJoy"),

        html.H3("Files"),
        dcc.Upload(
            id="file-upload",
            children=html.Div(["Drag & drop .hdr files or ", html.A("browse")]),
            style={
                "borderWidth": "1px", "borderStyle": "dashed",
                "borderColor": "#4ecdc4", "borderRadius": "4px",
                "padding": "12px", "textAlign": "center",
                "cursor": "pointer", "color": "#888",
            },
            multiple=True,
        ),
        html.Div(id="loaded-files-list"),

        html.H3("RGB Visualization"),
        dcc.RadioItems(
            id="rgb-mode",
            options=[
                {"label": "Band Resampling", "value": "resample"},
                {"label": "Bands 54, 32, 22", "value": "bands_default"},
                {"label": "Bands 20, 40, 60", "value": "bands_alt"},
            ],
            value="resample",
            style={"fontSize": "0.85em"},
        ),

        html.H3("Contrast"),
        html.Label("Low %", style={"fontSize": "0.8em"}),
        dcc.Slider(id="low-pct", min=0, max=10, value=1, step=0.5,
                   marks=None, tooltip={"placement": "right"}),
        html.Label("High %", style={"fontSize": "0.8em"}),
        dcc.Slider(id="high-pct", min=90, max=100, value=99, step=0.5,
                   marks=None, tooltip={"placement": "right"}),
        html.Label("Gain", style={"fontSize": "0.8em"}),
        dcc.Slider(id="gain", min=0.5, max=2.0, value=1.0, step=0.05,
                   marks=None, tooltip={"placement": "right"}),
        html.Label("Offset", style={"fontSize": "0.8em"}),
        dcc.Slider(id="offset", min=-0.5, max=0.5, value=0.0, step=0.05,
                   marks=None, tooltip={"placement": "right"}),

        html.H3("ROI Matching"),
        dcc.RadioItems(
            id="matching-mode",
            options=[
                {"label": "Feature-based (ORB)", "value": "feature"},
                {"label": "Copy coordinates", "value": "copy"},
                {"label": "Off", "value": "off"},
            ],
            value="feature",
            style={"fontSize": "0.85em"},
        ),

        html.H3("Tools"),
        html.Button("Draw ROI", id="btn-draw", className="dash-button",
                     style={"width": "100%", "marginBottom": "8px"}),
        html.Button("Delete Selected", id="btn-delete", className="dash-button",
                     style={"width": "100%", "marginBottom": "8px",
                            "background": "#e74c3c"}),
        html.Button("Export All", id="btn-export", className="dash-button",
                     style={"width": "100%"}),

        html.Div(id="status-msg", style={
            "marginTop": "16px", "fontSize": "0.8em", "color": "#888"
        }),
    ])

    # Image panels (up to 6)
    image_panels = []
    for i in range(MAX_PANELS):
        panel = html.Div(
            className="image-panel",
            id=f"panel-container-{i}",
            style={"display": "none"},  # Hidden until file loaded
            children=[
                html.Div(className="panel-header", children=[
                    html.Span(id=f"panel-title-{i}", children=f"Image {i+1}"),
                    html.Button("x", id=f"panel-close-{i}",
                                style={"background": "none", "border": "none",
                                       "color": "#888", "cursor": "pointer"}),
                ]),
                dcc.Graph(
                    id=f"image-graph-{i}",
                    figure=make_empty_figure(),
                    config={
                        "modeBarButtonsToAdd": [
                            "drawclosedpath", "eraseshape",
                        ],
                        "modeBarButtonsToRemove": ["autoScale2d"],
                        "scrollZoom": True,
                    },
                    style={"height": "400px"},
                ),
            ],
        )
        image_panels.append(panel)

    # Main content area
    main = html.Div(className="main-content", children=[
        html.Div(className="image-grid", children=image_panels),

        html.Div(className="spectrum-panel", children=[
            dcc.Graph(
                id="spectrum-graph",
                figure=make_spectrum_figure(),
                config={"scrollZoom": True},
            ),
        ]),

        html.Div(className="roi-table", children=[
            html.H3("ROIs", style={"margin": "0 0 8px 0", "fontSize": "0.85em",
                                    "textTransform": "uppercase", "letterSpacing": "1px",
                                    "color": "#888"}),
            html.Div(id="roi-table-content", children="No ROIs yet."),
        ]),
    ])

    # State stores
    stores = [
        dcc.Store(id="image-data-store", data={}),    # {idx: {path, cube_id, wavelengths, rgb_b64}}
        dcc.Store(id="roi-store", data=[]),             # [{id, vertices, image_idx, color, confirmed}]
        dcc.Store(id="active-panel", data=None),        # Which panel is being drawn on
    ]

    return html.Div(children=[sidebar, main] + stores)
```

**Step 3: Update app.py to use the layout**

```python
# roijoy/app.py
"""ROIJoy Dash application entry point."""
import dash
from roijoy.layout import create_layout

app = dash.Dash(
    __name__,
    title="ROIJoy",
    suppress_callback_exceptions=True,
)

app.layout = create_layout()

# Import callbacks to register them (callbacks.py will be created in Task 6)
# from roijoy import callbacks  # uncomment after Task 6

if __name__ == "__main__":
    app.run(debug=True, port=8050)
```

**Step 4: Run to verify layout renders**

```bash
cd /Users/alfarolab/Dropbox/git/ROIJoy
source .venv/bin/activate
python roijoy/app.py
```

Expected: Browser at http://localhost:8050 shows dark-themed layout with sidebar and empty image grid.

**Step 5: Commit**

```bash
git add roijoy/layout.py roijoy/app.py roijoy/assets/style.css
git commit -m "feat: multi-panel Dash layout with sidebar and image grid"
```

---

### Task 6: Core Callbacks — File Loading & Image Display

**Files:**
- Create: `roijoy/callbacks.py`
- Modify: `roijoy/app.py` (uncomment callbacks import)

**Step 1: Write the callbacks module**

This handles: file path input (since ENVI files are too large for browser upload, we use path input), image rendering, contrast adjustment, and zoom sync.

```python
# roijoy/callbacks.py
"""Dash callbacks for ROIJoy interactivity."""
import json
import numpy as np
from dash import Input, Output, State, callback, no_update, ctx, ALL, html
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from roijoy.envi_io import load_envi, render_rgb, apply_contrast, rgb_to_base64
from roijoy.layout import make_empty_figure, MAX_PANELS

# Server-side cache for loaded cubes (keyed by panel index)
_cube_cache = {}


@callback(
    [Output(f"panel-container-{i}", "style") for i in range(MAX_PANELS)] +
    [Output(f"panel-title-{i}", "children") for i in range(MAX_PANELS)] +
    [Output(f"image-graph-{i}", "figure") for i in range(MAX_PANELS)] +
    [Output("image-data-store", "data"),
     Output("status-msg", "children"),
     Output("loaded-files-list", "children")],
    Input("file-upload", "contents"),
    State("file-upload", "filename"),
    State("image-data-store", "data"),
    State("rgb-mode", "value"),
    State("low-pct", "value"),
    State("high-pct", "value"),
    State("gain", "value"),
    State("offset", "value"),
    prevent_initial_call=True,
)
def load_files(contents_list, filenames, current_data, rgb_mode, low, high, gain, offset):
    """Handle file upload / path loading."""
    # For now this is a placeholder — ENVI files are too large for browser upload.
    # We'll add a path-based input in a follow-up.
    # This callback structure shows the pattern for updating all panels.
    raise PreventUpdate


def load_envi_to_panel(hdr_path: str, panel_idx: int, rgb_mode: str,
                       low: float, high: float, gain: float, offset: float) -> tuple:
    """Load an ENVI file and prepare it for display in a panel.

    Returns (figure, panel_data_dict) for the given panel.
    """
    cube, wavelengths = load_envi(hdr_path)
    _cube_cache[panel_idx] = {"cube": cube, "wavelengths": wavelengths, "path": hdr_path}

    # Render RGB
    if rgb_mode == "resample":
        rgb = render_rgb(cube, wavelengths, mode="resample")
    elif rgb_mode == "bands_default":
        rgb = render_rgb(cube, wavelengths, mode="bands", band_indices=[54, 32, 22])
    else:
        rgb = render_rgb(cube, wavelengths, mode="bands", band_indices=[20, 40, 60])

    # Apply contrast
    rgb = apply_contrast(rgb, low, high, gain, offset)

    # Convert to base64 for display
    b64 = rgb_to_base64(rgb)

    # Build Plotly figure with image
    fig = make_empty_figure()
    fig.add_layout_image(
        source=b64,
        xref="x", yref="y",
        x=0, y=0,
        sizex=cube.shape[1], sizey=cube.shape[0],
        sizing="stretch",
        layer="below",
    )
    fig.update_xaxes(range=[0, cube.shape[1]])
    fig.update_yaxes(range=[cube.shape[0], 0])  # Flip Y for image convention

    panel_data = {
        "path": hdr_path,
        "shape": list(cube.shape),
        "n_wavelengths": len(wavelengths),
    }

    return fig, panel_data


# Contrast adjustment callback — re-renders all loaded panels
@callback(
    [Output(f"image-graph-{i}", "figure", allow_duplicate=True) for i in range(MAX_PANELS)],
    [Input("rgb-mode", "value"),
     Input("low-pct", "value"),
     Input("high-pct", "value"),
     Input("gain", "value"),
     Input("offset", "value")],
    prevent_initial_call=True,
)
def update_contrast(rgb_mode, low, high, gain, offset):
    """Re-render all loaded images with new contrast settings."""
    figures = []
    for i in range(MAX_PANELS):
        if i in _cube_cache:
            fig, _ = load_envi_to_panel(
                _cube_cache[i]["path"], i, rgb_mode, low, high, gain, offset
            )
            figures.append(fig)
        else:
            figures.append(no_update)
    return figures
```

**Step 2: Uncomment the callbacks import in app.py**

In `roijoy/app.py`, change:
```python
# from roijoy import callbacks  # uncomment after Task 6
```
to:
```python
from roijoy import callbacks
```

**Step 3: Run to verify contrast sliders work**

```bash
python roijoy/app.py
```

Expected: App runs without errors. Contrast sliders don't crash (no images loaded yet).

**Step 4: Commit**

```bash
git add roijoy/callbacks.py roijoy/app.py
git commit -m "feat: core callbacks for contrast adjustment and image display"
```

---

### Task 7: File Path Input & Panel Population

**Files:**
- Modify: `roijoy/layout.py` (add path input)
- Modify: `roijoy/callbacks.py` (add path-based loading)

Since ENVI binary files are 136MB each, we can't use browser upload. Instead, add a text input where users type/paste the path to a `.hdr` file, plus a "Load" button.

**Step 1: Add path input to sidebar in layout.py**

After the Upload component in `create_layout()`, add:

```python
        html.H3("Or Load by Path"),
        dcc.Input(
            id="file-path-input",
            type="text",
            placeholder="/path/to/file.hdr",
            style={"width": "100%", "marginBottom": "8px",
                   "background": "#22252e", "border": "1px solid #2a2d36",
                   "color": "#e0e0e0", "padding": "8px", "borderRadius": "4px"},
        ),
        html.Button("Load File", id="btn-load-file", className="dash-button",
                     style={"width": "100%", "marginBottom": "8px"}),
```

**Step 2: Add the loading callback in callbacks.py**

```python
@callback(
    [Output(f"panel-container-{i}", "style", allow_duplicate=True) for i in range(MAX_PANELS)] +
    [Output(f"panel-title-{i}", "children", allow_duplicate=True) for i in range(MAX_PANELS)] +
    [Output(f"image-graph-{i}", "figure", allow_duplicate=True) for i in range(MAX_PANELS)] +
    [Output("image-data-store", "data", allow_duplicate=True),
     Output("status-msg", "children", allow_duplicate=True),
     Output("loaded-files-list", "children", allow_duplicate=True)],
    Input("btn-load-file", "n_clicks"),
    State("file-path-input", "value"),
    State("image-data-store", "data"),
    State("rgb-mode", "value"),
    State("low-pct", "value"),
    State("high-pct", "value"),
    State("gain", "value"),
    State("offset", "value"),
    prevent_initial_call=True,
)
def load_file_by_path(n_clicks, path, current_data, rgb_mode, low, high, gain, offset):
    """Load an ENVI file by path and display it in the next available panel."""
    if not path or not path.endswith('.hdr'):
        raise PreventUpdate

    import os
    if not os.path.exists(path):
        styles = [no_update] * MAX_PANELS
        titles = [no_update] * MAX_PANELS
        figures = [no_update] * MAX_PANELS
        return styles + titles + figures + [no_update, f"File not found: {path}", no_update]

    # Find next available panel
    if current_data is None:
        current_data = {}

    panel_idx = len(current_data)
    if panel_idx >= MAX_PANELS:
        styles = [no_update] * MAX_PANELS
        titles = [no_update] * MAX_PANELS
        figures = [no_update] * MAX_PANELS
        return styles + titles + figures + [no_update, "Maximum 6 images loaded.", no_update]

    # Load the file
    fig, panel_data = load_envi_to_panel(path, panel_idx, rgb_mode, low, high, gain, offset)

    # Build outputs
    styles = [no_update] * MAX_PANELS
    titles = [no_update] * MAX_PANELS
    figures = [no_update] * MAX_PANELS

    styles[panel_idx] = {"display": "block"}
    titles[panel_idx] = os.path.basename(path).replace('.hdr', '')
    figures[panel_idx] = fig

    current_data[str(panel_idx)] = panel_data

    # Build file list display
    file_list = html.Ul([
        html.Li(d.get("path", "").split("/")[-1].replace(".hdr", ""),
                style={"fontSize": "0.8em", "color": "#aaa"})
        for d in current_data.values()
    ])

    return (styles + titles + figures +
            [current_data, f"Loaded: {os.path.basename(path)}", file_list])
```

**Step 3: Test by running the app and loading a test file**

```bash
python roijoy/app.py
```

In the browser, paste this path into the input and click "Load File":
```
/Users/alfarolab/Dropbox/git/ROIJoy/data/RO_004_5_2025-04-17_01-08-18_.hdr
```

Expected: Image appears in Panel 1. Load a second file to see it in Panel 2.

**Step 4: Commit**

```bash
git add roijoy/layout.py roijoy/callbacks.py
git commit -m "feat: path-based ENVI file loading into image panels"
```

---

### Task 8: Polygon Drawing & ROI Extraction Callback

**Files:**
- Modify: `roijoy/callbacks.py`

**Step 1: Add polygon capture callback**

When a user draws a closed path on any panel, Plotly fires a `relayout` event with the shape data. We capture this, extract the spectrum, and propagate to other panels.

```python
# Add to callbacks.py

from roijoy.roi import extract_spectrum
from roijoy.matching import match_roi, copy_roi

# 10 distinct ROI colors
ROI_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#2980b9", "#27ae60", "#c0392b",
]


@callback(
    Output("roi-store", "data", allow_duplicate=True),
    Output("spectrum-graph", "figure", allow_duplicate=True),
    Output("roi-table-content", "children", allow_duplicate=True),
    [Input(f"image-graph-{i}", "relayoutData") for i in range(MAX_PANELS)],
    State("roi-store", "data"),
    State("image-data-store", "data"),
    State("matching-mode", "value"),
    prevent_initial_call=True,
)
def on_shape_drawn(*args):
    """Handle polygon drawing on any image panel."""
    # Last 3 args are states
    relayout_data_list = args[:MAX_PANELS]
    roi_data, image_data, matching_mode = args[MAX_PANELS:]

    if roi_data is None:
        roi_data = []

    # Find which panel triggered
    triggered = ctx.triggered_id
    if triggered is None:
        raise PreventUpdate

    # Extract panel index from triggered id like "image-graph-2"
    panel_idx = None
    for i in range(MAX_PANELS):
        if triggered == f"image-graph-{i}":
            panel_idx = i
            break

    if panel_idx is None:
        raise PreventUpdate

    relayout = relayout_data_list[panel_idx]
    if relayout is None:
        raise PreventUpdate

    # Check for new shape in relayout data
    # Plotly stores shapes as 'shapes' key or 'shapes[N].path' for individual additions
    new_shapes = None
    for key in relayout:
        if 'shapes' in key:
            # Get all shapes from the figure
            # The relayout contains the full shapes list when a shape is added
            if key == 'shapes':
                new_shapes = relayout[key]
            break

    if new_shapes is None:
        raise PreventUpdate

    # Process only the newest shape (last in list)
    if not new_shapes:
        raise PreventUpdate

    latest_shape = new_shapes[-1]
    if latest_shape.get('type') != 'path':
        raise PreventUpdate

    # Parse SVG path to vertices
    path_str = latest_shape.get('path', '')
    vertices = parse_svg_path(path_str)
    if len(vertices) < 3:
        raise PreventUpdate

    # Create new ROI
    roi_id = len(roi_data) + 1
    color = ROI_COLORS[(roi_id - 1) % len(ROI_COLORS)]

    new_roi = {
        "id": roi_id,
        "color": color,
        "panels": {
            str(panel_idx): {
                "vertices": vertices,
                "confirmed": True,
            }
        },
    }

    # Feature matching to propagate to other panels
    if matching_mode != "off" and image_data:
        source_cache = _cube_cache.get(panel_idx)
        if source_cache:
            source_rgb = render_rgb(source_cache["cube"], source_cache["wavelengths"])
            source_rgb_uint8 = (source_rgb * 255).astype(np.uint8)

            for idx_str, pdata in image_data.items():
                other_idx = int(idx_str)
                if other_idx == panel_idx:
                    continue
                target_cache = _cube_cache.get(other_idx)
                if target_cache is None:
                    continue

                target_rgb = render_rgb(target_cache["cube"], target_cache["wavelengths"])
                target_rgb_uint8 = (target_rgb * 255).astype(np.uint8)

                if matching_mode == "feature":
                    result = match_roi(source_rgb_uint8, target_rgb_uint8, vertices)
                else:
                    result = copy_roi(vertices)

                if result["success"]:
                    new_roi["panels"][str(other_idx)] = {
                        "vertices": result["vertices"],
                        "confirmed": False,  # Tentative
                        "method": result["method"],
                    }

    roi_data.append(new_roi)

    # Update spectrum plot
    spectrum_fig = build_spectrum_figure(roi_data)

    # Update ROI table
    table = build_roi_table(roi_data, image_data or {})

    return roi_data, spectrum_fig, table


def parse_svg_path(path_str: str) -> list[tuple[float, float]]:
    """Parse an SVG path string from Plotly into a list of (x, y) vertices.

    Plotly drawclosedpath produces paths like:
    'M100,200L150,250L200,200Z'
    """
    vertices = []
    parts = path_str.replace('M', '').replace('Z', '').split('L')
    for part in parts:
        part = part.strip()
        if ',' in part:
            x, y = part.split(',')
            vertices.append((float(x), float(y)))
    return vertices


def build_spectrum_figure(roi_data: list) -> go.Figure:
    """Build the spectrum comparison figure from all ROIs."""
    from roijoy.layout import make_spectrum_figure
    fig = make_spectrum_figure()

    line_styles = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]

    for roi in roi_data:
        color = roi["color"]
        roi_id = roi["id"]

        for panel_str, panel_roi in roi["panels"].items():
            panel_idx = int(panel_str)
            if not panel_roi["confirmed"]:
                continue

            cache = _cube_cache.get(panel_idx)
            if cache is None:
                continue

            result = extract_spectrum(
                cache["cube"], cache["wavelengths"], panel_roi["vertices"]
            )

            line_style = line_styles[panel_idx % len(line_styles)]

            fig.add_trace(go.Scatter(
                x=result["wavelengths"],
                y=result["mean"],
                mode="lines",
                name=f"ROI {roi_id} - Panel {panel_idx + 1}",
                line=dict(color=color, dash=line_style),
            ))

    return fig


def build_roi_table(roi_data: list, image_data: dict) -> html.Table:
    """Build an HTML table showing ROI status across panels."""
    if not roi_data:
        return "No ROIs yet."

    n_panels = len(image_data)
    header_cells = [html.Th("ROI"), html.Th("Color")]
    for i in range(n_panels):
        header_cells.append(html.Th(f"Img {i+1}"))

    rows = [html.Tr(header_cells)]

    for roi in roi_data:
        cells = [
            html.Td(str(roi["id"])),
            html.Td(html.Div(style={
                "width": "16px", "height": "16px", "borderRadius": "50%",
                "background": roi["color"], "display": "inline-block",
            })),
        ]
        for i in range(n_panels):
            panel_roi = roi["panels"].get(str(i))
            if panel_roi is None:
                cells.append(html.Td("-"))
            elif panel_roi["confirmed"]:
                cells.append(html.Td("✓", style={"color": "#2ecc71"}))
            else:
                cells.append(html.Td("~", style={"color": "#f39c12"}))

        rows.append(html.Tr(cells))

    return html.Table(rows, style={"width": "100%", "fontSize": "0.85em"})
```

**Step 2: Test interactively**

```bash
python roijoy/app.py
```

Load two test images, draw a polygon on one, verify:
1. Polygon appears
2. Tentative ROI appears on other panel(s)
3. Spectrum comparison chart updates
4. ROI table shows status

**Step 3: Commit**

```bash
git add roijoy/callbacks.py
git commit -m "feat: polygon drawing, ROI extraction, feature matching, and spectrum comparison"
```

---

### Task 9: Synchronized Zoom/Pan Across Panels

**Files:**
- Modify: `roijoy/callbacks.py`

**Step 1: Add zoom sync callback**

```python
# Add to callbacks.py

@callback(
    [Output(f"image-graph-{i}", "figure", allow_duplicate=True) for i in range(MAX_PANELS)],
    [Input(f"image-graph-{i}", "relayoutData") for i in range(MAX_PANELS)],
    [State(f"image-graph-{i}", "figure") for i in range(MAX_PANELS)],
    State("image-data-store", "data"),
    prevent_initial_call=True,
)
def sync_zoom(*args):
    """Sync zoom/pan across all image panels."""
    relayout_list = args[:MAX_PANELS]
    figure_list = args[MAX_PANELS:2*MAX_PANELS]
    image_data = args[2*MAX_PANELS]

    triggered = ctx.triggered_id
    if triggered is None:
        raise PreventUpdate

    # Find source panel
    source_idx = None
    for i in range(MAX_PANELS):
        if triggered == f"image-graph-{i}":
            source_idx = i
            break

    if source_idx is None:
        raise PreventUpdate

    relayout = relayout_list[source_idx]
    if relayout is None:
        raise PreventUpdate

    # Extract zoom range from relayout
    x_range = None
    y_range = None
    if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
        x_range = [relayout["xaxis.range[0]"], relayout["xaxis.range[1]"]]
    if "yaxis.range[0]" in relayout and "yaxis.range[1]" in relayout:
        y_range = [relayout["yaxis.range[0]"], relayout["yaxis.range[1]"]]

    if x_range is None and y_range is None:
        raise PreventUpdate

    # Apply to all other loaded panels
    outputs = []
    for i in range(MAX_PANELS):
        if image_data and str(i) in image_data and i != source_idx:
            fig = go.Figure(figure_list[i])
            if x_range:
                fig.update_xaxes(range=x_range)
            if y_range:
                fig.update_yaxes(range=y_range)
            outputs.append(fig)
        else:
            outputs.append(no_update)

    # Don't update the source panel
    outputs[source_idx] = no_update

    return outputs
```

**Step 2: Test by loading two images and zooming on one**

Expected: Both panels zoom together.

**Step 3: Commit**

```bash
git add roijoy/callbacks.py
git commit -m "feat: synchronized zoom/pan across image panels"
```

---

### Task 10: Export & Final Polish

**Files:**
- Modify: `roijoy/callbacks.py` (export callback)
- Modify: `roijoy/roi.py` (add combined CSV export)

**Step 1: Add combined export function to roi.py**

```python
# Add to roijoy/roi.py

def export_combined_csv(path: str, all_roi_data: list, cube_cache: dict) -> None:
    """Export all ROI spectra across all images into a single comparison CSV."""
    rows = []
    header_written = False

    for roi in all_roi_data:
        roi_id = roi["id"]
        for panel_str, panel_roi in roi["panels"].items():
            if not panel_roi["confirmed"]:
                continue
            panel_idx = int(panel_str)
            cache = cube_cache.get(panel_idx)
            if cache is None:
                continue

            result = extract_spectrum(cache["cube"], cache["wavelengths"], panel_roi["vertices"])

            if not header_written:
                wl_headers = [f"{w:.1f}" for w in result["wavelengths"]]
                header = ["ROI_ID", "Image_Index", "Image_Path", "N_Pixels",
                          "Stat"] + wl_headers
                rows.append(header)
                header_written = True

            rows.append([roi_id, panel_idx, cache.get("path", ""), result["n_pixels"],
                         "mean"] + [f"{v:.6e}" for v in result["mean"]])
            rows.append([roi_id, panel_idx, cache.get("path", ""), result["n_pixels"],
                         "std"] + [f"{v:.6e}" for v in result["std"]])

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
```

**Step 2: Add export callback in callbacks.py**

```python
@callback(
    Output("status-msg", "children", allow_duplicate=True),
    Input("btn-export", "n_clicks"),
    State("roi-store", "data"),
    State("image-data-store", "data"),
    prevent_initial_call=True,
)
def export_all(n_clicks, roi_data, image_data):
    """Export all confirmed ROI data to CSV files."""
    if not roi_data:
        return "No ROIs to export."

    import os
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    from roijoy.roi import export_spectrum_csv, export_combined_csv

    # Export individual ROI spectra
    for roi in roi_data:
        for panel_str, panel_roi in roi["panels"].items():
            if not panel_roi["confirmed"]:
                continue
            panel_idx = int(panel_str)
            cache = _cube_cache.get(panel_idx)
            if cache is None:
                continue

            result = extract_spectrum(cache["cube"], cache["wavelengths"], panel_roi["vertices"])

            outpath = os.path.join(output_dir,
                f"roi_{roi['id']}_panel_{panel_idx}_spectrum.csv")
            export_spectrum_csv(outpath, result["wavelengths"], result["mean"], result["std"])

    # Export combined comparison
    combined_path = os.path.join(output_dir, "combined_comparison.csv")
    export_combined_csv(combined_path, roi_data, _cube_cache)

    return f"Exported {len(roi_data)} ROIs to {output_dir}/"
```

**Step 3: Test export by drawing ROIs and clicking Export All**

Expected: CSV files appear in `output/` directory.

**Step 4: Commit**

```bash
git add roijoy/roi.py roijoy/callbacks.py
git commit -m "feat: CSV export for individual and combined ROI spectra"
```

---

### Task 11: Create Feature Branch, Final Integration Test, Push

**Step 1: Create feature branch**

All development so far has been on main (initial scaffolding). Create a feature branch for the Dash GUI work:

```bash
git checkout -b feat/dash-gui
```

Note: If Tasks 2-10 were already committed to main, that's fine for initial setup. The feature branch captures the polished state.

**Step 2: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass.

**Step 3: Run the app and do a full manual integration test**

```bash
python roijoy/app.py
```

Test checklist:
- [ ] Load 3 ENVI files via path input
- [ ] All 3 appear in image grid
- [ ] Adjust contrast sliders — all panels update
- [ ] Switch RGB mode — all panels update
- [ ] Draw a polygon on Panel 1
- [ ] Tentative ROI appears on Panels 2 and 3 (dashed)
- [ ] Spectrum comparison chart shows data
- [ ] ROI table shows ✓ for Panel 1, ~ for Panels 2 and 3
- [ ] Click Export All — CSV files created
- [ ] Zoom on Panel 1 — Panels 2 and 3 follow

**Step 4: Push feature branch**

```bash
git push -u origin feat/dash-gui
```

---

## Summary of Commits

1. `feat: environment setup with Dash app skeleton`
2. `feat: ENVI I/O module with actual wavelength parsing`
3. `feat: ROI data model with spectrum extraction and CSV export`
4. `feat: ORB feature-based ROI matching with copy fallback`
5. `feat: multi-panel Dash layout with sidebar and image grid`
6. `feat: core callbacks for contrast adjustment and image display`
7. `feat: path-based ENVI file loading into image panels`
8. `feat: polygon drawing, ROI extraction, feature matching, and spectrum comparison`
9. `feat: synchronized zoom/pan across image panels`
10. `feat: CSV export for individual and combined ROI spectra`

## Test Files to Use

All in `data/` (symlinked):
- `RO_004_5_2025-04-17_01-08-18_.hdr` — 570x570, 110 bands
- `RO_007_8_2025-04-17_01-15-56_.hdr`
- `RO_008_9_2025-04-17_01-18-20_.hdr`
- `RO_010_11_2025-04-17_01-23-46_.hdr`
- `RO_017_18_2025-04-17_01-46-16_.hdr`
