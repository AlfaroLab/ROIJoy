# ROIJoy Design Document

**Date:** 2026-02-16
**Status:** Approved
**Origin:** Based on `extract-reflectance-gui.py` from AlfaroLab/oist-field-collection-2025

## Purpose

ROIJoy is an interactive browser-based tool for selecting regions of interest (ROIs) on hyperspectral images and extracting reflectance spectra. Its key differentiator from the original matplotlib-based script is **multi-image comparison**: users can load up to 6 ENVI images of the same specimen under different lighting conditions, draw a polygon ROI on one image, and have the tool automatically suggest homologous ROIs on the other images using feature-based matching.

## Architecture

**Framework:** Plotly Dash (browser-based, all-Python backend)
**Approach:** Pure Dash/Plotly with built-in polygon drawing (`drawclosedpath` mode)

### Project Structure

```
ROIJoy/
├── roijoy/
│   ├── __init__.py
│   ├── app.py              # Dash app entry point
│   ├── layout.py           # UI layout
│   ├── callbacks.py        # Dash callbacks
│   ├── envi_io.py          # ENVI file I/O, RGB rendering
│   ├── roi.py              # ROI data model, extraction, CSV export
│   ├── matching.py         # Feature-based ROI matching (ORB)
│   └── assets/style.css
├── legacy/                 # Original matplotlib scripts for reference
├── data/                   # ENVI test files (git-ignored, symlinked)
├── output/                 # Exported data (git-ignored)
├── docs/plans/
├── requirements.txt
└── .gitignore
```

## UI Layout

The app displays a responsive grid of image panels (1-3 images = single row, 4-6 = 2x3 grid), a sidebar with controls, a spectrum comparison panel, and an ROI table.

### Sidebar Controls
- File loader (multi-file upload or path selection)
- RGB visualization mode (band resampling vs specific band triplets)
- Contrast sliders (low/high percentile, gain, offset)
- Matching strategy selector (feature-based, copy-coordinates, off)
- Drawing tools (Draw, Edit, Delete)

### Image Panels
- Each panel shows one ENVI image rendered as RGB
- Plotly `drawclosedpath` mode for polygon drawing
- Synchronized zoom/pan across all panels via `relayout` callback
- Confirmed ROIs shown as solid polygons; tentative (suggested) ROIs as dashed

### Spectrum Panel
- Combined line chart showing mean reflectance per ROI per image
- Color-coded by ROI number, line style by image
- Standard deviation shown as filled region

### ROI Table
- Lists all ROIs with confirmation status per image
- Per-ROI export buttons

## Data Flow

### ENVI Loading
1. `spectral.envi.open()` loads cube from .hdr + .bin
2. Parse actual wavelengths from .hdr (not hardcoded)
3. Generate RGB preview via band resampling or triplet selection
4. Apply contrast adjustments
5. Encode as base64 PNG for browser display
6. Full cube stays in server memory for spectrum extraction

### Feature-Based ROI Matching
1. User draws polygon on Image 1
2. Extract ORB keypoints + descriptors from both images
3. Match features with ratio test filtering
4. Compute homography via RANSAC
5. Transform polygon vertices using homography
6. Display as dashed tentative ROI on other images
7. User accepts, adjusts, or rejects
8. Fallback to same-coordinates copy if matching fails (low inlier count)

### Export Formats
- Per-ROI polygon coordinates CSV (normalized)
- Per-ROI spectrum summary CSV (wavelength, mean, std)
- Per-ROI random subsample CSV (100 pixels with full spectra)
- Combined comparison CSV (all ROIs across all images)

## Key Improvements Over Original

1. **Multi-image support** (up to 6 vs original's 1)
2. **Browser-based UI** (responsive, modern, no matplotlib window management)
3. **Feature-based ROI propagation** across images
4. **Actual wavelength parsing** from .hdr files (fixes hardcoded linspace bug)
5. **Synchronized zoom/pan** across image panels
6. **Spectrum comparison** across images in a single chart

## Dependencies

- dash, plotly (web framework + visualization)
- numpy (array operations)
- spectral (ENVI I/O)
- scikit-image (polygon rasterization)
- opencv-python (ORB feature matching)
- pandas (data export)
