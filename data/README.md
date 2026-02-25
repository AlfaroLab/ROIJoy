# ROIJoy Example Data

This directory contains example ENVI hyperspectral image files for testing ROIJoy.

## What's Included

Five cropped demo images (center 67% of originals, 380x380 pixels):

| File | Size |
|------|------|
| `RO_004_5_2025-04-17_01-08-18_` | 63.5 MB |
| `RO_007_8_2025-04-17_01-15-56_` | 63.5 MB |
| `RO_008_9_2025-04-17_01-18-20_` | 63.5 MB |
| `RO_010_11_2025-04-17_01-23-46_` | 63.5 MB |
| `RO_017_18_2025-04-17_01-46-16_` | 63.5 MB |

Each image has a `.hdr` (header) and `.bin` (binary data) file pair.

## Data Specifications

**Image dimensions:** 380 x 380 pixels (cropped from 570 x 570 originals)
**Spectral bands:** 110 bands
**Wavelength range:** 370-806 nm
**File size:** ~63.5 MB per image (.bin file)
**Format:** ENVI BIP (Band Interleaved by Pixel), float32
**Source:** 2025 OIST field collection, cropped to center region

## Quick Start

1. Start ROIJoy: `python roijoy/app.py`
2. Use the file dropdown to select an example image
3. Draw polygons to extract ROI spectra
4. Compare across multiple images with feature matching

## Using Your Own Data

ROIJoy works with any ENVI format hyperspectral images. You can:
- Place `.hdr` + `.bin` file pairs in this directory (they appear in the dropdown)
- Type the full path to any `.hdr` file in the app's text input

## Running Tests

The test suite uses `RO_004_5_2025-04-17_01-08-18_` as sample data:

```bash
pytest tests/
```
