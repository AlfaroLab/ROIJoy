# ROIJoy Example Data

This directory contains example ENVI hyperspectral image files for testing ROIJoy.

## 📁 What's Included

**Header files (.hdr)** - ✅ Included in repository:
- `RO_004_5_2025-04-17_01-08-18_.hdr`
- `RO_007_8_2025-04-17_01-15-56_.hdr` 
- `RO_008_9_2025-04-17_01-18-20_.hdr`
- `RO_010_11_2025-04-17_01-23-46_.hdr`
- `RO_017_18_2025-04-17_01-46-16_.hdr`

**Binary files (.bin)** - ❌ Not included (136MB each, too large for GitHub):
- `RO_004_5_2025-04-17_01-08-18_.bin`
- `RO_007_8_2025-04-17_01-15-56_.bin`
- `RO_008_9_2025-04-17_01-18-20_.bin` 
- `RO_010_11_2025-04-17_01-23-46_.bin`
- `RO_017_18_2025-04-17_01-46-16_.bin`

## 🔧 Getting Binary Data

### Option 1: Use Your Own Data
ROIJoy works with any ENVI format hyperspectral images. Place your `.hdr` and `.bin` file pairs in this directory.

### Option 2: Download Example Data (UCLA Lab Members)
Contact the Alfaro Lab for access to the full example dataset from the 2025 OIST field collection.

### Option 3: Generate Synthetic Data
For testing purposes, you can generate synthetic hyperspectral data:

```python
import numpy as np
from spectral import envi

# Create synthetic 200x200x50 hyperspectral cube
rows, cols, bands = 200, 200, 50
synthetic_data = np.random.rand(rows, cols, bands).astype(np.float32)

# Save as ENVI format
envi.save_image('synthetic_test.hdr', synthetic_data, dtype=np.float32, force=True)
```

## 📊 Data Specifications

**Image dimensions:** 570 × 570 pixels  
**Spectral bands:** 110 bands  
**Wavelength range:** 370-806 nm  
**File size:** ~136MB per image (.bin file)  
**Format:** ENVI BSQ (Band Sequential)  

## 🚀 Quick Start

1. Place `.bin` files corresponding to the `.hdr` files in this directory
2. Start ROIJoy: `python roijoy/app.py`  
3. Use the file dropdown to select an example image
4. Draw polygons to extract ROI spectra
5. Compare across multiple images with feature matching

## 🧪 Running Tests

The test suite expects at least `RO_004_5_2025-04-17_01-08-18_.hdr/.bin` to be present:

```bash
pytest tests/
```