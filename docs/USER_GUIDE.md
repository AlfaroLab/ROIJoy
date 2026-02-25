# ROIJoy User Guide

ROIJoy is a browser-based tool for selecting regions of interest (ROIs) on hyperspectral ENVI images and extracting reflectance spectra. It supports loading up to 6 images simultaneously, drawing polygon ROIs, automatically propagating ROIs across images using feature matching, and exporting spectral data to CSV.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Launching the App](#launching-the-app)
4. [Interface Overview](#interface-overview)
5. [Loading ENVI Images](#loading-envi-images)
6. [Adjusting the Display](#adjusting-the-display)
7. [Drawing ROIs](#drawing-rois)
8. [ROI Matching Across Images](#roi-matching-across-images)
9. [Viewing Spectra](#viewing-spectra)
10. [Exporting Data](#exporting-data)
11. [Multi-Lighting Condition Workflow](#multi-lighting-condition-workflow)
12. [Tips and Troubleshooting](#tips-and-troubleshooting)

---

## Requirements

- Python 3.10 or later
- An ENVI-format hyperspectral image (`.hdr` header + `.bin` data file)
- A modern web browser (Chrome, Firefox, or Safari)

## Installation

Clone the repository and create a virtual environment:

```bash
git clone <repo-url> ROIJoy
cd ROIJoy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Launching the App

```bash
source .venv/bin/activate
python roijoy/app.py
```

The terminal will print something like:

```
Dash is running on http://127.0.0.1:8050/
```

Open that URL in your browser. ROIJoy runs entirely in your browser but all computation happens locally on your machine.

## Interface Overview

The interface has two main areas:

```
┌─────────────────┬──────────────────────────────────┐
│    SIDEBAR       │    IMAGE GRID                     │
│                  │    (up to 6 panels)               │
│  • File loading  │                                   │
│  • RGB mode      ├──────────────────────────────────┤
│  • Contrast      │    SPECTRUM COMPARISON            │
│  • Matching mode ├──────────────────────────────────┤
│  • Export        │    ROI TABLE                      │
└─────────────────┴──────────────────────────────────┘
```

**Sidebar** (left): All controls for loading files, adjusting display settings, choosing matching mode, and exporting data.

**Image Grid** (right, top): Displays loaded hyperspectral images. Each panel shows an RGB rendering of the ENVI data cube. Panels appear as you load files.

**Spectrum Comparison** (right, middle): Shows the extracted reflectance spectra for all confirmed ROIs, plotted as mean with shaded standard deviation.

**ROI Table** (right, bottom): Lists all defined ROIs and their status across loaded images.

## Loading ENVI Images

ROIJoy reads ENVI-format files consisting of a `.hdr` header file and a `.bin` binary data file. Both files must be in the same directory and share the same base name:

```
my_image.hdr    ← header (wavelengths, dimensions, data type)
my_image.bin    ← raw spectral data
```

### To load a file:

1. **From the dropdown**: If your `.hdr` files are in the `data/` directory, they appear automatically in the dropdown. Select one and click **Load File**.

2. **By path**: Type or paste the full path to any `.hdr` file on your system into the text input, then click **Load File**.

Each loaded file appears in the next available panel (up to 6 total). The panel title shows the filename.

### Loading multiple files

Simply repeat the load process for each file. Images are added left to right, top to bottom in the grid. For comparing lighting conditions, load all conditions for a single specimen before moving on.

## Adjusting the Display

### RGB Visualization Mode

Under **RGB Visualization** in the sidebar, choose how the hyperspectral cube is rendered as a visible-light image:

- **Band Resampling** (default): Averages wavelength ranges that correspond to red (620-750 nm), green (495-570 nm), and blue (450-495 nm). Usually gives the most natural-looking color image.
- **Bands 54, 32, 22**: Uses specific band indices. Useful if the default rendering is washed out.
- **Bands 20, 40, 60**: Alternative band triplet.

### Contrast Controls

Four sliders adjust the image display:

| Slider   | Range       | Default | Effect                                    |
|----------|-------------|---------|-------------------------------------------|
| Low %    | 0 – 10      | 1       | Percentile below which pixels go to black |
| High %   | 90 – 100    | 99      | Percentile above which pixels go to white |
| Gain     | 0.5 – 2.0   | 1.0     | Brightness multiplier                     |
| Offset   | -0.5 – 0.5  | 0.0     | Brightness offset added after gain        |

Contrast changes apply to all loaded panels simultaneously. Existing ROI polygons are preserved when you adjust contrast.

### Zooming and Panning

- **Scroll wheel**: Zoom in/out on any panel
- **Click and drag** (when not in draw mode): Pan the view
- Zoom and pan are **synchronized** across all panels — when you zoom in on one image, all others follow. This makes it easy to compare the same region across lighting conditions.

## Drawing ROIs

ROIs (Regions of Interest) are polygon selections drawn directly on the image.

### To draw a polygon ROI:

1. Make sure the **draw closed path** tool is active in the Plotly toolbar at the top of each image panel (it should be active by default — the icon looks like a pencil with a closed shape).
2. Click on the image to place vertices of your polygon. Each click adds a point.
3. Close the polygon by clicking near the starting point, or double-click to auto-close.
4. The polygon appears with a teal outline and is automatically assigned a unique color from the ROI palette.

Each ROI gets a sequential ID (ROI 1, ROI 2, etc.) and its own color for identification across panels and in the spectrum plot.

### ROI Status Symbols

In the ROI table, each ROI's status per panel is shown as:

| Symbol | Meaning |
|--------|---------|
| ✓      | Confirmed — polygon was directly drawn on this panel |
| ~      | Tentative — polygon was propagated by feature matching |
| —      | No ROI data for this panel |

## ROI Matching Across Images

When you draw an ROI on one panel, ROIJoy can automatically place a corresponding polygon on the other loaded panels. This is controlled by the **ROI Matching** setting in the sidebar:

### Feature-based (ORB)

Uses computer vision to find matching features between images. The algorithm:

1. Detects ORB keypoints in both the source and target images
2. Matches descriptors using a ratio test
3. Computes a geometric transformation (homography) via RANSAC
4. Transforms your polygon vertices to the correct position in the target image

This works well when the images show the same subject from a similar angle. The matched ROI appears as **tentative** (~) in the table.

### Copy coordinates

Simply copies the polygon vertices as-is to all other panels. This is useful when the images are perfectly aligned (same camera position, same framing) so the patch is in the exact same pixel location across all images.

### Off

No propagation. The ROI only exists on the panel where you drew it.

**Recommendation for bird specimens under different lighting**: If the bird is in the same position across all three lighting conditions, use **Copy coordinates** mode. If the bird may have shifted slightly between shots, use **Feature-based (ORB)**.

## Viewing Spectra

### Main Spectrum Comparison Panel

Below the image grid, the **Spectrum Comparison** chart shows the reflectance spectra for all confirmed ROIs.

- **X-axis**: Wavelength in nm (parsed from the ENVI header)
- **Y-axis**: Reflectance (raw values from the data cube)
- Each trace is labeled as `ROI {id} / Img {panel}` (e.g., "ROI 1 / Img 1")
- **Shaded bands** show ±1 standard deviation around the mean
- Different line styles (solid, dashed, dotted) distinguish panels for the same ROI

### Per-Panel Inset Spectra

When you select an ROI by clicking the ROI table:

- A small spectrum overlay appears in the bottom-right corner of each relevant panel
- Shows mean line and ±1 SD shading for that specific ROI on that panel
- Helps visually confirm that the ROI is sampling the region you intended

Click the ROI table to cycle through ROIs. Clicking past the last ROI deselects all.

## Exporting Data

Click the **Export All** button in the sidebar to save CSV files to the `output/` directory.

### Output files

1. **Per-ROI files**: `roi_{id}_panel_{idx}_spectrum.csv`
   - One file per confirmed ROI per panel
   - Columns: `Wavelength (nm)`, `Mean Reflectance`, `Std Dev`
   - One row per wavelength band

2. **Combined file**: `combined_comparison.csv`
   - All ROIs and panels in a single file
   - Columns: `ROI_ID`, `Image_Index`, `Image_Path`, `N_Pixels`, `Stat`, then one column per wavelength
   - Two rows per ROI-panel combination: one for mean, one for standard deviation

### Important note on export

Currently, **only confirmed (✓) ROIs are exported**. Tentative (~) ROIs from feature matching are skipped. If you are using feature-based or copy matching mode and want to export data from all panels, you need to be aware that the propagated ROIs need to be confirmed before export. (This is a known limitation — see workaround in the tips section.)

## Multi-Lighting Condition Workflow

This workflow is designed for comparing reflectance of the same patch on a specimen photographed under different lighting conditions.

### Setup (per bird)

1. Launch ROIJoy: `python roijoy/app.py`
2. Load the 3 ENVI images for one bird (one per lighting condition)
   - Select each `.hdr` file from the dropdown or type the path
   - Click **Load File** for each — they appear as panels 1, 2, and 3

### Digitize the patch

3. Set **ROI Matching** to **Copy coordinates** (if images are aligned) or **Feature-based** (if the bird shifted between shots)
4. Zoom in on the target patch using scroll wheel — all three panels zoom together
5. Draw a polygon around the patch on any one of the three panels
6. The ROI is automatically propagated to the other two panels
7. Visually check that the polygon lands on the correct patch in all panels. If it looks off, the feature matching may have missed — try drawing directly on each panel with matching set to **Off**.

### Review and export

8. Check the **Spectrum Comparison** chart — you should see one trace per lighting condition for your ROI
9. Click the ROI table to select the ROI and verify the per-panel inset spectra look correct
10. Click **Export All** to save the CSV data to `output/`
11. The `combined_comparison.csv` will have the mean and std for each lighting condition

### Repeating for multiple birds

For a 20-bird dataset:

- After exporting one bird's data, **rename or move the output files** before starting the next bird (otherwise they'll be overwritten)
- A good convention: create a folder per bird
  ```bash
  mkdir output/bird_01
  mv output/*.csv output/bird_01/
  ```
- Restart the app (Ctrl+C in terminal, then `python roijoy/app.py`) to clear all panels, or simply load new files into available panels

### Batch naming convention

For organized data management across 20 birds × 3 conditions, consider naming your output folders:

```
output/
  bird_01/
    combined_comparison.csv
    roi_1_panel_0_spectrum.csv
    roi_1_panel_1_spectrum.csv
    roi_1_panel_2_spectrum.csv
  bird_02/
    ...
```

## Tips and Troubleshooting

### The image looks too dark or washed out
Adjust the contrast sliders. Try increasing **Gain** to 1.5 or lowering **Low %** to 0. You can also try a different RGB visualization mode.

### Feature matching placed the ROI in the wrong spot
Switch to **Copy coordinates** mode if images are well-aligned, or set matching to **Off** and draw manually on each panel.

### I drew a bad polygon
Currently there is no undo for individual ROIs. You can use the **eraser** tool in the Plotly toolbar to remove shapes from a panel. For a fresh start, restart the app.

### The app is slow to load images
Each ENVI data cube is loaded fully into memory. With 6 panels of 570×570×110 images, that's about 200 MB of RAM. This is normal. Loading takes a few seconds per file.

### Wavelength axis shows wrong values
ROIJoy reads wavelengths from the `.hdr` file metadata. If your header is missing the `wavelength` field, it falls back to a 350-1000 nm linear range, which may not match your instrument. Check your `.hdr` file for a `wavelength = { ... }` section.

### Browser won't connect to localhost:8050
Make sure the terminal shows "Dash is running on http://127.0.0.1:8050/" with no errors. If port 8050 is in use, kill the other process or edit `app.py` to use a different port.

### Tentative ROIs not exporting
This is a known limitation. Only ROIs marked ✓ (confirmed) are exported. As a workaround, set matching to **Off** and draw the polygon manually on each panel — this marks each as confirmed. Alternatively, use **Copy coordinates** mode and note that the propagated copies also need manual confirmation.
