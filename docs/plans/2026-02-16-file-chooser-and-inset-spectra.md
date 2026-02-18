# File Chooser + Per-Panel Spectrum Inset

**Date**: 2026-02-16
**Branch**: feat/dash-gui

## Features

### 1. File Chooser Dropdown
Replace the raw text input with a searchable dropdown listing `.hdr` files from `data/`.
- On startup, scan `data/` directory for `.hdr` files
- `dcc.Dropdown` with searchable filenames, auto-loads on selection
- Keep text input as "Custom path" fallback below the dropdown
- Default directory: `data/` relative to project root

### 2. Per-Panel Spectrum Inset (Active ROI)
When a user selects an ROI row in the table, each image panel shows a small inset chart.
- Add `dcc.Store(id="selected-roi")` for selection state
- ROI table rows are clickable (toggle selection)
- Overlay `dcc.Graph` mini-chart per panel (~180x120px, bottom-right corner)
- Shows mean reflectance line + shaded +/-1 SD band in the ROI's color
- Hidden when no ROI is selected
- Semi-transparent dark background to not obscure image

### 3. Text Visibility
Increase contrast of all text levels for readability on dark backgrounds.

## Data Flow

```
ROI table row click -> selected-roi store -> inset callback
inset callback reads: selected-roi, roi-store, _cube_cache
  for each panel with that ROI:
    extract_spectrum() -> mean, std
    build mini plotly figure with fill_between
    update inset-graph-{i}
```
