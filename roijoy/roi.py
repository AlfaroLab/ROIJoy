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
    """Extract a random subsample of pixel spectra from a polygon ROI."""
    r = np.array([v[1] for v in vertices])
    c = np.array([v[0] for v in vertices])
    rr, cc = polygon(r, c, cube.shape[:2])

    spectra = cube[rr, cc, :]
    coords = np.column_stack((cc, rr))

    if len(spectra) > n_samples:
        idx = np.random.choice(len(spectra), n_samples, replace=False)
        return coords[idx], spectra[idx]
    return coords, spectra


def normalize_vertices(vertices: list[tuple[float, float]],
                       image_shape: tuple[int, int]) -> list[tuple[float, float]]:
    """Normalize pixel vertices to [0, 1] range."""
    rows, cols = image_shape
    return [(x / cols, y / rows) for x, y in vertices]


def export_spectrum_csv(path: str, wavelengths: np.ndarray,
                        mean: np.ndarray, std: np.ndarray) -> None:
    """Export spectrum summary to CSV."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Wavelength (nm)", "Mean Reflectance", "Std Dev"])
        for w, m, s in zip(wavelengths, mean, std):
            writer.writerow([f"{w:.1f}", f"{m:.6e}", f"{s:.6e}"])


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
