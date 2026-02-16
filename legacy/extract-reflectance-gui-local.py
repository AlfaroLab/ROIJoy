
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from spectral import envi, get_rgb
from skimage.draw import polygon
import os


script_dir = os.path.dirname(os.path.abspath(__file__))


hdr_path = os.path.join(script_dir, '..', 'DATA', 'RO_004_5_2025-04-17_01-08-18_.hdr')
bin_path = os.path.join(script_dir, '..', 'DATA', 'RO_004_5_2025-04-17_01-08-18_.bin')


# Load ENVI image and binary data
data = envi.open(hdr_path, image=bin_path)
cube = data.load()

# Define RGB combinations
band_options = {
    'Default (20,40,60)': [20, 40, 60],
    'Alt 1 (10,30,50)': [10, 30, 50],
    'Alt 2 (30,50,70)': [30, 50, 70]
}
selected_bands = band_options['Default (20,40,60)']
rgb_raw = get_rgb(data, selected_bands).astype(np.float32)

# Setup figure and sliders
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(left=0.3, bottom=0.4)
img_disp = ax.imshow(rgb_raw, vmin=0, vmax=1)
ax.set_title("Adjust sliders, then click 'Continue to Polygon'")

# GUI elements
ax_low = plt.axes([0.35, 0.26, 0.55, 0.03])
ax_high = plt.axes([0.35, 0.22, 0.55, 0.03])
ax_gain = plt.axes([0.35, 0.18, 0.55, 0.03])
ax_offset = plt.axes([0.35, 0.14, 0.55, 0.03])
ax_reset = plt.axes([0.82, 0.02, 0.12, 0.04])
ax_save = plt.axes([0.65, 0.02, 0.15, 0.04])
ax_continue = plt.axes([0.35, 0.02, 0.25, 0.04])
ax_radio = plt.axes([0.02, 0.6, 0.25, 0.25], frameon=True)

# Sliders
low_slider = Slider(ax_low, 'Low %', 0, 10, valinit=1)
high_slider = Slider(ax_high, 'High %', 90, 100, valinit=99)
gain_slider = Slider(ax_gain, 'Gain', 0.5, 2.0, valinit=1)
offset_slider = Slider(ax_offset, 'Offset', -0.5, 0.5, valinit=0)
radio = RadioButtons(ax_radio, list(band_options.keys()))

# Global for final image
final_rgb = None

def update(val=None):
    low = low_slider.val
    high = high_slider.val
    gain = gain_slider.val
    offset = offset_slider.val
    p_low, p_high = np.percentile(rgb_raw, (low, high))
    rgb_adj = np.clip((rgb_raw - p_low) / (p_high - p_low), 0, 1)
    rgb_adj = np.clip(gain * rgb_adj + offset, 0, 1)
    img_disp.set_data(rgb_adj)
    fig.canvas.draw_idle()

def reset(event):
    low_slider.reset()
    high_slider.reset()
    gain_slider.reset()
    offset_slider.reset()

def save_rgb(event):
    update()
    out_path = '/Users/rosamariorduna/Downloads/binandhdr_folder/current_rgb_preview.png'
    plt.imsave(out_path, img_disp.get_array())
    print(f"Saved RGB preview to: {out_path}")

def change_band(label):
    global rgb_raw
    selected = band_options[label]
    rgb_raw = get_rgb(data, selected).astype(np.float32)
    update()

def continue_to_polygon(event):
    global final_rgb
    low = low_slider.val
    high = high_slider.val
    gain = gain_slider.val
    offset = offset_slider.val
    p_low, p_high = np.percentile(rgb_raw, (low, high))
    final_rgb = np.clip((rgb_raw - p_low) / (p_high - p_low), 0, 1)
    final_rgb = np.clip(gain * final_rgb + offset, 0, 1)

    ax.clear()
    ax.imshow(final_rgb)
    ax.set_title("Draw a polygon around the patch (Right-click or Enter to finish)")
    fig.canvas.draw_idle()

    pts = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close(fig)

    r = np.array([p[1] for p in pts])
    c = np.array([p[0] for p in pts])
    rr, cc = polygon(r, c, cube.shape[:2])
    mask = np.zeros(cube.shape[:2], dtype=bool)
    mask[rr, cc] = True

    spectra = cube[mask, :]
    avg_spectrum = spectra.mean(axis=0)
    std_spectrum = spectra.std(axis=0)

    wavelengths = np.linspace(350, 1000, cube.shape[2])
    plt.figure()
    plt.plot(wavelengths, avg_spectrum, label='Mean Reflectance')
    plt.fill_between(wavelengths, avg_spectrum - std_spectrum, avg_spectrum + std_spectrum, alpha=0.3, label='Std Dev')
    plt.title("Average Reflectance Spectrum with Standard Deviation")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.grid(True)
    plt.show()

    output_data = np.column_stack((wavelengths, avg_spectrum, std_spectrum))
    output_path = '/Users/rosamariorduna/Downloads/binandhdr_folder/average_spectrum_with_std.csv'
    np.savetxt(output_path, output_data, delimiter=',', header='Wavelength (nm),Mean Reflectance,Std Dev', comments='')
    print(f"Spectrum with standard deviation saved to: {output_path}")

# Connect widgets
low_slider.on_changed(update)
high_slider.on_changed(update)
gain_slider.on_changed(update)
offset_slider.on_changed(update)

plt.show()
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

selector = None
img_data = img.load()
fig_reflectance, ax_reflectance = plt.subplots()

def onselect(verts):
    global img_data
    print("Polygon vertices selected:")
    path = Path(verts)
    y_pixels, x_pixels = np.meshgrid(np.arange(img_data.shape[0]), np.arange(img_data.shape[1]), indexing='ij')
    x_flat = x_pixels.flatten()
    y_flat = y_pixels.flatten()
    coords = np.vstack((y_flat, x_flat)).T
    mask = path.contains_points(coords).reshape(img_data.shape[0], img_data.shape[1])

    roi_pixels = img_data[mask]
    if roi_pixels.size == 0:
        print("No pixels selected!")
        return

    mean_spectrum = np.mean(roi_pixels, axis=0)
    std_spectrum = np.std(roi_pixels, axis=0)

    # Plot
    ax_reflectance.clear()
    bands = np.arange(len(mean_spectrum))
    ax_reflectance.plot(bands, mean_spectrum, label='Mean Reflectance')
    ax_reflectance.fill_between(bands, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, alpha=0.3, label='±1 SD')
    ax_reflectance.set_title("ROI Reflectance")
    ax_reflectance.set_xlabel("Band")
    ax_reflectance.set_ylabel("Reflectance")
    ax_reflectance.legend()
    fig_reflectance.canvas.draw()




    # Save to CSV
    output_dir = "../OUTPUT"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "roi_reflectance.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Band", "Mean Reflectance", "Standard Deviation"])
        for i, (mean, std) in enumerate(zip(mean_spectrum, std_spectrum)):
            writer.writerow([i, mean, std])
    print(f"ROI reflectance saved to: {csv_file}")

    # Save plot
    plot_file = os.path.join(output_dir, "roi_reflectance_plot.png")
    fig_reflectance.savefig(plot_file)
    print(f"Reflectance plot saved to: {plot_file}")



def on_reset(event):
    global selector, ax_reflectance, fig_reflectance
    print("Resetting polygon and reflectance plot...")
    if selector:
        selector.disconnect_events()
        selector = None
    ax_reflectance.clear()
    ax_reflectance.set_title("ROI Reflectance")
    ax_reflectance.set_xlabel("Band")
    ax_reflectance.set_ylabel("Reflectance")
    fig_reflectance.canvas.draw()


btn_reset = Button(ax_reset, 'Reset')
btn_reset.on_clicked(on_reset)
btn_save = Button(ax_save, 'Save')
btn_save.on_clicked(on_save_rgb)
btn_continue = Button(ax_continue, 'Continue')
btn_continue.on_clicked(on_continue)
radio.on_clicked(change_band)