"""Dash callbacks for ROIJoy interactivity."""
import os
import numpy as np
from dash import Input, Output, State, callback, no_update, ctx, html
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from roijoy.envi_io import load_envi, render_rgb, apply_contrast, rgb_to_base64
from roijoy.roi import extract_spectrum, export_spectrum_csv, export_combined_csv
from roijoy.matching import match_roi, copy_roi
from roijoy.layout import make_empty_figure, make_spectrum_figure, make_inset_figure, MAX_PANELS

# Server-side cache for loaded cubes (keyed by panel index)
_cube_cache = {}

# 10 distinct ROI colors — high contrast on dark backgrounds
ROI_COLORS = [
    "#ff6b6b", "#4dabf7", "#51cf66", "#fcc419", "#cc5de8",
    "#20c997", "#ff922b", "#339af0", "#69db7c", "#f06595",
]

LINE_STYLES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex color like '#ff6b6b' to 'rgba(255, 107, 107, alpha)'.

    Plotly does not support 8-digit hex (#RRGGBBAA), so we must use rgba().
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _build_image_figure(cube, wavelengths, rgb_mode, low, high, gain, offset):
    """Render an ENVI cube into a Plotly figure with image overlay."""
    if rgb_mode == "resample":
        rgb = render_rgb(cube, wavelengths, mode="resample")
    elif rgb_mode == "bands_default":
        rgb = render_rgb(cube, wavelengths, mode="bands", band_indices=[54, 32, 22])
    else:
        rgb = render_rgb(cube, wavelengths, mode="bands", band_indices=[20, 40, 60])

    rgb = apply_contrast(rgb, low, high, gain, offset)
    b64 = rgb_to_base64(rgb)

    fig = make_empty_figure()
    fig.add_layout_image(
        source=b64,
        xref="x", yref="y",
        x=0, y=0,
        sizex=cube.shape[1], sizey=cube.shape[0],
        sizing="stretch",
        layer="below",
    )
    fig.update_xaxes(range=[0, cube.shape[1]], gridcolor="#1c2233")
    fig.update_yaxes(range=[cube.shape[0], 0], gridcolor="#1c2233")
    return fig


# ---------------------------------------------------------------------------
# Callback: Sync dropdown selection to text input
# ---------------------------------------------------------------------------
@callback(
    Output("file-path-input", "value"),
    Input("file-dropdown", "value"),
    prevent_initial_call=True,
)
def sync_dropdown_to_input(dropdown_value):
    """When user selects a file from the dropdown, put the path in the input."""
    if dropdown_value:
        return dropdown_value
    raise PreventUpdate


# ---------------------------------------------------------------------------
# Callback: Load ENVI file by path
# ---------------------------------------------------------------------------
@callback(
    [Output(f"panel-container-{i}", "style") for i in range(MAX_PANELS)] +
    [Output(f"panel-title-{i}", "children") for i in range(MAX_PANELS)] +
    [Output(f"image-graph-{i}", "figure") for i in range(MAX_PANELS)] +
    [Output("image-data-store", "data"),
     Output("status-msg", "children"),
     Output("loaded-files-list", "children")],
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
    if not path or not path.strip().endswith('.hdr'):
        raise PreventUpdate

    path = path.strip()
    if not os.path.exists(path):
        styles = [no_update] * MAX_PANELS
        titles = [no_update] * MAX_PANELS
        figures = [no_update] * MAX_PANELS
        return styles + titles + figures + [no_update, f"File not found: {path}", no_update]

    if current_data is None:
        current_data = {}

    panel_idx = len(current_data)
    if panel_idx >= MAX_PANELS:
        styles = [no_update] * MAX_PANELS
        titles = [no_update] * MAX_PANELS
        figures = [no_update] * MAX_PANELS
        return styles + titles + figures + [no_update, "Maximum 6 images loaded.", no_update]

    # Load the ENVI file
    cube, wavelengths = load_envi(path)
    _cube_cache[panel_idx] = {"cube": cube, "wavelengths": wavelengths, "path": path}

    fig = _build_image_figure(cube, wavelengths, rgb_mode, low, high, gain, offset)

    # Build outputs
    styles = [no_update] * MAX_PANELS
    titles = [no_update] * MAX_PANELS
    figures = [no_update] * MAX_PANELS

    styles[panel_idx] = {"display": "block"}
    titles[panel_idx] = os.path.basename(path).replace('.hdr', '')
    figures[panel_idx] = fig

    current_data[str(panel_idx)] = {
        "path": path,
        "shape": list(cube.shape),
        "n_wavelengths": len(wavelengths),
    }

    file_list = html.Ul([
        html.Li(
            os.path.basename(d.get("path", "")).replace(".hdr", ""),
        )
        for d in current_data.values()
    ], style={"padding": "0 20px", "margin": "4px 0"})

    return (styles + titles + figures +
            [current_data, f"Loaded: {os.path.basename(path)}", file_list])


# ---------------------------------------------------------------------------
# Callback: Contrast / RGB mode adjustment
# ---------------------------------------------------------------------------
@callback(
    [Output(f"image-graph-{i}", "figure", allow_duplicate=True) for i in range(MAX_PANELS)],
    [Input("rgb-mode", "value"),
     Input("low-pct", "value"),
     Input("high-pct", "value"),
     Input("gain", "value"),
     Input("offset", "value")],
    State("image-data-store", "data"),
    [State(f"image-graph-{i}", "figure") for i in range(MAX_PANELS)],
    prevent_initial_call=True,
)
def update_contrast(rgb_mode, low, high, gain, offset, image_data, *current_figures):
    """Re-render all loaded images with new contrast/RGB settings."""
    if not image_data:
        raise PreventUpdate

    figures = []
    for i in range(MAX_PANELS):
        if i in _cube_cache:
            cache = _cube_cache[i]
            fig = _build_image_figure(
                cache["cube"], cache["wavelengths"],
                rgb_mode, low, high, gain, offset,
            )
            # Preserve existing shapes (drawn polygons)
            if current_figures[i] and isinstance(current_figures[i], dict):
                old_shapes = current_figures[i].get("layout", {}).get("shapes", [])
                if old_shapes:
                    fig.update_layout(shapes=old_shapes)
            figures.append(fig)
        else:
            figures.append(no_update)
    return figures


# ---------------------------------------------------------------------------
# Callback: Polygon drawing & ROI extraction
# ---------------------------------------------------------------------------
@callback(
    Output("roi-store", "data"),
    Output("spectrum-graph", "figure"),
    Output("roi-table-content", "children"),
    [Input(f"image-graph-{i}", "relayoutData") for i in range(MAX_PANELS)],
    State("roi-store", "data"),
    State("image-data-store", "data"),
    State("matching-mode", "value"),
    prevent_initial_call=True,
)
def on_shape_drawn(*args):
    """Handle polygon drawing on any image panel."""
    relayout_data_list = args[:MAX_PANELS]
    roi_data, image_data, matching_mode = args[MAX_PANELS:]

    if roi_data is None:
        roi_data = []

    triggered = ctx.triggered_id
    if triggered is None:
        raise PreventUpdate

    # Find which panel triggered
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
    new_shapes = None
    for key in relayout:
        if key == "shapes":
            new_shapes = relayout[key]
            break

    if new_shapes is None or not new_shapes:
        raise PreventUpdate

    # Process the newest shape
    latest_shape = new_shapes[-1]
    if latest_shape.get("type") != "path":
        raise PreventUpdate

    path_str = latest_shape.get("path", "")
    vertices = _parse_svg_path(path_str)
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

            for idx_str in image_data:
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
                        "confirmed": False,
                        "method": result["method"],
                    }

    roi_data.append(new_roi)

    spectrum_fig = _build_spectrum_figure(roi_data)
    table = _build_roi_table(roi_data, image_data or {})

    return roi_data, spectrum_fig, table


# ---------------------------------------------------------------------------
# Callback: Synchronized zoom/pan across panels
# ---------------------------------------------------------------------------
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
    figure_list = args[MAX_PANELS:2 * MAX_PANELS]
    image_data = args[2 * MAX_PANELS]

    triggered = ctx.triggered_id
    if triggered is None:
        raise PreventUpdate

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

    # Only sync actual zoom changes, not shape drawing
    x_range = None
    y_range = None
    if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
        x_range = [relayout["xaxis.range[0]"], relayout["xaxis.range[1]"]]
    if "yaxis.range[0]" in relayout and "yaxis.range[1]" in relayout:
        y_range = [relayout["yaxis.range[0]"], relayout["yaxis.range[1]"]]

    if x_range is None and y_range is None:
        raise PreventUpdate

    outputs = []
    for i in range(MAX_PANELS):
        if image_data and str(i) in image_data and i != source_idx:
            fig = go.Figure(figure_list[i]) if figure_list[i] else make_empty_figure()
            if x_range:
                fig.update_xaxes(range=x_range)
            if y_range:
                fig.update_yaxes(range=y_range)
            outputs.append(fig)
        else:
            outputs.append(no_update)

    outputs[source_idx] = no_update
    return outputs


# ---------------------------------------------------------------------------
# Callback: ROI table row click -> select ROI
# ---------------------------------------------------------------------------
@callback(
    Output("selected-roi", "data"),
    Output("roi-table-content", "children", allow_duplicate=True),
    Input("roi-table-content", "n_clicks"),
    State("selected-roi", "data"),
    State("roi-store", "data"),
    State("image-data-store", "data"),
    prevent_initial_call=True,
)
def on_roi_table_click(n_clicks, current_selected, roi_data, image_data):
    """Toggle ROI selection when the table area is clicked.

    We use a simple cycling approach: each click selects the next ROI,
    or deselects if we've cycled through all.
    """
    if not roi_data:
        raise PreventUpdate

    # Cycle through ROI IDs: None -> 1 -> 2 -> ... -> None
    roi_ids = [r["id"] for r in roi_data]
    if current_selected is None:
        new_selected = roi_ids[0]
    else:
        try:
            idx = roi_ids.index(current_selected)
            if idx + 1 < len(roi_ids):
                new_selected = roi_ids[idx + 1]
            else:
                new_selected = None  # Deselect
        except ValueError:
            new_selected = roi_ids[0]

    table = _build_roi_table(roi_data, image_data or {}, selected_roi_id=new_selected)
    return new_selected, table


# ---------------------------------------------------------------------------
# Callback: Update inset spectrum overlays based on selected ROI
# ---------------------------------------------------------------------------
@callback(
    [Output(f"inset-container-{i}", "style") for i in range(MAX_PANELS)] +
    [Output(f"inset-graph-{i}", "figure") for i in range(MAX_PANELS)],
    Input("selected-roi", "data"),
    State("roi-store", "data"),
    State("image-data-store", "data"),
    prevent_initial_call=True,
)
def update_inset_spectra(selected_roi_id, roi_data, image_data):
    """Show/hide per-panel inset spectrum for the selected ROI."""
    inset_styles = [{"display": "none"}] * MAX_PANELS
    inset_figures = [make_inset_figure()] * MAX_PANELS

    if selected_roi_id is None or not roi_data:
        return inset_styles + inset_figures

    # Find the selected ROI
    selected_roi = None
    for roi in roi_data:
        if roi["id"] == selected_roi_id:
            selected_roi = roi
            break

    if selected_roi is None:
        return inset_styles + inset_figures

    color = selected_roi["color"]

    for panel_str, panel_roi in selected_roi["panels"].items():
        panel_idx = int(panel_str)
        if panel_idx >= MAX_PANELS:
            continue

        cache = _cube_cache.get(panel_idx)
        if cache is None:
            continue

        result = extract_spectrum(
            cache["cube"], cache["wavelengths"], panel_roi["vertices"]
        )

        wl = result["wavelengths"].tolist()
        mean = result["mean"].tolist()
        std = result["std"].tolist()

        fig = make_inset_figure()

        # Shaded +/- 1 SD band
        upper = [m + s for m, s in zip(mean, std)]
        lower = [m - s for m, s in zip(mean, std)]

        fig.add_trace(go.Scatter(
            x=wl + wl[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor=_hex_to_rgba(color, 0.13),
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Mean line
        fig.add_trace(go.Scatter(
            x=wl,
            y=mean,
            mode="lines",
            line=dict(color=color, width=1.5),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Status label
        status = "\u2713" if panel_roi.get("confirmed") else "\u223c"
        fig.add_annotation(
            x=0.02, y=0.95,
            xref="paper", yref="paper",
            text=f"<b>ROI {selected_roi_id} {status}</b>",
            showarrow=False,
            font=dict(size=8, color=color),
            xanchor="left", yanchor="top",
        )

        inset_styles[panel_idx] = {"display": "block"}
        inset_figures[panel_idx] = fig

    return inset_styles + inset_figures


# ---------------------------------------------------------------------------
# Callback: Export all ROIs
# ---------------------------------------------------------------------------
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

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(output_dir, exist_ok=True)

    count = 0
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
            count += 1

    combined_path = os.path.join(output_dir, "combined_comparison.csv")
    export_combined_csv(combined_path, roi_data, _cube_cache)

    return f"Exported {count} spectra + combined CSV to output/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_svg_path(path_str: str) -> list[tuple[float, float]]:
    """Parse an SVG path string from Plotly into (x, y) vertices.

    Plotly drawclosedpath produces: 'M100,200L150,250L200,200Z'
    """
    vertices = []
    parts = path_str.replace("M", "").replace("Z", "").split("L")
    for part in parts:
        part = part.strip()
        if "," in part:
            x, y = part.split(",")
            vertices.append((float(x), float(y)))
    return vertices


def _build_spectrum_figure(roi_data: list) -> go.Figure:
    """Build the spectrum comparison figure from all ROIs."""
    fig = make_spectrum_figure()

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

            wl = result["wavelengths"].tolist()
            mean = result["mean"].tolist()
            std = result["std"].tolist()
            trace_name = f"ROI {roi_id} / Img {panel_idx + 1}"

            # Shaded +/- 1 SD band
            upper = [m + s for m, s in zip(mean, std)]
            lower = [m - s for m, s in zip(mean, std)]

            fig.add_trace(go.Scatter(
                x=wl + wl[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.15),
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))

            # Mean line
            fig.add_trace(go.Scatter(
                x=wl,
                y=mean,
                mode="lines",
                name=trace_name,
                line=dict(color=color, dash=LINE_STYLES[panel_idx % len(LINE_STYLES)]),
            ))

    return fig


def _build_roi_table(roi_data: list, image_data: dict, selected_roi_id=None) -> html.Div:
    """Build an HTML table showing ROI status across panels.

    Rows for the selected ROI are highlighted.
    """
    if not roi_data:
        return html.Div("Draw polygons on loaded images to create ROIs.")

    n_panels = len(image_data)
    header_cells = [html.Th("#"), html.Th("")]
    for i in range(n_panels):
        header_cells.append(html.Th(f"IMG {i + 1}"))

    rows = [html.Tr(header_cells)]

    for roi in roi_data:
        is_selected = (roi["id"] == selected_roi_id)
        row_style = {}
        if is_selected:
            row_style = {
                "background": _hex_to_rgba(roi["color"], 0.08),
                "borderLeft": f"2px solid {roi['color']}",
            }

        cells = [
            html.Td(str(roi["id"]), style={"fontWeight": "600", "color": roi["color"]}),
            html.Td(html.Div(style={
                "width": "8px", "height": "8px", "borderRadius": "50%",
                "background": roi["color"], "display": "inline-block",
                "boxShadow": f"0 0 6px {_hex_to_rgba(roi['color'], 0.27)}",
            })),
        ]
        for i in range(n_panels):
            panel_roi = roi["panels"].get(str(i))
            if panel_roi is None:
                cells.append(html.Td("\u2014", style={"color": "#5c6578"}))
            elif panel_roi["confirmed"]:
                cells.append(html.Td("\u2713", style={"color": "#51cf66", "fontWeight": "700"}))
            else:
                cells.append(html.Td("\u223c", style={"color": "#fcc419"}))

        rows.append(html.Tr(cells, style=row_style))

    hint = "Click table to cycle ROI selection" if roi_data else ""
    return html.Div([
        html.Table(rows),
        html.Div(hint, style={
            "fontSize": "10px", "color": "#5c6578", "marginTop": "6px",
            "fontFamily": "var(--font-mono)", "fontStyle": "italic",
        }) if hint else None,
    ])
