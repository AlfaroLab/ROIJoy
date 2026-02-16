"""Dash layout for ROIJoy multi-panel interface."""
import dash
from dash import html, dcc
import plotly.graph_objects as go


def make_empty_figure():
    """Create an empty Plotly figure with dark theme for image display."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0d14",
        plot_bgcolor="#0a0d14",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1], scaleanchor="x"),
        dragmode="drawclosedpath",
        newshape=dict(
            line=dict(color="#00e5cc", width=2),
            fillcolor="rgba(0, 229, 204, 0.08)",
        ),
    )
    return fig


def make_spectrum_figure():
    """Create an empty spectrum comparison figure."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0d14",
        plot_bgcolor="#0a0d14",
        margin=dict(l=60, r=20, t=30, b=40),
        xaxis_title="Wavelength (nm)",
        yaxis_title="Reflectance",
        title=dict(text="SPECTRUM COMPARISON", font=dict(size=11)),
        height=280,
        xaxis=dict(
            gridcolor="#1c2233",
            zerolinecolor="#1c2233",
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            gridcolor="#1c2233",
            zerolinecolor="#1c2233",
            tickfont=dict(size=9),
        ),
        legend=dict(
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
    )
    return fig


MAX_PANELS = 6


def create_layout():
    """Build the full Dash layout."""
    # Sidebar
    sidebar = html.Div(className="sidebar", children=[
        html.H1("ROIJoy"),

        html.H3("Load Images"),
        dcc.Input(
            id="file-path-input",
            type="text",
            placeholder="/path/to/file.hdr",
            style={"width": "100%", "marginBottom": "8px",
                   "background": "#22252e", "border": "1px solid #2a2d36",
                   "color": "#e0e0e0", "padding": "8px", "borderRadius": "4px",
                   "boxSizing": "border-box"},
        ),
        html.Button("Load File", id="btn-load-file", className="dash-button",
                     style={"width": "100%", "marginBottom": "8px"}),
        html.Div(id="loaded-files-list"),

        html.H3("RGB Visualization"),
        dcc.RadioItems(
            id="rgb-mode",
            options=[
                {"label": " Band Resampling", "value": "resample"},
                {"label": " Bands 54, 32, 22", "value": "bands_default"},
                {"label": " Bands 20, 40, 60", "value": "bands_alt"},
            ],
            value="resample",
            style={"fontSize": "0.85em"},
            className="radio-items",
        ),

        html.H3("Contrast"),
        html.Label("Low %", style={"fontSize": "0.8em", "color": "#888"}),
        dcc.Slider(id="low-pct", min=0, max=10, value=1, step=0.5,
                   marks=None, tooltip={"placement": "right"}),
        html.Label("High %", style={"fontSize": "0.8em", "color": "#888"}),
        dcc.Slider(id="high-pct", min=90, max=100, value=99, step=0.5,
                   marks=None, tooltip={"placement": "right"}),
        html.Label("Gain", style={"fontSize": "0.8em", "color": "#888"}),
        dcc.Slider(id="gain", min=0.5, max=2.0, value=1.0, step=0.05,
                   marks=None, tooltip={"placement": "right"}),
        html.Label("Offset", style={"fontSize": "0.8em", "color": "#888"}),
        dcc.Slider(id="offset", min=-0.5, max=0.5, value=0.0, step=0.05,
                   marks=None, tooltip={"placement": "right"}),

        html.H3("ROI Matching"),
        dcc.RadioItems(
            id="matching-mode",
            options=[
                {"label": " Feature-based (ORB)", "value": "feature"},
                {"label": " Copy coordinates", "value": "copy"},
                {"label": " Off", "value": "off"},
            ],
            value="feature",
            style={"fontSize": "0.85em"},
            className="radio-items",
        ),

        html.H3("Tools"),
        html.Button("Export All", id="btn-export", className="dash-button",
                     style={"width": "100%", "marginBottom": "8px"}),

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
            style={"display": "none"},
            children=[
                html.Div(className="panel-header", children=[
                    html.Span(id=f"panel-title-{i}", children=f"Image {i + 1}"),
                    html.Button("\u00d7", id=f"panel-close-{i}",
                                style={"background": "none", "border": "none",
                                       "color": "#888", "cursor": "pointer",
                                       "fontSize": "1.2em"}),
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
            html.Div(id="roi-table-content", children="No ROIs yet. Load images and draw polygons to get started."),
        ]),
    ])

    # State stores
    stores = [
        dcc.Store(id="image-data-store", data={}),
        dcc.Store(id="roi-store", data=[]),
        dcc.Store(id="active-panel", data=None),
    ]

    return html.Div(children=[sidebar, main] + stores)
