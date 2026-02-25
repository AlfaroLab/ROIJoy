"""ROIJoy Dash application entry point."""
import argparse
import dash
from roijoy.layout import create_layout

app = dash.Dash(
    __name__,
    title="ROIJoy",
    suppress_callback_exceptions=True,
)

app.layout = create_layout()

# Register callbacks (importing the module triggers @callback decorators)
from roijoy import callbacks  # noqa: F401, E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROIJoy hyperspectral ROI selector")
    parser.add_argument("--port", type=int, default=8050, help="Port to run on (default: 8050)")
    args = parser.parse_args()
    app.run(debug=True, dev_tools_ui=False, port=args.port)
