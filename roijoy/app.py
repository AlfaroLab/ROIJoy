"""ROIJoy Dash application entry point."""
import dash
from dash import html

app = dash.Dash(__name__, title="ROIJoy")
app.layout = html.Div("ROIJoy is running!")

if __name__ == "__main__":
    app.run(debug=True, port=8050)
