import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.io as pio
from dash.dependencies import Input, Output

import odm

pd.options.display.max_columns = None
pio.templates.default = "plotly_white"
# get data
filename = "Data/Template - Data Model - 20210127.xls"
model = odm.Odm()
model.read_excel(filename)
samples = model.combine_per_sample()
geo = model.geo
map_center = model.map_center


def draw_map():
    fig = px.choropleth_mapbox(
        samples,
        geojson=geo,
        locations="Site.polygonID",
        featureidkey="properties.polygonID",
        custom_data=["Site.name"],
        color="Site.name",
        labels={"name": "Sampling Location"},
        center=map_center,
        opacity=0.5,
        mapbox_style="open-street-map",
        zoom=6,
    )
    fig.update_layout(
        clickmode="event+select",
        height=800,
        legend_orientation="h",
        legend_yanchor="top",
        legend_xanchor="left"
    )
    return fig


def get_timeseries_names(names):
    return [name for name in names if "date" in name.lower()]


def get_values_names(names):
    return [name for name in names if "value" in name.lower()]


# Build App
app = dash.Dash(__name__)  # JupyterDash(__name__)
app.layout = html.Div([
    dcc.Store(id="site-store"),
    html.H1(
        "Data exploration - COVID Wastewater data",
        style={"textAlign": "center"}
    ),
    html.Br(),
    html.Div([
        html.Div([
            html.Label("X-Axis"),
            html.Br(),
            dcc.Dropdown(
                id="x-dropdown-1",
            )],
            style={'width': '45%', 'display': 'inlineBlock', 'float': 'left'}),
        html.Div([
            html.Label("Y-Axis"),
            html.Br(),
            dcc.Dropdown(
                id="y-dropdown-1",
            )],
            style={'width': '45%', 'display': 'inlineBlock', 'float': 'left'}
        ),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),

        dcc.Graph(id='timeseries-1'),
    ], style={'width': '45%', 'display': 'inlineBlock', 'float': 'right'}),
    html.Div([
        dcc.Graph(id='map-1', figure=draw_map()),
    ], style={'width': '45%', 'display': 'inlineBlock', 'float': 'right'}),
    html.Br(),
])


# Define callback to update graphs
@app.callback(
    Output('timeseries-1', 'figure'),
    [Input("x-dropdown-1", "value"),
     Input("y-dropdown-1", "value"),
     Input("site-store", "data")])
def time_series_1(x_col, y_col, data):
    if x_col is None or y_col is None:
        return px.scatter()
    df = samples if data is None else pd.read_json(data)

    return px.scatter(
        df, x=x_col, y=y_col,
        color_continuous_scale="Viridis",
        title=f"{y_col} over {x_col}"
    )


@app.callback(
    Output("site-store", "data"),
    Input('map-1', 'clickData'))
def filter_by_clicked_location(clickData):
    if clickData is None:
        return None
    point = clickData["points"][0]
    site_name = point["customdata"][0]
    return samples.loc[samples["Site.name"] == site_name] \
        .to_json(date_format='iso')


@app.callback(
    Output("y-dropdown-1", "options"),
    Input('site-store', 'data'))
def update_y_dropdown(data):
    df = samples if data is None else pd.read_json(data)
    return [
        {'label': c, 'value': c}
        for c in samples.columns if c in get_values_names(df.columns.to_list())
    ]


@app.callback(
    Output("x-dropdown-1", "options"),
    Input('site-store', 'data'))
def update_x_dropdown(data):
    df = samples if data is None else pd.read_json(data)
    return [
        {'label': c, 'value': c}
        for c in samples.columns
        if c in get_timeseries_names(df.columns.to_list())
    ]


if __name__ == "__main__":
    app.run_server(debug=True)  # inline
