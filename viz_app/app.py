import dash
import dash_core_components as dcc
import dash_html_components as html
import glob
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from odm import Odm, CustomEncoder
from odm import visualization_helpers

pd.options.display.max_columns = None
pio.templates.default = "plotly_white"

def draw_map(sample_data, geo):
    fig = px.choropleth_mapbox(
        sample_data,
        geojson=geo,
        locations="Site.polygonID",
        featureidkey="properties.polygonID",
        custom_data=["Site.name"],
        color="Site.name",
        labels={"name": "Sampling Location"},
        center=visualization_helpers.get_map_center(geo),
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
    dcc.Store(id='odm-store'),
    dcc.Store(id="sewershed-store"),
    dcc.Store(id="sample-store"),
    dcc.Store(id="site-store"),

    html.H1(
        "Data exploration - COVID Wastewater data",
        style={"textAlign": "center"}
    ),
    html.Br(),
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Button(
                'Upload Excel Template File',
                id='upload-button',
                className='button-primary',
            ),
            multiple=True,
        ),
    ]),
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
        dcc.Graph(id='map-1', figure=draw_map(sample_data=None, geo=None)),
    ], style={'width': '45%', 'display': 'inlineBlock', 'float': 'right'}),
    html.Br(),
])


def parse_uploaded_files(contents, filename, date):
    pass

# Define callback to parse the uploaded file(s)
@app.callback(
    Output('odm-store', 'data'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
     State('odm-store', 'data')]
)
def read_uploaded_excel(contents, filename):
    if contents is None:
        raise PreventUpdate
    serialized = json.dumps(sensors, indent=4, cls=Sensors.CustomEncoder)
    return serialized

# Define callback to update graphs
@app.callback(
    Output('timeseries-1', 'figure'),
    [Input("x-dropdown-1", "value"),
     Input("y-dropdown-1", "value"),
     Input("sewershed-store", "data")])
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
    Output("sewershed-store", "data"),
    [Input('map-1', 'clickData'),
     Input('sample-store', 'data')])
def filter_by_clicked_sewershed(clickData, samples):
    if clickData is None:
        return None
    point = clickData["points"][0]
    site_name = point["customdata"][0]
    return samples.loc[samples["Site.name"] == site_name] \
        .to_json(date_format='iso')


@app.callback(
    [Output("x-dropdown-1", "options"),
     Output('y-dropdown-1', 'options')],
    [Input('sewershed-store', 'data'),
     Input('sample-store', 'data')])
def update_dropdowns_1(sewershed_data, samples_data):
    if not samples_data:
        return None
    df = pd.read_json(samples_data) if sewershed_data is None else pd.read_json(sewershed_data)
    x_options = [
        {'label': c, 'value': c}
        for c in df.columns.to_list()
    ]
    y_options = x_options
    return x_options, y_options



if __name__ == "__main__":
    # Set up test data
    # get data
    filename = "/workspaces/ODM-Import/Data/Ville de Québec 202102.xlsx"
    model = Odm()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    fake_wkt_path = "/".join(dir_path.split("/")[:-1])

    fake_wkt_path += "/Data/polygons/*.wkt"
    polygon_files = glob.glob(fake_wkt_path)

    fake_poly = visualization_helpers.create_dummy_polygons(polygon_files)
    fake_poly = table_parsers.parse_polygon(fake_poly)

    model.data["Polygon"] = fake_poly
    model.load_from_excel(filename)


    model.ingest_geometry()
    samples = model.combine_per_sample()

    #edit the samples data so that they point to the dummy polygons
    polys = model.data["Polygon"]
    east_poly_id = polys.loc[polys["Polygon.name"].str.contains("east"), ["Polygon.polygonID"]].values[0][0]
    west_poly_id = polys.loc[polys["Polygon.name"].str.contains("west"), ["Polygon.polygonID"]].values[0][0]

    def fill_poly_id(row):
        if "quebec est" in row["Site.name"].lower():
            return east_poly_id
        elif "quebec ouest" in row["Site.name"].lower():
            return  west_poly_id
        return np.nan

    samples["Site.polygonID"] = samples.apply(lambda x: fill_poly_id(x), axis=1)



    geo = model.geo
    map_center = visualization_helpers.get_map_center(geo)

    app.run_server(debug=True)  # inline
