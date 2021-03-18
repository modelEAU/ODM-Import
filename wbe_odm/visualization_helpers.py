from pyproj import Geod
import math
import geojson
import numpy as np
import pandas as pd
import random
from shapely.geometry import shape, Point, LineString


def find_time_columns_to_merge(df):
    categories = {}
    for col in df.columns.to_list():
        col = col.lower()
        if "date" not in col:
            continue
        info = col.split(".")
        for bit in info:
            if "date" in bit:
                if bit in categories:
                    categories[bit].append(col)
                else:
                    categories[bit] = [col]
    return categories


def add_missing_columns(df, needed_names):
    existing_names = df.columns.to_list()
    for name in needed_names:
        if name not in existing_names:
            df[name] = None
    return df


def recombine_times(df):
    categories = find_time_columns_to_merge(df)
    df = add_missing_columns(df, categories)


def create_dummy_polygons(filepaths):
    polygons = {}
    default_polygon = {
        "polygonID": None,
        "name": None,
        "pop": None,
        "type": None,
        "wkt": None,
        "file": None,
        "link": None
    }
    for i, file in enumerate(filepaths):
        polygon = default_polygon.copy()
        with open(file, "r") as f:
            polygon["wkt"] = f.read()
        polygon["name"] = file.split("/")[-1].split(".")[-2]
        polygon["pop"] = random.randint(250000, 350000)
        polygon["type"] = "swrCat"
        polygon["polygonID"] = polygon["name"]
        polygons[i] = polygon
    return pd.DataFrame.from_dict(polygons, orient="index")


def get_map_center(geo_json):
    default_center = {'lat': 46.731183, "lon": -71.321217}
    x_s = []
    y_s = []
    if geo_json is None:
        return None
    if len(geo_json["features"]) == 0:
        return None
    for feat in geo_json["features"]:
        geometry = feat["geometry"]
        if not geometry:
            continue
        # convert the geometry to shapely
        geom = shape(geometry)
        # obtain the coordinates of the feature's centroid
        x_s.append(geom.centroid.x)
        y_s.append(geom.centroid.y)
    if len(x_s) == 0:
        return default_center
    x_m = sum(x_s) / len(x_s)
    y_m = sum(y_s) / len(y_s)
    return {"lat": y_m, "lon": x_m}


def points_to_meters(point_1, point_2):
    line_string = LineString([point_1, point_2])
    geod = Geod(ellps="WGS84")
    return geod.geometry_length(line_string)


def get_bounding_box(geometry):
    coords = np.array(list(geojson.utils.coords(geometry)))
    return coords[:, 0].min(), coords[:, 0].max(),\
        coords[:, 1].min(), coords[:, 1].max()


def interpolate_zoom(row, low_lat, high_lat, lat):
    high_label = "Latitude " + str(high_lat)
    low_label = "Latitude " + str(low_lat)
    x1, x2 = low_lat, high_lat
    y1, y2 = row[low_label], row[high_label]
    return interpolate(x1, x2, y1, y2, lat)


def interpolate(x1, x2, y1, y2, x):
    return y1 + (x - x1) * (y2 - y1)/(x2 - x1)


def find_neighbours(value, df, colname):
    exactmatch = df[df[colname] == value]
    if not exactmatch.empty:
        return exactmatch.index
    else:
        lowerneighbour_ind = df[df[colname] < value][colname].idxmax()
        upperneighbour_ind = df[df[colname] > value][colname].idxmin()
        return [lowerneighbour_ind, upperneighbour_ind]


def get_zoom_level(geo_json, map_height_px):
    default_bounding_box = (
        -71.383618, 46.746301, -71.168241, 46.840914
    )

    bounding_box = get_bounding_box(geo_json)
    if bounding_box is None:
        bounding_box = default_bounding_box
    center_lon = (bounding_box[0] + bounding_box[1]) / 2
    center_lat = (bounding_box[2] + bounding_box[3]) / 2
    point_1 = Point(center_lon, bounding_box[2])
    point_2 = Point(center_lon, bounding_box[3])

    distance = points_to_meters(point_1, point_2)
    required_pixel_density = distance/map_height_px * 2

    mapbox_lat_low = math.floor(abs(center_lat)/20) * 20
    mapbox_lat_high = math.ceil(abs(center_lat)/20) * 20
    if mapbox_lat_high > 80:
        mapbox_lat_high = 80
    mapbox_zoom = pd.read_csv(
        "/workspaces/ODM Import/mapbox_zoom.csv",
        index_col="Zoom level"
    )

    mapbox_zoom["calc"] = mapbox_zoom.apply(
        lambda row:
            interpolate_zoom(
                row, mapbox_lat_low, mapbox_lat_high, center_lat),
        axis=1)

    [zoom_min, zoom_max] = find_neighbours(
        required_pixel_density, mapbox_zoom, "calc")

    y1, y2 = zoom_max, zoom_min
    x1 = mapbox_zoom["calc"].iloc[zoom_max]
    x2 = mapbox_zoom["calc"].iloc[zoom_min]
    zoom_level = interpolate(
        x1, x2, y1, y2,
        required_pixel_density
    )
    return zoom_level

