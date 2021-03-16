import pandas as pd
import random
from shapely.geometry import asShape


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
    x_s = []
    y_s = []
    if len(geo_json["features"]) == 0:
        return None
    for feat in geo_json["features"]:
        # convert the geometry to shapely
        geom = asShape(feat["geometry"])
        # obtain the coordinates of the feature's centroid
        x_s.append(geom.centroid.x)
        y_s.append(geom.centroid.y)
    x_m = sum(x_s) / len(x_s)
    y_m = sum(y_s) / len(y_s)
    return {"lat": y_m, "lon": x_m}
