import pandas as pd
import glob
import os.path
import random

def find_time_columns_to_merge(df):
    categories = {}
    for col in df.columns.to_list():
        col = col.lower()
        if "date" not in col: continue
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
    df = add_missing_columns(df)

def create_dummy_polygons(filepath):
    polygons = {}
    default_polygon = {"polygonID": None,
     "name": None,
     "pop": None,
     "type": None,
     "wkt": None,
     "file": None,
     "link": None
     }
    complete_path = os.path.join(filepath,"*.wkt")
    polygon_files = glob.glob(complete_path)
    fields = []
    for i, file in enumerate(polygon_files):
        polygon = default_polygon.copy()
        with open(file, "r") as f:
            polygon["wkt"] = f.read()
        polygon["name"] = file.split("/")[-1].split(".")[-2]
        polygon["pop"] = random.randint(250000, 350000)
        polygon["type"] = "swrCat"
        polygon["polygonID"] = polygon["name"]
        polygons[i] = polygon
    return pd.DataFrame.from_dict(polygons, orient="index")
