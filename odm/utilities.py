from functools import reduce
import numpy as np
import pandas as pd
from geojson_rewind import rewind
from pygeoif import geometry
import geodaisy.converters as convert
import constants
import table_parsers

def reduce_dt(x, y):
    if pd.isna(x) and pd.isna(y):
        return pd.NaT
    elif pd.isna(x) and not pd.isna(y):
        return y
    elif not pd.isna(x) and pd.isna(y):
        return x
    else:
        return pd.NaT


def reduce_text(x, y):
    if x is None and y is None:
        return ""
    if x is None:
        return y
    if y is None:
        return x
    if x == "" and y == "":
        return ""
    elif x == "":
        return y
    elif y == "" or x == y:
        return x
    else:
        return ";".join([x, y])


def reduce_nums(x, y):
    if pd.isna(x) and pd.isna(y):
        return np.nan
    elif pd.isna(x) and not pd.isna(y):
        return y
    elif not pd.isna(x) and pd.isna(y):
        return x
    else:
        return x+y/2


def remove_columns(cols, df):
    to_remove = [col for col in cols if col in df.columns]
    df.drop(to_remove, axis=1, inplace=True)
    return df


def reduce_by_type(series):
    data_type = str(series.dtype)
    name = series.name
    if "datetime" in data_type:
        return reduce(reduce_dt, series)

    if "object" in data_type:
        return reduce(reduce_text, series)

    if "float64" in data_type or "int" in data_type:
        return reduce(reduce_nums, series)
    else:
        raise TypeError(f"could not parse series of dtype {name}")


def convert_wkt_to_geojson(s):
    if s in ["-", ""]:
        return None  # {"type":"Polygon", "coordinates":None}
    from_wkt = geometry.from_wkt(s)
    geo_interface = from_wkt.__geo_interface__
    geojson_feature = convert.geo_interface_to_geojson(geo_interface)
    geojson_feature = rewind(geojson_feature, rfc7946=False)
    return geojson_feature


def get_attribute_from_name(name):
    for attribute, dico in TABLE_LOOKUP.items():
        if name in dico.values():
            return attribute
    return None


def parse_types(odm_name, series):
    def clean_bool(x):
        return str(x)\
            .lower()\
            .strip()\
            .replace("yes", "true")\
            .replace("oui", "true")\
            .replace("non", "false")\
            .replace("no", "false")

    def clean_string(x):
        if str(x).lower() in constants.UNKNOWN_TOKENS:
            x = ""
        return str(x).lower().strip()

    def clean_num(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    def clean_category(x, name):
        x = clean_string(x)
        return f"{name} unknown" if x == "" else x
    name = series.name
    desired_type = constants.TYPES[odm_name].get(name, "string")
    if desired_type == "bool":
        series = series.apply(lambda x: clean_bool(x))
    elif desired_type == "string":
        series = series.apply(lambda x: clean_string(x))
    elif desired_type in ["inst64", "float64"]:
        series = series.apply(lambda x: clean_num(x))
    elif desired_type == "category":
        series = series.apply(lambda x: clean_category(x, name))
    series = series.astype(desired_type)
    return series


TABLE_LOOKUP = {
    "ww_measure": {
        "odm_name": "WWMeasure",
        "excel_name": "WWMeasure",
        "parser": table_parsers.parse_ww_measure,
    },
    "site_measure": {
        "odm_name": "SiteMeasure",
        "excel_name": "SiteMeasure",
        "parser": table_parsers.parse_site_measure,
    },
    "sample": {
        "odm_name": "Sample",
        "excel_name": "Sample",
        "parser": table_parsers.parse_sample,
    },
    "site": {
        "odm_name": "Site",
        "excel_name": "Site",
        "parser": table_parsers.parse_site,
    },
    "polygon": {
        "odm_name": "Polygon",
        "excel_name": "Polygon",
        "parser":table_parsers.parse_polygon,
    },
    "cphd": {
        "odm_name": "CovidPublicHealthData",
        "excel_name": "CPHD",
        "parser": table_parsers.parse_cphd,
    },
}