import json
import re
from functools import reduce

import numpy as np
import pandas as pd
from geojson_rewind import rewind
from geomet import wkt


UNKNOWN_TOKENS = [
    "nan",
    "na",
    "nd"
    "n.d",
    "none",
    "-",
    "unknown",
    "n/a",
    "n/d"
]


def get_data_types():
    url = "https://raw.githubusercontent.com/Big-Life-Lab/covid-19-wastewater/main/site/Variables.csv"  # noqa
    variables = pd.read_csv(url)
    variables["variableName"] = variables["variableName"].str.lower()
    variables["variableType"] = variables["variableType"].apply(
        lambda x: re.sub(r"date(time)?", "datetime64[ns]", x)
    )
    variables["variableType"] = variables["variableType"].apply(
        lambda x:
            x.replace("boolean", "bool")
            .replace("float", "float64")
            .replace("integer", "int64")
            .replace("blob", "object")
    )
    return variables\
        .groupby("tableName")[['variableName', 'variableType']] \
        .apply(lambda x: x.set_index('variableName').to_dict(orient='index')) \
        .to_dict()


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
    geojson_feature = json.dumps(wkt.loads(s))
    geojson_feature = rewind(geojson_feature, rfc7946=False)
    return geojson_feature


def parse_types(table_name, series):
    def clean_bool(x):
        return str(x)\
            .lower()\
            .strip()\
            .replace("yes", "true")\
            .replace("oui", "true")\
            .replace("non", "false")\
            .replace("no", "false")

    def clean_string(x):
        if str(x).lower() in UNKNOWN_TOKENS:
            x = ""
        return str(x).strip()

    def clean_num(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    def clean_category(x, name):
        x = clean_string(x)
        return f"{name} unknown" if x == "" else x

    variable_name = series.name.lower()
    types = get_data_types()
    lookup_table = types[table_name]
    lookup_type = lookup_table.get(variable_name, dict())
    desired_type = lookup_type.get("variableType", "string")
    if desired_type == "bool":
        series = series.apply(lambda x: clean_bool(x))
    elif desired_type == "string" and variable_name != "wkt":
        series = series.apply(lambda x: clean_string(x).lower())
    elif desired_type == "string" and variable_name == "wkt":
        series = series.apply(lambda x: clean_string(x))
    elif desired_type in ["inst64", "float64"]:
        series = series.apply(lambda x: clean_num(x))
    elif desired_type == "category":
        series = series.apply(lambda x: clean_category(x, variable_name))
    series = series.astype(desired_type)
    return series


def keep_only_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a DataFrame that only contains features information
    about a sample that can be used for machine learning.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of joined ODM tables with each row representing a
        sample.

    Returns
    -------
    pd.DataFrame
        A DataFrame without id's notes, or data access columns.
    """
    return df
