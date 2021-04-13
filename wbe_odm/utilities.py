import json
from functools import reduce

import numpy as np
import pandas as pd
from geojson_rewind import rewind
from geomet import wkt


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
    geojson_feature = json.loads(json.dumps(wkt.loads(s)))
    geojson_feature = rewind(geojson_feature, rfc7946=False)
    return geojson_feature
