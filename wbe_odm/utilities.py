import json
from functools import reduce

import numpy as np
import pandas as pd
from geojson_rewind import rewind
from geomet import wkt


def get_midpoint_time(date1, date2):
    if pd.isna(date1) or pd.isna(date2):
        return pd.NaT
    return date1 + (date2 - date1)/2


def get_plot_datetime(df):
    # grb -> "dateTime"
    # ps and cp -> if start and end are present: midpoint
    # ps and cp -> if only end is present: end
    df["Sample.plotDate"] = pd.NaT
    grb_filt = df["Sample.collection"].str.contains("grb")
    s_filt = ~df["Sample.dateTimeStart"].isna()
    e_filt = ~df["Sample.dateTimeEnd"].isna()

    df.loc[grb_filt, "Sample.plotDate"] = df.loc[grb_filt, "Sample.dateTime"]
    df.loc[s_filt & e_filt, "Sample.plotDate"] = df.apply(
        lambda row: get_midpoint_time(
            row["Sample.dateTimeStart"], row["Sample.dateTimeEnd"]
        ),
        axis=1
    )
    df.loc[
        e_filt & ~s_filt, "Sample.plotDate"] = df.loc[
            e_filt & ~s_filt, "Sample.dateTimeEnd"]
    return df["Sample.plotDate"]

def reduce_dt(x, y):
    if pd.isna(x) and pd.isna(y):
        return pd.NaT
    elif pd.isna(x):
        return y
    elif not pd.isna(y):
        return x
    return pd.NaT


def reduce_text(x, y):
    if x is None and y is None:
        return ""
    elif x is None:
        return y
    elif y is None:
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
    elif pd.isna(x):
        return y
    elif pd.isna(y):
        return x
    return x+y/2


def reduce_by_type(series):
    data_type = str(series.dtype)
    name = series.name
    if "datetime" in data_type:
        return reduce(reduce_dt, series)

    if "object" in data_type:
        return reduce(reduce_text, series)

    if data_type in ["float64", "int"]:
        return reduce(reduce_nums, series)
    else:
        raise TypeError(f"could not parse series of dtype {name}")


def convert_wkt_to_geojson(s):
    if s in ["-", ""]:
        return None  # {"type":"Polygon", "coordinates":None}
    geojson_feature = json.loads(json.dumps(wkt.loads(s)))
    geojson_feature = rewind(geojson_feature, rfc7946=False)
    return geojson_feature


UNKNOWN_REGEX = r"$^|n\.?[a|d|/|n]+\.?|^-$|unk.*|none"


def get_data_types():
    url = "https://raw.githubusercontent.com/Big-Life-Lab/covid-19-wastewater/main/site/Variables.csv"  # noqa
    variables = pd.read_csv(url)
    variables["variableName"] = variables["variableName"].str.lower()
    variables["variableType"] = variables["variableType"]\
        .replace(r"date(time)?", "datetime64[ns]", regex=True) \
        .replace("boolean", "bool") \
        .replace("float", "float64") \
        .replace("integer", "int64") \
        .replace("blob", "object") \
        .replace("category", "string")

    return variables\
        .groupby("tableName")[['variableName', 'variableType']] \
        .apply(lambda x: x.set_index('variableName').to_dict(orient='index')) \
        .to_dict()


def get_table_fields(table_name):
    url = "https://raw.githubusercontent.com/Big-Life-Lab/covid-19-wastewater/main/site/Variables.csv"  # noqa
    variables = pd.read_csv(url)
    return variables.loc[variables["tableName"] == table_name, "variableName"]
