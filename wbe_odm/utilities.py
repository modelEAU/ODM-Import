import json
from functools import reduce
import re
import warnings

import numpy as np
import pandas as pd
from geojson_rewind import rewind
import shapely.wkt
import geomet.wkt


def typecast_wide_table(df):
    for col in df.columns:
        name = df[col].name
        if "date" in name or "timestamp" in name:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif "value" in name:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype(str)
    return df


def has_cphd_data(x, uniques):
    if pd.isna(x):
        return None
    ls = x.split(";")
    ls = [i for i in ls if i in uniques]
    return ";".join(ls) if ls else None


def pick_cphd_poly_by_size(x, poly):
    if pd.isna(x):
        return None
    ls = x.split(";")
    areas = []
    for id_ in ls:
        area = poly.loc[poly["Polygon.polygonID"] == id_, "area"]
        areas.append(area)
    min_area = min(areas)
    min_idx = areas.index(min_area)
    return ls[min_idx]


def convert_wkt(x):
    try:
        return shapely.wkt.loads(x)
    except Exception:
        return None


def get_polygon_for_cphd(merged, poly, cphd):
    poly["shape"] = poly["Polygon.wkt"].apply(lambda x: convert_wkt(x))
    poly["area"] = poly["shape"].apply(lambda x: x.area)
    unique_cphd_polys = cphd["CPHD.polygonID"].unique()
    merged["polys_w_cphd"] = merged["Calculated.polygonList"].apply(
        lambda x: has_cphd_data(x, unique_cphd_polys))
    merged["Calculated.polygonIDForCPHD"] = merged["polys_w_cphd"].apply(
        lambda x: pick_cphd_poly_by_size(x, poly))
    poly.drop(columns=["shape", "area"], inplace=True)
    merged.drop(columns=["polys_w_cphd"], inplace=True)
    return merged


def get_encompassing_polygons(row, poly):
    poly["contains"] = poly["shape"].apply(
        lambda x: x.contains(row["temp_point"])
        if x is not None else False)
    poly_ids = poly[
        "Polygon.polygonID"].loc[poly["contains"]].to_list()
    poly.drop(columns=["contains"], inplace=True)
    return ";".join(poly_ids)


def get_midpoint_time(date1, date2):
    if pd.isna(date1) or pd.isna(date2):
        return pd.NaT
    return date1 + (date2 - date1)/2


def clean_grab_datetime(df):
    one_day = pd.to_timedelta("24 hours")
    result_end = ["Calculated.dateTimeEnd"]
    result_start = ["Calculated.dateTimeStart"]
    grab_date = "Sample.dateTime"
    grab_token = "grb"
    collection = "Sample.collection"
    df[result_start] = pd.to_datetime(None)
    df[result_end] = pd.to_datetime(None)

    grb_filt = df[collection].str.contains(grab_token)
    na_filt = ~df[grab_date].isna()
    filt = na_filt & grb_filt
    df2 = df.loc[filt, df.columns.to_list()]
    df2[result_start] = df2[grab_date].dt.normalize()
    df2[result_end] = df2[result_start] + one_day

    df.loc[filt] = df2
    return df


def calc_start_date(end_date, type_):
    if pd.isna(end_date) or pd.isna(type_):
        return pd.NaT
    x = type_
    hours = None
    if re.match(r"cp[tf]p[0-9]+h", x):
        hours = int(x[4:-1])
    elif re.match(r"ps[0-9]+h", x):
        hours = int(x[2:-1])
    if hours is not None:
        interval = pd.to_timedelta(f"{hours}h")
        return end_date - interval
    return pd.NaT


def clean_composite_data_intervals(df):
    """This function implements the following rules for
    associating composite samples with a time interval:
        1. The Composite ends at midnight of the day recorded as the end date.
        2. The start date is back-calculated from the interval of the composite
            and the calculated end date.

    Args:
        df (pd.DataFrame): The (combined) dataframe with the samples to treat.
        It should have the columns:
            "Sample.dateTimeStart",
            "Sample.dateTimeEnd",
            "Sample.collection",

    Returns:
        pd.DataFrame: The entrance DataFrame with modified
        values in the columns:
            "Sample.dateTimeStart",
            "Sample.dateTimeEnd"
    """
    coll = "Sample.collection"
    end = "Sample.dateTimeEnd"
    result_end = "Calculated.dateTimeEnd"
    result_start = "Calculated.dateTimeStart"

    one_day = pd.to_timedelta("23 hours 59 minutes")
    df[result_end] = pd.to_datetime(df[end] + one_day).dt.date
    df[result_start] = df.apply(
        lambda row: calc_start_date(row[result_end], row[coll]), axis=1)
    return df


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
    return (x+y)/2


def reduce_by_type(series):
    if series.empty:
        return np.nan
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
    geojson_feature = json.loads(json.dumps(geomet.wkt.loads(s)))
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


def clean_primary_key(key):
    key = str(key)
    if key.startswith('u'):
        key = key[1:]
    if key[0].isupper():
        key = key[0].lower() + key[1:]
    return key.replace('Ww', 'ww')


def get_primary_key(table_name=None):
    url = "https://raw.githubusercontent.com/Big-Life-Lab/covid-19-wastewater/main/site/Variables.csv"  # noqa
    variables = pd.read_csv(url)
    keys = variables.loc[variables["key"] == "Primary Key", ["tableName", "variableName"]].set_index("tableName")
    keys = keys.apply(lambda x: clean_primary_key(x["variableName"]), axis=1)
    keys = keys.to_dict()
    if table_name is None:
        return keys
    return keys[table_name]


def build_site_specific_dataset(df, site_id):
    if df.empty:
        return df
    filt_site1 = df["Site.siteID"] == site_id
    if "SiteMeasure.siteID" in df.columns:
        filt_site2 = df["SiteMeasure.siteID"] == site_id
        filt_site = filt_site1 | filt_site2
    else:
        filt_site = filt_site1
    df1 = df[filt_site]

    filt_cphd_df = df.loc[filt_site1, "Calculated.polygonIDForCPHD"]
    if not filt_cphd_df.empty:
        cphd_poly_id = str(
            filt_cphd_df.iloc[0]).lower()
        poly_filt = df["CPHD.polygonID"]\
            .fillna("").str.lower().str.match(cphd_poly_id)
        df2 = df[poly_filt]
        dataset = pd.concat([df1, df2], axis=0)
    else:
        dataset = df1

    dataset = dataset.set_index(["Calculated.timestamp"])
    dataset.sort_index()
    return dataset.reindex(sorted(dataset.columns), axis=1)


def resample_per_day(df):
    if df.empty:
        return df
    return df.resample('1D').agg(reduce_by_type)


def reduce_with_warnings(series):
    values = series.repalce('', np.nan).dropna().unique()
    n = len(values)
    if n == 0:
        return np.nan
    if n > 1:
        mismatched_values = series.loc[~series.duplicated()]
        warnings.warn(f"Several values for the same field of items with the same id: Name: {series.name},\nmismatched_values: {mismatched_values}")
    return list(values)[0]
