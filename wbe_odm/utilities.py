import json
import re
import warnings
from functools import reduce
from typing import Any, Optional

import geomet.wkt
import numpy as np
import pandas as pd
import shapely.wkt
from geojson_rewind import rewind

Geometry = Any
UNKNOWN_REGEX = re.compile(r"$^|n\.?[a|d|/|n]+\.?|^-$|unk.*|none", flags=re.I)


def hex_color_adder(color1: str, color2: str) -> str:
    """Takes two hex color representations,
    and adds the hex numbers together, and returns
    the string representation of the resulting color.

    Parameters
    ----------
    color1 : str
        "#xxxxxx"
    color2 : str
        "#xxxxxx"

    Returns
    -------
    str
        the resulting color between "#000000" and "#FFFFFF"
    """
    hex_pattern = re.compile("^#([A-F]|[0-9]){6}$", flags=re.I)
    if not re.search(hex_pattern, color1) or not re.search(hex_pattern, color2):
        raise ValueError(f"color strings are not valid: {color1}, {color2}")
    reds = (int(color1[1:3], 16), int(color2[1:3], 16))
    greens = (int(color1[3:5], 16), int(color2[3:5], 16))
    blues = (int(color1[5:], 16), int(color2[5:], 16))
    final_color = ["#"]
    final_color.extend(
        hex(min(255, sum(channel)))[2:] for channel in [reds, greens, blues]
    )

    return "".join(final_color)


def typecast_wide_table(df):
    for col in df.columns:
        name = df[col].name
        if "date" in name or "timestamp" in name:
            df[col] = pd.to_datetime(
                df[col], errors="coerce", infer_datetime_format=True
            )
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
        area = poly.loc[poly["Polygon_polygonID"] == id_, "area"]
        areas.append(area)
    min_area = min(areas)
    min_idx = areas.index(min_area)
    return ls[min_idx]


def convert_wkt(x: str) -> Optional[Geometry]:
    return shapely.wkt.loads(x) if x else None


def get_minimal_polygon_for_cphd(merged, poly, cphd):
    poly["shape"] = poly["Polygon_wkt"].apply(lambda x: convert_wkt(x))
    poly["area"] = poly["shape"].apply(lambda x: x.area)
    unique_cphd_polys = cphd["CPHD_polygonID"].unique()
    merged["polys_w_cphd"] = merged["Calculated_polygonList"].apply(
        lambda x: has_cphd_data(x, unique_cphd_polys)
    )
    merged["Calculated_polygonIDForCPHD"] = merged["polys_w_cphd"].apply(
        lambda x: pick_cphd_poly_by_size(x, poly)
    )
    poly.drop(columns=["shape", "area"], inplace=True)
    merged.drop(columns=["polys_w_cphd"], inplace=True)
    return merged


def get_intersecting_polygons(
    site_poly_id: Optional[str], polygon_table: pd.DataFrame
) -> str:
    if not site_poly_id or pd.isna(site_poly_id):
        return ""

    polygon_table["shape"] = polygon_table["Polygon_wkt"].apply(
        lambda x: convert_wkt(x)  # type: ignore
    )
    sewershed_shape = polygon_table.loc[
        polygon_table["Polygon_polygonID"] == site_poly_id, "shape"
    ].values[0]

    if sewershed_shape is None or pd.isna(sewershed_shape):
        return ""

    polygon_table["intersects"] = polygon_table["shape"].apply(
        lambda shape: shape.intersects(sewershed_shape) if shape else False
    )
    poly_ids = (
        polygon_table["Polygon_polygonID"].loc[polygon_table["intersects"]].to_list()
    )
    polygon_table.drop(columns=["intersects"], inplace=True)
    del polygon_table["shape"]
    return ";".join(poly_ids)


def get_midpoint_time(date1, date2):
    if pd.isna(date1) or pd.isna(date2):
        return pd.NaT
    return date1 + (date2 - date1) / 2


def clean_grab_datetime(df):
    one_day = pd.to_timedelta("24 hours")
    result_end = ["Calculated_dateTimeEnd"]
    result_start = ["Calculated_dateTimeStart"]
    grab_date = "Sample_dateTime"
    grab_token = "grb"
    collection = "Sample_collection"
    df[result_start] = pd.to_datetime(np.nan)
    df[result_end] = pd.to_datetime(np.nan)

    grb_filt = df[collection].str.contains(grab_token)
    na_filt = ~df[grab_date].isna()
    filt = na_filt & grb_filt
    df2 = df.loc[filt, df.columns.to_list()]
    df2[result_start[0]] = df2[grab_date].dt.normalize()
    df2[result_end[0]] = df2[result_start] + one_day

    df.loc[filt] = df2
    return df


def calc_start_date(end_date: pd.Timestamp, type_: str) -> pd.Timestamp:
    if pd.isna(end_date) or pd.isna(type_):
        return pd.to_datetime(np.nan)
    x = type_
    hours = None
    if re.match(r"cp[tf]p[0-9]+h", x):
        hours = int(x[4:-1])
    elif re.match(r"ps[0-9]+h", x):
        hours = int(x[2:-1])
    if hours is not None:
        interval = pd.to_timedelta(f"{hours}h")
        return end_date - interval
    return pd.to_datetime(np.nan)


def clean_composite_data_intervals(df: pd.DataFrame) -> pd.DataFrame:
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
    if df.empty:
        return df
    coll = "Sample_collection"
    end = "Sample_dateTimeEnd"
    if coll not in df.columns or end not in df.columns:
        raise ValueError(f"Missing columns in DataFrame: {coll} or {end}")
    df[coll] = df[coll].str.lower().astype(str)
    result_end = "Calculated_dateTimeEnd"
    result_start = "Calculated_dateTimeStart"

    little_less_than_one_day = pd.to_timedelta("23 hours 59 minutes")
    df[result_end] = pd.to_datetime(df[end] + little_less_than_one_day).dt.date
    df[result_start] = df.apply(
        lambda row: calc_start_date(row[result_end], row[coll]), axis=1
    )
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
    x, y = str(x), str(y)
    if pd.isna(x):
        x = "na"
    if pd.isna(y):
        y = "na"
    if x == y:
        return x
    elif re.match(UNKNOWN_REGEX, x) and re.match(UNKNOWN_REGEX, y):
        return ""
    elif re.match(UNKNOWN_REGEX, x):
        return y
    elif re.match(UNKNOWN_REGEX, y):
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
    return (x + y) / 2


def reduce_bool(x, y):
    if pd.isna(x) and pd.isna(y):
        return np.nan
    elif pd.isna(x):
        return np.nan
    elif pd.isna(y):
        return np.nan
    return x or y


def reduce_by_type(series):
    if series.empty:
        return np.nan
    data_type = str(series.dtype)
    name = series.name
    if "datetime" in data_type:
        return reduce(reduce_dt, series)

    if "object" in data_type:
        return reduce(reduce_text, series)

    if data_type in {"float64", "int"}:
        return reduce(reduce_nums, series)
    if data_type in {"boolean", "bool"}:
        return reduce(reduce_bool, series)
    else:
        raise TypeError(f"could not parse series of dtype {name}")


def convert_wkt_to_geojson(s):
    if s in ["-", ""]:
        return None  # {"type":"Polygon", "coordinates":None}
    geojson_feature = json.loads(json.dumps(geomet.wkt.loads(s)))
    geojson_feature = rewind(geojson_feature, rfc7946=False)
    return geojson_feature


def rank_polygons_by_desc_area(poly_df):
    df = poly_df.copy()
    df["area"] = df["wkt"].apply(lambda x: shapely.wkt.loads(x).area)
    df["order"] = df["area"].rank(ascending=False)
    return df["order"]


def get_data_types():
    url = "https://raw.githubusercontent.com/Big-Life-Lab/covid-19-wastewater/main/site/Variables.csv"  # noqa
    variables = pd.read_csv(url)
    variables["variableName"] = variables["variableName"].str.lower()
    variables["variableType"] = (
        variables["variableType"]
        .replace(r"date(time)?", "datetime64[ns]", regex=True)
        .replace("boolean", "bool")
        .replace("float", "float64")
        .replace("integer", "int64")
        .replace("blob", "object")
        .replace("category", "string")
    )

    return (
        variables.groupby("tableName")[["variableName", "variableType"]]
        .apply(lambda x: x.set_index("variableName").to_dict(orient="index"))
        .to_dict()
    )


def get_table_fields(table_name):
    url = "https://raw.githubusercontent.com/Big-Life-Lab/covid-19-wastewater/main/site/Variables.csv"  # noqa
    variables = pd.read_csv(url)
    return variables.loc[variables["tableName"] == table_name, "variableName"]


def clean_primary_key(key):
    key = str(key)
    key = key.removeprefix("u")
    if key[0].isupper():
        key = key[0].lower() + key[1:]
    return key.replace("Ww", "ww")


def get_primary_key(table_name=None):
    url = "https://raw.githubusercontent.com/Big-Life-Lab/covid-19-wastewater/main/site/Variables.csv"  # noqa
    variables = pd.read_csv(url)
    keys = variables.loc[
        variables["key"] == "Primary Key", ["tableName", "variableName"]
    ].set_index("tableName")
    keys = keys.apply(lambda x: clean_primary_key(x["variableName"]), axis=1)
    keys = keys.to_dict()
    if table_name is None:
        return keys
    return keys[table_name]


def get_relevant_polygons(site_id, df):
    poly_ids = df.loc[df["Site_siteID"] == site_id, "Calculated_polygonList"].unique()
    poly_lists = [x.split(";") for x in poly_ids]
    # turn list of lists into a single list
    poly_ids = set([item for sublist in poly_lists for item in sublist])

    poly_ids = [x for x in poly_ids if x not in ["-", ""]]
    return poly_ids


def build_site_specific_dataset(df, site_id):
    idx_col = "Calculated_timestamp"
    if df.empty:
        return df
    filt_site = df["Site_siteID"] == site_id

    df1 = df[filt_site]
    df1.set_index(idx_col, inplace=True)

    df1 = df1.sort_index()
    df1 = df1.dropna(how="all", axis=1)
    return df1.reindex(sorted(df1.columns), axis=1)


def resample_per_day(df):
    return df if df.empty else df.resample("1D").agg(reduce_by_type)


def reduce_with_warnings(series):
    values = series.repalce("", np.nan).dropna().unique()
    n = len(values)
    if n == 0:
        return np.nan
    if n > 1:
        mismatched_values = series.loc[~series.duplicated()]
        warnings.warn(
            f"Several values for the same field of items with the same id: Name: {series.name},\nmismatched_values: {mismatched_values}"
        )  # noqa
    return list(values)[0]
