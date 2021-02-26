import pandas as pd
import numpy as np
from functools import reduce
from shapely.geometry import asShape
from geojson_rewind import rewind
import geodaisy.converters as convert
from pygeoif import geometry
from collections import defaultdict
class Odm:
    def __init__(self, xls):
        self.tables = dict(xls)
        #TODO: categorical columns could be found programmatically
        self.CATEGORICALS = {
            "Sample": ["type", "collection", "pooled"],
            "WWMeasure": ["fractionAnalyzed", "type", "unit", "aggregation", "access"],
            "SiteMeasure": ["type", "aggregation", "unit", "access"],
            "CPHD": ["type", "typeDate"],
            "Site": ["type"],
            "AssayMethod": ["unit"],
            "Instrument": ["type"],
            "Polygon": ["type"]
        }
        self.data = {}
        self.geo = {
            "type": "FeatureCollection",
            "features": []
        }

        # parsing every sheet
        self.data["WWMeasure"] = parse_ww_measure(self.tables["WWMeasure"].copy(deep=True))
        self.data["SiteMeasure"] = parse_site_measure(self.tables["SiteMeasure"].copy(deep=True))
        self.data["Sample"] = parse_sample(self.tables["Sample"].copy(deep=True))
        self.data["Site"] = parse_site(self.tables["Site"].copy(deep=True))
        self.data["Site"] = parse_site(self.tables["Site"].copy(deep=True))
        self.data["Polygon"] = parse_polygon(self.tables["Polygon"].copy(deep=True))
        self.data["CPHD"] = parse_cphd(self.tables["CPHD"].copy(deep=True))
        self.geo = extract_geo_features(self.data["Polygon"])
        self.map_center = get_map_center(self.geo)



    def combine_per_sample(self):
        #TODO: Combine with CPHD data
        ww_measure = self.data["WWMeasure"]
        sample = self.data["Sample"]
        site_measure = self.data["SiteMeasure"]
        site = self.data["Site"]

        merge = pd.merge(sample, ww_measure, how="left", left_on="Sample.sampleID", right_on="WWMeasure.sampleID")

        merge = pd.merge(merge, site_measure, how="left", left_on="Sample.siteID", right_on="SiteMeasure.siteID")

        merge = pd.merge(merge, site, how="left", left_on="Sample.siteID", right_on="Site.siteID")
        merge.set_index("Sample.sampleID", inplace=True)
        merge = remove_columns(["SiteMeasure.SiteID", "Sample.SiteID"], merge)
        return merge


def parse_dt(x,y):
    if pd.isna(x) and pd.isna(y):
        return pd.NaT
    elif pd.isna(x) and not pd.isna(y):
        return y
    elif not pd.isna(x) and pd.isna(y):
        return x
    else:
        return pd.NaT

def parse_text(x,y):
    if x == "" and y == "":
        return ""
    elif x == "":
        return y
    elif y == "":
        return x
    else:
        return ";".join([x, y])

def parse_nums(x,y):
    if pd.isna(x) and pd.isna(y):
        return np.nan
    elif pd.isna(x) and not pd.isna(y):
        return y
    elif not pd.isna(x) and pd.isna(y):
        return x
    else:
        return x+y/2;

def agg_by_type(series):
    data_type = str(series.dtype)
    name = series.name
    if "datetime" in data_type:
        return reduce(parse_dt, series)

    if "object" in data_type:
        return reduce(parse_text, series)

    if "float64" in data_type:
        return reduce(parse_nums, series)
    else:
        raise TypeError(f"could notr parse series {name}")

def parse_ww_measure(df):
    # Parsing date columns into the datetime format
    date_column_names = ["analysisDate", "reportDate"]
    for col in date_column_names:
        df[col] = pd.to_datetime(df[col])
    df[["assayMethodID", "notes"]].fillna("", inplace=True)
    df["qualityFlag"].fillna("NO", inplace=True)
    #making a copy of the df I can iterate over while I modify the original DataFrame
    df_copy = df.copy(deep=True)
    for i, row in df_copy.iterrows():
        value = row["value"]
        value_fraction = row["fractionAnalyzed"]
        value_type = row["type"]
        value_unit = row["unit"].replace("/", "_")
        value_aggregate = row["aggregation"]
        value_issue = row["qualityFlag"]
        notes = row["notes"]
        analysisDate = row["analysisDate"]
        reportDate = row["reportDate"]
        combined_name = ".".join([value_fraction, value_type, value_unit, value_aggregate, value_issue])
        combined_value_name = ".".join([combined_name, "value"])
        combined_notes_name = ".".join([combined_name, "notes"])
        combined_analysisDate_name = ".".join([combined_name, "analysisDate"])
        combined_reportDate_name = ".".join([combined_name, "reportDate"])

        if combined_value_name not in df.columns.tolist():
            df[combined_value_name] = np.nan
        df.loc[i, combined_value_name] = value

        if combined_notes_name not in df.columns.tolist():
            df[combined_notes_name] = ""
        df.loc[i, combined_notes_name] = notes

        if combined_analysisDate_name not in df.columns.tolist():
                df[combined_analysisDate_name] = pd.NaT
        df.loc[i, combined_analysisDate_name] = analysisDate

        if combined_reportDate_name not in df.columns.tolist():
                df[combined_reportDate_name] = pd.NaT
        df.loc[i, combined_reportDate_name] =reportDate

    del df_copy

    #removing access data and columns that have been used in the value spread
    to_remove = ["notes", "analysisDate", "reportDate" "value", "fractionAnalyzed", "type", "aggregation", "unit", "qualityFlag", "accessToPublic", "accessToAllOrg", "accessToPHAC", "accessToLocalHA", "accessToProvHA", "accessToOtherProv", "accessToDetails"]
    df = remove_columns(to_remove, df)
    #df = df.groupby("sampleID").agg(agg_by_type)
    #df.reset_index(inplace=True)
    df = df.add_prefix("WWMeasure.")
    return df

def parse_site_measure(df):
    # Parsing date columns into the datetime format
    date_column_names = ["dateTime"]
    for col in date_column_names:
        df[col] = pd.to_datetime(df[col])

    #making a copy of the df I can iterate over while I modify the original DataFrame
    df_copy = df.copy(deep=True)
    for i, row in df_copy.iterrows():
        value = row["value"]

        value_type = row["type"]
        value_unit = row["unit"].replace("/", "-")
        value_aggregate = row["aggregation"]

        notes = row["notes"]
        dateTime = row["dateTime"]

        combined_name = ".".join([value_type, value_unit, value_aggregate])
        combined_value_name = ".".join([combined_name, "value"])
        combined_notes_name = ".".join([combined_name, "notes"])
        combined_dateTime_name = ".".join([combined_name, "dateTime"])

        if combined_value_name not in df.columns.tolist():
            df[combined_value_name] = np.nan
        df.loc[i, combined_value_name] = value

        if combined_notes_name not in df.columns.tolist():
            df[combined_notes_name] = ""
        df.loc[i, combined_notes_name]= notes

        if combined_dateTime_name not in df.columns.tolist():
            df[combined_dateTime_name] = pd.NaT
        df.loc[i, combined_dateTime_name] = dateTime
    "aggregationDesc"

    del df_copy
    to_remove = ["dateTime", "type", "aggregation", "value", "unit", "accessToPublic", "accessToAllOrgs", "accessToPHAC", "accessToLocalHA", "accessToProvHA", "accessToOtherProv", "accessToDetails", "notes"]
    df = remove_columns(to_remove, df)
    df = df.groupby("siteID").agg(agg_by_type)
    df.reset_index(inplace=True)
    df = df.add_prefix("SiteMeasure.")
    return df

def remove_columns(cols, df):
    to_remove = [col for col in cols if col in df.columns]
    df.drop(to_remove, axis=1, inplace=True)
    return df

def parse_sample(df):
    # Parsing date columns into the datetime format
    date_column_names = ["dateTime", "dateTimeStart", "dateTimeEnd"]
    for col in date_column_names:
        df[col] = df[col].apply(lambda x: pd.NaT if x == "None" else x)
        df[col] = pd.to_datetime(df[col])

    df_copy = df.copy(deep=True)
    for i, row in df_copy.iterrows():
        sites = row["siteID"]
        if ";" in sites:
            ids = [ x.strip() for x in sites.split(";")]
            df["siteID"].iloc[i] = ids.pop(0)
            for ii in ids:
                new_row = df.iloc[i].copy()
                new_row["siteID"] = ii
                df = df.append(new_row, ignore_index=True)
    df = df.add_prefix("Sample.")
    return df

def parse_site(df):
    df = df.add_prefix("Site.")
    return df

def parse_polygon(df):
    df["geoJSON"] = df["wkt"].apply(lambda x: convert_geojson(x))
    df = df.add_prefix("Polygon.")
    return df

def parse_cphd(df):
    date_column_names = ["date"]
    for col in date_column_names:
        df[col] = df[col].apply(lambda x: pd.NaT if x == "None" else x)
        df[col] = pd.to_datetime(df[col])
    #making a copy of the df I can iterate over while I modify the original DataFrame
    df_copy = df.copy(deep=True)
    for i, row in df_copy.iterrows():
        value = row["value"]
        value_type = row["type"]

        date = row["date"]
        date_type = row["dateType"]

        notes = row["notes"]

        combined_value_name = ".".join([value_type, "value"])
        combined_notes_name = ".".join([value_type, "notes"])
        combined_date_name = ".".join([date_type, "date"])

        if combined_value_name not in df.columns.tolist():
            df[combined_value_name] = np.nan
        df.loc[i, combined_value_name] = value

        if combined_notes_name not in df.columns.tolist():
            df[combined_notes_name] = ""
        df.loc[i, combined_notes_name] = notes

        if combined_date_name not in df.columns.tolist():
                df[combined_date_name] = pd.NaT
        df.loc[i, combined_date_name] = date


    del df_copy
    to_remove = ["reporterID","date", "dateType", "type", "value","notes"]
    df = remove_columns(to_remove, df)
    df = df.groupby("cphdID").agg(agg_by_type)
    df.reset_index(inplace=True)
    df = df.add_prefix("CPHD.")
    return df

def convert_geojson(s):
    if s in ["-", ""] or np.isnan(s):
        return None  #{"type":"Polygon", "coordinates":None}
    from_wkt = geometry.from_wkt(s)
    geo_interface = from_wkt.__geo_interface__
    geojson_feature = convert.geo_interface_to_geojson(geo_interface)
    geojson_feature = rewind(geojson_feature, rfc7946=False)
    return geojson_feature

def get_map_center(geo_json):
    x_s = []
    y_s = []
    if len(geo_json["features"])==0: return None
    for feat in geo_json["features"]:
        # convert the geometry to shapely
        geom = asShape(feat["geometry"])
        # obtain the coordinates of the feature's centroid
        x_s.append(geom.centroid.x)
        y_s.append(geom.centroid.y)
    x_m = sum(x_s)/len(x_s)
    y_m = sum(y_s)/len(y_s)
    return {"lat":y_m, "lon":x_m}

def extract_geo_features(polygon_df):
    geo = {
        "type": "FeatureCollection",
        "features": []
    }
    for i, row in polygon_df.iterrows():
        if row["Polygon.geoJSON"] is None:
            continue
        new_feature = {
            "type": "Feature",
            "geometry": row["Polygon.geoJSON"],
            "properties":{
                "polygonID":row["Polygon.polygonID"],
            },
            "id":i
        }
        geo["features"].append(new_feature)
    return geo

if __name__ == "__main__":
    # run with example data
    filename = "Data/Template - Data Model - 20210127.xls"
    xls = pd.read_excel(filename, engine="xlrd", sheet_name=None)
    odm_data = Odm(xls)
    samples = odm_data.combine_per_sample()

