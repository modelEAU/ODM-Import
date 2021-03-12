import pandas as pd
import numpy as np
import utilities


def parse_ww_measure(df):
    # Parsing date columns into the datetime format
    date_column_names = ["analysisDate", "reportDate"]

    for col in date_column_names:
        df[col] = pd.to_datetime(df[col])

    for col in df.columns.to_list():
        if "notes" in col:
            df[col].fillna("", inplace=True)
    df["index"] = df["index"].astype(str)
    assay_col = "assayID" if "assayID" in df.columns.to_list() \
        else "assayMethodID"

    df.loc[:, (assay_col, "notes")].fillna("", inplace=True)
    df["qualityFlag"].fillna("NO", inplace=True)
    # making a copy of the df I can iterate over
    # while I modify the original DataFrame
    df_copy = df.copy(deep=True)
    for i, row in df_copy.iterrows():
        value = row["value"]
        value_fraction = row["fractionAnalyzed"]
        value_type = row["type"]
        value_unit = row["unit"].replace("/", "-")
        value_aggregate = row["aggregation"]
        value_issue = row["qualityFlag"]

        notes = row["notes"]
        analysisDate = row["analysisDate"]
        reportDate = row["reportDate"]
        combined_name = ".".join([
            value_fraction,
            value_type,
            value_unit,
            value_aggregate,
            str(value_issue)
            .replace("True", "Issue")
            .replace("False", "No Issue")
        ])
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
        df.loc[i, combined_reportDate_name] = reportDate

    del df_copy
    to_remove = [
        "notes", "analysisDate", "reportDate",
        "value", "fractionAnalyzed", "type",
        "aggregation", "unit", "qualityFlag",
        "accessToPublic", "accessToAllOrg", "accessToPHAC",
        "accessToLocalHA", "accessToProvHA", "accessToOtherProv",
        "accessToDetails"
    ]
    df = utilities.remove_columns(to_remove, df)
    df = df.add_prefix("WWMeasure.")
    return df


def parse_site_measure(df):
    # Parsing date columns into the datetime format
    date_column_names = ["dateTime"]
    for col in date_column_names:
        df[col] = pd.to_datetime(df[col])

    # making a copy of the df I can iterate over
    # while I modify the original DataFrame
    df_copy = df.copy(deep=True)
    for i, row in df_copy.iterrows():
        value = row["value"]

        value_type = str(row["type"])
        value_unit = str(row["unit"]).replace("/", "-")
        value_aggregate = str(row["aggregation"])
        notes = row["notes"]

        combined_name = ".".join([value_type, value_unit, value_aggregate])
        combined_value_name = ".".join([combined_name, "value"])
        combined_notes_name = ".".join([combined_name, "notes"])

        if combined_value_name not in df.columns.tolist():
            df[combined_value_name] = np.nan
        df.loc[i, combined_value_name] = value

        if combined_notes_name not in df.columns.tolist():
            df[combined_notes_name] = ""
        df.loc[i, combined_notes_name] = notes

    del df_copy
    to_remove = [
        "type", "aggregation",
        "value", "unit", "accessToPublic",
        "accessToAllOrgs", "accessToPHAC", "accessToLocalHA",
        "accessToProvHA", "accessToOtherProv", "accessToDetails",
        "notes"
    ]
    df = utilities.remove_columns(to_remove, df)
    df = df.groupby("dateTime").agg(utilities.reduce_by_type)
    df.reset_index(inplace=True)
    df = df.add_prefix("SiteMeasure.")
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
            ids = [x.strip() for x in sites.split(";")]
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
    df["wkt"].fillna("", inplace=True)
    df = df.add_prefix("Polygon.")
    return df


def parse_cphd(df):
    date_column_names = ["date"]
    for col in date_column_names:
        df[col] = df[col].apply(lambda x: pd.NaT if x == "None" else x)
        df[col] = pd.to_datetime(df[col])
    # making a copy of the df I can iterate over
    # while I modify the original DataFrame
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
    to_remove = ["reporterID", "date", "dateType", "type", "value", "notes"]
    df = utilities.remove_columns(to_remove, df)
    df = df.groupby("cphdID").agg(utilities.reduce_by_type)
    df.reset_index(inplace=True)
    df = df.add_prefix("CPHD.")
    return df
