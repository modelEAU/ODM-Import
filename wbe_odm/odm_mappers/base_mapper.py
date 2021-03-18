from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import re


UNKNOWN_TOKENS = [
    "nan",
    "na",
    "nd"
    "n.d",
    "none",
    "-",
    "unknown",
    "n/a",
    "n/d",
    ""
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


def parse_types(table_name, series):
    def clean_bool(x, name):
        x = str(x).lower()
        if x in UNKNOWN_TOKENS:
            x = "false" if "quality" in name else "true"
        x = x\
            .lower()\
            .strip()\
            .replace("non", "false")
        x = x.replace("no", "false")
        return False if x == "false" else True

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
        series = series.apply(lambda x: clean_bool(x, variable_name))
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


class BaseMapper(ABC):
    sample = None
    ww_measure = None
    site = None
    site_measure = None
    reporter = None
    lab = None
    assay_method = None
    instrument = None
    polygon = None
    cphd = None
    # Attribute name to source name
    conversion_dict = {
        "ww_measure": {
            "odm_name": "WWMeasure",
            "source_name": ""
            },
        "site_measure": {
            "odm_name": "SiteMeasure",
            "source_name": ""
            },
        "sample": {
            "odm_name": "Sample",
            "source_name": ""
            },
        "site": {
            "odm_name": "Site",
            "source_name": ""
            },
        "polygon": {
            "odm_name": "Polygon",
            "source_name": ""
            },
        "cphd": {
            "odm_name": "CovidPublicHealthData",
            "source_name": ""
            },
        "reporter": {
            "odm_name": "Reporter",
            "source_name": ""
            },
        "lab": {
            "odm_name": "Lab",
            "source_name": ""
            },
        "assay_method": {
            "odm_name": "AssayMethod",
            "source_name": ""
            },
        "instrument": {
            "odm_name": "Instrument",
            "source_name": ""
            },
    }

    @abstractmethod
    def read():
        pass

    @abstractmethod
    def validates():
        pass

    def type_cast_table(odm_name, df):
        return df.apply(
                lambda x: parse_types(odm_name, x),
                axis=0)