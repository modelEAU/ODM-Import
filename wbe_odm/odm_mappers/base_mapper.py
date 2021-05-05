from abc import ABC, abstractmethod
import pandas as pd
import re
from wbe_odm import utilities


DATA_TYPES = utilities.get_data_types()
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


def replace_unknown_by_default(string, default):
    if re.fullmatch(utilities.UNKNOWN_REGEX, string, re.IGNORECASE):
        return default
    return string


def parse_types(table_name, series):
    variable_name = series.name.lower()
    types = DATA_TYPES
    lookup_table = types[table_name]
    lookup_type = lookup_table.get(variable_name, dict())
    desired_type = lookup_type.get("variableType", "string")
    if desired_type == "bool":
        series = series.astype(str)
        default_bool = "false" if "qualityFlag" in variable_name else "true"
        series = series.str.strip().str.lower()
        series = series.apply(
            lambda x: replace_unknown_by_default(x, default_bool))
        series = series.str.replace("oui", "true", case=False)
        series = series.str.replace("yes", "true", case=False)
        series = series.str.startswith("true")
    elif desired_type in ["string", "category"]:
        series = series.astype(str)
        series = series.str.strip()
        series = series.apply(lambda x: replace_unknown_by_default(x, ""))
        if variable_name != "wkt":
            series = series.str.lower()
    elif desired_type == "datetime64[ns]":
        series = series.astype(str)
        series = series.apply(lambda x: replace_unknown_by_default(x, ""))
    elif desired_type in ["int64", "float64"]:
        series = pd.to_numeric(series, errors="coerce")
        return series

    return series


class BaseMapper(ABC):
    sample = pd.DataFrame(
        columns=utilities.get_table_fields("Sample"))
    ww_measure = pd.DataFrame(
        columns=utilities.get_table_fields("WWMeasure"))
    site = pd.DataFrame(
        columns=utilities.get_table_fields("Site"))
    site_measure = pd.DataFrame(
        columns=utilities.get_table_fields("SiteMeasure"))
    reporter = pd.DataFrame(
        columns=utilities.get_table_fields("Reporter"))
    lab = pd.DataFrame(
        columns=utilities.get_table_fields("Lab"))
    assay_method = pd.DataFrame(
        columns=utilities.get_table_fields("AssayMethod"))
    instrument = pd.DataFrame(
        columns=utilities.get_table_fields("Instrument"))
    polygon = pd.DataFrame(
        columns=utilities.get_table_fields("Polygon"))
    cphd = pd.DataFrame(
        columns=utilities.get_table_fields("CPHD"))
    # Attribute name to source name
    conversion_dict = {
        "ww_measure": {
            "odm_name": "WWMeasure",
            "primary_key": "wwMeasureID",
            "source_name": ""
            },
        "site_measure": {
            "odm_name": "SiteMeasure",
            "primary_key": "siteMeasureID",
            "source_name": ""
            },
        "sample": {
            "odm_name": "Sample",
            "primary_key": "sampleID",
            "source_name": ""
            },
        "site": {
            "odm_name": "Site",
            "primary_key": "siteID",
            "source_name": ""
            },
        "polygon": {
            "odm_name": "Polygon",
            "primary_key": "polygonID",
            "source_name": ""
            },
        "cphd": {
            "odm_name": "CovidPublicHealthData",
            "primary_key": "cphdID",
            "source_name": ""
            },
        "reporter": {
            "odm_name": "Reporter",
            "primary_key": "reporterID",
            "source_name": ""
            },
        "lab": {
            "odm_name": "Lab",
            "primary_key": "labID",
            "source_name": ""
            },
        "assay_method": {
            "odm_name": "AssayMethod",
            "primary_key": "assayMethodID",
            "source_name": ""
            },
        "instrument": {
            "odm_name": "Instrument",
            "primary_key": "instrumentID",
            "source_name": ""
            },
    }

    @abstractmethod
    def read():
        pass

    @abstractmethod
    def validates(self):
        pass

    def remove_duplicates(self):
        for attribute, dico in self.conversion_dict.items():
            value = self.conversion_dict[attribute]
            table_name = dico["odm_name"]
            if not isinstance(value, pd.DataFrame):
                return pd.DataFrame(
                    columns=utilities.get_table_fields(table_name))
            if value.empty:
                return pd.DataFrame(
                    columns=utilities.get_table_fields(table_name))
            return value.drop_duplicates(
                keep="first", ignore_index=True
            )

    def type_cast_table(self, odm_name, df):
        return df.apply(
                lambda x: parse_types(odm_name, x),
                axis=0)

    def get_odm_names(self):
        return [
            self.conversion_dict[x]["odm_name"]
            for x in self.conversion_dict.keys()]

    def get_attribute_from_odm_name(self, odm_name):
        for attribute, dico in self.conversion_dict.items():
            table_name = dico["odm_name"]
            if table_name == odm_name:
                return attribute
        raise NameError("Could not find attribute for table %s", odm_name)
