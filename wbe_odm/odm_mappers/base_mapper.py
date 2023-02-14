import re
from abc import ABC, abstractmethod

import pandas as pd

from wbe_odm import utilities

DATA_TYPES = utilities.get_data_types()
UNKNOWN_TOKENS = ["nan", "na", "nd" "n.d", "none", "-", "unknown", "n/a", "n/d", ""]
CONVERSION_DICT = {
    "ww_measure": {"odm_name": "WWMeasure", "source_name": ""},
    "site_measure": {"odm_name": "SiteMeasure", "source_name": ""},
    "sample": {"odm_name": "Sample", "source_name": ""},
    "site": {"odm_name": "Site", "source_name": ""},
    "polygon": {"odm_name": "Polygon", "source_name": ""},
    "cphd": {"odm_name": "CovidPublicHealthData", "source_name": ""},
    "reporter": {"odm_name": "Reporter", "source_name": ""},
    "lab": {"odm_name": "Lab", "source_name": ""},
    "assay_method": {"odm_name": "AssayMethod", "source_name": ""},
    "instrument": {"odm_name": "Instrument", "source_name": ""},
}


def replace_unknown_by_default(string, default):
    return default if re.fullmatch(utilities.UNKNOWN_REGEX, string) else string


def parse_types(table_name, series):
    if series.empty:
        return series
    variable_name = series.name.lower()
    types = DATA_TYPES
    lookup_table = types[table_name]
    lookup_type = lookup_table.get(variable_name, dict())
    desired_type = lookup_type.get("variableType", "string")
    if desired_type == "bool":
        series = series.astype(str)
        default_bool = "false" if "qualityFlag" in variable_name else "true"
        series = series.str.strip().str.lower()
        series = series.apply(lambda x: replace_unknown_by_default(x, default_bool))
        series = series.str.replace("oui", "true", case=False)
        series = series.str.replace("yes", "true", case=False)
        series = series.str.startswith("true")
        series = series.astype("bool")
    elif desired_type in ["string", "category"]:
        series = series.astype(str)
        series = series.str.strip()
        series = series.apply(lambda x: replace_unknown_by_default(x, ""))
        if variable_name != "wkt":
            series = series.str.lower()
    elif desired_type == "datetime64[ns]":
        series = series.astype(str)
        series = series.apply(lambda x: replace_unknown_by_default(x, ""))
        series = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    elif desired_type in ["int64", "float64"]:
        series = pd.to_numeric(series, errors="coerce")

    return series


class BaseMapper(ABC):
    sample = pd.DataFrame(columns=utilities.get_table_fields("Sample"))
    ww_measure = pd.DataFrame(columns=utilities.get_table_fields("WWMeasure"))
    site = pd.DataFrame(columns=utilities.get_table_fields("Site"))
    site_measure = pd.DataFrame(columns=utilities.get_table_fields("SiteMeasure"))
    reporter = pd.DataFrame(columns=utilities.get_table_fields("Reporter"))
    lab = pd.DataFrame(columns=utilities.get_table_fields("Lab"))
    assay_method = pd.DataFrame(columns=utilities.get_table_fields("AssayMethod"))
    instrument = pd.DataFrame(columns=utilities.get_table_fields("Instrument"))
    polygon = pd.DataFrame(columns=utilities.get_table_fields("Polygon"))
    cphd = pd.DataFrame(columns=utilities.get_table_fields("CPHD"))
    # Attribute name to source name
    conversion_dict = CONVERSION_DICT

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def validates(self):
        pass

    def remove_duplicates(self):
        for attribute, dico in self.conversion_dict.items():
            value = self.conversion_dict[attribute]
            table_name = dico["odm_name"]
            if not isinstance(value, pd.DataFrame):
                return pd.DataFrame(columns=utilities.get_table_fields(table_name))
            if value.empty:
                return pd.DataFrame(columns=utilities.get_table_fields(table_name))
            return value.drop_duplicates(keep="first", ignore_index=True)

    def type_cast_table(self, odm_name, df):
        return df.apply(lambda x: parse_types(odm_name, x), axis=0)

    def get_attribute_from_odm_name(self, odm_name):
        for attribute, dico in self.conversion_dict.items():
            table_name = dico["odm_name"]
            if table_name == odm_name:
                return attribute
        raise NameError("Could not find attribute for table %s", odm_name)


def get_odm_names(attr=None):
    if attr is None:
        return [CONVERSION_DICT[x]["odm_name"] for x in CONVERSION_DICT.keys()]
    return CONVERSION_DICT[attr]["odm_name"]
