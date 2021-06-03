#%%
import pandas as pd
import numpy as np
from datetime import datetime
import os
import re
import warnings
from wbe_odm import utilities
from wbe_odm.odm_mappers import base_mapper
from wbe_odm.odm_mappers import excel_template_mapper
from wbe_odm.odm_mappers.csv_mapper import CsvMapper


LABEL_REGEX = r"[a-zA-Z]+_[0-9]+(\.[0-9])?_[a-zA-Z0-9]+_[a-zA-Z0-9]+"

directory = os.path.dirname(__file__)

MCGILL_MAP_NAME = directory + "/" + "mcgill_map.csv"

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

class MapperFuncs:
    # def excel_style(col):
    #     """ Convert given column number to an Excel-style column name. """
    #     result = []
    #     while col:
    #         col, rem = divmod(col-1, 26)
    #         result[:0] = LETTERS[rem]
    #     return "".join(result)

    @classmethod
    def parse_date(cls, item):
        if isinstance(item, (str, datetime)):
            return pd.to_datetime(item)
        return pd.NaT


    # def str_date_from_timestamp(timestamp_series):
    #     return timestamp_series.dt.strftime("%Y-%m-%d").fillna("")

    @classmethod
    def clean_up(cls, df, molecular_cols, meas_cols):
        # removed rows that aren't samples
        type_col = "sampling.general information.type sample.na"
        df = df.loc[~df[type_col].isin(["Reference", "Negative"])]

        df.loc[:, molecular_cols+meas_cols] = df.loc[
            :, molecular_cols+meas_cols]\
            .apply(pd.to_numeric, errors="coerce")

        df = df.dropna(subset=molecular_cols, how="all")
        # Parse other measurement columns:
        # we need to convert resistivity to conductivity
        cond_col = 'concentration.key parametres.conductivity megohm.na'
        df[cond_col] = df[cond_col].apply(
            lambda x: 1/x if str(x).isnumeric
            else np.nan)
        # Parse date columns to datetime
        for col in df.columns:
            if "date" in col:
                df[col] = df[col].apply(
                    lambda x: cls.parse_date(x))
        return df


    # def typecast_column(desired_type, series):
    #     if desired_type == "bool":
    #         series = series.astype(str)
    #         series = series.str.strip().str.lower()
    #         series = series.apply(
    #             lambda x: base_mapper.replace_unknown_by_default(x, ""))
    #         series = series.str.replace("oui", "true", case=False)\
    #             .str.replace("yes", "true", case=False)\
    #             .str.startswith("true")
    #     elif desired_type in ["string", "category"]:
    #         series = series.astype(str)
    #         series = series.str.lower()
    #         series = series.str.strip()
    #         series = series.apply(
    #             lambda x: base_mapper.replace_unknown_by_default(x, ""))
    #     elif desired_type in ["int64", "float64"]:
    #         series = pd.to_numeric(series, errors="coerce")
    #     elif desired_type == "datetime64[ns]":
    #         series = pd.to_datetime(series, errors="coerce")
    #     series = series.astype(desired_type)
    #     return series


    # def typecast_lab(lab, types):
    #     clean_types = []
    #     for datatype in types:
    #         if datatype in base_mapper.UNKNOWN_TOKENS:
    #             datatype = "string"
    #         datatype = str(datatype)\
    #             .replace("date", "datetime64[ns]") \
    #             .replace("mixed", "object") \
    #             .replace("boolean", "bool") \
    #             .replace("float", "float64") \
    #             .replace("integer", "int64") \
    #             .replace("number", "float64") \
    #             .replace("text", "string") \
    #             .replace("blob", "object")
    #         clean_types.append(datatype)
    #     for i, col_name in enumerate(lab.columns):
    #         lab[col_name] = typecast_column(clean_types[i], lab[col_name])
    #     return lab

    @classmethod
    def clean_labels(cls, label):
        parts = str(label).lower().split("_")
        parts = [part.strip() for part in parts]
        return "_".join(parts)

    @classmethod
    def get_sample_type(cls, sample_type):
        """acceptable_types = [
            "qtips", "filter", "gauze",
            "swrsed", "pstgrit", "psludge",
            "pefflu", "ssludge", "sefflu",
            "water", "faeces", "rawww", ""
        ]"""
        sample_type = sample_type.str.strip().str.lower()
        return sample_type

    @classmethod
    def get_start_date(cls, start_col, end_col, sample_type):
        df = pd.concat([start_col, end_col, sample_type], axis=1)
        df.columns = ["start", "end", "type"]
        df["s"] = df.apply(
            lambda row: utilities.calc_start_date(row["end"], row["type"]), axis=1)
        return df["s"]

    @classmethod
    def get_grab_date(cls, end_series, type_series):
        df = pd.concat([end_series, type_series], axis=1)
        df.columns = ["end", "type"]
        df["date_grab"] = pd.NaT
        filt = df["type"].str.contains("grb")
        df.loc[filt, "date_grab"] = df.loc[filt, "end"]
        return df["date_grab"]

    @classmethod
    def get_collection_method(cls, collection):
        def check_collection_method(x):
            if re.match(r"cp[tf]p[0-9]+h", x) or x == "grb":
                return x
            elif "grb" in collection:
                added_bit = collection[len("grb"):]
                return "grb" + "cp" + added_bit
            else:
                return ""
        collection = collection.str.strip()
        collection = collection.apply(lambda x: check_collection_method(x))
        return collection


    # def pass_raw(*args):
    #     if len(args) == 0:
    #         return None
    #     elif len(args) == 1:
    #         return args[0]
    #     arguments = pd.concat([arg for arg in args], axis=1)
    #     return arguments.agg(",".join, axis=1)

    @classmethod
    def get_assay_method_id(cls, sample_type, concentration_method, assay_date):
        formatted_date = CsvMapper.str_date_from_timestamp(assay_date)
        clean_series = []
        for series in [sample_type, concentration_method, formatted_date]:
            series = series.fillna("").astype(str)
            clean_series.append(series)
        df = pd.concat(clean_series, axis=1)
        return df.agg("_".join, axis=1)

    @classmethod
    def get_assay_instrument(cls, static_methods, sample_type, concentration_method):
        clean_series = []
        for series in [sample_type, concentration_method]:
            series = series.fillna("").astype(str)
            clean_series.append(series)
        df = pd.concat(clean_series, axis=1)
        df["general_id"] = df.agg("_".join, axis=1).str.lower()
        merged = pd.merge(
            left=static_methods,
            right=df,
            left_on="assayMethodID",
            right_on="general_id")
        return merged["instrumentID"].fillna("")

    @classmethod
    def get_assay_name(cls, static_methods, sample_type, concentration_method):
        clean_series = []
        for series in [sample_type, concentration_method]:
            series = series.fillna("").astype(str)
            clean_series.append(series)
        df = pd.concat(clean_series, axis=1)
        df["general_id"] = df.agg("_".join, axis=1).str.lower()
        merged = pd.merge(
            left=static_methods,
            right=df,
            left_on="assayMethodID",
            right_on="general_id")
        return merged["name"].fillna("")

    @classmethod
    def write_concentration_method(cls, conc_method, conc_volume, ph_final):
        clean_series = []
        names = ["conc", "conc_volume", "ph_final"]
        for series, name in zip([conc_method, conc_volume, ph_final], names):
            series = series.fillna("unknown").astype(str)
            series.name = name
            clean_series.append(series)
        df = pd.concat(clean_series, axis=1)

        df["text"] = df.apply(
            lambda row: f"{row['conc']}, Volume:{row['conc_volume']} mL, Final pH:{row['ph_final']}", # noqa
            axis=1)
        return df["text"]

    @classmethod
    def get_site_id(cls, labels):
        def extract_from_label(label_id):
            if re.match(LABEL_REGEX, label_id):
                label_parts = label_id.split("_")
                return "_".join(label_parts[0:2])
            else:
                return ""
        clean_label_series = labels.apply(lambda x: cls.clean_labels(x))
        return clean_label_series.apply(lambda x: extract_from_label(x))

    @classmethod
    def sample_is_pooled(cls, pooled):
        # It isn't clear what the sheet wants the user to do - either say "Yes"
        # if the sample is pooled, or actually put in the sample ids
        # of the children. For now, let's only check if it is pooled or not
        return pooled != ""

    @classmethod
    def get_children_samples(cls, pooled, sample_date):
        def make_children_ids(row):
            split_pooled = row["pooled"].split(",")if "," in pooled else ""
            children_ids = []
            for item in split_pooled:
                if re.match(LABEL_REGEX, item):
                    child_id = "_".join([item, row["clean_date"]])
                    children_ids.append(child_id)
            if not children_ids:
                return ""
            else:
                return ",".join(children_ids)
        clean_date = CsvMapper.str_date_from_timestamp(sample_date)
        df = pd.concat([pooled, clean_date], axis=1)
        df.columns = ["pooled", "clean_date"]
        df["children_ids"] = df.apply(lambda row: make_children_ids(row))
        return df["children_ids"]

    @classmethod
    def get_sample_id(cls, label_id, sample_date, spike_batch, lab_id, sample_index):
        # TODO: Deal with index once it's been implemented in McGill sheet
        clean_date = CsvMapper.str_date_from_timestamp(sample_date)
        clean_label = label_id.apply(lambda x: cls.clean_labels(x))

        df = pd.concat([clean_label, clean_date, spike_batch], axis=1)
        df["lab_id"] = lab_id
        df["index_no"] = str(sample_index) \
            if not isinstance(sample_index, pd.Series) \
            else sample_index.astype(str)
        df.columns = [
            "clean_label", "clean_date", "spike_batch",
            "lab_id", "index_no"
        ]
        df["sample_ids"] = ""
        regex_filt = df["clean_label"].str.match(LABEL_REGEX, case=False)

        df.loc[regex_filt, "sample_ids"] = df.loc[
            regex_filt,
            ["clean_label", "clean_date", "index_no"]
        ].agg("_".join, axis=1)

        df.loc[~regex_filt, "sample_ids"] = df.loc[
            ~regex_filt,
            ["lab_id", "spike_batch", "clean_label", "index_no"]
        ].agg("_".join, axis=1)
        return df["sample_ids"]

    @classmethod
    def get_wwmeasure_id(
            cls,
            label_id,
            sample_date,
            spike_batch,
            lab_id,
            sample_index,
            meas_type,
            meas_date,
            index):
        # TODO: Deal with index once it's been implemented in McGill sheet
        sample_id = cls.get_sample_id(
            label_id,
            sample_date,
            spike_batch,
            lab_id,
            sample_index
        )
        meas_date = CsvMapper.str_date_from_timestamp(meas_date)
        df = pd.concat([sample_id, meas_date], axis=1)
        df["meas_type"] = meas_type
        df["index_no"] = str(index) if not isinstance(index, pd.Series) \
            else index.astype(str)
        return df.agg("_".join, axis=1)

    @classmethod
    def get_reporter_id(cls, static_reporters, name):
        def get_reporter_name(x):
            reporters_w_name = static_reporters.loc[
                static_reporters["reporterID"].str.lower().str.contains(x)]
            if len(reporters_w_name) > 0:
                return reporters_w_name.iloc[0]["reporterID"]
            else:
                return x

        name = name.str.replace(", ", "/")\
            .str.replace(",", "/")\
            .str.replace(";", "/")
        name = name.str.lower().apply(lambda x: x.split("/")[0] if "/" in x else x)
        name = name.str.strip()
        reporters_ids = name.apply(get_reporter_name)
        return reporters_ids

    @classmethod
    def has_quality_flag(cls, flag):
        return flag != ""

    @classmethod
    def get_sample_volume(cls, vols, default):
        vols = vols.apply(lambda x: x if not pd.isna(x) else default)
        return vols

    @classmethod
    def get_field_sample_temp(cls, series):
        temp_map = {
            "refrigerated": 4.0,
            "ice": 0.0,
            "norefrigaration": 20.0,
            # "norefrigeration": np.nan
        }
        series = series.str.lower().map(temp_map)
        return series

    @classmethod
    def get_shipped_on_ice(cls, series):
        series = series.str.lower()
        map_to = {
            "yes": True,
            "no": False
        }
        return series.map(map_to)

    @classmethod
    def grant_access(cls, access):
        return access.str.lower().isin(["", "1", "yes", "true"])

    @classmethod
    def validate_fraction_analyzed(cls, series):
        filt = (
            series.str.contains("mixed") |
            series.str.contains("liquid") |
            series.str.contains("solids")
        )
        series.loc[~filt] = ""
        return series

    @classmethod
    def validate_value(cls, values):
        return pd.to_numeric(values, errors="coerce")
    
    @classmethod
    def get_lab_id(cls, lab_id):
        if isinstance(lab_id, str):
            return lab_id.lower().strip()
        elif isinstance(lab_id, pd.Series):
            return lab_id.str.lower().strip()
        raise TypeError(f"What is this lab_id?: {lab_id}")


# processing_functions = {
#     "get_grab_date": get_grab_date,
#     "get_start_date": get_cp_start_date,
#     "get_collection_method": get_collection_method,
#     "get_sample_type": get_sample_type,
#     "get_assay_method_id": get_assay_method_id,
#     "get_assay_instrument": get_assay_instrument,
#     "get_assay_name": get_assay_name,
#     "write_concentration_method": write_concentration_method,
#     "get_site_id": get_site_id,
#     "sample_is_pooled": sample_is_pooled,
#     "get_children_samples": get_children_samples,
#     "get_sample_id": get_sample_id,
#     "get_wwmeasure_id": get_wwmeasure_id,
#     "get_reporter_id": get_reporter_id,
#     "has_quality_flag": CsvMapper.has_quality_flag,
#     "grant_access": grant_access,
#     "get_sample_volume": get_sample_volume,
#     "get_field_sample_temp": get_field_sample_temp,
#     "get_shipped_on_ice": get_shipped_on_ice,
#     "validate_fraction_analyzed": validate_fraction_analyzed,
#     "validate_value": validate_value,

# }
def append_new_entry(new_entry, current_table_data):
    if current_table_data is None:
        new_entry = {0: new_entry}
        return pd.DataFrame.from_dict(new_entry, orient='index')
    new_index = current_table_data.index.max() + 1
    current_table_data.loc[new_index] = new_entry
    return current_table_data


def get_lod(lab, label_col_name, spike_col_name, lod_value_col):
    new_cols = ['LOD', 'LOQ']
    filt = lab[label_col_name] == "negative"
    cols_to_keep = [
        label_col_name,
        spike_col_name,
        lod_value_col,
    ]
    lod_df = lab.loc[filt][cols_to_keep]
    for col in new_cols:
        lab.loc[:, col] = np.nan
    lod_df[spike_col_name] = lod_df[spike_col_name].replace("", np.nan)
    lod_df = lod_df.dropna(subset=[spike_col_name])
    spike_ids = list(lod_df[spike_col_name].dropna().unique())
    for spike_id in spike_ids:
        lod_filt = lod_df[spike_col_name] == spike_id
        lab_filt = lab[spike_col_name] == spike_id
        lod = lod_df.loc[lod_filt].iloc[0].loc[lod_value_col]
        for col in new_cols:
            lab.loc[lab_filt, col] = lod
    return lab


# def filter_by_date(df, date_col, start, end):
#     if start is not None:
#         startdate = pd.to_datetime(start)
#         start_filt = (df[date_col] > startdate)
#     else:
#         start_filt = None
#     if end is not None:
#         enddate = pd.to_datetime(end)
#         end_filt = (df[date_col] < enddate)
#     else:
#         end_filt = None
#     if start_filt is None and end_filt is None:
#         return df
#     elif start_filt is None:
#         return df[end_filt]
#     elif end_filt is None:
#         return df[start_filt]
#     else:
#         return df[start_filt & end_filt]


def validate_date_text(date_text):
    date_text = str(date_text)
    try:
        if date_text != datetime.strptime(
                date_text, "%Y-%m-%d").strftime('%Y-%m-%d'):
            raise ValueError
        return True
    except ValueError:
        return False


def remove_bad_rows(lab):
    """"LabelID column should contain something for all valid rows.
    If it's something else than an empty value and that this empty value
    doesn't cast to datetime, the row should be deleted"""
    LABEL_ID_COL = "D"
    filt = (~pd.isnull(lab[LABEL_ID_COL]))
    return lab.loc[filt]


# def get_labsheet_inputs(map_row, lab_data, lab_id):
#     lab_input = map_row["labInputs"]
#     if lab_input == "":
#         return None
#     var_name = map_row["variableName"]
#     raw_inputs = lab_input.split(";")
#     final_inputs = []
#     for input_ in raw_inputs:
#         if re.match(r"__const__.*:.*", input_):
#             value, type_ = input_[len("__const__"):].split(":")
#             if type_ == "str":
#                 value = str(value)
#             elif type_ == "int":
#                 value = int(value)
#         elif input_ == "__labID__":
#             value = lab_id
#         elif input_ == "__varName__":
#             value = var_name
#         elif input_ == "__default__":
#             value = map_row["defaultValue"]
#         else:
#             value = lab_data[input_]
#         final_inputs.append(value)
#     return tuple(final_inputs)


# def get_static_inputs(map_row, static_data):
#     input_sources = map_row["inputSources"]
#     if "static" in input_sources:
#         static_table = input_sources.split("+")[0]
#         static_table = static_table[len("static "):]
#         return static_data[static_table]
#     else:
#         return None


# def get_all_inputs(row):
#     static_input = row["static"]
#     lab_inputs = row["lab_arguments"]
#     if static_input is None and lab_inputs is None:
#         inputs = None
#     elif static_input is None:
#         inputs = lab_inputs
#     else:
#         inputs = (static_input, *lab_inputs)
#     if inputs is None:
#         inputs = tuple([row["defaultValue"]])
#     return inputs


# def parse_sheet(mapping, static, lab_data, processing_functions, lab_id,):
#     mapping["lab_arguments"] = mapping.apply(
#         lambda row: get_labsheet_inputs(row, lab_data, lab_id), axis=1)
#     mapping["static"] = mapping.apply(
#         lambda row: get_static_inputs(row, static), axis=1)
#     mapping["final_inputs"] = mapping.apply(
#         lambda row: get_all_inputs(row), axis=1)
#     mapping["func"] = mapping["processingFunction"].apply(
#         lambda x: processing_functions.get(x, pass_raw))

#     mapping["columnName"] = mapping[
#         ["table", "elementName", "variableName"]].agg("_".join, axis=1)
#     to_apply = mapping.loc[
#         :, ["columnName", "func", "final_inputs"]]
#     for _, apply_row in to_apply.iterrows():
#         col_name = apply_row["columnName"]
#         lab_data[col_name] = apply_row["func"](*apply_row["final_inputs"])
#     tables = {table: None for table in mapping["table"].unique()}
#     for table in tables:
#         elements = mapping.loc[
#             mapping["table"] == table, "elementName"
#         ].unique()
#         sub_dfs = []
#         for element in elements:
#             table_element_filt = (mapping["table"] == table)\
#                  & (mapping["elementName"] == element)
#             col_names = mapping.loc[table_element_filt, "columnName"]
#             var_names = mapping.loc[table_element_filt, "variableName"]
#             sub_df = lab_data[col_names]
#             sub_df.columns = var_names
#             sub_dfs.append(sub_df)
#         table_df = pd.concat(sub_dfs, axis=0, ignore_index=True)
#         if table in ["WWMeasure", "SiteMeasure"]:
#             table_df = table_df.dropna(subset=["value"])
#         tables[table] = table_df
#     return tables


class McGillMapper(CsvMapper):
    def __init__(self, processing_functions=MapperFuncs):
        super().__init__(processing_functions=processing_functions)
    def get_attr_from_table_name(self, table_name):
        for attr, dico in self.conversion_dict.items():
            odm_name = dico["odm_name"]
            if odm_name == table_name:
                return attr

    def read_static_data(self, staticdata_path):
        # Get the static data
        static_tables = [
            "Lab",
            "Reporter",
            "Site",
            "Instrument",
            "Polygon",
            "AssayMethod"
        ]
        attrs = []
        for table in static_tables:
            attr = self.get_attr_from_table_name(table)
            attrs.append(attr)
        static_data = {}
        excel_mapper = excel_template_mapper.ExcelTemplateMapper()
        if staticdata_path is not None:
            excel_mapper.read(staticdata_path, sheet_names=static_tables)
        for table, attr in zip(static_tables, attrs):
            static_data[table] = getattr(excel_mapper, attr)
            setattr(self, attr, static_data[table])
        return static_data

    def read(self,
             labsheet_path,
             staticdata_path,
             worksheet_name,
             lab_id,
             map_path=MCGILL_MAP_NAME,
             startdate=None,
             enddate=None):
        # get the lab data
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore")
            lab = pd.read_excel(labsheet_path,
                                sheet_name=worksheet_name,
                                header=None,
                                usecols="A:BV")
        # parse the headers to deal with merged cells and get unique names
        lab.columns = self.get_excel_style_columns(lab)

        lab_datatypes = lab.iloc[5].values
        lab = lab.iloc[6:]
        lab = remove_bad_rows(lab)
        lab = self.typecast_lab(lab, lab_datatypes)
        lab = lab.dropna(how="all")
        mapping = pd.read_csv(map_path, header=0)
        mapping.fillna("", inplace=True)
        mapping = mapping.astype(str)
        label_col_name = "D"  # sampleID column
        spike_col_name = "AB"  # spikeID
        lod_value_col = "BI"  # sars-cov-2 gc/rxn
        sample_date_col = "B"  # end date
        lab = get_lod(lab, label_col_name, spike_col_name, lod_value_col)
        lab = self.filter_by_date(lab, sample_date_col, startdate, enddate)
        static_data = self.read_static_data(staticdata_path)
        dynamic_tables = self.parse_sheet(
            mapping,
            static_data,
            lab,
            self.processing_functions,
            lab_id
        )
        for table_name, table in dynamic_tables.items():
            attr = self.get_attr_from_table_name(table_name)
            table = self.type_cast_table(table_name, table)
            setattr(self, attr, table)
        return

    def validates(self):
        return True


if __name__ == "__main__":
    mapper = McGillMapper(processing_functions=MapperFuncs)
    lab_data = "/Users/jeandavidt/OneDrive - Université Laval/COVID/Latest Data/Input/CentrEau-COVID_Resultats_Quebec_final.xlsx" # noqa
    static_data = "/Users/jeandavidt/OneDrive - Université Laval/COVID/Latest Data/Input/CentrEAU-COVID_Static_Data.xlsx"  # noqa
    # lab_data = "/Users/martinwellman/Documents/Health/Wastewater/McGillLabData/CentrEau-COVID_Resultats_Quebec_final.xlsx" # noqa
    # static_data = "/Users/martinwellman/Documents/Health/Wastewater/McGillLabData/mcgill_static.xlsx"  # noqa
    sheet_name = "QC Data Daily Samples (McGill)"
    lab_id = "frigon_lab"
    mapper.read(lab_data,
                static_data,
                sheet_name,
                lab_id,
                map_path=MCGILL_MAP_NAME,
                startdate=None,
                enddate=None)
    print(mapper.site)
    mapper.save_all("test.xlsx")
