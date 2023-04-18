import os
import re
import warnings
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from wbe_odm import utilities
from wbe_odm.odm_mappers import excel_template_mapper
from wbe_odm.odm_mappers.csv_mapper import CsvMapper

LABEL_REGEX = r"[a-zA-Z]+_[0-9]+(\.[0-9])?_[a-zA-Z0-9]+_[a-zA-Z0-9]+"

directory = os.path.dirname(__file__)

MCGILL_MAP_2021 = f"{directory}/mcgill_map.csv"
MCGILL_MAP_2022 = f"{directory}/mcgill_map_2022.csv"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class MapperFuncs:
    @classmethod
    def parse_date(cls, item, format="%Y-%m-%d"):
        date = pd.NaT
        try:
            date = pd.to_datetime(item, format=format)
        except TypeError:
            return date

    @classmethod
    def clean_up(cls, df, molecular_cols, meas_cols, date_format):
        # removed rows that aren't samples
        type_col = "sampling.general information.type sample.na"
        df = df.loc[~df[type_col].isin(["Reference", "Negative"])]

        df.loc[:, molecular_cols + meas_cols] = df.loc[
            :, molecular_cols + meas_cols
        ].apply(pd.to_numeric, errors="coerce")

        df = df.dropna(subset=molecular_cols, how="all")
        # Parse other measurement columns:
        # we need to convert resistivity to conductivity
        cond_col = "concentration.key parametres.conductivity megohm.na"
        df[cond_col] = df[cond_col].apply(
            lambda x: 1 / x if str(x).isnumeric else np.nan
        )
        # Parse date columns to datetime
        for col in df.columns:
            if "date" in col:
                df[col] = df[col].apply(lambda x: cls.parse_date(x, date_format))
        return df

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
            lambda row: utilities.calc_start_date(row["end"], row["type"]), axis=1
        )
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
            if (
                re.match(r"cp[tf]p[0-9]+h", x)
                or x == "grb"
                or re.match(r"ps[0-9]+h", x)
            ):
                return x
            elif "grb" in collection:
                added_bit = collection[len("grb") :]
                return "grb" + "cp" + added_bit
            else:
                return ""

        collection = collection.str.strip()
        collection = collection.apply(lambda x: check_collection_method(x))
        return collection

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
            right_on="general_id",
        )
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
            right_on="general_id",
        )
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
            lambda row: f"{row['conc']}, Volume:{row['conc_volume']} mL, Final pH:{row['ph_final']}",  # noqa
            axis=1,
        )
        return df["text"]

    @classmethod
    def get_site_id(cls, labels):
        def extract_from_label(label_id):
            if re.match(LABEL_REGEX, label_id):
                label_parts = label_id.split("_")
                return "_".join(label_parts[:2])
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
            split_pooled = row["pooled"].split(",") if "," in pooled else ""
            children_ids = [
                "_".join([item, row["clean_date"]])
                for item in split_pooled
                if re.match(LABEL_REGEX, item)
            ]
            return ",".join(children_ids) if children_ids else ""

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
        df["index_no"] = (
            sample_index.astype(str)
            if isinstance(sample_index, pd.Series)
            else str(sample_index)
        )

        df.columns = ["clean_label", "clean_date", "spike_batch", "lab_id", "index_no"]
        df["sample_ids"] = ""
        regex_filt = df["clean_label"].str.match(LABEL_REGEX, case=False)

        df.loc[regex_filt, "sample_ids"] = df.loc[
            regex_filt, ["clean_label", "clean_date", "index_no"]
        ].agg("_".join, axis=1)

        df.loc[~regex_filt, "sample_ids"] = df.loc[
            ~regex_filt, ["lab_id", "spike_batch", "clean_label", "index_no"]
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
        index,
    ):
        # TODO: Deal with index once it's been implemented in McGill sheet
        sample_id = cls.get_sample_id(
            label_id, sample_date, spike_batch, lab_id, sample_index
        )
        meas_date = CsvMapper.str_date_from_timestamp(meas_date)
        df = pd.concat([sample_id, meas_date], axis=1)
        df["meas_type"] = meas_type
        df["index_no"] = (
            index.astype(str) if isinstance(index, pd.Series) else str(index)
        )

        return df.agg("_".join, axis=1)

    @classmethod
    def get_reporter_id(cls, static_reporters, name):
        def get_reporter_name(x):
            reporters_w_name = static_reporters.loc[
                static_reporters["reporterID"].str.lower().str.contains(x)
            ]
            if len(reporters_w_name) > 0:
                return reporters_w_name.iloc[0]["reporterID"]
            else:
                return x

        name = name.str.replace(", ", "/").str.replace(",", "/").str.replace(";", "/")
        name = name.str.lower().apply(lambda x: x.split("/")[0] if "/" in x else x)
        name = name.str.strip()
        reporters_ids = name.apply(get_reporter_name)
        return reporters_ids

    @classmethod
    def has_quality_flag(cls, flag):
        negative_flags = ["", "0", "no", "n", "f", "false", "none", "nan"]
        return ~flag.astype(str).str.lower().isin(negative_flags)

    @classmethod
    def get_sample_volume(cls, vols, default):
        vols = vols.apply(lambda x: default if pd.isna(x) else x)
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
        map_to = {"yes": True, "no": False}
        return series.map(map_to)

    @classmethod
    def grant_access(cls, access):
        return access.str.lower().isin(["", "1", "yes", "true"])

    @classmethod
    def validate_fraction_analyzed(cls, series):
        filt = (
            series.str.contains("mixed")
            | series.str.contains("liquid")
            | series.str.contains("solids")
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


def append_new_entry(new_entry, current_table_data):
    if current_table_data is None:
        new_entry = {0: new_entry}
        return pd.DataFrame.from_dict(new_entry, orient="index")
    new_index = current_table_data.index.max() + 1
    current_table_data.loc[new_index] = new_entry
    return current_table_data


def get_lod(lab, label_col_name, spike_col_name, lod_value_col):
    new_cols = ["LOD"]
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


def get_loq(lab):
    lab["LOQ"] = np.nan
    return lab


def validate_date_text(date_text):
    date_text = str(date_text)
    try:
        if date_text != datetime.strptime(date_text, "%Y-%m-%d").strftime("%Y-%m-%d"):
            raise ValueError
        return True
    except ValueError:
        return False


def remove_bad_rows(lab):
    """ "LabelID column should contain something for all valid rows.
    If it's something else than an empty value and that this empty value
    doesn't cast to datetime, the row should be deleted"""
    LABEL_ID_COL = "D"
    filt1 = ~pd.isnull(lab[LABEL_ID_COL])
    filt2 = ~lab[LABEL_ID_COL].str.lower().str.contains("reference").astype(bool)
    filt3 = ~lab[LABEL_ID_COL].str.lower().str.contains("negative").astype(bool)
    return lab.loc[filt1 & filt2 & filt3]


def get_labsheet_inputs(map_row, lab_data, lab_id):
    lab_input = map_row["labInputs"]
    if lab_input == "":
        return None
    var_name = map_row["variableName"]
    raw_inputs = lab_input.split(";")
    final_inputs = []
    for input_ in raw_inputs:
        if re.match(r"__const__.*:.*", input_):
            value, type_ = input_[len("__const__") :].split(":")
            if type_ == "str":
                value = str(value)
            elif type_ == "int":
                value = int(value)
        elif input_ == "__labID__":
            value = lab_id
        elif input_ == "__varName__":
            value = var_name
        elif input_ == "__default__":
            value = map_row["defaultValue"]
        else:
            value = lab_data[input_]
        final_inputs.append(value)
    return tuple(final_inputs)


def get_static_inputs(map_row, static_data):
    input_sources = map_row["inputSources"]
    if "static" in input_sources:
        static_table = input_sources.split("+")[0]
        static_table = static_table[len("static ") :]
        return static_data[static_table]
    else:
        return None


def get_all_inputs(row):
    static_input = row["static"]
    lab_inputs = row["lab_arguments"]
    if static_input is None and lab_inputs is None:
        inputs = None
    elif static_input is None:
        inputs = lab_inputs
    else:
        inputs = (static_input, *lab_inputs)
    if inputs is None:
        inputs = (row["defaultValue"],)
    return inputs


def parse_sheet(
    mapping,
    static,
    lab_data,
    processing_functions,
    lab_id,
):
    mapping["lab_arguments"] = mapping.apply(
        lambda row: get_labsheet_inputs(row, lab_data, lab_id), axis=1
    )
    mapping["static"] = mapping.apply(
        lambda row: get_static_inputs(row, static), axis=1
    )
    mapping["final_inputs"] = mapping.apply(lambda row: get_all_inputs(row), axis=1)
    mapping["func"] = mapping["processingFunction"].apply(
        lambda x: processing_functions.get(x, CsvMapper.pass_raw)
    )

    mapping["columnName"] = mapping[["table", "elementName", "variableName"]].agg(
        "_".join, axis=1
    )
    to_apply = mapping.loc[:, ["columnName", "func", "final_inputs"]]
    for _, apply_row in to_apply.iterrows():
        col_name = apply_row["columnName"]
        lab_data[col_name] = apply_row["func"](*apply_row["final_inputs"])
    tables = {table: pd.DataFrame() for table in mapping["table"].unique()}
    for table in tables:
        elements = mapping.loc[mapping["table"] == table, "elementName"].unique()
        sub_dfs = []
        for element in elements:
            table_element_filt = (mapping["table"] == table) & (
                mapping["elementName"] == element
            )
            col_names = mapping.loc[table_element_filt, "columnName"]
            var_names = mapping.loc[table_element_filt, "variableName"]
            sub_df = lab_data[col_names]
            sub_df.columns = var_names
            sub_dfs.append(sub_df)
        table_df = pd.concat(sub_dfs, axis=0, ignore_index=True)
        if table in ["WWMeasure", "SiteMeasure"]:
            table_df = table_df.dropna(subset=["value"])
        tables[table] = table_df
    return tables


class QcChecker:
    def __init__(self, version: Literal[2021, 2022], date_check: bool):
        self.version = version
        self.date_check = date_check

    def _find_df_borders(self, sheet_cols, idx_col_pos):
        pos_of_cols_w_headers = []
        for i, col in enumerate(sheet_cols):
            if i == idx_col_pos:
                continue
            if "Unnamed" not in col:
                pos_of_cols_w_headers.append(i + 1)
        last_sheet_col = len(sheet_cols)
        pos_of_cols_w_headers.append(last_sheet_col)

        xl_start_cols = []
        xl_end_cols = []

        pos_of_last_item = len(pos_of_cols_w_headers) - 1
        for i in range(len(pos_of_cols_w_headers.copy())):
            if i == pos_of_last_item:
                # This is the end of the last df, so stop
                break

            start_pos = pos_of_cols_w_headers[i]

            if i == pos_of_last_item - 1:
                end_pos = pos_of_cols_w_headers[i + 1]

            else:
                end_pos = pos_of_cols_w_headers[i + 1] - 1

            start_idx = CsvMapper.excel_style(start_pos + 1)
            end_idx = CsvMapper.excel_style(end_pos + 1)
            xl_start_cols.append(start_idx)
            xl_end_cols.append(end_idx)
        return xl_start_cols, xl_end_cols

    def _get_type_codes(self, sheet_df):
        return sheet_df.iloc[1].dropna().to_list()

    def _get_sample_collection(self, type_codes):
        return [str(x).lower() for x in type_codes[::3]]

    def _get_last_dates(self, sheet_df):
        dates = sheet_df.iloc[2].dropna().to_list()
        temp_dates = []
        for item in dates:
            item = pd.to_datetime(item, errors="raise", infer_datetime_format=True)
            temp_dates.append(item)
        return temp_dates

    def _get_label_ids(self, type_codes):
        return type_codes[2::3]

    def _get_site_ids(self, label_ids):
        sites = []
        for item in label_ids:
            split = item.split("_")[:2]
            site = "_".join(split).lower()
            sites.append(site)
        return sites

    def _get_values_df(self, path, sheet_name, start, end, header_row_pos):
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore")
            return pd.read_excel(
                path,
                sheet_name=sheet_name,
                header=header_row_pos,
                usecols=f"{start}:{end}",
            )

    def _get_index_series(
        self, path: str, sheet_name: str, idx_col: str, header_row_pos: int
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore")
            idx_series = pd.read_excel(
                path,
                sheet_name=sheet_name,
                header=header_row_pos,
                usecols=idx_col,
                squeeze=True,
            )  # type:ignore
        return pd.to_datetime(idx_series, infer_datetime_format=True)

    def _clean_names(self, df):
        rejected_col_template = "Rejected by"
        cols = df.columns
        renamed_cols = {}
        incrementer = 0
        for col in cols:
            new_col = col
            if "rejected" in col.lower():
                new_col = (
                    f"{rejected_col_template}.{incrementer}"
                    if incrementer
                    else rejected_col_template
                )
                incrementer += 1
            elif re.match(r".*\.[0-9]", col):
                dot_idx = col.find(".")
                new_col = col[:dot_idx]

            renamed_cols[col] = new_col
        return df.rename(columns=renamed_cols)

    def _patch_pmmv_names(self, df):
        bad_name = "PMMoV (gc/ml)"
        correct_name = "PMMV (gc/ml)"
        if bad_name in df.columns:
            df = df.rename(columns={bad_name: correct_name})
        return df

    def _extract_dfs(self, path, sheet_name, idx_col_pos=0, header_row_pos=4):
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore")
            sheet_df = pd.read_excel(path, sheet_name=sheet_name, header=0, index_col=0)
        sheet_cols = [str(col) for col in sheet_df.columns]
        start_borders, end_borders = self._find_df_borders(sheet_cols, idx_col_pos)
        idx_col = CsvMapper.excel_style(idx_col_pos + 1)

        dfs = []
        if self.version == 2021:
            cols_to_keep = [
                "BRSV (%rec)",
                "Rejected by",
                "PMMV (gc/ml)",
                "Rejected by.1",
                "SARS (gc/ml)",
                "Rejected by.2",
                "Quality Note",
            ]
        elif self.version == 2022:
            cols_to_keep = [
                "BRSV (%rec)",
                "Quality_flag_BRSV",
                "PMMV (gc/ml)",
                "Quality_flag_PMMoV",
                "SARS_N1 (gc/ml)",
                "Quality_flag_SARS_N1",
                "SARS_N2 (gc/ml)",
                "Quality_flag_SARS_N2",
                "Quality_flags_explication",
            ]
        else:
            raise ValueError(f"Version {self.version} not supported")
        for start, end in zip(start_borders, end_borders):
            vals = self._get_values_df(path, sheet_name, start, end, header_row_pos)
            idx = self._get_index_series(path, sheet_name, idx_col, header_row_pos)
            df = vals.set_index(idx)
            df = self._clean_names(df)
            df = self._patch_pmmv_names(df)
            df = df.loc[:, cols_to_keep]
            df = df.dropna(how="all")
            df.fillna("", inplace=True)
            dfs.append(df)
        return sheet_df, dfs

    def _parse_dates(self, df):
        for col in df.columns:
            if "dateTime" in col:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
        return df

    def _validation_has_started(self, last_date):
        # If there is no 'last checked date', then validation isn't happening at this site, but the data shouldn't be removed
        return not pd.isna(pd.to_datetime(last_date, infer_datetime_format=True))

    def _apply_quality_checks(
        self, mapper, v_df, last_date, site_id, sample_collection
    ):
        if self.version == 2021:
            charac = {
                "BRSV (%rec)": {
                    "rejected_col": "Rejected by",
                    "unit": "pctrecovery",
                    "type": "brsv",
                },
                "PMMV (gc/ml)": {
                    "rejected_col": "Rejected by.1",
                    "unit": "gcml",
                    "type": "pmmov",
                },
                "SARS (gc/ml)": {
                    "rejected_col": "Rejected by.2",
                    "unit": "gcml",
                    "type": "covn1",
                },
            }
        elif self.version == 2022:
            charac = {
                "BRSV (%rec)": {
                    "rejected_col": "Quality_flag_BRSV",
                    "unit": "pctrecovery",
                    "type": "brsv",
                },
                "PMMV (gc/ml)": {
                    "rejected_col": "Quality_flag_PMMoV",
                    "unit": "gcml",
                    "type": "pmmov",
                },
                "SARS_N1 (gc/ml)": {
                    "rejected_col": "Quality_flag_SARS_N1",
                    "unit": "gcml",
                    "type": "covn1",
                },
                "SARS_N2 (gc/ml)": {
                    "rejected_col": "Quality_flag_SARS_N2",
                    "unit": "gcml",
                    "type": "covn2",
                },
            }
        else:
            raise ValueError(f"Version {self.version} not supported")

        samples = mapper.sample.copy()
        samples = self._parse_dates(samples)
        ww = mapper.ww_measure.copy()

        quality_note_col = (
            "Quality Note" if self.version == 2021 else "Quality_flags_explication"
        )
        sample_collection_filt = (
            samples["collection"].str.lower().str.contains(sample_collection)
        )
        sample_sites_filt = samples["siteID"].str.lower().str.contains(site_id)
        for _, row in v_df.iterrows():
            sample_date_filt1 = (
                samples["dateTimeEnd"].dt.date
                == pd.to_datetime(row.name, infer_datetime_format=True).date()
            )
            sample_date_filt2 = (
                samples["dateTime"].dt.date
                == pd.to_datetime(row.name, infer_datetime_format=True).date()
            )
            sample_date_filt = sample_date_filt1 | sample_date_filt2
            sample_tot_filt = (
                sample_date_filt & sample_sites_filt & sample_collection_filt
            )

            samples.loc[sample_tot_filt, ["qualityFlag", "notes"]] = [
                True,
                row[quality_note_col],
            ]

            sample_list = (
                samples.loc[sample_tot_filt, "sampleID"].drop_duplicates().to_list()
            )

            for wwm_info in charac.values():
                if row[wwm_info["rejected_col"]]:
                    # print("Applying flag for ", row.name, col, row[wwm_info["rejected_col"]])
                    ww_type_filt = ww["type"].str.lower().str.contains(wwm_info["type"])
                    ww_unit_filt = ww["unit"].str.lower().str.contains(wwm_info["unit"])
                    ww_sample_filt = ww["sampleID"].isin(sample_list)
                    ww_tot_filt = ww_type_filt & ww_unit_filt & ww_sample_filt

                    ww.loc[ww_tot_filt, ["qualityFlag", "notes"]] = [
                        True,
                        row[quality_note_col],
                    ]

        if "grb" in sample_collection:
            sample_last_date_filt = samples["dateTime"] > last_date
        else:
            sample_last_date_filt = samples["dateTimeEnd"] > last_date

        if self.date_check:
            unchecked_filt = (
                sample_collection_filt & sample_sites_filt & sample_last_date_filt
            )
            samples.loc[unchecked_filt, ["qualityFlag", "notes"]] = [
                True,
                "Unchecked viral measurements",
            ]

            unchecked_sample_ids = (
                samples.loc[unchecked_filt, "sampleID"].drop_duplicates().to_list()
            )

            ww_u_type_filt = (
                ww["type"].str.lower().isin([x["type"] for x in charac.values()])
            )
            ww_u_sample_filt = ww["sampleID"].isin(unchecked_sample_ids)
            ww.loc[ww_u_type_filt & ww_u_sample_filt, ["qualityFlag", "notes"]] = [
                True,
                "Unchecked viral measurement",
            ]

        mapper.sample = samples
        mapper.ww_measure = ww
        return mapper

    def read_validation(self, mapper, path, sheet_name):
        sheet_df, dfs = self._extract_dfs(path, sheet_name)

        last_dates = self._get_last_dates(sheet_df)

        type_codes = self._get_type_codes(sheet_df)
        sample_collections = self._get_sample_collection(type_codes)
        label_ids = self._get_label_ids(type_codes)
        site_ids = self._get_site_ids(label_ids)

        for v_df, last_date, site_id, sample_type in zip(
            dfs, last_dates, site_ids, sample_collections
        ):
            if not self._validation_has_started(last_date):
                continue
            mapper = self._apply_quality_checks(
                mapper, v_df, last_date, site_id, sample_type
            )
        return mapper


class McGillMapper(CsvMapper):
    def __init__(self, version: Literal[2021, 2022], processing_functions=MapperFuncs):
        self.version = version
        super().__init__(processing_functions=processing_functions)

    def get_attr_from_table_name(self, table_name: str) -> str:
        result = None
        for attr, dico in self.conversion_dict.items():
            odm_name = dico["odm_name"]
            if odm_name == table_name:
                result = attr
                break
        if result is None:
            raise ValueError(f"Could not find attribute for table {table_name}")
        return result

    def read_static_data(self, staticdata_path):
        # Get the static data
        static_tables = [
            "Lab",
            "Reporter",
            "Site",
            "Instrument",
            "Polygon",
            "AssayMethod",
        ]
        attrs = []
        for table in static_tables:
            attr = self.get_attr_from_table_name(table)
            if attr is None:
                raise ValueError(f"Could not find attribute for table {table}")
            attrs.append(attr)
        static_data = {}
        excel_mapper = excel_template_mapper.ExcelTemplateMapper()
        if staticdata_path is not None:
            excel_mapper.read(staticdata_path, sheet_names=static_tables)
        for table, attr in zip(static_tables, attrs):
            static_data[table] = getattr(excel_mapper, attr)
            setattr(self, attr, static_data[table])
        return static_data

    def read(
        self,
        labsheet_path,
        staticdata_path,
        worksheet_name,
        lab_id,
        startdate=None,
        enddate=None,
    ):
        # choose the right map file
        if self.version == 2021:
            map_path = MCGILL_MAP_2021
            col_range = "A:BV"
        elif self.version == 2022:
            map_path = MCGILL_MAP_2022
            col_range = "A:CI"
        else:
            raise ValueError(f"Version {self.version} not supported")

        # get the lab data
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore")
            lab = pd.read_excel(
                labsheet_path, sheet_name=worksheet_name, header=None, usecols=col_range
            )
        # parse the headers to deal with merged cells and get unique names
        lab.columns = self.get_excel_style_columns(lab)

        lab_datatypes = lab.iloc[5].values
        lab = lab.iloc[6:]
        lab = remove_bad_rows(lab)
        lab = self.typecast_lab(lab, lab_datatypes)
        # lab = lab.dropna(how="all")
        mapping = pd.read_csv(map_path, header=0)
        mapping.fillna("", inplace=True)
        mapping = mapping.astype(str)
        label_col_name = "D"  # sampleID column
        spike_col_name = "AB"  # spikeID
        lod_value_col = "BI"  # sars-cov-2 gc/rxn
        sample_date_col = "B"  # end date
        lab = get_lod(lab, label_col_name, spike_col_name, lod_value_col)
        lab = get_loq(lab)
        lab = self.filter_by_date(lab, sample_date_col, startdate, enddate)
        static_data = self.read_static_data(staticdata_path)
        dynamic_tables = self.parse_sheet(
            mapping, static_data, lab, self.processing_functions, lab_id
        )
        for table_name, table in dynamic_tables.items():
            attr = self.get_attr_from_table_name(table_name)
            table = self.type_cast_table(table_name, table)
            setattr(self, attr, table)
        return

    def validates(self):
        return True


def debug_test():
    mapper = McGillMapper(2021, processing_functions=MapperFuncs)
    lab_data = "/Users/jeandavidt/Library/CloudStorage/OneDrive-UniversitéLaval/Université/Doctorat/COVID/Latest Data/Input/2021/CentrEau-COVID_Resultats_test.xlsx"  # noqa
    static_data = "/Users/jeandavidt/Library/CloudStorage/OneDrive-UniversitéLaval/Université/Doctorat/COVID/Latest Data/Input/Ongoing/CentrEAU-COVID_Static_Data.xlsx"  # noqa
    sheet_name = "QC Data Daily Samples (McGill)"
    lab_id = "frigon_lab"
    mapper.read(
        lab_data,
        static_data,
        sheet_name,
        lab_id,
        startdate=None,
        enddate=None,
    )
    print(mapper.ww_measure.loc[mapper.ww_measure["qualityFlag"]])
    qc_quality_checker = QcChecker(2021, date_check=True)
    qc_lab = qc_quality_checker.read_validation(
        mapper, lab_data, "QC_Compil_STEP (int)"
    )
    with_flag = qc_lab.ww_measure.loc[qc_lab.ww_measure["qualityFlag"]]
    print(with_flag)


def debug_2021():
    mapper = McGillMapper(2021, processing_functions=MapperFuncs)
    lab_data = "/Users/jeandavidt/Library/CloudStorage/OneDrive-UniversitéLaval/Université/Doctorat/COVID/Latest Data/Input/2021/CentrEau-COVID_Resultats_Quebec_final.xlsx"  # noqa
    static_data = "/Users/jeandavidt/Library/CloudStorage/OneDrive-UniversitéLaval/Université/Doctorat/COVID/Latest Data/Input/Ongoing/CentrEAU-COVID_Static_Data.xlsx"  # noqa

    sheet_name = "QC Data Daily Samples (McGill)"
    lab_id = "frigon_lab"
    mapper.read(
        lab_data,
        static_data,
        sheet_name,
        lab_id,
        startdate=None,
        enddate=None,
    )
    print(mapper.ww_measure.loc[mapper.ww_measure["qualityFlag"]])
    qc_quality_checker = QcChecker(2021, date_check=True)
    qc_lab = qc_quality_checker.read_validation(
        mapper, lab_data, "QC_Compil_STEP (int)"
    )
    with_flag = qc_lab.ww_measure.loc[qc_lab.ww_measure["qualityFlag"]]
    print(with_flag)


def debug_2022():
    mapper = McGillMapper(2022, processing_functions=MapperFuncs)
    lab_data = "/Users/jeandavidt/Library/CloudStorage/OneDrive-UniversitéLaval/Université/Doctorat/COVID/Latest Data/Input/2022/CentrEau-COVID_Resultats_Quebec_2022.xlsx"  # noqa
    static_data = "/Users/jeandavidt/Library/CloudStorage/OneDrive-UniversitéLaval/Université/Doctorat/COVID/Latest Data/Input/Ongoing/CentrEAU-COVID_Static_Data.xlsx"  # noqa

    sheet_name = "QC Data Daily Samples (McGill)"
    lab_id = "frigon_lab"
    mapper.read(
        lab_data,
        static_data,
        sheet_name,
        lab_id,
        startdate=None,
        enddate=None,
    )
    print(mapper.ww_measure.loc[mapper.ww_measure["qualityFlag"]])
    qc_quality_checker = QcChecker(2022, date_check=True)
    qc_lab = qc_quality_checker.read_validation(
        mapper, lab_data, "QC_Compil_STEP (int)"
    )

    print(qc_lab.ww_measure.loc[qc_lab.ww_measure["qualityFlag"]])


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings(action="error")
        # debug_test()
        debug_2021()
        # debug_2022()
