import pandas as pd
import numpy as np
from datetime import datetime
import os
import re
import warnings
from wbe_odm import utilities
from wbe_odm.odm_mappers import base_mapper
from wbe_odm.odm_mappers import excel_template_mapper


LABEL_REGEX = r"[a-zA-Z]+_[0-9]+(\.[0-9])?_[a-zA-Z0-9]+_[a-zA-Z0-9]+"

directory = os.path.dirname(__file__)

MCGILL_MAP_NAME = directory + "/" + "mcgill_map.csv"

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def excel_style(col):
    """ Convert given column number to an Excel-style column name. """
    result = []
    while col:
        col, rem = divmod(col-1, 26)
        result[:0] = LETTERS[rem]
    return "".join(result)


def parse_date(item):
    if isinstance(item, (str, datetime)):
        return pd.to_datetime(item)
    return pd.NaT


def str_date_from_timestamp(timestamp_series):
    return timestamp_series.dt.strftime("%Y-%m-%d").fillna("")


def clean_up(df, molecular_cols, meas_cols):
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
                lambda x: parse_date(x))
    return df


def typecast_column(desired_type, series):
    if desired_type == "bool":
        series = series.astype(str)
        series = series.str.strip().str.lower()
        series = series.apply(
            lambda x: base_mapper.replace_unknown_by_default(x, ""))
        series = series.str.replace("oui", "true", case=False)\
            .str.replace("yes", "true", case=False)\
            .str.startswith("true")
    elif desired_type in ["string", "category"]:
        series = series.astype(str)
        series = series.str.lower()
        series = series.str.strip()
        series = series.apply(
            lambda x: base_mapper.replace_unknown_by_default(x, ""))
    elif desired_type in ["int64", "float64"]:
        series = pd.to_numeric(series, errors="coerce")
    elif desired_type == "datetime64[ns]":
        series = pd.to_datetime(series, errors="coerce")
    series = series.astype(desired_type)
    return series


def typecast_lab(lab, types):
    clean_types = []
    for datatype in types:
        if datatype in base_mapper.UNKNOWN_TOKENS:
            datatype = "string"
        datatype = str(datatype)\
            .replace("date", "datetime64[ns]") \
            .replace("mixed", "object") \
            .replace("boolean", "bool") \
            .replace("float", "float64") \
            .replace("integer", "int64") \
            .replace("number", "float64") \
            .replace("text", "string") \
            .replace("blob", "object")
        clean_types.append(datatype)
    for i, col_name in enumerate(lab.columns):
        lab[col_name] = typecast_column(clean_types[i], lab[col_name])
    return lab


def clean_labels(label):
    parts = str(label).lower().split("_")
    parts = [part.strip() for part in parts]
    return "_".join(parts)


def get_sample_type(sample_type):
    """acceptable_types = [
        "qtips", "filter", "gauze",
        "swrsed", "pstgrit", "psludge",
        "pefflu", "ssludge", "sefflu",
        "water", "faeces", "rawww", ""
    ]"""
    sample_type = sample_type.str.strip().str.lower()
    return sample_type


def get_cp_start_date(start_col, end_col, sample_type):
    df = pd.concat([start_col, end_col, sample_type], axis=1)
    df.columns = ["start", "end", "type"]
    df["s"] = df.apply(
        lambda row: utilities.calc_start_date(row["end"], row["type"]), axis=1)
    return df["s"]


def get_grab_date(end_series, type_series):
    df = pd.concat([end_series, type_series], axis=1)
    df.columns = ["end", "type"]
    df["date_grab"] = pd.NaT
    filt = df["type"].str.contains("grb")
    df.loc[filt, "date_grab"] = df.loc[filt, "end"]
    return df["date_grab"]


def get_collection_method(collection):
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


def pass_raw(*args):
    if len(args) == 0:
        return None
    elif len(args) == 1:
        return args[0]
    arguments = pd.concat([arg for arg in args], axis=1)
    return arguments.agg(",".join, axis=1)


def get_assay_method_id(sample_type, concentration_method, assay_date):
    formatted_date = str_date_from_timestamp(assay_date)
    clean_series = []
    for series in [sample_type, concentration_method, formatted_date]:
        series = series.fillna("").astype(str)
        clean_series.append(series)
    df = pd.concat(clean_series, axis=1)
    return df.agg("_".join, axis=1)


def get_assay_instrument(static_methods, sample_type, concentration_method):
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


def get_assay_name(static_methods, sample_type, concentration_method):
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


def write_concentration_method(conc_method, conc_volume, ph_final):
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


def get_site_id(labels):
    def extract_from_label(label_id):
        if re.match(LABEL_REGEX, label_id):
            label_parts = label_id.split("_")
            return "_".join(label_parts[0:2])
        else:
            return ""
    clean_label_series = labels.apply(lambda x: clean_labels(x))
    return clean_label_series.apply(lambda x: extract_from_label(x))


def sample_is_pooled(pooled):
    # It isn't clear what the sheet wants the user to do - either say "Yes"
    # if the sample is pooled, or actually put in the sample ids
    # of the children. For now, let's only check if it is pooled or not
    return pooled != ""


def get_children_samples(pooled, sample_date):
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
    clean_date = str_date_from_timestamp(sample_date)
    df = pd.concat([pooled, clean_date], axis=1)
    df.columns = ["pooled", "clean_date"]
    df["children_ids"] = df.apply(lambda row: make_children_ids(row))
    return df["children_ids"]


def get_sample_id(label_id, sample_date, spike_batch, lab_id, sample_index):
    # TODO: Deal with index once it's been implemented in McGill sheet
    clean_date = str_date_from_timestamp(sample_date)
    clean_label = label_id.apply(lambda x: clean_labels(x))

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


def get_wwmeasure_id(
        label_id,
        sample_date,
        spike_batch,
        lab_id,
        sample_index,
        meas_type,
        meas_date,
        index):
    # TODO: Deal with index once it's been implemented in McGill sheet
    sample_id = get_sample_id(
        label_id,
        sample_date,
        spike_batch,
        lab_id,
        sample_index
    )
    meas_date = str_date_from_timestamp(meas_date)
    df = pd.concat([sample_id, meas_date], axis=1)
    df["meas_type"] = meas_type
    df["index_no"] = str(index) if not isinstance(index, pd.Series) \
        else index.astype(str)
    return df.agg("_".join, axis=1)


def get_reporter_id(static_reporters, name):
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


def has_quality_flag(flag):
    return flag != ""


def get_sample_volume(vols, default):
    vols = vols.apply(lambda x: x if not pd.isna(x) else default)
    return vols


def get_field_sample_temp(series):
    temp_map = {
        "refrigerated": 4.0,
        "ice": 0.0,
        "norefrigaration": 20.0,
        # "norefrigeration": np.nan
    }
    series = series.str.lower().map(temp_map)
    return series


def get_shipped_on_ice(series):
    series = series.str.lower()
    map_to = {
        "yes": True,
        "no": False
    }
    return series.map(map_to)


def grant_access(access):
    return access.str.lower().isin(["", "1", "yes", "true"])


def validate_fraction_analyzed(series):
    filt = (
        series.str.contains("mixed") |
        series.str.contains("liquid") |
        series.str.contains("solids")
    )
    series.loc[~filt] = ""
    return series


def validate_value(values):
    return pd.to_numeric(values, errors="coerce")


processing_functions = {
    "get_grab_date": get_grab_date,
    "get_start_date": get_cp_start_date,
    "get_collection_method": get_collection_method,
    "get_sample_type": get_sample_type,
    "get_assay_method_id": get_assay_method_id,
    "get_assay_instrument": get_assay_instrument,
    "get_assay_name": get_assay_name,
    "write_concentration_method": write_concentration_method,
    "get_site_id": get_site_id,
    "sample_is_pooled": sample_is_pooled,
    "get_children_samples": get_children_samples,
    "get_sample_id": get_sample_id,
    "get_wwmeasure_id": get_wwmeasure_id,
    "get_reporter_id": get_reporter_id,
    "has_quality_flag": has_quality_flag,
    "grant_access": grant_access,
    "get_sample_volume": get_sample_volume,
    "get_field_sample_temp": get_field_sample_temp,
    "get_shipped_on_ice": get_shipped_on_ice,
    "validate_fraction_analyzed": validate_fraction_analyzed,
    "validate_value": validate_value,

}


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


def filter_by_date(df, date_col, start, end):
    if start is not None:
        startdate = pd.to_datetime(start)
        start_filt = (df[date_col] > startdate)
    else:
        start_filt = None
    if end is not None:
        enddate = pd.to_datetime(end)
        end_filt = (df[date_col] < enddate)
    else:
        end_filt = None
    if start_filt is None and end_filt is None:
        return df
    elif start_filt is None:
        return df[end_filt]
    elif end_filt is None:
        return df[start_filt]
    else:
        return df[start_filt & end_filt]


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


def get_labsheet_inputs(map_row, lab_data, lab_id):
    lab_input = map_row["labInputs"]
    if lab_input == "":
        return None
    var_name = map_row["variableName"]
    raw_inputs = lab_input.split(";")
    final_inputs = []
    for input_ in raw_inputs:
        if re.match(r"__const__.*:.*", input_):
            value, type_ = input_[len("__const__"):].split(":")
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
        static_table = static_table[len("static "):]
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
        inputs = tuple([row["defaultValue"]])
    return inputs


def parse_sheet(mapping, static, lab_data, processing_functions, lab_id,):
    mapping["lab_arguments"] = mapping.apply(
        lambda row: get_labsheet_inputs(row, lab_data, lab_id), axis=1)
    mapping["static"] = mapping.apply(
        lambda row: get_static_inputs(row, static), axis=1)
    mapping["final_inputs"] = mapping.apply(
        lambda row: get_all_inputs(row), axis=1)
    mapping["func"] = mapping["processingFunction"].apply(
        lambda x: processing_functions.get(x, pass_raw))

    mapping["columnName"] = mapping[
        ["table", "elementName", "variableName"]].agg("_".join, axis=1)
    to_apply = mapping.loc[
        :, ["columnName", "func", "final_inputs"]]
    for _, apply_row in to_apply.iterrows():
        col_name = apply_row["columnName"]
        lab_data[col_name] = apply_row["func"](*apply_row["final_inputs"])
    tables = {table: None for table in mapping["table"].unique()}
    for table in tables:
        elements = mapping.loc[
            mapping["table"] == table, "elementName"
        ].unique()
        sub_dfs = []
        for element in elements:
            table_element_filt = (mapping["table"] == table)\
                 & (mapping["elementName"] == element)
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
    def _find_df_borders(self, sheet_cols, idx_col_pos):
        pos_of_cols_w_headers = []
        for i, col in enumerate(sheet_cols):
            if i==idx_col_pos:
                continue
            if 'Unnamed' not in col:
                pos_of_cols_w_headers.append(i+1)
        last_sheet_col = len(sheet_cols)
        pos_of_cols_w_headers.append(last_sheet_col)

        xl_start_cols = []
        xl_end_cols = []

        pos_of_last_item = len(pos_of_cols_w_headers) - 1
        for i in range(len(pos_of_cols_w_headers.copy())):
            if i == pos_of_last_item:
                #This is the end of the last df, so stop
                break

            start_pos = pos_of_cols_w_headers[i]
            
            if i == pos_of_last_item-1:
                end_pos = pos_of_cols_w_headers[i+1]
                
            else:
                end_pos = pos_of_cols_w_headers[i+1] -1


            start_idx = excel_style(start_pos+1)
            end_idx = excel_style(end_pos+1)
            xl_start_cols.append(start_idx)
            xl_end_cols.append(end_idx)
        return xl_start_cols, xl_end_cols


    def _get_type_codes(self, sheet_df):
        return sheet_df.iloc[1].dropna().to_list()


    def _get_sample_collection(self, type_codes):
        return type_codes[::3]


    def _get_last_dates(self, sheet_df):
        dates = sheet_df.iloc[2].dropna().to_list()
        temp_dates =[]
        for item in dates:
            try:
                item = pd.to_datetime(item)
                temp_dates.append(item)
            except Exception:
                temp_dates.append(pd.NaT)
        return temp_dates


    def _get_label_ids(self, type_codes):
        return type_codes[2::3]


    def _get_site_ids(self, label_ids):
        sites = []
        for item in label_ids:
            split = item.split("_")[0:2]
            site = "_".join(split).lower()
            sites.append(site)
        return sites

    def _get_values_df(self, path, sheet_name, start, end, header_row_pos):
        return pd.read_excel(
                path,
                sheet_name=sheet_name,
                header=header_row_pos,
                usecols =f"{start}:{end}"
                )

    def _get_index_series(self, path, sheet_name, idx_col, header_row_pos):
        idx_series = pd.read_excel(
                path,
                sheet_name=sheet_name,
                header=header_row_pos,
                usecols = idx_col,
                squeeze=True
            )
        idx_series = pd.to_datetime(idx_series).dt.strftime("%Y-%m-%d")
        return pd.to_datetime(idx_series)
        
    def _clean_names(self, df):
        rejected_col_template = "Rejected by"
        cols = df.columns
        renamed_cols = {}
        incrementer = 0
        for col in cols:
            new_col = col
            if 'rejected' in col.lower():
                if not incrementer:
                    new_col = rejected_col_template
                else:
                    new_col = rejected_col_template + "." + str(incrementer)
                incrementer +=1
            elif re.match(".*\.[0-9]", col):
                dot_idx = col.find(".")
                new_col = col[0:dot_idx]

            renamed_cols[col] = new_col
        return df.rename(columns = renamed_cols) 

    def _extract_dfs(self, path, sheet_name, idx_col_pos=0, header_row_pos=4):
        sheet_df = pd.read_excel(path, sheet_name=sheet_name, header=0, index_col=0)
        sheet_cols = [str(col) for col in sheet_df.columns]
        start_borders, end_borders = self._find_df_borders(sheet_cols, idx_col_pos)
        idx_col = excel_style(idx_col_pos+1)
        
        dfs = []
        i=0
        for start, end in zip(start_borders, end_borders):
            vals = self._get_values_df(path, sheet_name, start, end, header_row_pos)
            idx = self._get_index_series(path, sheet_name, idx_col, header_row_pos)
            df = vals.set_index(idx)
            df = self._clean_names(df)
            cols_to_keep = ["BRSV (%rec)","Rejected by", "PMMV (gc/ml)","Rejected by.1", "SARS (gc/ml)", "Rejected by.2", "Quality Note"]
            df = df[cols_to_keep]
            df = df.dropna(how='all')
            df.fillna("", inplace=True)
            dfs.append(df)
            i += 1
        return sheet_df, dfs


    def _parse_dates(self, df):
        for col in df.columns:
            if 'dateTime' in col:
                df[col] = pd.to_datetime(df[col])
        return df

    def _validation_has_started(self, last_date):
        # If there is no 'last checked date', then validation isn't happening at this site, but the data shouldn't be removed
        return not pd.isna(pd.to_datetime(last_date))
    
    def _apply_quality_checks(self, mapper,  v_df, last_date, site_id, sample_collection):
        charac = {
            "BRSV (%rec)": {
                "rejected_col": "Rejected by",
                "unit": "pctrecovery",
                "type": "nbrsv",
            },
            "PMMV (gc/ml)": {
                "rejected_col": "Rejected by.1",
                "unit": "gcml",
                "type": "npmmov",
            },
            "SARS (gc/ml)": {
                "rejected_col": "Rejected by.2",
                "unit": "gcml",
                "type": "covn2",
            }
        }
        
        samples = mapper.sample
        samples = self._parse_dates(samples)
        ww = mapper.ww_measure
        
        sample_collection_filt = samples["collection"].str.contains(sample_collection)
        sample_sites_filt = samples["siteID"].str.lower().str.contains(site_id)
        for _, row in v_df.iterrows():
            sample_date_filt1 = samples["dateTimeEnd"].dt.date == pd.to_datetime(row.name).date()
            sample_date_filt2 = samples["dateTime"].dt.date == pd.to_datetime(row.name).date()
            sample_date_filt = sample_date_filt1 | sample_date_filt2
            sample_tot_filt = sample_date_filt & sample_sites_filt & sample_collection_filt

            samples.loc[sample_tot_filt, ["qualityFlag", "notes"]] = [True, row["Quality Note"]]

            sample_list = samples.loc[sample_tot_filt, "sampleID"].drop_duplicates().to_list()

            for col, wwm_info in charac.items():
                if row[wwm_info["rejected_col"]]:
                    # print("Applying flag for ", row.name, col, row[wwm_info["rejected_col"]])
                    ww_type_filt = ww["type"].str.lower().str.contains(wwm_info["type"])
                    ww_unit_filt = ww["unit"].str.lower().str.contains(wwm_info["unit"])
                    ww_sample_filt = ww["sampleID"].isin(sample_list)
                    ww_tot_filt = ww_type_filt & ww_unit_filt & ww_sample_filt

                    ww.loc[ww_tot_filt, ["qualityFlag", "notes"]] = [True, row["Quality Note"]]

        if 'grb' in sample_collection:
            sample_last_date_filt = samples['dateTime'] > last_date
        else:
            sample_last_date_filt = samples['dateTimeEnd'] > last_date
        
        unchecked_filt = sample_collection_filt & sample_sites_filt & sample_last_date_filt
        samples.loc[unchecked_filt, ["qualityFlag", "notes"]] = [True, "Unchecked viral measurements"]
        
        unchecked_sample_ids = samples.loc[unchecked_filt, "sampleID"].drop_duplicates().to_list()

        ww_u_type_filt = ww["type"].str.lower().isin([x["type"] for x in charac.values()])
        ww_u_sample_filt = ww['sampleID'].isin(unchecked_sample_ids)
        ww.loc[ww_u_type_filt & ww_u_sample_filt, ["qualityFlag", "notes"]] = [True, "Unchecked viral measurement"]
        
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

        for v_df, last_date, site_id, sample_type in zip(dfs, last_dates, site_ids, sample_collections):
            if not self._validation_has_started(last_date):
                continue
            mapper = self._apply_quality_checks(mapper, v_df, last_date, site_id, sample_type)
        return mapper


class McGillMapper(base_mapper.BaseMapper):
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
        lab.columns = [
            excel_style(i+1)
            for i, _ in enumerate(lab.columns.to_list())
        ]
        lab_datatypes = lab.iloc[5].values
        lab = lab.iloc[6:]
        lab = remove_bad_rows(lab)
        lab = typecast_lab(lab, lab_datatypes)
        lab = lab.dropna(how="all")
        mapping = pd.read_csv(map_path, header=0)
        mapping.fillna("", inplace=True)
        mapping = mapping.astype(str)
        label_col_name = "D"  # sampleID column
        spike_col_name = "AB"  # spikeID
        lod_value_col = "BI"  # sars-cov-2 gc/rxn
        sample_date_col = "B"  # end date
        lab = get_lod(lab, label_col_name, spike_col_name, lod_value_col)
        lab = filter_by_date(lab, sample_date_col, startdate, enddate)
        static_data = self.read_static_data(staticdata_path)
        dynamic_tables = parse_sheet(
            mapping,
            static_data,
            lab,
            processing_functions,
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
    
    def add_test_cov(ww):
        ww["qualityFlag"] = ww["qualityFlag"].fillna(False)
        type_filt = ww["type"].str.lower() == "covn2"
        quality_filt = ~ww["qualityFlag"]
        df = ww.loc[type_filt & quality_filt]
        return df["wwMeasureID"].to_list()
    covs = []

    mapper = McGillMapper()
    lab_data = "/Users/jeandavidt/OneDrive - Université Laval/COVID/Latest Data/Input/CentrEau-COVID_Resultats_Quebec_final.xlsx" # noqa
    static_data = "/Users/jeandavidt/OneDrive - Université Laval/COVID/Latest Data/Input/CentrEAU-COVID_Static_Data.xlsx"  # noqa
    sheet_name = "QC Data Daily Samples (McGill)"
    lab_id = "frigon_lab"
    mapper.read(lab_data,
                static_data,
                sheet_name,
                lab_id,
                map_path=MCGILL_MAP_NAME,
                startdate=None,
                enddate=None)
    covs.append(add_test_cov(mapper.ww_measure))
    qual_check = QcChecker()
    mapper_quality = qual_check.read_validation(mapper, lab_data, "QC_Compil_STEP (int)")
    covs.append(add_test_cov(mapper_quality.ww_measure))
    set_ini = set(covs[0])
    set_final = set(covs[1])
    # print(set_ini - set_final)
