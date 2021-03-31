import pandas as pd
import numpy as np
import datetime
import re
from wbe_odm.odm_mappers import base_mapper
from wbe_odm.odm_mappers import excel_template_mapper

LABEL_REGEX = r"[a-zA-Z]+_[0-9]+(\.[0-9])?_[a-zA-Z0-9]+_[a-zA-Z0-9]+"


def parse_mcgill_headers(array):
    column_titles = []
    empty_token = "na"
    arr = array.copy()
    # iterate over each row
    for i, row in enumerate(arr):
        # iterate over every item in the row
        for j, item in enumerate(row):
            # if the field is empty, we must figure out how to fill it
            if re.match(base_mapper.UNKNOWN_REGEX, item):
                # if we're in the top row, the only option is to get the item
                # on the left
                if i == 0:
                    if j == 0:
                        array[i, j] = empty_token
                    else:
                        array[i, j] = array[i, j-1]
                # if we're on another row, we must check tow things:
                # 1) The value of the row above
                # 2) The value of the row above the item on the left
                else:
                    # Of course, the first column doesn't have a left neighbor
                    if j == 0:
                        above_left = empty_token
                        above = empty_token
                        left = empty_token
                    else:
                        above_left = array[i-1, j-1]
                        above = array[i-1, j]
                        left = array[i, j-1]
                    # Now, if the item above the current item isn't the same as
                    # the one above the one on the left, we are in a different
                    #  catgeory, so we can't fill with what's on the left.
                    if above_left != above:
                        array[i, j] = empty_token
                    # If they are the same however, we can fill in with the
                    # left value
                    else:
                        array[i, j] = left
            # If the itemisn't empty, we clean it up
            else:
                # if it's a number, make sure it appears as an integer
                if re.match(r".*\.0", item):
                    item = item[:-2]
                array[i, j] = str(item).lower().strip()

    for i, _ in enumerate(array[0, :]):
        column_name = ".".join(array[:, i])
        column_titles.append(column_name)

    return column_titles


def parse_date(item):
    if isinstance(item, str) or isinstance(item, datetime.datetime):
        return pd.to_datetime(item)
    return pd.NaT


def str_date_from_timestamp(timestamp):
    return str(timestamp.date())


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
        series = series.str.replace(
            base_mapper.UNKNOWN_REGEX, "", regex=True)\
            .str.replace("oui", "true", case=False)\
            .str.replace("yes", "true", case=False)\
            .str.startswith("true")
    elif desired_type == "string" or desired_type == "category":
        series = series.astype(str)
        series = series.str.lower()
        series = series.str.strip()
        series = series.str.replace(
            base_mapper.UNKNOWN_REGEX, "", regex=True, case=False)
    elif desired_type in ["int64", "float64"]:
        series = pd.to_numeric(series, errors="coerce")
    series = series.astype(desired_type)
    return series


def typecast_lab(lab, types):
    types["type"] = types["type"]\
        .replace(r"date(time)?", "datetime64[ns]", regex=True) \
        .replace("boolean", "bool") \
        .replace("float", "float64") \
        .replace("integer", "int64") \
        .replace("blob", "object")
    for col in lab.columns:
        try:
            desired_type = types[types["column"] == col].iloc[0]["type"]
        except Exception:
            desired_type = "string"
        lab[col] = typecast_column(desired_type, lab[col])
    return lab


def get_labsheet_inputs(map_row, lab_row, lab_id):
    lab_input = map_row["labInputs"]
    if lab_input == "":
        return None
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
        else:
            value = lab_row[input_]
        final_inputs.append(value)
    return tuple(final_inputs)


def pass_raw(*args):
    if len(args) == 1:
        return args[0]
    arguments = [str(arg) for arg in args]
    return ",".join(arguments)


def get_assay_method_id(sample_type, concentration_method, assay_date):
    formatted_date = str_date_from_timestamp(assay_date)
    return "_".join([sample_type, concentration_method, formatted_date])


def get_assay_instrument(static_methods, sample_type, concentration_method):
    general_id = str("_".join([sample_type, concentration_method])).lower()
    filt = static_methods["assayMethodID"] == general_id
    generic_methods = static_methods.loc[filt]
    try:
        generic_method = generic_methods.iloc[0]
    except IndexError:
        return ""
    instrument_id = generic_method["instrumentID"]
    return instrument_id


def get_assay_name(static_methods, sample_type, concentration_method):
    general_id = str("_".join([sample_type, concentration_method])).lower()
    filt = static_methods["assayMethodID"] == general_id
    generic_methods = static_methods.loc[filt]
    try:
        generic_method = generic_methods.iloc[0]
    except IndexError:
        return ""
    name = generic_method["name"]
    return name


def write_concentration_method(conc_method, conc_volume, ph_final):
    return f"{conc_method}, Volume:{conc_volume} mL, Final pH:{ph_final}"


def get_site_id(label_id):
    if re.match(LABEL_REGEX, label_id):
        label_parts = label_id.split("_")
        return "_".join(label_parts[0:2])
    else:
        return ""


def sample_is_pooled(pooled):
    # It isn't clear what the sheet wants the user to do - either say "Yes"
    # if the sample is pooled, or actually put in the sample ids
    # of the children. For now, let's only check if it is pooled or not
    return pooled != ""


def get_children_samples(pooled, sample_date):
    clean_date = str_date_from_timestamp(sample_date)
    if "," in pooled:
        pooled = pooled.split(",")
    children_ids = []
    for item in pooled:
        if re.match(LABEL_REGEX, item):
            child_id = "_".join([item, clean_date])
            children_ids.append(child_id)
    return children_ids


def get_sample_id(label_id, sample_date, lab_id):
    clean_date = str_date_from_timestamp(sample_date)
    clean_label = str(label_id).lower()
    if re.match(LABEL_REGEX, clean_label):
        return "_".join([clean_label, clean_date])
    else:
        return "_".join([lab_id, clean_label, clean_date])


def get_wwmeasure_id(label_id,
                     sample_date,
                     lab_id,
                     meas_type,
                     meas_date,
                     index):
    sample_id = get_sample_id(label_id, sample_date, lab_id)
    meas_date = str_date_from_timestamp(meas_date)
    index = str(index)
    return "_".join([sample_id, meas_date, meas_type, index])


def get_reporter_id(name, lab_id):
    if "/" in name:
        name = name.split("/")[0]
    return f"{lab_id}_{name}"


def has_quality_flag(flag):
    return flag != ""


def grant_access(access):
    return str(access).lower() in ["", "1", "yes", "true"]


processing_functions = {
    "get_assay_method_id": get_assay_method_id,
    "get_assay_instrument": get_assay_instrument,
    "get_assay_name": get_assay_name,
    "write_concentration_method": write_concentration_method,
    "get_site_id": get_site_id,
    "sample_is_pooled": sample_is_pooled,
    "get_children_samples": get_children_samples,
    "get_wwmeasure_id": get_wwmeasure_id,
    "get_reporter_id": get_reporter_id,
    "has_quality_flag": has_quality_flag,
    "grant_access": grant_access,
}


def append_new_entry(new_entry, current_table_data):
    if current_table_data is None:
        new_entry = {0: new_entry}
        new_table_data = pd.DataFrame.from_dict(new_entry, orient='index')
        return new_table_data
    new_index = current_table_data.index.max() + 1
    current_table_data.loc[new_index] = new_entry
    return current_table_data


def parse_lab_row(lab_row, mapping, static_dico, lab_id, entries_to_store):
    elements = list(mapping["elementName"].unique())
    for i, element in enumerate(elements):
        filt = mapping["elementName"] == element
        odm_table = mapping.loc[filt].iloc[0]["table"]
        fields = mapping.loc[mapping["elementName"] == element]
        new_entry = {}
        for _, field in fields.iterrows():
            field_name = field["variableName"]

            input_sources = field["inputSources"]
            if "static sheet" in input_sources:
                static_input = static_dico[odm_table]
            else:
                static_input = None
            lab_inputs = get_labsheet_inputs(field, lab_row, lab_id)
            if static_input is None and lab_inputs is None:
                inputs = None
            elif static_input is None:
                inputs = lab_inputs
            else:
                inputs = (static_input, *lab_inputs)
            if inputs is None:
                value = field["defaultValue"]
            else:
                func = field["processingFunction"]
                value = processing_functions.get(func, pass_raw)(*inputs)
            new_entry[field_name] = value
        current_table_data = entries_to_store[odm_table]
        updated_table_data = append_new_entry(
            new_entry,
            current_table_data
        )
        entries_to_store[odm_table] = updated_table_data
    return entries_to_store


def concat_entries(store_dict, new_dict):
    pass


def get_lod(lab):
    new_cols = ['pcr.sars-cov-2.lod.na', 'pcr.sars-cov-2.loq.na']
    label_col = "na.na.label_id.na"
    spike_col = "concentration.key parametres.spikebatch_id.na"
    filt = lab[label_col] == "negative"
    lod_source = "pcr.sars-cov-2.gc/rx.na"
    cols_to_keep = [
        label_col,
        spike_col,
        lod_source,
    ]
    lod_df = lab.loc[filt][cols_to_keep]
    lab[new_cols] = np.nan
    lod_df[spike_col] = lod_df[spike_col].replace("", np.nan)
    lod_df = lod_df.dropna(subset=[spike_col])
    spike_ids = list(lod_df[spike_col].dropna().unique())
    for spike_id in spike_ids:
        lod_filt = lod_df[spike_col] == spike_id
        lab_filt = lab[spike_col] == spike_id
        lod = lod_df.loc[lod_filt].iloc[0].loc[lod_source]
        lab_rows = lab.loc[lab_filt]
        for col in new_cols:
            lab_rows[col] = lod
        lab.loc[lab_filt] = lab_rows
    return lab


class McGillMapper(base_mapper.BaseMapper):
    def get_attr_from_table_name(self, table_name):
        for attr, dico in self.conversion_dict.items():
            odm_name = dico["odm_name"]
            if odm_name == table_name:
                return attr

    def read(self,
             labsheet_path,
             staticdata_path,
             typesheet_path,
             mapsheet_path,
             worksheet_name,
             lab_id,
             startdate=None,
             enddate=None):
        # get the lab data
        lab = pd.read_excel(labsheet_path,
                            sheet_name=worksheet_name,
                            header=None,
                            usecols="A:BS")
        # parse the headers to deal with merged cells and get unique names
        lab.columns = parse_mcgill_headers(lab.iloc[0:4].to_numpy(dtype=str))
        lab = lab.iloc[4:]
        lab = lab.dropna(how="all")
        types = pd.read_csv(typesheet_path, header=0)
        mapping = pd.read_csv(mapsheet_path, header=0)
        mapping.fillna("", inplace=True)
        mapping = mapping.astype(str)

        lab = typecast_lab(lab, types)
        lab = get_lod(lab)
        # Get the static data
        static_tables = [
            "Lab",
            "Reporter",
            "Site",
            "AssayMethod",
            "Instrument",
            "Polygon",
        ]
        attrs = []
        for table in static_tables:
            attr = self.get_attr_from_table_name(table)
            attrs.append(attr)
        static_data = {}
        excel_mapper = excel_template_mapper.ExcelTemplateMapper()
        excel_mapper.read(staticdata_path)
        for table, attr in zip(static_tables, attrs):
            static_data[table] = getattr(excel_mapper, attr)
        entries_to_store = {
            "WWMeasure": None,
            "Sample": None,
            "AssayMethod": None,
            "Reporter": None,
        }
        for _, row in lab.iterrows():
            entries_to_store = parse_lab_row(
                row,
                mapping,
                static_data,
                lab_id,
                entries_to_store
            )
        for key, table in entries_to_store.items():
            attr = self.get_attr_from_table_name(key)
            setattr(self, attr, table)
        return

    def validates(self):
        return True


if __name__ == "__main__":
    mapper = McGillMapper()
    start = "2021-03-01"
    end = "2021-03-15"
    labsheet_path = "Data/Lab/McGill/mcgill_lab.xlsx"
    staticdata_path = "Data/Lab/McGill/mcgill_static.xlsx"
    worksheet_name = "Mtl Data Daily Samples (McGill)"
    typesheet_path = "Data/Lab/McGill/mcgill_types.csv"
    mapsheet_path = "Data/Lab/McGill/mcgill_map.csv"
    mapper.read(labsheet_path,
                staticdata_path,
                typesheet_path,
                mapsheet_path,
                worksheet_name,
                lab_id="frigon_lab",
                startdate=start, enddate=end)
    
