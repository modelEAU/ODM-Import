import pandas as pd
import numpy as np
from datetime import datetime
import re
from wbe_odm.odm_mappers import base_mapper
from wbe_odm.odm_mappers import excel_template_mapper

LABEL_REGEX = r"[a-zA-Z]+_[0-9]+(\.[0-9])?_[a-zA-Z0-9]+_[a-zA-Z0-9]+"


LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def get_sample_type(sample_type):
    acceptable_types = [
        "qtips", "filter", "gauze",
        "swrsed", "pstgrid", "psludge",
        "pefflu", "ssludge", "sefflu",
        "water", "faeces"
    ]
    sample_type = sample_type.strip()
    if sample_type.lower() == "raw":
        return "rawWW"
    elif sample_type.lower() in acceptable_types:
        return sample_type
    else:
        return ""


def get_collection_method(collection_str):
    collection = collection_str.strip()
    if re.match(r"cp[TF]P[0-9]+h", collection):
        return collection
    elif collection == "grb":
        return collection
    elif "grb" in collection:
        added_bit = collection[3:]
        return "grb" + "Cp" + added_bit
    else:
        return ""


def excel_style(col):
    """ Convert given column number to an Excel-style column name. """
    result = []
    while col:
        col, rem = divmod(col-1, 26)
        result[:0] = LETTERS[rem]
    return "".join(result)


def parse_date(item):
    if isinstance(item, str) or isinstance(item, datetime):
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
    clean_types = []
    for datatype in types:
        if datatype in base_mapper.UNKNOWN_TOKENS:
            datatype = "string"
        datatype = str(datatype)\
            .replace("date", "datetime64[ns]") \
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


def get_labsheet_inputs(map_row, lab_row, lab_id):
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
    if len(children_ids) == 0:
        return ""
    else:
        return ",".join(children_ids)


def get_sample_id(label_id, sample_date, lab_id, index=1):
    clean_date = str_date_from_timestamp(sample_date)
    clean_label = str(label_id).lower()
    if lab_id == "modeleau_lab":
        clean_label = clean_label.replace("raw", "pstgrit")
    if re.match(LABEL_REGEX, clean_label):
        return "_".join([clean_label, clean_date, str(index)])
    else:
        return "_".join([lab_id, clean_label, clean_date, str(index)])


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


def get_reporter_id(static_reporters, name):
    name = name.lower()
    if "/" in name:
        name = name.split("/")[0]
    name = name.strip()
    reporters_w_name = static_reporters.loc[
        static_reporters["reporterID"].str.lower().str.contains(name)]
    if len(reporters_w_name) > 0:
        reporterID = reporters_w_name.iloc[0]["reporterID"]
    else:
        reporterID = name
    return reporterID


def has_quality_flag(flag):
    return flag != ""


def grant_access(access):
    return str(access).lower() in ["", "1", "yes", "true"]


processing_functions = {
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
}


def append_new_entry(new_entry, current_table_data):
    if current_table_data is None:
        new_entry = {0: new_entry}
        new_table_data = pd.DataFrame.from_dict(new_entry, orient='index')
        return new_table_data
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
    """First column should contain datetime. If it's something
    else than an empty value and that this empty value
    doesn't cast to datetime."""
    filt = (pd.isnull(lab["A"])) | (
        lab["A"].apply(validate_date_text))
    return lab[filt]


class McGillMapper(base_mapper.BaseMapper):
    def get_attr_from_table_name(self, table_name):
        for attr, dico in self.conversion_dict.items():
            odm_name = dico["odm_name"]
            if odm_name == table_name:
                return attr

    def parse_lab_row(self, lab_row, mapping, lab_id, entries_to_store):
        elements = list(mapping["elementName"].unique())
        for _, element in enumerate(elements):
            if element == "":
                continue
            filt = mapping["elementName"] == element
            odm_table = mapping.loc[filt].iloc[0]["table"]
            fields = mapping.loc[mapping["elementName"] == element]
            new_entry = {}
            for _, field in fields.iterrows():
                field_name = field["variableName"]

                input_sources = field["inputSources"]
                if "static" in input_sources:
                    static_table = input_sources.split("+")[0]
                    static_table = static_table[len("static "):]
                    static_attr = self.get_attr_from_table_name(static_table)
                    static_input = getattr(self, static_attr)
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

    def read(self,
             labsheet_path,
             staticdata_path,
             mapsheet_path,
             worksheet_name,
             lab_id,
             startdate=None,
             enddate=None):
        # get the lab data
        lab = pd.read_excel(labsheet_path,
                            sheet_name=worksheet_name,
                            header=None,
                            usecols="A:BV")
        # parse the headers to deal with merged cells and get unique names
        lab.columns = [
            excel_style(i+1)
            for i, _ in enumerate(lab.columns.to_list())
        ]
        lab_datatypes = lab.iloc[4].values
        lab = lab.iloc[5:]
        lab = remove_bad_rows(lab)
        lab = typecast_lab(lab, lab_datatypes)
        lab = lab.dropna(how="all")
        mapping = pd.read_csv(mapsheet_path, header=0)
        mapping.fillna("", inplace=True)
        mapping = mapping.astype(str)
        label_col_name = "D"  # sampleID column
        spike_col_name = "AB"  # spikeID
        lod_value_col = "BI"  # sars-cov-2 gc/rxn
        sample_date_col = "B"  # end date
        lab = get_lod(lab, label_col_name, spike_col_name, lod_value_col)
        lab = filter_by_date(lab, sample_date_col, startdate, enddate)

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
        excel_mapper.read(staticdata_path)
        for table, attr in zip(static_tables, attrs):
            static_data[table] = getattr(excel_mapper, attr)
            setattr(self, attr, static_data[table])
        entries_to_store = {
            "WWMeasure": None,
            "Sample": None,
            "AssayMethod": None,
        }
        for _, row in lab.iterrows():
            entries_to_store = self.parse_lab_row(
                row,
                mapping,
                lab_id,
                entries_to_store
            )

        for key, table in entries_to_store.items():
            table = table.drop_duplicates()
            table = table.dropna(how="all")
            table = table.loc[table.iloc[:, 0] != ""]
            table = table.reset_index(drop=True)
            attr = self.get_attr_from_table_name(key)
            setattr(self, attr, table)
        return

    def validates(self):
        return True


if __name__ == "__main__":
    mapper = McGillMapper()
    path_to_static = "Data/Lab/McGill/Final/"
    lab_data = "/Users/jeandavidt/Desktop/latest-data/CentrEau-COVID_Resultats_Montreal_final.xlsx" # noqa
    static_data = path_to_static + "mcgill_static.xlsx"
    mapping = path_to_static + "mcgill_map.csv"
    sheet_name = "Mtl Data Daily Samples (Poly)"
    lab_id = "frigon_lab"
    start_date = "2021-01-01"
    end_date = None
    mapper.read(lab_data,
                static_data,
                mapping,
                sheet_name,
                lab_id=lab_id,
                startdate=None, enddate=None)
    print(mapper.site)
