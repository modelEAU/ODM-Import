import pandas as pd
import numpy as np
import datetime
import re
from wbe_odm.odm_mappers import base_mapper
# from wbe_odm.odm_mappers import excel_template_mapper

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
        except IndexError:
            desired_type = "string"
        lab[col] = typecast_column(desired_type, lab[col])
    return lab


def get_labsheet_inputs(map_row, lab_row, lab_id):
    raw_inputs = map_row["labInputs"].split(";")
    final_inputs = []
    for input_ in raw_inputs:
        if re.match(r"__const__.*:.*", input_):
            value, type_ = input_[len("__const__")-1:].split(":")
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
        return args
    arguments = [str(arg) for arg in args]
    return ",".join(arguments)


def get_assay_method_id(sample_type, concentration_method, assay_date):
    formatted_date = str(assay_date.date())
    return "_".join([sample_type, concentration_method, formatted_date])


def write_concentration_method(conc_method, ph_final):
    pass


def write_extraction_method(extraction):
    pass


def write_pcr_method(pcr):
    pass

def write_assay_notes():
    pass

def get_site_id():
    pass

def sample_is_pooled():
    pass

def create_children_samples():
    pass

def get_wwmeasure_id():
    pass

def get_reporter_id():
    pass

def get_sample_id():
    pass

def get_lab_id():
    pass

def validate_value():
    pass

def has_quality_flag():
    pass

def grant_access():
    pass

processing_functions = {
    "get_assay_method_id": get_assay_method_id,
    "write_concentration_method": write_concentration_method,
    "write_extraction_method": write_extraction_method,
    "write_pcr_method": write_pcr_method,
    "write_assay_notes": write_assay_notes,
    "get_site_id": get_site_id,
    "sample_is_pooled": sample_is_pooled,
    "create_children_samples": create_children_samples,
    "get_wwmeasure_id": get_wwmeasure_id,
    "get_reporter_id": get_reporter_id,
    "get_sample_id": get_sample_id,
    "get_lab_id": get_lab_id,
    "validate_value": validate_value,
    "has_quality_flag": has_quality_flag,
    "grant_access": grant_access,
}


def parse_lab_row(lab_row, mapping, static_dico, lab_id):
    elements = list(mapping["elementName"].unique())
    for element in elements:
        odm_table = mapping.loc[mapping["elementName"] == element].iloc[0]["table"]
        fields = mapping.loc[mapping["elementName"] == element]
        for _, field in fields.iterrows():
            odm_column = field["variableName"]
            input_sources = field["inputSources"]
            if "static sheet" in input_sources:
                static_input = static_dico[odm_table]
            else:
                static_input = tuple()
            lab_inputs = get_labsheet_inputs(field, lab_row, lab_id)
            inputs = (*static_input, *lab_inputs)
            if len(inputs) == 0:
                field["value"] == field["defaultValue"]
            else:
                func = field["processingFunction"]
                field["value"] = processing_functions.get(func, pass_raw)(*inputs)


class McGillMapper(base_mapper.BaseMapper):
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
        lab = pd.read_excel(labsheet_path, sheet_name=worksheet_name, header=None, usecols="A:BS")
        # parse the headers to deal with merged cells and get unique names
        lab.columns = parse_mcgill_headers(lab.iloc[0:4].to_numpy(dtype=str))
        lab = lab.iloc[4:]
        types = pd.read_csv(typesheet_path, header=0)
        mapping = pd.read_csv(mapsheet_path, header=0)
        lab = typecast_lab(lab, types)
        # Get the static data
        static_sheets = [
            "Lab",
            "Reporter",
            "Site",
            "AssayMethod",
            "Instrument",
            "Polygon",
        ]
        static_data = {}
        with pd.ExcelFile(staticdata_path) as xls:
            for sheet in static_sheets:
                static_data[sheet] = pd.read_excel(xls, sheet)
        
        for i, row in lab.iterrows():
            parse_lab_row(row, mapping, static_data, lab_id)
        

        # This mapper should:
        # 1) create sample rows 
        # 2) create assay method rows
        # 3) create ww measurement rows

        

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
                "frigon_lab",
                startdate=start, enddate=end)
