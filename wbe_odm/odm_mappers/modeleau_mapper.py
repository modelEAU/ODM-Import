import pandas as pd
import datetime as dt
from wbe_odm.odm_mappers import base_mapper


def replace_excel_dates(series):
    return series.apply(
        lambda x: pd.to_timedelta(x, unit='d') +
        dt.datetime(1899, 12, 30) if isinstance(x, float) else x
    )


def clean_up(df):
    for col in df.columns.to_list():
        df[col] = pd.to_datetime(replace_excel_dates(df[col])) \
            if "date" in col.lower() else df[col]
        df[col] = df[col].apply(lambda x: None if x in ["None", ""] else x)
        if col == "Measurement":
            df[col] = df[col].apply(lambda x: x.replace("*", "").strip())
        if "Unnamed" in col:
            del df[col]
    return df


default_sample = {
        "sampleID": None,
        "siteID": None,
        "instrumentID": None,
        "reporterID": "MaryamTohidi",
        "dateTime": None,
        "dateTimeStart": None,
        "dateTimeEnd": None,
        "type": "pstGrit",
        "collection": "cpTP24h",
        "preTreatment": None,
        "pooled": None,
        "children": None,
        "parent": None,
        "sizeL": 1,
        "index": 1,
        "fieldSampleTempC": 4,
        "shippedOnIce": "Yes",
        "storageTempC": 4,
        "qualityFlag": "NO",
        "notes": "",
    }

default_measurement = {
    # "WWMeasureID": None,
    "WWMeasureID": None,
    "reporterID": "MaryamTohidi",
    "sampleID": None,
    "labID": "ModelEau_lab",
    "assayMethodID": None,
    "analysisDate": None,
    "reportDate": None,
    "fractionAnalyzed": None,
    "type": None,
    "value": None,
    "unit": None,
    "aggregation": "single",
    "index": 0,
    "qualityFlag": "NO",
    "accessToPublic": "YES",
    "accessToAllOrg": "YES",
    "accessToPHAC": "YES",
    "accessToLocalHA": "YES",
    "accessToProvHA": "YES",
    "accessToOtherProv": "YES",
    "accessToDetails": "YES",
    "notes": None,
}

measurement_dico = {
    "Turbidity": "wqTurb",
    "covN1": "covN1",
    "covN2": "covN2",
    "covN3": "covN3",
    "covE": "covE",
    "covRdRp": "covRdRp",
    "nPMMoV": "nPMMoV",
    "nCrA": "nCrA",
    "nBrsv": "nBrsv",
    "TS": "wqTS",
    "TSS": "wqTSS",
    "VSS": "wqVSS",
    "COD": "wqCOD",
    "P": "wqOPhos",
    "NH4": "wqNH4N",
    "TN": "wqTN",
    "pH": "wqPh",
    "Conductivity": "wqCond",
}


def check_if_grab(row):
    return pd.isna(row["Sample end date"]) or pd.isna(row["Sample start date"])


def build_sample_id(measurement_sheet_row):
    is_grab = check_if_grab(measurement_sheet_row)
    if is_grab:
        id_date = measurement_sheet_row["Date (enter end date here)"]
    else:
        id_date = measurement_sheet_row["Sample start date"]

    id_date = id_date.strftime(r"%Y-%m-%d")

    if "sampleType" in measurement_sheet_row:
        sample_type = measurement_sheet_row["sampleType"]
    else:
        sample_type = default_sample["type"]

    if "sampleIndex" in measurement_sheet_row:
        sample_index = measurement_sheet_row["sampleIndex"]
    else:
        sample_index = default_sample["index"]

    site_id = measurement_sheet_row["siteID"]
    return ("_").join([site_id, id_date, sample_type, str(sample_index)])


def build_measurement_id(measurement_sheet_row):
    sample_id = build_sample_id(measurement_sheet_row)
    lab_id = default_measurement["labID"]
    measurement_type = measurement_dico[measurement_sheet_row["Measurement"]]
    date = measurement_sheet_row["Analysis Date"].strftime(r"%Y-%m-%d")
    if "measurementIndex" in measurement_sheet_row:
        measurement_index = measurement_sheet_row["measurementIndex"]
    else:
        measurement_index = default_measurement["index"]
    return ("_").join([
        sample_id,
        lab_id,
        measurement_type,
        date,
        str(measurement_index)
    ])


def create_sample_row(row):
    new_sample = default_sample.copy()
    new_sample["sampleID"] = build_sample_id(row)
    is_grab = check_if_grab(row)
    if is_grab:
        new_sample["dateTime"] = row["Date (enter end date here)"]
    else:
        new_sample["dateTimeEnd"] = row["Sample end date"]
        new_sample["dateTimeStart"] = row["Sample start date"]

    if "reporterID" in row:
        new_sample["reporterID"] = row["reporterID"]
    if "sampleIndex" in row:
        new_sample["index"] = row["sampleIndex"]

    new_sample["siteID"] = row["siteID"]

    return pd.Series(new_sample)


def reorder_sample_columns(df):
    ordered_columns = [
        "sampleID",
        "siteID",
        "instrumentID",
        "reporterID",
        "dateTime",
        "dateTimeStart",
        "dateTimeEnd",
        "type",
        "collection",
        "preTreatment",
        "pooled",
        "children",
        "parent",
        "sizeL",
        "index",
        "fieldSampleTempC",
        "shippedOnIce",
        "storageTempC",
        "qualityFlag",
        "notes"
    ]
    return df[ordered_columns]


def reorder_ww_measure_columns(df):
    ordered_columns = [
        "WWMeasureID",
        "reporterID",
        "sampleID",
        "labID",
        "assayMethodID",
        "analysisDate",
        "reportDate",
        "fractionAnalyzed",
        "type",
        "value",
        "unit",
        "aggregation",
        "index",
        "qualityFlag",
        "accessToPublic",
        "accessToAllOrg",
        "accessToPHAC",
        "accessToLocalHA",
        "accessToProvHA",
        "accessToOtherProv",
        "accessToDetails",
        "notes"
    ]
    return df[ordered_columns]


def get_samples_from_lab_sheet(df):
    samples = df.apply(lambda x: create_sample_row(x), axis=1)
    samples = samples.drop_duplicates()
    samples.reset_index(drop=True, inplace=True)
    samples = reorder_sample_columns(samples)
    return samples


def create_measurement_row(row):
    new_measurement = default_measurement.copy()
    new_measurement["sampleID"] = build_sample_id(row)
    new_measurement["WWMeasureID"] = build_measurement_id(row)
    new_measurement["analysisDate"] = row["Analysis Date"]
    new_measurement["type"] = measurement_dico[row["Measurement"]]
    new_measurement["value"] = row["Value"]
    new_measurement["unit"] = row["Unit"]
    new_measurement["fractionAnalyzed"] = row["fraction analyzed"]
    new_measurement["qualityFlag"] = row["qualityFlag"]
    new_measurement["notes"] = row["notes"]

    if "index" in row:
        new_measurement["index"] = row["index"]
    return pd.Series(new_measurement)


def edit_index_in_id(row):
    current_index = row["WWMeasureID"]
    row["WWMeasureID"] = current_index[:-1] + str(row["index"])
    return row


def build_missing_indices(df):
    uniques = df["WWMeasureID"].drop_duplicates()
    for _, unique in enumerate(uniques):
        replicates = df.loc[df["WWMeasureID"] == unique]
        indices = [x+1 for x in range(len(replicates))]
        df.loc[df["WWMeasureID"] == unique, ["index"]] = indices
    df = df.apply(lambda x: edit_index_in_id(x), axis=1)
    return df


def get_measurements_from_lab_sheet(df):
    measurements = df.apply(lambda x: create_measurement_row(x), axis=1)
    measurements.reset_index(drop=True, inplace=True)
    if len(measurements.loc[measurements["index"] == 0]) > 0:
        measurements = build_missing_indices(measurements)
    measurements = reorder_ww_measure_columns(measurements)
    return measurements


class ModelEauMapper(base_mapper.BaseMapper):
    def read(self, filepath, sheet_name):
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        df = clean_up(df)
        df.drop_duplicates(keep="first", inplace=True)
        self.sample = get_samples_from_lab_sheet(df)
        self.ww_measure = get_measurements_from_lab_sheet(df)
        self.remove_duplicates()
        return

    def validates(self):
        return True


if __name__ == "__main__":
    path = "Data/Lab/modelEAU/COVIDProject_Lab Measurements.xlsx"
    sheet_name = "Lab analyses"
    mapper = ModelEauMapper()
    mapper.read(path, sheet_name)
    mapper.ww_measure.to_excel("Data/Lab/modelEAU/to_paste_wwmeasure.xlsx")
    mapper.sample.to_excel("Data/Lab/modelEAU/to_paste_sample.xlsx")
    print(mapper.ww_measure)
