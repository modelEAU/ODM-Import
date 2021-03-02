import pandas as pd
import datetime as dt

class ModelEau:
    def replace_excel_dates(series):
        return series.apply(lambda x: pd.to_timedelta(x, unit='d') + dt.datetime(1899, 12, 30) if isinstance(x, float) else x)
    def clean_up(df):
        for col in df.columns.to_list():
            df[col] = pd.to_datetime(ModelEau.replace_excel_dates(df[col])) if "date" in col.lower() else df[col]
            df[col] = df[col].apply(lambda x: None if x in ["None", ""] else x)
            if col == "Measurement":
                df[col] = df[col].apply(lambda x: x.replace("*", "").strip())
            if "Unnamed" in col:
                del df[col]
        return df

    default_sample = {
            "sampleID": None,
            "siteID": None,
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
            "fieldSampleTempC": 4,
            "shippedOnIce": "Yes",
            "storageTempC": 4,
            "qualityFlag": "NO",
            "notes": "",
            "index": 1
        }
    default_measurement = {
        "uWwMeasureID": None, # generate
        "WwMeasureID": None,
        "reporterID": "MaryamTohidi", #read optional
        "sampleID": None, #generate
        "labID": "ModelEau_lab",
        "assayMethodID": None,
        "analysisDate": None, # read
        "reportDate": None, # current date
        "fractionAnalyzed": None,   #read optional
        "type": None, #read
        "value": None,# read
        "unit": None,# read
        "aggregation": "single",
        "index": 0, # read
        "qualityFlag": "NO", # read optional
        "accessToPublic": "YES",
        "accessToAllOrg": "YES",
        "accessToPHAC": "YES",
        "accessToLocalHA": "YES",
        "accessToProvHA": "YES",
        "accessToOtherProv": "YES",
        "accessToDetails": "YES",
        "notes": None, # read
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
        is_grab = ModelEau.check_if_grab(measurement_sheet_row)
        if is_grab:
            id_date = measurement_sheet_row["Date (enter end date here)"]
        else:
            id_date = measurement_sheet_row["Sample start date"]

        id_date = id_date.strftime(r"%Y-%m-%d")

        if "sampleType" in measurement_sheet_row:
            sample_type = measurement_sheet_row["sampleType"]
        else:
            sample_type = ModelEau.default_sample["type"]

        if "sampleIndex" in measurement_sheet_row:
            sample_index = measurement_sheet_row["sampleIndex"]
        else:
            sample_index = ModelEau.default_sample["index"]

        site_id = measurement_sheet_row["siteID"]
        return ("_").join([site_id, id_date, sample_type, str(sample_index)])

    def build_measurement_id(measurement_sheet_row):
        sample_id = ModelEau.build_sample_id(measurement_sheet_row)
        lab_id = ModelEau.default_measurement["labID"]
        measurement_type = ModelEau.measurement_dico[measurement_sheet_row["Measurement"]]
        date = measurement_sheet_row["Analysis Date"].strftime(r"%Y-%m-%d")
        if "measurementIndex" in measurement_sheet_row:
            measurement_index = measurement_sheet_row["measurementIndex"]
        else:
            measurement_index = ModelEau.default_measurement["index"]
        return ("_").join([sample_id, lab_id, measurement_type, date, str(measurement_index)])


    def create_sample_row(measurement_sheet_row):
        new_sample = ModelEau.default_sample.copy()
        new_sample["sampleID"] = ModelEau.build_sample_id(measurement_sheet_row)
        is_grab = ModelEau.check_if_grab(measurement_sheet_row)
        if is_grab:
            new_sample["dateTime"] = measurement_sheet_row["Date (enter end date here)"]
        else:
            new_sample["dateTimeEnd"] = measurement_sheet_row["Sample end date"]
            new_sample["dateTimeStart"] = measurement_sheet_row["Sample start date"]

        if "reporterID" in measurement_sheet_row:
            new_sample["reporterID"] = measurement_sheet_row["reporterID"]
        if "sampleIndex" in measurement_sheet_row:
            new_sample["index"] = measurement_sheet_row["sampleIndex"]

        new_sample["siteID"] = measurement_sheet_row["siteID"]

        return pd.Series(new_sample)

    def get_samples_from_lab_sheet(df):
        samples = df.apply(lambda x: ModelEau.create_sample_row(x), axis=1)
        samples = samples.drop_duplicates()
        samples.reset_index(drop=True, inplace=True)
        return samples


    def create_measurement_row(measurement_sheet_row):
        new_measurement = ModelEau.default_measurement.copy()
        new_measurement["sampleID"] = ModelEau.build_sample_id(measurement_sheet_row)
        
        new_measurement["uWwMeasureID"] = ModelEau.build_measurement_id(measurement_sheet_row)
        new_measurement["analysisDate"] = measurement_sheet_row["Analysis Date"]
        new_measurement["type"] = ModelEau.measurement_dico[measurement_sheet_row["Measurement"]]
        new_measurement["value"] = measurement_sheet_row["Value"]
        new_measurement["unit"] = measurement_sheet_row["Unit"]
        new_measurement["fractionAnalyzed"] = measurement_sheet_row["fraction analyzed"]
        new_measurement["qualityFlag"] = measurement_sheet_row["qualityFlag"]
        new_measurement["notes"] = measurement_sheet_row["notes"]
        
        if "index" in measurement_sheet_row:
            new_measurement["index"] = measurement_sheet_row["index"]
        
        return pd.Series(new_measurement)

    def edit_index_in_id(row):
        current_index = row["uWwMeasureID"]
        row["uWwMeasureID"] = current_index[:-1] + str(row["index"])
        return row

    def build_missing_indices(df):
        uniques = df["uWwMeasureID"].drop_duplicates()
        for i, unique in enumerate(uniques):
            replicates = df.loc[df["uWwMeasureID"] == unique]
            indices = [x+1 for x in range (len(replicates))]
            df.loc[df["uWwMeasureID"] == unique, ["index"]] = indices
        df = df.apply(lambda x: ModelEau.edit_index_in_id(x), axis=1)
        return df

    def get_measurements_from_lab_sheet(df):
        measurements = df.apply(lambda x: ModelEau.create_measurement_row(x), axis=1)
        measurements.reset_index(drop=True, inplace=True)
        if len(measurements.loc[measurements["index"] == 0]) > 0:
            measurements = ModelEau.build_missing_indices(measurements)
        return measurements

if __name__ == "__main__":
    path = "Data/Lab/COVIDProject_Lab Measurements.xlsx"
    sheet_name = "Lab analyses"
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = ModelEau.clean_up(df)
    samples = ModelEau.get_samples_from_lab_sheet(df)
    measurements = ModelEau.get_measurements_from_lab_sheet(df)