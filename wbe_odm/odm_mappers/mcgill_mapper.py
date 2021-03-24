import pandas as pd
import datetime
import re
from wbe_odm.odm_mappers import base_mapper

default_sample = {
        "sampleID": None,
        "siteID": None,
        "reporterID": None,
        "dateTime": None,
        "dateTimeStart": None,
        "dateTimeEnd": None,
        "type": "pstGrit",
        "collection": "cpTP24h",  # Flow proportional?
        "preTreatment": None,  # Is there one?
        "pooled": None,
        "children": None,
        "parent": None,
        "sizeL": 1,  # ?
        "fieldSampleTempC": 4,  # ?
        "shippedOnIce": "Yes",
        "storageTempC": 4,  # ?
        "qualityFlag": "NO",
        "notes": "",
        "index": 1
    }

default_measurement = {
    "uWwMeasureID": None,
    "WwMeasureID": None,
    "reporterID": "MaryamTohidi",
    "sampleID": None,
    "labID": "McGill_lab",  # Create the correct ID
    "assayMethodID": None,  # Should be filled in
    "analysisDate": None,
    "reportDate": None,
    "fractionAnalyzed": None,
    "type": None,
    "value": None,
    "unit": None,
    "aggregation": "single",
    "index": 1,
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
    "molecular detection.external control (brsv).ct.1": {
        "type": "nBrsv",
        "unit": "Ct"
    },
    "molecular detection.external control (brsv).ct.2": {
        "type": "nBrsv",
        "unit": "Ct"
    },
    "molecular detection.external control (brsv).ct.3": {
        "type": "nBrsv",
        "unit": "Ct"
    },
    "molecular detection.external control (brsv).% recovery.na": {
        "type": "nBrsv",
        "unit": "pctRecovery"
    },
    "molecular detection.pmmv.ct.1": {
        "type": "nPMMoV",
        "unit": "Ct"
    },
    "molecular detection.pmmv.ct.2": {
        "type": "nPPMoV",
        "unit": "Ct"
    },
    "molecular detection.pmmv.ct.3": {
        "type": "nPPMov",
        "unit": "Ct"
    },
    "molecular detection.pmmv.gc/ml.na": {
        "type": "nPPMov",
        "unit": "gc/ml"
    },
    "molecular detection.sars-cov-2.ct.1": {
        "type": "covN2",
        "unit": "Ct"
    },
    "molecular detection.sars-cov-2.ct.2": {
        "type": "covN2",
        "unit": "Ct"
    },
    "molecular detection.sars-cov-2.ct.3": {
        "type": "covN2",
        "unit": "Ct"
    },
    "molecular detection.sars-cov-2.gc/ml.na": {
        "type": "covN2",
        "unit": "gc/ml"
    },
    'concentration.key parametres.ph.initial': {
        "type": "wqPh",
        "unit": "ph"
    },
    'concentration.key parametres.turbidity (ntu).na': {
        "type": "wqTurb",
        "unit": "NTU"
    },
    'concentration.key parametres.conductivity megohm.na': {
        "type": "wqCond",
        "unit": "Ct"
    },
    'concentration.key parametres.tss (mg/l).na': {
        "type": "wqTss",
        "unit": "mg/l"
    },
}


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
                    array[i, j] = array[i, j-1]
                # if we're on another row, we must check tow things:
                # 1) The value of the row above
                # 2) The value of the row above the item on the left
                else:
                    # Of course, the first column doesn't have a left neighbor
                    if j == 0:
                        above_left = empty_token
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
                array[i, j] = str(array[i, j]).lower().strip()

    for i, _ in enumerate(array[0, :]):
        column_name = ".".join(array[:, i])
        column_titles.append(column_name)

    return column_titles


def parse_date(item):
    if isinstance(item, str):
        return pd.to_datetime(item)
    elif isinstance(item, datetime.datetime):
        date = pd.to_datetime(item)
        return date
    return pd.NaT


def clean_up(df):
    # removed rows that aren't samples
    type_col = "sampling.general information.type sample.na"
    df = df.loc[df[type_col] != "Reference"]
    df = df.loc[df[type_col] != "Negative"]
    # remove rowas that don't contain molecular measures
    molecular_columns = [
        "molecular detection.external control (brsv).ct.1",
        "molecular detection.external control (brsv).ct.2",
        "molecular detection.external control (brsv).ct.3",
        "molecular detection.external control (brsv).% recovery.na",
        "molecular detection.pmmv.ct.1",
        "molecular detection.pmmv.ct.2",
        "molecular detection.pmmv.ct.3",
        "molecular detection.pmmv.gc/ml.na",
        "molecular detection.sars-cov-2.ct.1",
        "molecular detection.sars-cov-2.ct.2",
        "molecular detection.sars-cov-2.ct.3",
        "molecular detection.sars-cov-2.gc/ml.na",
    ]
    measure_columns = [
        "concentration.general.concentrated volume (ml).na",
        'concentration.key parametres.ph.initial',
        'concentration.key parametres.ph.final',
        'concentration.key parametres.turbidity (ntu).na',
        'concentration.key parametres.conductivity megohm.na',
        'concentration.key parametres.tss (mg/l).na',
        'molecular detection.final elution in the extraction (µl).na.na',
        'molecular detection.final volume in the pcr (µl).na.na',
        'molecular detection.external control (brsv).ct.mean',
        'molecular detection.external control (brsv).gc/rxn.na',
        'molecular detection.pmmv.ct.mean',
        'molecular detection.pmmv.gc/rxn.na',
        'molecular detection.sars-cov-2.gc/rx.na',
        'molecular detection.sars-cov-2.gc/normalized to pmmv.na',
    ]

    df.loc[:, molecular_columns+measure_columns] = df.loc[
        :, molecular_columns+measure_columns]\
        .apply(pd.to_numeric, errors="coerce")

    df = df.dropna(subset=molecular_columns, how="all")
    # Parse other measurement columns:

    # Parse date columns to datetime
    for col in df.columns:
        if "date" in col:
            df[col] = df[col].apply(
                lambda x: parse_date(x))
    return df


def get_samples_from_lab_sheet(df):
    # Sample Ids will be generated using:
    # 1) Site ID
    # 2) Sample date
    # Since ModelEAU sample id's are built the same way, we should end up
    # with matched id's between the labs

    pass


def get_measurements_from_lab_sheet():
    pass


def get_reporters_from_lab_sheet(df):
    pass


class McGillMapper(base_mapper.BaseMapper):
    def read(self, filepath, sheet_name):
        df = pd.read_excel(path, sheet_name=sheet_name)
        df = df.iloc[:, 1:]
        df.columns = parse_mcgill_headers(df.iloc[0:4].to_numpy(dtype=str))
        df = clean_up(df)

        self.samples = get_samples_from_lab_sheet(df)
        self.ww_measure = get_measurements_from_lab_sheet(df)
        return

    def validates(self):
        return True


if __name__ == "__main__":
    path = "/workspaces/ODM Import/Data/Lab/McGill/Results Template.xlsx"
    sheet_name = "Qc Data Daily Samples 2"
    mapper = McGillMapper()
    mapper.read(path, sheet_name)
    print(mapper.ww_measure)
