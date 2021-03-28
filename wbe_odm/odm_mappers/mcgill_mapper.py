import pandas as pd
import numpy as np
import datetime
import re
from wbe_odm.odm_mappers import base_mapper
# from wbe_odm.odm_mappers import excel_template_mapper


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
        "unit": "uS/cm"
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


def get_sample_ids_from_lab_sheet(df, site_map):
    # Sample Ids will be generated using:
    # 1) Site ID
    # 2) Sample date
    date_col = "sampling.general information.date (d/m/y).na"
    site_col = "sampling.general information.sampling point.na"
    index_col = "sampling.general information.index.na"
    df["str_col"] = df[date_col].dt.strftime(r"%Y-%m-%d")
    df["site_ID"] = df[site_col].map(site_map)
    # Since ModelEAU sample id's are built the same way, we should end up
    # with matched id's between the labs
    df["sampleID"] = df["siteID"] \
        + "_" + df["str_col"] \
        + "_" + df[index_col].astype(str)
    return df["sampleID"]


def get_samples_from_lab_sheet(df):
    pass


def build_maps(measurement_dico):
    maps = {
        "type": {},
        "unit": {},
    }
    for col_header, types_dico in measurement_dico.items():
        for var_name, props_dico in maps.items():
            props_dico[col_header] = measurement_dico[col_header][var_name]
    return maps


def get_measurements_from_lab_sheet(
        df,
        idx_cols,
        meas_cols,
        site_map,
        measurement_dico,
        default_measurement):

    df = pd.melt(
        df,
        id_vars=idx_cols,
        value_vars=meas_cols
    )

    df["sampleID"] = get_sample_ids_from_lab_sheet(df, site_map)
    maps = build_maps(measurement_dico)
    for var_name, _map in maps.items():
        df[var_name] = df["variable"].map(_map)
    for var, default in default_measurement.items():
        df[var] = default
    df["siteID"] = site_map[sheet_name]
    df["str_date"] = df["Date"].dt.strftime('%Y-%m-%d')


class McGillMapper(base_mapper.BaseMapper):
    def read(self, filepath, sheet_name,
             default_sample,
             default_measures,
             start=None, end=None, create_samples=True):
        df = pd.read_excel(path, sheet_name=sheet_name)
        # df = df.iloc[:, 1:]
        df.columns = parse_mcgill_headers(df.iloc[0:4].to_numpy(dtype=str))
        id_vars = [
            "sampling.general information.date (d/m/y).na",
            "sampling.general information.sampling point.na",
            "sampling.general information.type sample.na",
            "sampling.general information.worker/student.na",
            'concentration.general.date (d/m/y).na',
            'concentration.general.concentrated volume (ml).na',
            'concentration.general.worker/student.na',
            'molecular detection.date of rna extraction (d/m/y).na.na',
            'molecular detection.final elution in the extraction (µl).na.na',
            'molecular detection.worker/student (rna extraction).na.na',
            'molecular detection.date of pcr (d/m/y).na.na',
            'molecular detection.final volume in the pcr (µl).na.na',
            'molecular detection.worker/student (pcr).na.na',
            'comments.na.na.na',
            'conclusion.na.na.na'
        ]
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
            # "concentration.general.concentrated volume (ml).na",
            # 'concentration.key parametres.ph.initial',
            # 'concentration.key parametres.ph.final',
            'concentration.key parametres.turbidity (ntu).na',
            'concentration.key parametres.conductivity megohm.na',  # /cm? m?
            'concentration.key parametres.tss (mg/l).na',
            # 'molecular detection.final elution in the extraction (µl).na.na',
            # 'molecular detection.final volume in the pcr (µl).na.na',
            # 'molecular detection.external control (brsv).ct.mean',
            # 'molecular detection.external control (brsv).gc/rxn.na',
            # 'molecular detection.pmmv.ct.mean',
            # 'molecular detection.pmmv.gc/rxn.na',
            # 'molecular detection.sars-cov-2.gc/rx.na',
            # 'molecular detection.sars-cov-2.gc/normalized to pmmv.na',
        ]
        meas_cols = molecular_columns+measure_columns
        site_map = {
            "Est": "Quebec_Est_WWTP",
            "Ouest": "Quebec_Ouest_WWTP",
            "CHSLD": "Quebec_CHSLD_Charlesbourg",
        }
        df = clean_up(df, molecular_columns, measure_columns)
        self.ww_measure = get_measurements_from_lab_sheet(
            df, id_vars, meas_cols, site_map, default_measurement)
        if create_samples:
            self.samples = get_samples_from_lab_sheet(
                df, site_map, default_sample)
        return

    def validates(self):
        return True


if __name__ == "__main__":
    path = "Data/Lab/McGill/20210317(b)_Results Template_filled.xlsx"
    sheet_name = "Mtl Data Daily Samples (McGill)"
    mapper = McGillMapper()
    start = "2021-03-01"
    end = "2021-03-15"
    default_sample = {
        "sampleID": None,
        "siteID": None,
        "reporterID": None,
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
        "WwMeasureID": None,
        "reporterID": None,
        "sampleID": None,
        "labID": "McGill_lab",
        "assayMethodID": None,
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
    mapper.read(path, sheet_name,
                start=start, end=end, default_sample=default_sample,
                default_measures=default_measurement)
