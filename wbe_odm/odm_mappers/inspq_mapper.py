import pandas as pd
from wbe_odm.odm_mappers import base_mapper as bm
import requests
import io

poly_names = {
    '13 - Laval': "prov_qc_hlthReg_laval",
    '12 - Chaudière-Appalaches': "prov_qc_hlthReg_chaudiere-appalaches",
    '10 - Nord-du-Québec': "prov_qc_hlthReg_nord-du-quebec",
    '05 - Estrie': "prov_qc_hlthReg_estrie",
    '03 - Capitale-Nationale': "prov_qc_hlthReg_capitale-nationale",
    '04 - Mauricie et Centre-du-Québec': "prov_qc_hlthReg_mauricie-centre-du-quebec",  # noqa
    '16 - Montérégie': "prov_qc_hlthReg_monteregie",
    '08 - Abitibi-Témiscamingue': "prov_qc_hlthReg_abitibi-temiscamingue",
    '07 - Outaouais': "prov_qc_hlthReg_outaouais",
    '01 - Bas-Saint-Laurent': "prov_qc_hlthReg_bas-saint-laurent",
    '09 - Côte-Nord': "prov_qc_hlthReg_cote-nord",
    '06 - Montréal': "prov_qc_hlthReg_montreal",
    '15 - Laurentides': "prov_qc_hlthReg_laurentides",
    '14 - Lanaudière': "prov_qc_hlthReg_lanaudiere",
    '02 - Saguenay-Lac-Saint-Jean': "prov_qc_hlthReg_saguenay-lac-saint-jean",
    '11 - Gaspésie-Îles-de-la-Madeleine': "prov_qc_hlthReg_gaspesie-iles-de-la-madeleine",  # noqa
}


general_defaults = {
    "cphdID": None,
    "reporterID": "INSPQ",
    "polygonID": None,
    "date": None,
    "type": None,
    "dateType": None,
    "value": None,
    "notes": None,
}


values_to_save = {
    "cas_quo_tot_n": {
        "type": "conf",
        "dateType": "report"
    },
    "act_cum_tot_n": {
        "type": "active",
        "dateType": "report"
    },
    "dec_quo_tot_n": {
        "type": "death",
        "dateType": "report"
    },
    "hos_quo_tot_n": {
        "type": "hospCen",
        "dateType": "report"
    },
    "psi_quo_tes_n": {
        "type": "test",
        "dateType": "report"
    },
    "psi_quo_pos_n": {
        "type": "posTest",
        "dateType": "report"
    },
    "psi_quo_pos_t": {
        "type": "pPosRt",
        "dateType": "report"
    },
}

vaccine_to_save = {
    "vac_cum_1_n": {
        "type": "vaccineDose1",
        "dateType": "report",
    },

    "vac_cum_2_n": {
        "type": "vaccineDose2",
        "dateType": "report",
    }
}

def build_cphd_ids(reporter, region, type_, datetype, date):
    date = date.dt.strftime("%Y-%m-%d")
    df = pd.concat([reporter, region, type_, datetype, date], axis=1)
    return df.agg("_".join, axis=1)

def df_from_req(url):
    req = requests.get(url)
    if not req:
        raise requests.HTTPError(f'could not get data from {url}')
    content = req.content
    return pd.read_csv(io.StringIO(content.decode('utf-8')))

INSPQ_DATASET_URL = "https://www.inspq.qc.ca/sites/default/files/covid/donnees/covid19-hist.csv?randNum=27002747"
INSPQ_VACCINE_DATASET_URL = "https://www.inspq.qc.ca/sites/default/files/covid/donnees/vaccination.csv?randNum=27002747"

class INSPQ_mapper(bm.BaseMapper):
    def read(self, filepath=None):
        if filepath is None:
            hist = df_from_req(INSPQ_DATASET_URL)
        else:
            hist = pd.read_csv(filepath)
        hist = hist.loc[hist["Nom"].isin(poly_names.keys())]
        hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
        hist = hist.dropna(subset=["Date"])
        dfs = []
        for item, item_defaults in values_to_save.items():
            df = hist.copy()
            for variable, value in general_defaults.items():
                df[variable] = value
            df["date"] = df["Date"]
            for variable, value in item_defaults.items():
                df[variable] = value
            df["value"] = df[item]
            df["polygonID"] = df["Nom"].map(poly_names)
            df["cphdID"] = build_cphd_ids(
                df["reporterID"],
                df["polygonID"],
                df["type"],
                df["dateType"],
                df["date"])
            df = df[list(general_defaults.keys())]
            dfs.append(df)
        cphd = pd.concat(dfs, axis=0)
        cphd.drop_duplicates(keep="first", inplace=True)
        cphd = self.type_cast_table("CovidPublicHealthData", cphd)
        self.cphd = cphd
        return

    def validates(self):
        return True

class INSPQVaccineMapper(bm.BaseMapper):
    def read(self, filepath=None):
        if filepath is None:
            hist = df_from_req(INSPQ_VACCINE_DATASET_URL)
        else:
            hist = pd.read_csv(filepath)
        hist = hist.loc[hist["Nom"].isin(poly_names.keys())]
        hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
        hist = hist.dropna(subset=["Date"])
        dfs = []
        for item, item_defaults in vaccine_to_save.items():
            df = hist.copy()
            for variable, value in general_defaults.items():
                df[variable] = value
            df["date"] = df["Date"]
            for variable, value in item_defaults.items():
                df[variable] = value
            df["value"] = df[item]
            df["polygonID"] = df["Nom"].map(poly_names)
            df["cphdID"] = build_cphd_ids(
                df["reporterID"],
                df["polygonID"],
                df["type"],
                df["dateType"],
                df["date"])
            df = df[list(general_defaults.keys())]
            dfs.append(df)
        cphd = pd.concat(dfs, axis=0)
        cphd.drop_duplicates(keep="first", inplace=True)
        cphd = self.type_cast_table("CovidPublicHealthData", cphd)
        self.cphd = cphd
        return

    def validates(self):
        return True


if __name__ == "__main__":
    filepath = "/Users/jeandavidt/OneDrive - Université Laval/COVID/Latest Data/Input/INSPQ/covid19-hist.csv"  # noqa
    vac_path = "/Users/jeandavidt/Desktop/vaccination.csv"
    # print(df_from_req(INSPQ_VACCINE_DATASET_URL))

    mapper = INSPQ_mapper()
    mapper.read()
    print(mapper.cphd.head())
    
