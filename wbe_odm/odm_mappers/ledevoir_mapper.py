import json
import os
import requests
import pandas as pd
import unidecode
from wbe_odm.odm_mappers.csv_mapper import CsvMapper


directory = os.path.dirname(__file__)
LEDEVOIR_MAP_NAME = directory + "/" + "ledevoir_map.csv"


POLYGON_LOOKUP = {
    'laval': 'prov_qc_hlthReg_laval',
    'chaudiere-appalaches': 'prov_qc_hlthReg_chaudiere-appalaches',
    'nord-du-quebec': 'prov_qc_hlthReg_nord-du-quebec',
    'estrie': 'prov_qc_hlthReg_estrie',
    'capitale-nationale': 'prov_qc_hlthReg_capitale-nationale',
    'centre-du-quebec': 'prov_qc_hlthReg_centre-du-quebec',
    'monteregie': 'prov_qc_hlthReg_monteregie',
    'abitibi-temiscamingue': 'prov_qc_hlthReg_abitibi-temiscamingue',
    'outaouais': 'prov_qc_hlthReg_outaouais',
    'bas-saint-laurent': 'prov_qc_hlthReg_bas-saint-laurent',
    'cote-nord': 'prov_qc_hlthReg_cote-nord',
    'montreal': 'prov_qc_hlthReg_montreal',
    'laurentides': 'prov_qc_hlthReg_laurentides',
    'lanaudiere': 'prov_qc_hlthReg_lanaudiere',
    'saguenay-lac-saint-jean': 'prov_qc_hlthReg_saguenay-lac-saint-jean',
    'gaspesie-iles-de-la-madeleine':
        'prov_qc_hlthReg_gaspesie-iles-de-la-madeleine',
}


def get_cphd_id(reporter, region, type_, datetype, date):
    df = pd.DataFrame(
        CsvMapper.str_date_from_timestamp(date))
    df.columns = ["date"]
    df["reporter"] = reporter
    df["region"] = region
    df["type"] = type_
    df["datetype"] = datetype
    df = df[["reporter", "region", "type", "datetype", "date"]]
    return df.agg("_".join, axis=1)


def get_polygon_id(region):
    return region.map(POLYGON_LOOKUP)


def get_date(date):
    return pd.to_datetime(date)


cphd_funcs = {
    "get_cphd_id": get_cphd_id,
    "get_polygon_id": get_polygon_id,
    "get_date": get_date,
}


class LeDevoirMapper(CsvMapper):
    def __init__(self, config_file=None):
        super().__init__(processing_functions=cphd_funcs, config_file=config_file)

    def merge_regions_data(self, dfs, final_name):
        """Some regions are reported separately by INSPQ,
        but we don't have polygons for them. This function
        combines them together.

        Args:
            dfs (pd.DataFrame): The dataframes containig the case data.
                for the sub-regions to merge
            final_name (pd.DataFrame): The finale name of the region.

        Returns:
            pd.DataFrame: A combined case DataFrame with
                the cases for each subregion added together.
        """
        dc_cols = []
        dd_cols = []
        for df in dfs:
            dc_cols.append(df["dc"])
            dd_cols.append(df["dd"])
        dc_df = pd.concat(dc_cols, axis=1)
        dd_df = pd.concat(dd_cols, axis=1)
        dc_df["dc_tot"] = 0
        dd_df["dd_tot"] = 0
        for i in range(len(dfs)):
            dc_df["dc_tot"] += dc_df.iloc[:, i]
            dd_df["dd_tot"] += dd_df.iloc[:, i]
        tot = pd.concat([dc_df["dc_tot"], dd_df["dd_tot"]], axis=1)
        tot.columns = ["dc", "dd"]
        tot["region"] = final_name.lower()
        tot = tot[["region", "dc", "dd"]]
        return tot

    def load_ledevoir_data(self):
        response = requests.get("https://ledevoir-coronavirus.herokuapp.com/api/v2/reports/ca/qc")  # noqa
        j = json.loads(response.text)
        dfs = []
        dfs_to_add = []
        for i, entry in enumerate(j["regions"]):
            cases = pd.DataFrame(j["regions"][i]["data"])
            name = entry["name"]
            cases["region"] = unidecode.unidecode(name.lower())
            cases["date"] = pd.to_datetime(cases["date"])
            cases = cases.set_index("date")
            cases = cases[["region", "dc", "dd"]]
            if name.lower() in [
                "nunavik",
                "terres-cries-de-la-baie-james",
                "nord-du-quebec"
            ]:
                dfs_to_add.append(cases)
            else:
                dfs.append(cases)
        nord = self.merge_regions_data(dfs_to_add, "nord-du-quebec")
        dfs.append(nord)
        return pd.concat(dfs)

    def read(self, map_path=LEDEVOIR_MAP_NAME):
        mapping = pd.read_csv(map_path)
        mapping.fillna("", inplace=True)
        mapping = mapping.astype(str)
        lab_id = None
        static_data = self.read_static_data(None)

        case_data = self.load_ledevoir_data()
        case_data.reset_index(inplace=True)
        dynamic_tables = self.parse_sheet(
            mapping, static_data, case_data, self.processing_functions, lab_id
        )
        cphd = dynamic_tables["CovidPublicHealthData"]
        cphd.drop_duplicates(keep="first", inplace=True)
        cphd = self.type_cast_table("CovidPublicHealthData", cphd)
        self.cphd = cphd
        self.remove_duplicates()
        return
