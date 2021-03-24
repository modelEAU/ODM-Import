import pandas as pd
from wbe_odm.odm_mappers import base_mapper

default_site_measurement = {
    "sampleID": None,
    "reporterID": "NielsNicolai",
    "accessToPublic": "YES",
    "accessToAllOrg": "YES",
    "accessToPHAC": "YES",
    "accessToLocalHA": "YES",
    "accessToProvHA": "YES",
    "accessToOtherProv": "YES",
    "accessToDetails": "YES",
}


types = {
    "Précipitation": {
        "type": "envRnF",
        "aggregation": "single",
        "aggregationDesc": "Cumulative rainfall in one day.",
        "unit": "mm",
        "instrumentID": "Pluvio_VilledeQuebec"
    },
    "T° moyenne de l'eau": {
        "type": "wwTemp",
        "aggregation": "dailyAvg",
        "aggregationDesc": None,
        "unit": "°C",
        "instrumentID": None,
    },
    "pH": {
        "type": "wwPh",
        "aggregation": None,
        "aggregationDesc": None,
        "unit": "ph",
        "instrumentID": None,
    },
    "Débit affluent": {
        "type": "wwFlow",
        "aggregation": "single",
        "aggregationDesc": "Cumulative influent flow for one day.",
        "unit": "m3/d",
        "instrumentID": None,
    },
    "DCO affluent": {
        "type": "wwCOD",
        "aggregation": None,
        "aggregationDesc": None,
        "unit": "mg/L",
        "instrumentID": None,
    },
    "DBO5C affluent": {
        "type": "wwBOD5c",
        "aggregation": None,
        "aggregationDesc": None,
        "unit": "mg/L",
        "instrumentID": None,
    },
    "MES affluent": {
        "type": "wwTSS",
        "aggregation": None,
        "aggregationDesc": None,
        "unit": "mg/L",
        "instrumentID": None,
    },
    "Ptotal affluent": {
        "type": "wwPtot",
        "aggregation": None,
        "aggregationDesc": None,
        "unit": "mg/L",
        "instrumentID": None,
    },
    "N-NH4 effluent": {
        "type": "wwNH4N",
        "aggregation": None,
        "aggregationDesc": None,
        "unit": "mg/L",
        "instrumentID": None,
    },
}


def build_maps():
    maps = {
        "type": {},
        "aggregation": {},
        "aggregationDesc": {},
        "unit": {},
        "instrumentID": {},
    }
    for col_header, types_dico in types.items():
        for var_name, props_dico in maps.items():
            props_dico[col_header] = types[col_header][var_name]
    return maps


site_map = {
    "Données station Est": "Quebec_Est_WWTP",
    "Données station Ouest": "Quebec_Ouest_WWTP"
}


def reorder_columns(df):
    ordered_cols = [
        "siteMeasureID",
        "siteID",
        "instrumentID",
        "sampleID",
        "reporterID",
        "dateTime",
        "type",
        "aggregation",
        "aggregationDesc",
        "value",
        "unit",
        "accessToPublic",
        "accessToAllOrg",
        "accessToPHAC",
        "accessToLocalHA",
        "accessToProvHA",
        "accessToOtherProv",
        "accessToDetails",
        "notes",
    ]
    return df[ordered_cols]


def parse_plant_sheet(df, sheet_name):
    df = pd.melt(
        df,
        id_vars=["Date", "Conditions d'opération"],
        value_vars=types.keys())
    maps = build_maps()
    for var_name, _map in maps.items():
        df[var_name] = df["variable"].map(_map)
    for var, default in default_site_measurement.items():
        df[var] = default
    df["siteID"] = site_map[sheet_name]
    df["str_date"] = df["Date"].dt.strftime('%Y-%m-%d')

    df["siteMeasureID"] = \
        df["siteID"]\
        + "_" + df["type"] \
        + "_" + df["str_date"]
    del df["str_date"]
    df.rename(
        columns={"Conditions d'opération": "notes", "Date": "dateTime"},
        inplace=True
    )
    df = reorder_columns(df)
    return df


class VdQPlantMapper(base_mapper.BaseMapper):
    def read(self, filepath):
        sheet_names = ["Données station Est", "Données station Ouest"]
        odm_name = self.conversion_dict["site_measure"]["odm_name"]
        xls = pd.read_excel(
            filepath,
            sheet_name=sheet_names,
            header=0,
            skiprows=[1]
        )
        dfs = []
        for sheet_name, df in xls.items():
            if "pH moyen affluent" in df.columns:
                df.rename(columns={"pH moyen affluent": "pH"}, inplace=True)
            df = parse_plant_sheet(df, sheet_name)
            dfs.append(df)
        site_measure = pd.concat(dfs)

        site_measure.drop_duplicates(keep="first", inplace=True)
        site_measure.dropna(subset=["value"])
        site_measure = self.type_cast_table(odm_name, df)
        self.site_measure = site_measure
        return

    def validates(self):
        return True


if __name__ == "__main__":
    path = "/workspaces/ODM Import/Data/Site measure/Échantillonnage COVID ULaval.xlsx"  # noqa
    mapper = VdQPlantMapper()
    mapper.read(path)
