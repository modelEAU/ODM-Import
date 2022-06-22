import os

import numpy as np
import pandas as pd
from wbe_odm.odm_mappers import base_mapper as bm
from wbe_odm.odm_mappers.csv_mapper import CsvMapper

directory = os.path.dirname(__file__)
VDQ_LAB_MAP_NAME = directory + "/" + "vdqlab_map.csv"
VDQ_LAB_MAP_NAME_2022 = directory + "/" + "vdqlab2022_map.csv"
VDQ_PLUVIO_MAP_NAME = directory + "/" + "vdqpluvio_map.csv"
VDQ_SENSOR_MAP_NAME = directory + "/" + "vdqsensors_map.csv"


ST_PASCAL_CURVE = pd.DataFrame.from_dict({
    "flowrate": [
        15.029, 39.306, 76.301,
        117.919, 159.538, 198.844,
        231.214, 277.457, 319.075,
        358.382, 409.249, 446.243,
        483.237, 524.855, 550.289,
        587.283, 608.092, 638.15,
        656.647, 668.208, 672.832, 678.613],
    "height(m)": [
        0.046, 0.098, 0.173,
        0.225, 0.276, 0.317,
        0.351, 0.397, 0.432,
        0.455, 0.495, 0.518,
        0.553, 0.599, 0.651,
        0.697, 0.76, 0.864,
        0.95, 1.025, 1.612, 2.965]
}, orient="columns")


site_map = {
    "Données station Est": "QC_01",
    "Données station Ouest": "QC_02"
}


class MapperFuncs:
    @classmethod
    def get_qc_city_site_measure_id(cls, site_id, date, type_):
        df = pd.DataFrame(pd.to_datetime(date))
        df["type"] = type_
        df["site_id"] = site_id
        df.columns = ["dates", "type", "site_id"]
        df["formattedDates"] = df["dates"]\
            .dt.strftime("%Y-%m-%dT%H:%M:%S")\
            .fillna('').str.replace("T00:00:00", "")
        df = df[["site_id", "type", "formattedDates"]]
        return df.agg("_".join, axis=1)

    @classmethod
    def get_date(cls, dates):
        return pd.to_datetime(dates)

    @classmethod
    def maizerets_from_height(cls, height, is_open):
        calib = ST_PASCAL_CURVE
        df = pd.concat([height, is_open], axis=1)
        df.columns = ["height", "is_open"]
        # convert to mm
        df["height"] = df["height"] / 1000
        df["maizerets"] = df["height"].apply(
            lambda x: np.interp(x, calib["height(m)"], calib["flowrate"])
        )
        df["maizerets"] = pd.to_numeric(df["maizerets"], errors="coerce")
        df["is_open"] = pd.to_numeric(df["is_open"], errors="coerce")
        df.loc[df["is_open"].isna(), "maizerets"] = 0
        df["m3/d"] = df["maizerets"] * 3600 * 24 / 1000
        return df["m3/d"]

    @classmethod
    def charlesbourg_flow(cls, tot_flow, height, is_open):
        maizerets = pd.to_numeric(
            cls.maizerets_from_height(height, is_open),
            errors="coerce")
        return pd.to_numeric(tot_flow, errors="coerce") - maizerets

    @classmethod
    def limoilou_n_flow(cls, flow):
        return pd.to_numeric(flow, errors="coerce") / 3 * 24

    @classmethod
    def limoilou_s_flow(cls, flow):
        return pd.to_numeric(flow, errors="coerce") * 2 / 3 * 24

    @classmethod
    def m3h_to_m3d(cls, flow):
        return flow * 24


class VdQPlantMapper(CsvMapper):
    def __init__(self, processing_functions=MapperFuncs):
        super().__init__(processing_functions=processing_functions)

    def read(self, lab_path, lab_map=VDQ_LAB_MAP_NAME):
        sheet_names = ["Données station Est", "Données station Ouest"]
        static_data = self.read_static_data(None)
        xls = pd.read_excel(
            lab_path, sheet_name=sheet_names,
            header=0, skiprows=[1])
        mapping = pd.read_csv(lab_map)
        mapping.fillna("", inplace=True)
        mapping = mapping.astype(str)
        lab_id = None
        site_measure_dfs = []
        for sheet_name, df in xls.items():
            df.columns = [
                self.excel_style(i + 1)
                for i, _ in enumerate(df.columns.to_list())
            ]
            df["location"] = site_map[sheet_name]
            dynamic_tables = self.parse_sheet(
                mapping, static_data, df, self.processing_functions, lab_id
            )
            site_measure_dfs.append(dynamic_tables["SiteMeasure"])

        site_measure = pd.concat(site_measure_dfs)
        site_measure.drop_duplicates(keep="first", inplace=True)
        site_measure.dropna(subset=["value"], inplace=True)
        site_measure = self.type_cast_table("SiteMeasure", site_measure)
        self.site_measure = site_measure
        return


class VdQPlantMapper2022(CsvMapper):
    def __init__(self, processing_functions=MapperFuncs):
        super().__init__(processing_functions=processing_functions)

    def read(self, lab_path, lab_map=VDQ_LAB_MAP_NAME_2022):
        xls = pd.read_excel(lab_path, header=3)
        xls = xls.iloc[1:].copy()
        static_data = self.read_static_data(None)
        mapping = pd.read_csv(lab_map)
        mapping.fillna("", inplace=True)
        mapping = mapping.astype(str)

        lab_id = None
        xls.columns = [
            self.excel_style(i + 1)
            for i, _ in enumerate(xls.columns.to_list())
        ]
        dynamic_tables = self.parse_sheet(
            mapping, static_data, xls, self.processing_functions, lab_id
        )

        site_measure = dynamic_tables["SiteMeasure"]
        site_measure.drop_duplicates(keep="first", inplace=True)
        site_measure.dropna(subset=["value"], inplace=True)
        site_measure = self.type_cast_table("SiteMeasure", site_measure)
        self.site_measure = site_measure
        return


class VdQSensorsMapper(CsvMapper):
    def __init__(self, processing_functions=MapperFuncs):
        super().__init__(processing_functions=processing_functions)

    def read(self, sensors_path, sensors_map=VDQ_SENSOR_MAP_NAME):
        static_data = self.read_static_data(None)
        df = pd.read_excel(sensors_path, header=8, usecols="A:N")
        mapping = pd.read_csv(sensors_map)
        mapping.fillna("", inplace=True)
        mapping = mapping.astype(str)
        lab_id = None
        df.columns = [
            self.excel_style(i + 1)
            for i, _ in enumerate(df.columns.to_list())
        ]
        df = df.loc[
            ~df["D"].astype(str)
            .str.lower().isin(["moyen", "max", "min"])]
        df = df.dropna(how="all")
        date_cols = ["A", "B", "C", "D"]
        numeric_cols = [col for col in df.columns if col not in date_cols]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        dynamic_tables = self.parse_sheet(
            mapping, static_data, df, self.processing_functions, lab_id
        )
        site_measure = dynamic_tables["SiteMeasure"]
        site_measure.drop_duplicates(keep="first", inplace=True)
        site_measure = site_measure.dropna(subset=["value"])
        site_measure = self.type_cast_table("SiteMeasure", site_measure)
        self.site_measure = site_measure
        return


rain_default_site_measurement = {
    "siteMeasureID": None,
    "siteID": None,
    "reporterID": "CityQC",
    "instrumentID": "Pluvio_VilledeQuebec",
    "dateTime": None,
    "type": "envRnF",
    "aggregation": "single",
    "aggregationDesc": "Cumulative rainfall in one day.",
    "value": None,
    "unit": "mm",
    "accessToPublic": "No",
    "accessToAllOrgs": "No",
    "accessToPHAC": "No",
    "accessToLocalHA": "No",
    "accessToProvHA": "No",
    "accessToOtherProv": "No",
    "accessToDetails": "No",
    "notes": ""
}


class VdQRainMapper(bm.BaseMapper):
    def read(self, filepath):
        xl_file = pd.ExcelFile(filepath)
        odm_name = self.conversion_dict["site_measure"]["odm_name"]
        second_sheet_name = xl_file.sheet_names[1]
        xls = pd.read_excel(
            filepath,
            sheet_name=[second_sheet_name],
            header=0,
            skiprows=[1])

        df = xls[second_sheet_name][['Date', 'Pluvio', 'Hauteur totale (mm)']]
        df = df.dropna(subset=['Date'], axis=0)
        df = df.reset_index(drop=True)

        df['siteID'] = pd.Series(map(lambda x: f'QC_wstation_{str(x)}', df['Pluvio']))
        del df['Pluvio']

        for k, v in rain_default_site_measurement.items():
            if v is not None:
                df[k] = v

        df['siteMeasureID'] = df['siteID'] + "_" + df['type'] + "_" \
            + df["Date"].dt.strftime('%Y-%m-%d')
        df['siteMeasureID'].astype(str)
        df.rename(columns={
            "Hauteur totale (mm)": "value",
            "Date": "dateTime"
        }, inplace=True)
        df = df[list(rain_default_site_measurement.keys())]
        site_measure = self.type_cast_table(odm_name, df)
        self.site_measure = site_measure

    def validates(self):
        return True


if __name__ == "__main__":
    filepath = "/Users/jeandavidt/Library/CloudStorage/OneDrive-UniversitéLaval/Université/Doctorat/COVID/Latest Data/Input/2022/Qc_plant/ULaval_Covid2022.xlsm"
    mapper = VdQPlantMapper2022()
    mapper.read(filepath)
    print(mapper.site_measure.head())
