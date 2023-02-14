import os
from pathlib import Path

import numpy as np
import pandas as pd

from wbe_odm.odm_mappers.base_mapper import replace_unknown_by_default
from wbe_odm.odm_mappers.csv_mapper import CsvMapper

directory = Path(os.path.dirname(__file__))
MI_LAB_MAP_FILEPATH = directory / "mi_lab_map.csv"
MI_DEATHS_MAP_FILEPATH = directory / "mi_dcounty_map.csv"
MI_CASE_MAP_FILEPATH = directory / "mi_case_map.csv"
MI_HOSP_MAP_FILEPATH = directory / "mi_hcounty_map.csv"
MI_PCT_POS_MAP_FILEPATH = directory / "mi_pct_pos_map.csv"
MI_VACCINE_MAP_FILEPATH = directory / "mi_vaccine_map.csv"
MI_CONFIG_FILEPATH = directory / "mi_config.yaml"
OUTPUT_DT_FORMAT = "%Y-%m-%d"
MI_LAB_ID = "metroLab"

COUNTIES = [
    "Anoka County",
    "Washington County",
    "Dakota County",
    "Ramsey County",
    "Hennepin County",
    "Carver County",
    "Scott County",
]


class CaseMapperFuncs:
    @classmethod
    def clean_up_lab_sheet(cls, df):
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], format="%Y-%m-%d")
        return df

    @classmethod
    def build_cphd_id(
        cls,
        reporter: str,
        polygon: str,
        variable: str,
        date_type: str,
        dates: pd.Series,
    ) -> pd.Series:
        return f"{reporter}_{polygon}_{variable}_{date_type}_" + dates.dt.strftime(
            OUTPUT_DT_FORMAT
        )

    @classmethod
    def slice_on_dates(cls, df, date_col, start=None, end=None):
        if start:
            df = df.loc[df[date_col] >= start].copy()
        if end:
            df = df.loc[df[date_col] < end].copy()
        return df

    @classmethod
    def cases_100k_to_cases(
        cls, poly_table: pd.DataFrame, polygon: str, cases: pd.Series
    ) -> pd.Series:
        poly_pop = poly_table[poly_table["polygonID"] == polygon]["pop"].iloc[0]
        cases = cases * poly_pop / 100_000
        return cases.astype(int)


class VaccineMapperFuncs:
    @classmethod
    def clean_up_lab_sheet(cls, df: pd.DataFrame) -> pd.DataFrame:
        df["VAX_WEEK"] = df["VAX_WEEK"].ffill()
        df["VAX_WEEK"] = pd.to_datetime(df["VAX_WEEK"])

        vals = []
        for _, row in df.iterrows():
            for col in df.columns[4:]:
                if not np.isnan(row[col]):
                    vals.append(row[col])
                    continue
        df = df.drop(columns=df.columns[4:])
        df["percentages"] = vals

        df = df.drop(columns=["Vax week whitespace buffer", "Age"])

        df = df.rename(columns={"Unnamed: 3": "variable"})
        df = df[df["variable"] != "Up to date"]
        df["variable"] = df["variable"].map(
            {
                "At least one dose": "pctVaccineDose1",
                "Completed vaccine series": "pctVaccineDose2",
            }
        )
        return df

    @classmethod
    def build_cphd_id(
        cls,
        reporter: str,
        polygon: str,
        variables: pd.Series,
        date_type: str,
        dates: pd.Series,
    ) -> pd.Series:
        return (
            f"{reporter}_{polygon}_"
            + variables
            + "_"
            + date_type
            + dates.dt.strftime(OUTPUT_DT_FORMAT)
        )

    @classmethod
    def slice_on_dates(cls, df, date_col, start=None, end=None):
        if start:
            df = df.loc[df[date_col] >= start].copy()
        if end:
            df = df.loc[df[date_col] < end].copy()
        return df


class PctPosMapperFuncs:
    @classmethod
    def build_cphd_id(
        cls,
        reporter: str,
        polygon: str,
        variable: str,
        date_type: str,
        dates: pd.Series,
    ) -> pd.Series:
        prefix = "_".join([reporter, polygon, variable, date_type])
        formatted_dates = dates.dt.strftime("%Y-%m-%d")
        return prefix + formatted_dates

    @classmethod
    def slice_on_dates(cls, df, date_col, start=None, end=None):
        if start:
            df = df.loc[df[date_col] >= start].copy()
        if end:
            df = df.loc[df[date_col] < end].copy()
        return df

    @classmethod
    def clean_up_lab_sheet(cls, df: pd.DataFrame) -> pd.DataFrame:
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], format="%d-%b-%Y")
        return df

    @classmethod
    def coerce_float(cls, values: pd.Series) -> pd.Series:
        return pd.to_numeric(values, errors="coerce")


class HealthMapperFuncs:
    @classmethod
    def slice_on_dates(cls, df, date_col, start=None, end=None):
        if start:
            df = df.loc[df[date_col] >= start].copy()
        if end:
            df = df.loc[df[date_col] < end].copy()
        return df

    @classmethod
    def clean_up_lab_sheet(cls, df: pd.DataFrame) -> pd.DataFrame:
        df[df.columns[3]] = pd.to_datetime(df[df.columns[3]], format="%m/%d/%Y")
        df[df.columns[4]] = pd.to_datetime(df[df.columns[4]], format="%m/%d/%Y")
        return df

    @classmethod
    def build_cphd_id(
        cls,
        reporters: pd.DataFrame,
        county: pd.Series,
        type_: str,
        date_type: str,
        date: pd.Series,
    ) -> pd.Series:
        reporter_ids = cls.county_to_reporter(reporters, county)
        formatted_dates = cls.parse_health_date(date).dt.strftime("%Y-%m-%d")
        return reporter_ids + f"_{type_}_{date_type}_" + formatted_dates

    @classmethod
    def county_to_reporter(
        cls, reporters: pd.DataFrame, county: pd.Series
    ) -> pd.Series:
        def match_county_and_reporter(county: str) -> str:
            for reporter in reporters["reporterID"]:
                county_name = " ".join(county.split(" ")[:-1]).lower()
                if county_name in reporter.lower():
                    return reporter
                else:
                    continue
            return ""

        return county.map(match_county_and_reporter)

    @classmethod
    def county_to_polygon(cls, polygons: pd.DataFrame, county: pd.Series) -> pd.Series:
        def match_county_and_polygon(county: str) -> str:
            for polygon in polygons["polygonID"]:
                county_name = " ".join(county.split(" ")[:-1]).lower()
                if county_name in polygon.lower():
                    return polygon
                else:
                    continue
            return ""

        return county.map(match_county_and_polygon)

    @classmethod
    def parse_health_date(cls, county: pd.Series) -> pd.Series:
        return pd.to_datetime(county, format="%m/%d/%Y")

    @classmethod
    def coerce_float(cls, values: pd.Series) -> pd.Series:
        return pd.to_numeric(values, errors="coerce")


class LabMapperFuncs:
    @classmethod
    def create_sample_id(cls, site_id: str, date: pd.Series) -> pd.Series:
        date_strs = date.dt.strftime(OUTPUT_DT_FORMAT)
        return date_strs.apply(lambda x: f"{site_id}_{x}")

    @classmethod
    def create_wwmeasure_id(
        cls, site_id: str, date: pd.Series, measure: str
    ) -> pd.Series:
        date_strs = date.dt.strftime(OUTPUT_DT_FORMAT)
        return date_strs.apply(lambda x: f"{measure}_{site_id}_{x}")

    @classmethod
    def create_sitemeasure_id(
        cls, site_id: str, date: pd.Series, measure: str
    ) -> pd.Series:
        date_strs = date.dt.strftime(OUTPUT_DT_FORMAT)
        return date_strs.apply(lambda x: f"{measure}_{site_id}_{x}")

    @classmethod
    def coerce_float(cls, values: pd.Series) -> pd.Series:
        return pd.to_numeric(values, errors="coerce")

    @classmethod
    def convert_to_gc_ml(cls, values: pd.Series) -> pd.Series:
        # value comes in as gc/L
        return cls.coerce_float(values) / 1000

    @classmethod
    def mgd_to_m3d(cls, values: pd.Series) -> pd.Series:
        return cls.coerce_float(values) * 0.0037854

    @classmethod
    def clean_up_lab_sheet(cls, df: pd.DataFrame) -> pd.DataFrame:
        for i, col in enumerate(df.columns):
            if i == 0:
                df[col] = pd.to_datetime(df[col])
            else:
                df[col] = (
                    df[col]
                    .astype(str)
                    .apply(lambda x: replace_unknown_by_default(x, np.nan))
                )
                df[col] = df[col].astype(np.float32)
        return df

    @classmethod
    def slice_on_dates(cls, df, date_col, start=None, end=None):
        if start:
            df = df.loc[df[date_col] >= start].copy()
        if end:
            df = df.loc[df[date_col] < end].copy()
        return df


class MiLabMapper(CsvMapper):
    def __init__(self, config_file=None, processing_functions=LabMapperFuncs):
        super().__init__(
            config_file=config_file, processing_functions=processing_functions
        )

    def read(
        self,
        lab_filepath,
        static_filepath,
        map_filepath=MI_LAB_MAP_FILEPATH,
        start=None,
        end=None,
    ):
        if start:
            start = pd.to_datetime(start, format="%Y-%m-%d")
        if end:
            end = pd.to_datetime(end, format="%Y-%m-%d")

        lab = pd.read_excel(lab_filepath)
        lab = (
            self.processing_functions.clean_up_lab_sheet(lab)
            if self.processing_functions
            else lab
        )
        lab.columns = [
            self.excel_style(i + 1) for i, _ in enumerate(lab.columns.to_list())
        ]
        lab = (
            self.processing_functions.slice_on_dates(
                lab, date_col="A", start=start, end=end
            )
            if self.processing_functions
            else lab
        )
        mapping = pd.read_csv(MI_LAB_MAP_FILEPATH)
        mapping.fillna("", inplace=True)
        mapping = mapping.astype(str)

        static_data = self.read_static_data(static_filepath)

        dynamic_tables = self.parse_sheet(
            mapping, static_data, lab, self.processing_functions, lab_id=MI_LAB_ID
        )
        for table_name, table in dynamic_tables.items():
            table = table.drop_duplicates(keep="first")
            table = self.type_cast_table(table_name, table)
            attr = self.get_attr_from_table_name(table_name)
            setattr(self, attr, table)
        return

    def validates(self):
        return True


class MiHealthMapper(CsvMapper):
    def __init__(self, config_file=None, processing_functions=HealthMapperFuncs):
        super().__init__(
            config_file=config_file, processing_functions=processing_functions
        )

    def read(self, lab_filepath, static_filepath, map_filepath, start=None, end=None):
        if start:
            start = pd.to_datetime(start, format="%Y-%m-%d")
        if end:
            end = pd.to_datetime(end, format="%Y-%m-%d")

        lab = pd.read_csv(lab_filepath)
        lab = (
            self.processing_functions.clean_up_lab_sheet(lab)
            if self.processing_functions
            else lab
        )
        lab.columns = [
            self.excel_style(i + 1) for i, _ in enumerate(lab.columns.to_list())
        ]
        lab = (
            self.processing_functions.slice_on_dates(
                lab, date_col="D", start=start, end=end
            )
            if self.processing_functions
            else lab
        )
        lab = lab.loc[lab["B"].isin(COUNTIES)]
        mapping = pd.read_csv(map_filepath)
        mapping.fillna("", inplace=True)
        mapping = mapping.astype(str)

        static_data = self.read_static_data(static_filepath)

        dynamic_tables = self.parse_sheet(
            mapping, static_data, lab, self.processing_functions, lab_id=MI_LAB_ID
        )
        for table_name, table in dynamic_tables.items():
            table = table.drop_duplicates(keep="first")
            table = self.type_cast_table(table_name, table)
            attr = self.get_attr_from_table_name(table_name)
            setattr(self, attr, table)
        return

    def validates(self):
        return True


class MiPctPosMapper(CsvMapper):
    def __init__(self, config_file=None, processing_functions=PctPosMapperFuncs):
        super().__init__(
            config_file=config_file, processing_functions=processing_functions
        )

    def read(
        self,
        lab_filepath,
        static_filepath,
        map_filepath=MI_PCT_POS_MAP_FILEPATH,
        start=None,
        end=None,
    ):
        if start:
            start = pd.to_datetime(start, format="%Y-%m-%d")
        if end:
            end = pd.to_datetime(end, format="%Y-%m-%d")

        lab = pd.read_excel(lab_filepath, sheet_name="Layout 1")
        lab = (
            self.processing_functions.clean_up_lab_sheet(lab)
            if self.processing_functions
            else lab
        )
        lab.columns = [
            self.excel_style(i + 1) for i, _ in enumerate(lab.columns.to_list())
        ]
        lab = (
            self.processing_functions.slice_on_dates(
                lab, date_col="A", start=start, end=end
            )
            if self.processing_functions
            else lab
        )
        mapping = pd.read_csv(map_filepath)
        mapping.fillna("", inplace=True)
        mapping = mapping.astype(str)

        static_data = self.read_static_data(static_filepath)

        dynamic_tables = self.parse_sheet(
            mapping, static_data, lab, self.processing_functions, lab_id=MI_LAB_ID
        )
        for table_name, table in dynamic_tables.items():
            table = table.drop_duplicates(keep="first")
            table = self.type_cast_table(table_name, table)
            attr = self.get_attr_from_table_name(table_name)
            setattr(self, attr, table)
        return

    def validates(self):
        return True


class MiVaccineMapper(CsvMapper):
    def __init__(self, config_file=None, processing_functions=VaccineMapperFuncs):
        super().__init__(
            config_file=config_file, processing_functions=processing_functions
        )

    def read(
        self,
        lab_filepath,
        static_filepath,
        map_filepath=MI_VACCINE_MAP_FILEPATH,
        start=None,
        end=None,
    ):
        if start:
            start = pd.to_datetime(start, format="%Y-%m-%d")
        if end:
            end = pd.to_datetime(end, format="%Y-%m-%d")

        lab = pd.read_excel(lab_filepath, header=1)
        lab = (
            self.processing_functions.clean_up_lab_sheet(lab)
            if self.processing_functions
            else lab
        )
        lab.columns = [
            self.excel_style(i + 1) for i, _ in enumerate(lab.columns.to_list())
        ]
        lab = (
            self.processing_functions.slice_on_dates(
                lab, date_col="A", start=start, end=end
            )
            if self.processing_functions
            else lab
        )
        mapping = pd.read_csv(map_filepath)
        mapping.fillna("", inplace=True)
        mapping = mapping.astype(str)

        static_data = self.read_static_data(static_filepath)

        dynamic_tables = self.parse_sheet(
            mapping, static_data, lab, self.processing_functions, lab_id=MI_LAB_ID
        )
        for table_name, table in dynamic_tables.items():
            table = table.drop_duplicates(keep="first")
            table = self.type_cast_table(table_name, table)
            attr = self.get_attr_from_table_name(table_name)
            setattr(self, attr, table)
        return

    def validates(self):
        return True


class MiCaseMapper(CsvMapper):
    def __init__(self, config_file=None, processing_functions=CaseMapperFuncs):
        super().__init__(
            config_file=config_file, processing_functions=processing_functions
        )

    def read(
        self,
        lab_filepath,
        static_filepath,
        map_filepath=MI_CASE_MAP_FILEPATH,
        start=None,
        end=None,
    ):
        if start:
            start = pd.to_datetime(start, format="%Y-%m-%d")
        if end:
            end = pd.to_datetime(end, format="%Y-%m-%d")

        lab = pd.read_csv(lab_filepath)
        lab = (
            self.processing_functions.clean_up_lab_sheet(lab)
            if self.processing_functions
            else lab
        )
        lab.columns = [
            self.excel_style(i + 1) for i, _ in enumerate(lab.columns.to_list())
        ]
        lab = (
            self.processing_functions.slice_on_dates(
                lab, date_col="A", start=start, end=end
            )
            if self.processing_functions
            else lab
        )
        mapping = pd.read_csv(map_filepath)
        mapping.fillna("", inplace=True)
        mapping = mapping.astype(str)

        static_data = self.read_static_data(static_filepath)

        dynamic_tables = self.parse_sheet(
            mapping, static_data, lab, self.processing_functions, lab_id=MI_LAB_ID
        )
        for table_name, table in dynamic_tables.items():
            table = table.drop_duplicates(keep="first")
            table = self.type_cast_table(table_name, table)
            attr = self.get_attr_from_table_name(table_name)
            setattr(self, attr, table)
        return

    def validates(self):
        return True


if __name__ == "__main__":
    static_path = "/Users/jeandavidt/Library/CloudStorage/OneDrive-UniversitéLaval/Université/Doctorat/COVID/Latest Data/Input/Ongoing/MI_Static_Data.xlsx"  # noqa

    # lab_path = "/Users/jeandavidt/Library/CloudStorage/OneDrive-UniversitéLaval/Université/Doctorat/COVID/Latest Data/Input/MI/Metro Plant data.xlsx"  # noqa
    # mapper1 = MiLabMapper(config_file=MI_CONFIG_FILEPATH)
    # mapper1.read(lab_path, static_path)
    # print(mapper1.ww_measure)

    hosp_path = "/Users/jeandavidt/Library/CloudStorage/OneDrive-UniversitéLaval/Université/Doctorat/COVID/Latest Data/Input/MI/hcounty.csv"
    deaths_path = "/Users/jeandavidt/Library/CloudStorage/OneDrive-UniversitéLaval/Université/Doctorat/COVID/Latest Data/Input/MI/dcounty.csv"
    ppos_path = "/Users/jeandavidt/Library/CloudStorage/OneDrive-UniversitéLaval/Université/Doctorat/COVID/Latest Data/Input/MI/ww_MN metro percent positive.xlsx"
    vaccines_path = "/Users/jeandavidt/Library/CloudStorage/OneDrive-UniversitéLaval/Université/Doctorat/COVID/Latest Data/Input/MI/Cumulative vaccinations area.xlsx"
    cases_path = "/Users/jeandavidt/Developer/Metropolitan-Council/covid-poops/data/case_data.csv"

    # mapper2 = MiHealthMapper(config_file=MI_CONFIG_FILEPATH)
    # mapper2.read(hosp_path, static_path, MI_HOSP_MAP_FILEPATH)
    # print(mapper2.cphd)

    # mapper3 = MiHealthMapper(config_file=MI_CONFIG_FILEPATH)
    # mapper3.read(deaths_path, static_path, MI_DEATHS_MAP_FILEPATH)
    # print(mapper3.cphd)

    # mapper4 = MiPctPosMapper(config_file=MI_CONFIG_FILEPATH)
    # mapper4.read(ppos_path, static_path, MI_PCT_POS_MAP_FILEPATH)
    # print(mapper4.cphd)

    # mapper5 = MiVaccineMapper(config_file=MI_CONFIG_FILEPATH)
    # mapper5.read(vaccines_path, static_path, MI_VACCINE_MAP_FILEPATH)
    # print(mapper5.cphd)

    mapper6 = MiCaseMapper(config_file=MI_CONFIG_FILEPATH)
    mapper6.read(cases_path, static_path, MI_CASE_MAP_FILEPATH)
    print(mapper6.cphd)
