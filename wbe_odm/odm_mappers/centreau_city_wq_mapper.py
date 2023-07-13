import warnings

import pandas as pd

from wbe_odm.odm_mappers.csv_mapper import CsvMapper


class MapperFuncs:
    @classmethod
    def create_site_measure_id(cls, site_id, dates, parameters):
        df = pd.DataFrame(pd.to_datetime(dates))
        df["type"] = parameters
        df["site_id"] = site_id
        df.columns = ["dates", "type", "site_id"]
        df["formattedDates"] = (
            df["dates"]
            .dt.strftime("%Y-%m-%dT%H:%M:%S")
            .fillna("")
            .str.replace("T00:00:00", "")
        )
        df = df[["site_id", "type", "formattedDates"]]
        return df.agg("_".join, axis=1)

    @classmethod
    def parse_date(cls, dates):
        return pd.to_datetime(dates)


class WQCityMapper(CsvMapper):
    def __init__(self, processing_functions=MapperFuncs):
        super().__init__(processing_functions=processing_functions)

    def read(self, lab_data_path, sheet_name, lab_map_path):
        static_data = self.read_static_data(None)
        mapping = pd.read_csv(lab_map_path)
        mapping.fillna("", inplace=True)
        mapping = mapping.astype(str)

        lab_id = "None"
        sites, origin_site_dfs = self._extract_dfs(
            lab_data_path, sheet_name, idx_col_pos=0, header_row_pos=4
        )
        final_site_measure_dfs = []
        for site_id, site_df in zip(sites, origin_site_dfs):
            if site_id in ["lvl_05", "lvl_02"]:
                # for these, there is no COD or BOD, so we must change remove the rows that have wwCOD or wwBOD5c in the variableName column and for wwTemp, wwTurb and wwCond, we have to change the letter of the column containing the data to the current one minus 2

                variable_mapping = mapping[
                    ~mapping["elementName"].isin(["COD", "BOD5"])
                ].copy()
                # set the "labInputs" column to "H" for wwTemp, wwTurb and wwCond
                # based on two conditions: the defaultValue column must be wwTemp, wwTurb or wwCond and the variableName column must be type
                variable_mapping.loc[
                    (variable_mapping["elementName"] == "Water temp")
                    & (variable_mapping["variableName"] == "value"),
                    "labInputs",
                ] = "F"
                variable_mapping.loc[
                    (variable_mapping["elementName"] == "Turbidity")
                    & (variable_mapping["variableName"] == "value"),
                    "labInputs",
                ] = "G"
                variable_mapping.loc[
                    (variable_mapping["elementName"] == "EC")
                    & (variable_mapping["variableName"] == "value"),
                    "labInputs",
                ] = "H"

            else:
                variable_mapping = mapping
            site_df.columns = [
                self.excel_style(i + 1) for i, _ in enumerate(site_df.columns.to_list())
            ]
            site_df["location"] = site_id
            dynamic_tables = self.parse_sheet(
                variable_mapping,
                static_data,
                site_df,
                self.processing_functions,
                lab_id,
            )
            final_site_measure_dfs.append(dynamic_tables["SiteMeasure"])

        site_measure = pd.concat(final_site_measure_dfs)
        site_measure.drop_duplicates(keep="first", inplace=True)
        site_measure.dropna(subset=["value"], inplace=True)
        site_measure = self.type_cast_table("SiteMeasure", site_measure)
        self.site_measure = site_measure
        return

    def _find_df_borders(self, sheet_cols, idx_col_pos):
        pos_of_cols_w_headers = []
        for i, col in enumerate(sheet_cols):
            if i == idx_col_pos:
                continue
            if "Unnamed" not in col:
                pos_of_cols_w_headers.append(i + 1)
        last_sheet_col = len(sheet_cols)
        pos_of_cols_w_headers.append(last_sheet_col)

        xl_start_cols = []
        xl_end_cols = []

        pos_of_last_item = len(pos_of_cols_w_headers) - 1
        for i in range(len(pos_of_cols_w_headers.copy())):
            if i == pos_of_last_item:
                # This is the end of the last df, so stop
                break

            start_pos = pos_of_cols_w_headers[i]

            if i == pos_of_last_item - 1:
                end_pos = pos_of_cols_w_headers[i + 1]

            else:
                end_pos = pos_of_cols_w_headers[i + 1] - 1

            start_idx = CsvMapper.excel_style(start_pos + 1)
            end_idx = CsvMapper.excel_style(end_pos + 1)
            xl_start_cols.append(start_idx)
            xl_end_cols.append(end_idx)
        return xl_start_cols, xl_end_cols

    def _get_label_ids(self, type_codes):
        return type_codes[1::2]

    def _get_type_codes(self, sheet_df):
        return sheet_df.iloc[1].dropna().to_list()

    def _get_sample_collection(self, type_codes):
        return [str(x).lower() for x in type_codes[::3]]

    def _get_site_ids(self, label_ids):
        sites = []
        for item in label_ids:
            split = item.split("_")[:2]
            site = "_".join(split).lower()
            sites.append(site)
        return sites

    def _get_values_df(self, path, sheet_name, start, end, header_row_pos):
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore")
            return pd.read_excel(
                path,
                sheet_name=sheet_name,
                header=header_row_pos,
                usecols=f"{start}:{end}",
            )

    def _get_index_series(
        self, path: str, sheet_name: str, idx_col: str, header_row_pos: int
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore")
            idx_series = pd.read_excel(
                path,
                sheet_name=sheet_name,
                header=header_row_pos,
                usecols=idx_col,
                squeeze=True,
            )  # type:ignore
        return pd.to_datetime(idx_series, infer_datetime_format=True)

    def _extract_dfs(self, path, sheet_name, idx_col_pos=0, header_row_pos=4):
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore")
            sheet_df = pd.read_excel(path, sheet_name=sheet_name, header=0, index_col=0)

        type_codes = self._get_type_codes(sheet_df)
        label_ids = self._get_label_ids(type_codes)
        site_ids = self._get_site_ids(label_ids)

        sheet_cols = [str(col) for col in sheet_df.columns]
        start_borders, end_borders = self._find_df_borders(sheet_cols, idx_col_pos)
        # remove every second item in start_borders and end_borders
        if "mtl" not in site_ids[0]:
            start_borders = start_borders[::2]
            end_borders = end_borders[::2]

        idx_col = CsvMapper.excel_style(idx_col_pos + 1)

        dfs = []
        # cols_to_keep = [
        #     "BRSV (%rec)",
        #     "Rejected by",
        #     "PMMV (gc/ml)",
        #     "Rejected by.1",
        #     "SARS (gc/ml)",
        #     "Rejected by.2",
        #     "Quality Note",
        # ]
        for start, end in zip(start_borders, end_borders):
            vals = self._get_values_df(path, sheet_name, start, end, header_row_pos)
            idx = self._get_index_series(path, sheet_name, idx_col, header_row_pos)
            df = vals.set_index(idx)
            df = df.dropna(how="all")
            df = df.reset_index(drop=False)
            df.fillna("", inplace=True)
            dfs.append(df)
        return site_ids, dfs


if __name__ == "__main__":
    import os

    folder = "/Users/jeandavidt/Library/CloudStorage/OneDrive-UniversitéLaval/Université/Doctorat/COVID/Latest Data/Input/2022"
    lab_files = [
        file for file in os.listdir(folder) if ".xlsx" in file and "Quebec" not in file
    ]
    for lab_file in lab_files:
        sheet_name = ""
        if "Montreal" in lab_file:
            sheet_name = "MTL_Water_Quality"
            map_file = "./wbe_odm/odm_mappers/city_montreal_wq_map.csv"
        elif "Laval" in lab_file:
            sheet_name = "LVL_Water_Quality"
            map_file = "./wbe_odm/odm_mappers/city_laval_wq_map.csv"
        elif "Gatineau" in lab_file:
            sheet_name = "GTN_Water_Quality"
            map_file = "./wbe_odm/odm_mappers/city_gatineau_wq_map.csv"
        else:
            raise ValueError(f"Unknown lab file: {lab_file}")
        print(lab_file)
        path = os.path.join(folder, lab_file)
        mapper = WQCityMapper()
        mapper = mapper.read(
            path,
            sheet_name=sheet_name,
            lab_map_path=map_file,
        )
