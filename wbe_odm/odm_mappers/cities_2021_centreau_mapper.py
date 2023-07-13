import pandas as pd
import sys
from pathlib import Path
from wbe_odm.odm_mappers.csv_mapper import CsvMapper

# get this file's path
CITY_MAP_PATH = Path(str(sys.modules[__name__].__file__)).parent
CITY_MAP_FILE = "cities_2021_centreau_map.csv"

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
    

class WQCityMapper2021(CsvMapper):
    def __init__(self, processing_functions=MapperFuncs):
        super().__init__(processing_functions=processing_functions)
    
    def read(self, data_path, map_path = CITY_MAP_PATH / CITY_MAP_FILE):
        static_data = self.read_static_data(None)
        mapping = pd.read_csv(map_path)
        mapping.fillna("", inplace=True)
        mapping = mapping.astype(str)


        site_dfs = pd.read_excel(data_path, sheet_name=None, header=0)
        final_dfs = []
        for site_id, df in site_dfs.items():
            site_df = df.copy()
            # drop the first row
            site_df.drop(site_df.index[0], inplace=True)
            
            site_df.columns = [
                self.excel_style(i + 1) for i, _ in enumerate(site_df.columns.to_list())
            ]
            lab_id = f"City_{str(site_id).split('_')[0]}"
            site_df["location"] = site_id
            dynamic_tables = self.parse_sheet(
                mapping,
                static_data,
                site_df,
                self.processing_functions,
                lab_id,
            )
            final_dfs.append(dynamic_tables["SiteMeasure"])

        site_measure = pd.concat(final_dfs)
        site_measure.drop_duplicates(keep="first", inplace=True)
        site_measure.dropna(subset=["value"], inplace=True)
        site_measure = self.type_cast_table("SiteMeasure", site_measure)
        self.site_measure = site_measure
        return 
    