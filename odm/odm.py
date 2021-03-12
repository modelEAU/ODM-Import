import os
import sqlite3
import warnings

import pandas as pd
import requests
from sqlalchemy import create_engine

import table_parsers
import utilities
import visualization_helpers
import constants

pd.options.mode.chained_assignment = 'raise'


PARSERS = {
    "WWMeasure": {
        "sheet": "WWMeasure",
        "parser": table_parsers.parse_ww_measure,
    },
    "SiteMeasure": {
        "sheet": "SiteMeasure",
        "parser": table_parsers.parse_site_measure,
    },
    "Sample": {
        "sheet": "Sample",
        "parser": table_parsers.parse_sample,
    },
    "Site": {
        "sheet": "Site",
        "parser": table_parsers.parse_site,
    },
    "Polygon": {
        "sheet": "Polygon",
        "parser": table_parsers.parse_polygon,
    },
    "CovidPublicHealthData": {
        "sheet": "CPHD",
        "parser": table_parsers.parse_cphd,
    },
}


class Odm:
    def __init__(self):
        self.data = {}
        self.geo = {
            "type": "FeatureCollection",
            "features": []
        }

    def load_from_excel(self, filepath, table_names=None):
        if table_names is None:
            table_names = list(PARSERS.keys())
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore")
            xls = pd.read_excel(filepath, sheet_name=None)

        sheet_names = [PARSERS[name]["sheet"] for name in table_names]
        parsers = [PARSERS[name]["parser"] for name in table_names]

        for table, sheet, fn in zip(table_names, sheet_names, parsers):
            df = xls[sheet].copy(deep=True)
            df = df.apply(lambda x: utilities.parse_types(table, x), axis=0)
            df = fn(df)
            self.data[table] = df if table not in self.data.keys() \
                else self.data[table].append(df).drop_duplicates()

    def load_from_db(self, cnxn_str, table_names=None):
        if table_names is None:
            table_names = list(PARSERS.keys())
        engine = create_engine(cnxn_str)
        parsers = [PARSERS[name]["parser"] for name in table_names]

        for table, fn in zip(table_names, parsers):
            df = pd.read_sql(f"select * from {table}", engine)
            df = fn(df)
            self.data[table] = df if table not in self.data.keys() \
                else self.data[table].append(df).drop_duplicates()

    def ingest_geometry(self):
        def extract_geo_features(polygon_df):
            geo = {
                "type": "FeatureCollection",
                "features": []
            }
            for i, row in polygon_df.iterrows():
                if row["Polygon.wkt"] in [None, ""]:
                    continue
                new_feature = {
                    "type": "Feature",
                    "geometry": utilities.convert_wkt_to_geojson(row["Polygon.wkt"]),
                    "properties": {
                        "polygonID": row["Polygon.polygonID"],
                    },
                    "id": i
                }
                geo["features"].append(new_feature)
            return geo
        self.geo = extract_geo_features(self.data["Polygon"])

    def combine_per_sample(self):
        def agg_ww_measure_per_sample(ww):
            return ww.groupby("WWMeasure.sampleID").agg(utilities.reduce_by_type)

        def combine_ww_measure_and_sample(ww, sample):
            return pd.merge(
                sample, ww,
                how="left",
                left_on="Sample.sampleID",
                right_on="WWMeasure.sampleID")

        def combine_sample_site_measure(sample, site_measure):
            # Make the db in memory
            conn = sqlite3.connect(':memory:')
            # write the tables
            sample.to_sql('sample', conn, index=False)
            site_measure.to_sql("site_measure", conn, index=False)

            # write the query
            qry = "select * from sample" + \
                " left join site_measure on" + \
                " [SiteMeasure.dateTime] between" + \
                " [Sample.dateTimeStart] and [Sample.dateTimeEnd]"
            merged = pd.read_sql_query(qry, conn)
            conn.close()
            return merged

        def combine_site_sample(sample, site):
            return pd.merge(
                sample,
                site,
                how="left",
                left_on="Sample.siteID",
                right_on="Site.siteID")

        def combine_cphd_by_geo(merged, cphd):
            return merged

        def keep_only_features(df):
            return df

        ww_measure = self.data["WWMeasure"]
        ww_measure = agg_ww_measure_per_sample(ww_measure)

        sample = self.data["Sample"]
        merged = combine_ww_measure_and_sample(ww_measure, sample)

        site_measure = self.data["SiteMeasure"]
        merged = combine_sample_site_measure(merged, site_measure)

        site = self.data["Site"]
        merged = combine_site_sample(merged, site)

        cphd = self.data["CovidPublicHealthData"]
        merged = combine_cphd_by_geo(merged, cphd)

        merged.set_index("Sample.sampleID", inplace=True)
        merged = utilities.remove_columns(
            ["SiteMeasure.SiteID", "Sample.SiteID"],
            merged
        )

        features = keep_only_features(merged)

        return features

    def save_to_db(self, df, table_name, engine):
        df.to_sql(
            name='myTempTable',
            con=engine,
            if_exists='replace',
            index=False
        )
        cols = df.columns
        cols_str = f"{tuple(cols)}".replace("'", "\"")
        with engine.begin() as cn:
            sql = f"""REPLACE INTO {table_name} {cols_str}
                SELECT * from myTempTable """
            cn.execute(sql)
            cn.execute("drop table if exists myTempTable")
        return


def create_db(filepath):
    url = "https://raw.githubusercontent.com/Big-Life-Lab/covid-19-wastewater/dev/src/wbe_create_table_SQLITE_en.sql"
    sql = requests.get(url).text
    conn = None
    try:
        conn = sqlite3.connect(filepath)
        conn.executescript(sql)

    except Exception as e:
        print(e)
    finally:
        if conn:
            conn.close()


def destroy_db(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)


# testing functions
def test_samples_from_excel():
    # run with example excel data
    filename = "Data/Ville de Québec 202102.xlsx"
    odm_instance = Odm()
    odm_instance.load_from_excel(filename)
    odm_instance.ingest_geometry()
    return odm_instance.combine_per_sample()


def test_samples_from_db():
    # run with example db data
    path = "Data/WBE.db"
    connection_string = f"sqlite:///{path}"
    odm_instance = Odm()
    odm_instance.load_from_db(connection_string)
    odm_instance.ingest_geometry()
    return odm_instance.combine_per_sample()


def test_from_excel_and_db():
    # run with example db data
    path = "Data/WBE.db"
    connection_string = f"sqlite:///{path}"
    odm_instance = Odm()
    filename = "Data/Ville de Québec 202102.xlsx"
    odm_instance.load_from_excel(filename)
    odm_instance.load_from_db(connection_string)
    odm_instance.ingest_geometry()
    return odm_instance.combine_per_sample()


if __name__ == "__main__":
    #test_path = "Data/db/WBE.db"
    #create_db(test_path)
    #destroy_db(test_path)
    # samples = test_samples_from_db()
    samples = test_from_excel_and_db()
