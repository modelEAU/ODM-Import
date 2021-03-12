import json
import os
import sqlite3
import warnings

import pandas as pd
import requests
from sqlalchemy import create_engine

import utilities
import visualization_helpers
import constants

pd.options.mode.chained_assignment = 'raise'

lookup = utilities.TABLE_LOOKUP

class Odm:
    def __init__(self, ww_measure=None, site_measure=None, sample=None, site=None, polygon=None, cphd=None):
        self.ww_measure=ww_measure
        self.site_measure=site_measure
        self.sample=sample
        self.site=site
        self.polygon=polygon
        self.cphd=cphd

    def add_to_attr(self, attribute, new_df):
        current_value = getattr(self, attribute)
        if current_value is None:
            setattr(self, attribute, new_df)
            return
        try:
            combined_df = current_value.append(new_df).drop_duplicates()
            setattr(self, attribute, combined_df)
        except Exception as e:
            print(e)
            setattr(self, attribute, current_value)
        return

    def load_from_excel(self, filepath, sheet_names=None):
        if sheet_names is None:
            sheet_names = [x["excel_name"] for x in lookup.keys()]
        
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore")
            xls = pd.read_excel(filepath, sheet_name=sheet_names)

        attributes_to_fill = [utilities.get_attribute_from_name(sheet_name) for sheet_name in sheet_names]
        odm_names = [lookup[attribute]["odm_name"] for attribute in attributes_to_fill]
        parsers = [lookup[attribute]["parser"] for attribute in attributes_to_fill]

        for attribute, odm_name, sheet, fn in zip(attributes_to_fill, odm_names, sheet_names, parsers):
            df = xls[sheet].copy(deep=True)
            df = df.apply(lambda x: utilities.parse_types(odm_name, x), axis=0)
            df = fn(df)
            self.add_to_attr(self, attribute, df)


    def load_from_db(self, cnxn_str, table_names=None):
        if table_names is None:
            table_names = [attribute["odm_name"] for attribute in lookup.keys()]
        
        engine = create_engine(cnxn_str)
        attributes_to_fill = [utilities.get_attribute_from_name(table_name) for table_name in table_names]
        lookup = [lookup[attribute]["parser"] for attribute in attributes_to_fill]

        for attribute, table, fn in zip(attributes_to_fill, table_names, lookup):
            df = pd.read_sql(f"select * from {table}", engine)
            df = fn(df)
            self.add_to_attr(self, attribute, df)


    def get_geoJSON(self):
        geo = {
            "type": "FeatureCollection",
            "features": []
        }
        polygon_df = self.polygon
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

        ww_measure = self.ww_measure
        ww_measure = agg_ww_measure_per_sample(ww_measure)

        sample = self.sample
        merged = combine_ww_measure_and_sample(ww_measure, sample)

        site_measure = self.site_measure
        merged = combine_sample_site_measure(merged, site_measure)

        site = self.site_measure
        merged = combine_site_sample(merged, site)

        cphd = self.cphd
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
    
    def append_odm(self, other_odm):
        for attribute in self.__dict__:
            other_value = getattr(other_odm, attribute)
            self.add_to_attr(self, attribute, other_value)
        return


class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if (isinstance(o, Odm)):
            return {'__{}__'.format(o.__class__.__name__): o.__dict__}
        elif isinstance(o, pd.Timestamp):
            return {'__Timestamp__': str(o)}
        elif isinstance(o, pd.DataFrame):
            return {'__DataFrame__': o.to_json(date_format='epoch', orient='split')}
        else:
            return json.JSONEncoder.default(self, o)


def decode_object(o):
    if '__Odm__' in o:
        a = Odm(
            o['__Odm__']['data'],
        )    
        a.__dict__.update(o['__Odm__'])
        return a

    elif '__DataFrame__' in o:
        a = pd.read_json(o['__DataFrame__'], orient='split')
        return(a)
    elif '__Timestamp__' in o:
        return pd.to_datetime(o['__Timestamp__'])
    else:
        return o


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
    geo = odm_instance.get_geoJSON()
    return odm_instance.combine_per_sample()


def test_samples_from_db():
    # run with example db data
    path = "Data/WBE.db"
    connection_string = f"sqlite:///{path}"
    odm_instance = Odm()
    odm_instance.load_from_db(connection_string)
    geo = odm_instance.get_geoJSON()
    return odm_instance.combine_per_sample()


def test_from_excel_and_db():
    # run with example db data
    path = "Data/WBE.db"
    connection_string = f"sqlite:///{path}"
    odm_instance = Odm()
    filename = "Data/Ville de Québec 202102.xlsx"
    odm_instance.load_from_excel(filename)
    odm_instance.load_from_db(connection_string)
    geo = odm_instance.get_geoJSON()
    return odm_instance.combine_per_sample()


def test_serialization_deserialization():
    # run with example db data
    odm_instance = Odm()
    filename = "Data/Ville de Québec 202102.xlsx"
    odm_instance.load_from_excel(filename)
    odm_instance.get_geo()

    serialized = json.dumps(odm_instance, indent=4, cls=CustomEncoder)
    deserialized = json.loads(serialized, object_hook=decode_object)

    with open('orig.json', 'w') as f:
        json.dump(odm_instance, f, sort_keys=True, indent=4, cls=CustomEncoder)
    with open('deserialized.json', 'w') as g:
        json.dump(deserialized, g, sort_keys=True, indent=4, cls=CustomEncoder)

    print(deserialized == odm_instance)
    with open('orig.json', 'r') as file1:
        with open('deserialized.json', 'r') as file2:
            same = set(file1).intersection(file2)

    same.discard('\n')

    with open('some_output_file.txt', 'w') as file_out:
        for line in same:
            file_out.write(line)
    return None


if __name__ == "__main__":
    test_path = "Data/db/WBE.db"
    create_db(test_path)
    destroy_db(test_path)
    samples = test_samples_from_db()
    samples = test_from_excel_and_db()
    test_serialization_deserialization()
