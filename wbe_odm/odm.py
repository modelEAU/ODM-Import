import json
import os
import sqlite3

import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point

from wbe_odm import utilities
from wbe_odm.odm_mappers import base_mapper, csv_mapper
# Set pandas to raise en exception when using chained assignment,
# as that may lead to values being set on a view of the data
# instead of on the data itself.
pd.options.mode.chained_assignment = 'raise'


class Odm:
    """Data class that holds the contents of the
    tables defined in the Ottawa Data Model (ODM).
    The tables are stored as pandas DataFrames. Utility
    functions are provided to manipulate the data for further analysis.
    """
    def __init__(
        self,
        sample=pd.DataFrame(
            columns=utilities.get_table_fields("Sample")),
        ww_measure=pd.DataFrame(
            columns=utilities.get_table_fields("WWMeasure")),
        site=pd.DataFrame(
            columns=utilities.get_table_fields("Site")),
        site_measure=pd.DataFrame(
            columns=utilities.get_table_fields("SiteMeasure")),
        reporter=pd.DataFrame(
            columns=utilities.get_table_fields("Reporter")),
        lab=pd.DataFrame(
            columns=utilities.get_table_fields("Lab")),
        assay_method=pd.DataFrame(
            columns=utilities.get_table_fields("AssayMethod")),
        instrument=pd.DataFrame(
            columns=utilities.get_table_fields("Instrument")),
        polygon=pd.DataFrame(
            columns=utilities.get_table_fields("Polygon")),
        cphd=pd.DataFrame(
            columns=utilities.get_table_fields("CPHD")),
            ) -> None:

        self.sample = sample
        self.ww_measure = ww_measure
        self.site = site
        self.site_measure = site_measure
        self.reporter = reporter
        self.lab = lab
        self.assay_method = assay_method
        self.instrument = instrument
        self.polygon = polygon
        self.cphd = cphd

    def _default_value_by_dtype(
        self, dtype: str
            ):
        """gets you a default value of the correct data type to create new
        columns in a pandas DataFrame

        Parameters
        ----------
        dtype : str
            string name of the data type (found with df[column].dtype)

        Returns
        -------
        [pd.NaT, np.nan, str, None]
            The corresponding default value
        """
        null_values = {
            "datetime64[ns]": pd.NaT,
            "float64": np.nan,
            "int64": np.nan,
            "object": ""
        }
        return null_values.get(dtype, np.nan)

    def append_from(self, mapper) -> None:
        """Concatenates the Odm object's current data with
        that of a mapper.

        Parameters
        ----------
        mapper : odm_mappers.BaseMapper
            A mapper class implementing BaseMapper and adapted to one's
            specific use case
        """
        validates = True if isinstance(mapper, Odm) else mapper.validates()
        if not validates:
            raise ValueError("mapper object contains invalid data")

        self_attrs = self.__dict__
        for attr, current_df in self_attrs.items():
            new_df = getattr(mapper, attr)
            if current_df.empty:
                setattr(self, attr, new_df)
            elif new_df is None or new_df.empty:
                continue
            else:
                try:
                    combined = current_df.append(new_df)\
                        .drop_duplicates(keep="first", ignore_index=True)
                    setattr(self, attr, combined)
                except Exception as e:
                    setattr(self, attr, current_df)
                    raise e
        return

    def load_from(self, mapper: base_mapper.BaseMapper) -> None:
        """Reads an odm mapper object and loads the data into the Odm object.

        Parameters
        ----------
        mapper : odm_mappers.BaseMapper
            A mapper class implementing BaseMapper and adapted to one's
            specific use case

        """
        if mapper.validates():
            self_attrs = self.__dict__
            mapper_attrs = mapper.__dict__
            for key in self_attrs.keys():
                if key not in mapper_attrs:
                    continue
                new_df = mapper_attrs[key]
                self_attrs[key] = new_df.drop_duplicates(
                    keep="first", ignore_index=True)

    def get_polygon_geoJSON(self, types=None) -> dict:
        """[summary]

        Args:
            types ([type], optional): The types of polygons we want to plot.
            Defaults to None, which actually takes everything.

        Returns:
            dict: [description]
        """
        geo = {
            "type": "FeatureCollection",
            "features": []
        }
        polygon_df = self.polygon
        if types is not None:
            if isinstance(types, str):
                types = [types]
            types = [type_.lower() for type_ in types]
            polygon_df = polygon_df.loc[
                polygon_df["type"].str.lower().isin(types)
            ].copy()
        for col in polygon_df.columns:
            is_cat = polygon_df[col].dtype.name == "category"
            polygon_df[col] = polygon_df[col] if is_cat \
                else polygon_df[col].fillna("null")
        for i, row in polygon_df.iterrows():
            if row["wkt"] != "":
                new_feature = {
                    "type": "Feature",
                    "geometry": utilities.convert_wkt_to_geojson(
                        row["wkt"]
                    ),
                    "properties": {
                        col:
                        row[col] for col in polygon_df.columns
                            if "wkt" not in col
                    },
                    "id": i
                }
                geo["features"].append(new_feature)
        return geo

    def to_sqlite3(
        self,
        filepath,
        attrs_to_save: list = None,
            ) -> None:
        if attrs_to_save is None:
            attrs = self.__dict__
            attrs_to_save = [
                name for name, value in attrs.items()
                if not value.empty
            ]
        conversion_dict = base_mapper.BaseMapper.conversion_dict
        if not os.path.exists(filepath):
            create_db(filepath)
        con = sqlite3.connect(filepath)
        for attr in attrs_to_save:
            odm_name = conversion_dict[attr]["odm_name"]
            df = getattr(self, attr)
            if df.empty:
                continue
            df.to_sql(
                name='myTempTable',
                con=con,
                if_exists='replace',
                index=False
            )
            cols = df.columns
            cols_str = f"{tuple(cols)}".replace("'", "\"")

            sql = f"""REPLACE INTO {odm_name} {cols_str}
                    SELECT * from myTempTable """

            con.execute(sql)
            con.execute("drop table if exists myTempTable")
            con.close()
        return

    def to_csv(
        self,
        path: str,
        file_prefix: str,
        attrs_to_save: list = None
    ) -> None:
        if attrs_to_save is None:
            attrs_to_save = []
            attrs = self.__dict__
            for name, df in attrs.items():
                if df is None or df.empty:
                    continue
                attrs_to_save.append(name)

        conversion_dict = base_mapper.BaseMapper.conversion_dict
        if not os.path.exists(path):
            os.mkdir(path)
        for attr in attrs_to_save:
            odm_name = conversion_dict[attr]["odm_name"]
            filename = file_prefix + "_" + odm_name
            df = getattr(self, attr)
            if df is None or df.empty:
                continue
            complete_path = os.path.join(path, filename)
            df.to_csv(complete_path+".csv", sep=",", na_rep="na", index=False)
        return

    def append_odm(self, other_odm):
        for attribute in self.__dict__:
            other_value = getattr(other_odm, attribute)
            self.add_to_attr(attribute, other_value)
        return

    def combine_per_sample(self):
        return TableCombiner(self).combine_per_sample()


class TableWidener:
    wide = None

    def __init__(self, df, features, qualifiers):
        self.raw_df = df
        self.features = features
        self.qualifiers = qualifiers
        self.wide = None

    def clean_qualifier_columns(self):
        qualifiers = self.qualifiers
        df = self.raw_df
        if df.empty:
            return df
        for qualifier in qualifiers:
            filt1 = df[qualifier].isna()
            filt2 = df[qualifier] == ""
            df.loc[filt1 | filt2, qualifier] = f"unknown-{qualifier}"
            df[qualifier] = df[qualifier].str.replace("/", "-")
            if qualifier == "qualityFlag":
                df[qualifier] = df[qualifier].str\
                    .replace("True", "quality-issue")\
                    .replace("False", "no-quality-issue")
        return df

    def widen(self):
        """Takes important characteristics inside a table (features) and
        creates new columns to store them based on the value of other columns
        (qualifiers).

        Returns
        -------
        pd.DataFrame
            DataFrame with the original feature and qualifier columns removed
            and the features spread out over new columns named after the values
            of the qualifier columns.
        """
        df = self.raw_df.copy()
        if df.empty:
            return
        df = self.clean_qualifier_columns()
        for qualifier in self.qualifiers:
            df[qualifier] = df[qualifier].astype(str)
        df["col_qualifiers"] = df[self.qualifiers].agg("_".join, axis=1)
        unique_col_qualifiers = df["col_qualifiers"].unique()
        for col_qualifier in unique_col_qualifiers:
            for feature in self.features:
                col_name = "_".join([col_qualifier, feature])
                df[col_name] = np.nan
                filt = df["col_qualifiers"] == col_qualifier
                df.loc[filt, col_name] = df.loc[filt, feature]
        df.drop(columns=self.features+self.qualifiers, inplace=True)
        df.drop(columns=["col_qualifiers"], inplace=True)
        self.wide = df.copy()
        return self.wide


class TableCombiner(Odm):
    combined = None

    def __init__(self, source_odm):
        self.ww_measure = self.parse_ww_measure(source_odm.ww_measure)
        self.site_measure = self.parse_site_measure(source_odm.site_measure)
        self.sample = self.parse_sample(source_odm.sample)
        self.cphd = self.parse_cphd(source_odm.cphd)
        self.polygon = self.parse_polygon(source_odm.polygon)
        self.site = self.parse_site(source_odm.site)

    def remove_access(self, df: pd.DataFrame) -> pd.DataFrame:
        """removes all columns that set access rights

        Parameters
        ----------
        df : pd.DataFrame
            The tabel with the access rights columns

        Returns
        -------
        pd.DataFrame
            The same table with the access rights columns removed.
        """
        if df.empty:
            return df
        to_remove = [col for col in df.columns if "access" in col.lower()]
        return df.drop(columns=to_remove)

    # Parsers to go from the standard ODM tables to a unified samples table
    def parse_ww_measure(self, df) -> pd.DataFrame:
        """Prepares the WWMeasure Table for merging with
        the samples table to analyzer the data on a per-sample basis

        Returns
        -------
        pd.DataFrame
            Cleaned-up DataFrame indexed by sample.
            - Categorical columns from the WWMeasure table
                are separated into unique columns.
            - Boolean column's values are declared in the column title.
        """
        if df.empty:
            return df

        df = self.remove_access(df)
        features = ["value"]
        qualifiers = [
                "fractionAnalyzed",
                "type",
                "unit",
                "aggregation",
            ]
        wide = TableWidener(df, features, qualifiers).widen()
        wide.drop(columns=["index"], inplace=True)
        wide = wide.add_prefix("WWMeasure.")
        return wide

    def parse_site_measure(self, df) -> pd.DataFrame:
        if df.empty:
            return df
        df = self.remove_access(df)
        features = ["value"]
        qualifiers = [
            "type",
            "unit",
            "aggregation",
        ]
        wide = TableWidener(df, features, qualifiers).widen()

        wide = wide.add_prefix("SiteMeasure.")
        return wide

    def parse_sample(self, df) -> pd.DataFrame:
        if df.empty:
            return df
        df_copy = df.copy(deep=True)

        # we want the sample to show up in any site where it is relevant.
        # Here, we gather all the siteIDs present in the siteID column for a
        # given sample, and we spread them over additional new rows so that in
        # the end, each row of the sample table has only one siteID
        for i, row in df_copy.iterrows():
            # Get the value of the siteID field
            sites = row["siteID"]
            # Check whether there are saveral ids in the field
            if ";" in sites:
                # Get all the site ids in the list
                site_ids = {x.strip() for x in sites.split(";")}
                # Assign one id to the original row
                df["siteID"].iloc[i] = site_ids.pop()
                # Create new rows for each additional siteID and assign them
                # each a siteID
                for site_id in site_ids:
                    new_row = df.iloc[i].copy()
                    new_row["siteID"] = site_id
                    df = df.append(new_row, ignore_index=True)
        df = df.add_prefix("Sample.")
        return df

    def parse_site(self, df) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.add_prefix("Site.")
        return df

    def parse_polygon(self, df) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.add_prefix("Polygon.")
        return df

    def parse_cphd(self, df) -> pd.DataFrame:
        if df.empty:
            return df
        df = self.remove_access(df)
        features = ["value"]
        qualifiers = ["polygonID", "type", "dateType"]
        wide = TableWidener(df, features, qualifiers).widen()

        wide = wide.groupby("dateTime").agg(utilities.reduce_by_type)
        wide.reset_index(inplace=True)
        wide = wide.add_prefix("CPHD.")
        return wide

    def agg_ww_measure_per_sample(self, ww: pd.DataFrame) -> pd.DataFrame:
        """Helper function that aggregates the WWMeasure table by sample.

        Parameters
        ----------
        ww : pd.DataFrame
            The dataframe to rearrange. This dataframe should have gone
            through the _parse_ww_measure funciton before being passed in
            here. This is to ensure that categorical columns have been
            spread out.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the data from the WWMeasure table,
            re-ordered so that each row represents a sample.
        """
        if ww.empty:
            return ww
        return ww.groupby("WWMeasure.sampleID")\
            .agg(utilities.reduce_by_type)

    def resample_site_measures_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df.set_index("SiteMeasure.dateTime", inplace=True)
        df = df.resample("1D").agg(utilities.reduce_by_type)
        df.reset_index(inplace=True)
        return df

    def combine_ww_measure_and_sample(
        self,
        ww: pd.DataFrame,
        sample: pd.DataFrame
            ) -> pd.DataFrame:
        """Merges tables on sampleID

        Parameters
        ----------
        ww : pd.DataFrame
            WWMeasure table re-organized by sample
        sample : pd.DataFrame
            The sample table

        Returns
        -------
        pd.DataFrame
            A combined table containing the data from both DataFrames
        """
        if ww.empty and sample.empty:
            return pd.DataFrame()
        elif sample.empty:
            return ww
        elif ww.empty:
            return sample

        return pd.merge(
            sample, ww,
            how="left",
            left_on="Sample.sampleID",
            right_on="WWMeasure.sampleID")

    def combine_sample_site_measure(
        self,
        sample: pd.DataFrame,
        site_measure: pd.DataFrame
            ) -> pd.DataFrame:
        """Combines site measures and sample tables.

        Parameters
        ----------
        sample : pd.DataFrame
            sample DataFrame
        site_measure : pd.DataFrame
            Site Measure DataFrame

        Returns
        -------
        pd.DataFrame
            A combined DataFrame joined on sampling date
        """
        if sample.empty and site_measure.empty:
            return sample
        elif sample.empty:
            return site_measure
        elif site_measure.empty:
            return sample
        # Pandas doesn't provide good joining capability using dates, so we
        # go through SQLite to perform the join and come back to pandas
        # afterwards.
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

    def combine_site_sample(
        self,
        sample: pd.DataFrame,
        site: pd.DataFrame
            ) -> pd.DataFrame:
        """Combines the sample table with site-specific data.

        Parameters
        ----------
        sample : pd.DataFrame
            The sample table
        site : pd.DataFrame
            The site table

        Returns
        -------
        pd.DataFrame
            A combined DataFrame joined on siteID
        """
        if sample.empty and site.empty:
            return sample
        elif sample.empty:
            return site
        elif site.empty:
            return sample
        return pd.merge(
            sample,
            site,
            how="left",
            left_on="Sample.siteID",
            right_on="Site.siteID")

    def combine_polygons_per_sample(self, merged, polygons):
        """
            Adds a column called 'polygonIDs' containing a list
            of polygons that pertain to a site
        """
        merged["temp_point"] = merged.apply(
            lambda row: Point(
                row["Site.geoLong"], row["Site.geoLat"]
            ), axis=1)
        polygons["shape"] = polygons["Polygon.wkt"].apply(
            lambda x: utilities.convert_wkt(x))
        merged["polygonIDs"] = merged.apply(
            lambda row: utilities.get_encompassing_polygons(
                row, polygons), axis=1)
        merged.drop(["temp_point"], axis=1, inplace=True)
        return merged

    def remove_irrelevant_cphd(row, cphd_val_cols):
        relevant_polys = str(row["polygonIDs"]).split(";")
        for col in cphd_val_cols:
            cols_to_clean = cphd_val_cols.copy()
            for poly_name in relevant_polys:
                if poly_name in col:
                    continue
                cols_to_clean.append(col)
            row[cols_to_clean] = np.nan
        return row

    def combine_cphd_with_samples(
        self,
        sample: pd.DataFrame,
        cphd: pd.DataFrame
            ) -> pd.DataFrame:
        """Return the cphd data relevant to a given dsample using the
        geographical intersection between the sample's sewershed polygon
        and the cphd's health region polygon.

        Parameters
        ----------
        sample : pd.DataFrame
            Table containg sample information as well as a site polygonID
        cphd : pd.DataFrame
            Table containing public health data and a polygonID.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame containing bnoth sample data and public
            health data.
        """
        if sample.empty and cphd.empty:
            return sample
        elif sample.empty:
            return cphd
        elif cphd.empty:
            return sample

        sample["Sample.plotDate"] = utilities.get_plot_datetime(sample)
        sample.dropna(subset=["Sample.plotDate"], inplace=True)
        sample.sort_values("Sample.plotDate", inplace=True)
        cphd["CPHD.dateTime"] = pd.to_datetime(cphd["CPHD.dateTime"])
        cphd.sort_values("CPHD.dateTime", inplace=True)
        max_delta = pd.to_timedelta("24 hours")
        merged = pd.merge_asof(
            sample,
            cphd,
            left_on="Sample.plotDate",
            right_on="CPHD.dateTime",
            tolerance=max_delta)
        # cphd_value_cols = [
            # col for col in merged if "cphd" in col.lower() and "value" in col]
        # merged = merged.apply(lambda row: remove_irrelevant_cphd(row, cphd_value_cols), axis=1)
        return merged

    def combine_per_sample(self) -> pd.DataFrame:
        """Combines data from all tables containing sample-related information
        into a single DataFrame.
        To simplify data mining, the categorical columns are separated into
        distinct columns.
        Returns
        -------
        pd.DataFrame
            DataFrame with each row representing a sample
        """
        agg_ww_measure = self.agg_ww_measure_per_sample(self.ww_measure)

        merged = self.combine_ww_measure_and_sample(
            agg_ww_measure, self.sample)

        agg_site_measure = self.resample_site_measures_daily(self.site_measure)

        merged = self.combine_sample_site_measure(merged, agg_site_measure)

        merged = self.combine_site_sample(merged, self.site)

        merged = self.combine_polygons_per_sample(merged, self.polygon)

        merged = self.combine_cphd_with_samples(merged, self.cphd)

        merged.set_index("Sample.sampleID", inplace=True)
        merged.drop_duplicates(keep="first", inplace=True)
        self.combined = merged
        return merged


class OdmEncoder(json.JSONEncoder):
    def default(self, o):
        if (isinstance(o, Odm)):
            return {
                '__{}__'.format(o.__class__.__name__):
                o.__dict__
            }
        elif isinstance(o, pd.Timestamp):
            return {'__Timestamp__': str(o)}
        elif isinstance(o, pd.DataFrame):
            return {
                '__DataFrame__':
                o.to_json(date_format='iso', orient='split')
            }
        else:
            return json.JSONEncoder.default(self, o)


def create_db(filepath=None):
    url = "https://raw.githubusercontent.com/Big-Life-Lab/covid-19-wastewater/dev/src/wbe_create_table_SQLITE_en.sql"  # noqa
    sql = requests.get(url).text
    conn = None
    if filepath is None:
        filepath = "file::memory"
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


if __name__ == "__main__":
    CSV_FOLDER = "/Users/jeandavidt/OneDrive - Université Laval/COVID/Latest Data/short_csv"  # noqa
    mapper = csv_mapper.CsvMapper()
    mapper.read(CSV_FOLDER)
    store = Odm()
    store.load_from(mapper)
    samples = store.combine_per_sample()
    print("?")
