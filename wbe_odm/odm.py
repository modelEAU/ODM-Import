import json
import os
import sqlite3
from typing import Optional, Union, List

import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point


from wbe_odm import utilities
from wbe_odm.odm_mappers import base_mapper, csv_folder_mapper, mcgill_mapper
# Set pandas to raise en exception when using chained assignment,
# as that may lead to values being set on a view of the data
# instead of on the data itself.
pd.options.mode.chained_assignment = 'raise'


class Odm:
    """Data class that holds the contents of the
    tables defined in the Ottawa Data Model (ODM).
    The tables are stored as pandas DataFrames. Utility
    methods are provided to manipulate the data for further analysis.
    """
    def __init__(self,
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
        """Gets you a default value of the correct data type to create new
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

    def combine_table_instances(self, table_name: str, df1:pd.DataFrame, df2:pd.DataFrame) -> pd.DataFrame:
        """Merges two instances of an ODM table.

        Args:
            table_name (str): The ODM name of tables you want to combine
            df1 (pd.DataFrame): The first instance of the table
            df2 (pd.DataFrame): The second instance

        Returns:
            pd.DataFrame: The merged table with dupicate rows dropped
        """
        primary_key = utilities.get_primary_key(table_name)
        df = df1.append(df2)
        # This is way too slow, I'll have to find something else...
        # df = df.groupby(primary_key).agg(utilities.reduce_with_warnings).reset_index()
        df = df.drop_duplicates(subset=[primary_key])
        return df

    def append_from(self, mapper) -> None:              
        """Loads data from a mapper into the current ODM object.

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
                    table_name = base_mapper.get_odm_names(attr)
                    combined = self.combine_table_instances(table_name, current_df, new_df)
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
        """Transforms the Polygon table from the ODM into a geoJSON-like dict

        Parameters
        ----------
        types : Optional[Union[list, str]], optional
            The types of polygons we want to plot.
            Defaults to None, which actually takes everything.
        Returns
        -------
        dict
            geoJSON-formatted dict with the polygon data
        """

        geo = {
            "type": "FeatureCollection",
            "features": []
        }
        polygon_df = self.polygon.sort_values('polygonID')
        polygon_df['z'] = utilities.rank_polygons_by_desc_area(polygon_df)
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

    def to_sqlite3(self,
                   filepath: str,
                   attrs_to_save: list = None,
                ) -> None:
        """Stores the contents of the ODM object into a SQLite instance.

        Parameters
        ----------
        filepath : [str]
            Path to the SQLite instance
        attrs_to_save : list, optional
            The attributes of the ODM object to save to the database (each attribute representing a table). If None, all the tables are saved.
        """
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

    def to_csv(self,
               path: str,
               file_prefix: str = None,
               attrs_to_save: list = None
            ) -> None:
        """Saves the contents of the ODM object to CSV files.

        Parameters
        ----------
        path : str
            The path to the directory where files will be saved.
        file_prefix : str, optional
            The desired prefix that will go in front of the Table name in the .csv file name.
        attrs_to_save : list, optional
            The attributes of the ODM object to save to file (each attribute representing a table). If None, all the tables are saved.
        """
        if attrs_to_save is None:
            attrs = self.__dict__
            attrs_to_save = [
                name
                for name, df in attrs.items()
                if df is not None and not df.empty
            ]

        conversion_dict = base_mapper.BaseMapper.conversion_dict
        if not os.path.exists(path):
            os.mkdir(path)
        for attr in attrs_to_save:
            odm_name = conversion_dict[attr]["odm_name"]
            filename = file_prefix + "_" + odm_name if file_prefix else odm_name
            df = getattr(self, attr)
            if df is None or df.empty:
                continue
            complete_path = os.path.join(path, filename)
            df.to_csv(complete_path+".csv", sep=",", index=False)
        return

    def append_odm(self, other_odm) -> None:
        """ Joins the data inside another Odm instance to this one.

        Parameters
        ----------
        other_odm : [Odm]
            The Odm instance to join to the current one.
        """
        for attribute in self.__dict__:
            other_value = getattr(other_odm, attribute)
            self.add_to_attr(attribute, other_value)
        return

    def add_to_attr(self, attribute, other_value):
        raise NotImplementedError()

    def combine_dataset(self) -> pd.DataFrame:
        """Creates a Wide table out the data contained in the Odm object.

        Returns
        -------
        [pd.DataFrame]
            The combined data.
        """
        return TableCombiner(self).combine_per_sample()


class TableWidener:
    wide = None

    def __init__(self, df: pd.DataFrame, features: List[str], qualifiers: List[str]):
        """Creates the widener object and sets which columns are qualifiers and which are features

        Parameters
        ----------
        df : [type]
            The table to widen
        features : [type]
            [description]
        qualifiers : [type]
            [description]
        """
        self.raw_df = df
        self.features = features
        self.qualifiers = qualifiers
        self.wide = None

    def clean_qualifier_columns(self):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        qualifiers = self.qualifiers
        df = self.raw_df
        if df.empty:
            return df
        for qualifier in qualifiers:
            filt1 = df[qualifier].isna()
            filt2 = df[qualifier] == ""
            df.loc[filt1 | filt2, qualifier] = f"unknown-{qualifier}"
            df[qualifier] = df[qualifier].str.replace("/", "").str.lower()
            if qualifier == "qualityFlag":
                df[qualifier] = df[qualifier].str\
                    .replace("True", "quality-issue")\
                    .replace("False", "no-quality-issue")
        return df

    def widen(self, agg="mean"):
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
            df[qualifier] = df[qualifier].str.replace("single", f"single-to-{agg}")
        df["col_qualifiers"] = df[self.qualifiers].agg("_".join, axis=1)
        unique_col_qualifiers = df["col_qualifiers"].unique()
        for col_qualifier in unique_col_qualifiers:
            for feature in self.features:
                col_name = "_".join([col_qualifier, feature])
                df[col_name] = pd.Series()
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

    def typecast_combined(self, df: pd.DataFrame) -> pd.DataFrame:
        for col_name in df.columns:
            last_part = '_'.split(col_name)[-1].lower()
            if last_part in ['value', 'pop', 'temp', 'size']:
                df[col_name] = df[col_name].astype(np.float32)
            elif 'timestamp' in last_part or 'date' in last_part:
                df[col_name] = pd.to_datetime(df[col_name])
            elif 'flag' in col_name or 'pooled' in col_name or 'shippedOnIce' in col_name:
                df[col_name] = df[col_name].astype(np.bool)
            else:
                df[col_name] = df[col_name].fillna("").astype(str)
        return df

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
        features = ["value", "qualityFlag"]
        qualifiers = [
                # "fractionAnalyzed",
                "type",
                "unit",
                "aggregation",
            ]
        wide = TableWidener(df, features, qualifiers).widen()
        wide.drop(columns=["index"], inplace=True)
        wide = wide.add_prefix("WWMeasure_")
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
        return self.widen(
            df, features, qualifiers, "SiteMeasure_"
        )

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
        df = df.add_prefix("Sample_")
        return df

    def parse_site(self, df) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.add_prefix("Site_")
        return df

    def parse_polygon(self, df) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.add_prefix("Polygon_")
        return df

    def parse_cphd(self, df) -> pd.DataFrame:
        
        if df.empty:
            return df
        df = self.remove_access(df)
        features = ["value"]
        qualifiers = ["type", "dateType"]
        return self.widen(df, features, qualifiers, "CPHD_")

    def widen(self, df, features, qualifiers, table_name):
        wide = TableWidener(df, features, qualifiers).widen()
        wide = wide.add_prefix(table_name)
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
        return ww.groupby("WWMeasure_sampleID")\
            .agg(utilities.reduce_by_type)

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
            left_on="Sample_sampleID",
            right_on="WWMeasure_sampleID")

    def combine_site_measure(self, merged, site_measure):
        return pd.concat([merged, site_measure], axis=0)

    def combine_site_sample(self,
                            sample: pd.DataFrame,
                            site: pd.DataFrame) -> pd.DataFrame:
        if site.empty:
            return sample
        elif sample.empty:
            return site

        return pd.merge(sample, site, how="left",
                        left_on="Sample_siteID",
                        right_on="Site_siteID")

    def get_polygon_list(self, merged, polygons):
        """
            Adds a column called 'polygonIDs' containing a list
            of polygons that pertain to a site
        """
        merged["temp_point"] = merged.apply(
            lambda row: Point(
                row["Site_geoLong"], row["Site_geoLat"]
            ), axis=1)
        polygons["shape"] = polygons["Polygon_wkt"].apply(
            lambda x: utilities.convert_wkt(x))
        merged["Calculated_polygonList"] = merged.apply(
            lambda row: utilities.get_encompassing_polygons(
                row, polygons), axis=1)
        merged.drop(["temp_point"], axis=1, inplace=True)
        polygons.drop(["shape"], axis=1, inplace=True)
        return merged

    def combine_cphd_polygon_sample(self,
                                    df: pd.DataFrame,
                                    polygon: pd.DataFrame) -> pd.DataFrame:
        if polygon.empty:
            return df
        elif df.empty:
            return polygon
        polygon = polygon.copy()
        polygon = polygon.add_prefix("CPHD-")
        return pd.merge(df, polygon, how="left",
                        left_on="Calculated_polygonIDForCPHD",
                        right_on="CPHD-Polygon_polygonID")

    def combine_sewershed_polygon_sample(
            self,
            df: pd.DataFrame,
            polygon: pd.DataFrame) -> pd.DataFrame:
        if polygon.empty:
            return df
        elif df.empty:
            return polygon
        polygon = polygon.copy()
        polygon = polygon.add_prefix("Sewershed-")
        return pd.merge(df, polygon, how="left",
                        left_on="Site_polygonID",
                        right_on="Sewershed-Polygon_polygonID")

    def get_site_measure_ts(self, site_measure):
        if site_measure.empty:
            return site_measure
        site_measure["Calculated_timestamp"] = site_measure[
            "SiteMeasure_dateTime"]
        return site_measure

    def get_samples_timestamp(self, df):
        # grb -> "dateTime"
        # ps and cp -> if start and end are present: midpoint
        # ps and cp -> if only end is present: end
        df["Calculated_timestamp"] = pd.NaT
        grb_filt = df["Sample_collection"].str.contains("grb")
        s_filt = ~df["Sample_dateTimeStart"].isna()
        e_filt = ~df["Sample_dateTimeEnd"].isna()

        df.loc[grb_filt, "Calculated_timestamp"] =\
            df.loc[grb_filt, "Sample_dateTime"]
        df.loc[grb_filt, "Sample_dateTimeStart"] =\
            df.loc[grb_filt, "Sample_dateTime"]
        df.loc[grb_filt, "Sample_dateTimeEnd"] =\
            df.loc[grb_filt, "Sample_dateTime"]

        df.loc[s_filt & e_filt, "Calculated_timestamp"] = df.apply(
            lambda row: utilities.get_midpoint_time(
                row["Sample_dateTimeStart"], row["Sample_dateTimeEnd"]
            ),
            axis=1
        )
        df.loc[
            e_filt & ~s_filt, "Calculated_timestamp"] = df.loc[
                e_filt & ~s_filt, "Sample_dateTimeEnd"]
        return df

    def get_cphd_ts(self, df):
        df["Calculated_timestamp"] = df["CPHD_date"]
        return df

    def combine_cphd(self, merged, cphd_ts):
        if cphd_ts.empty:
            return merged
        elif merged.empty:
            return cphd_ts
        return pd.concat([merged, cphd_ts], axis=0)

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

        samples = self.combine_ww_measure_and_sample(
            agg_ww_measure, self.sample)

        # clean grab dates
        samples = utilities.clean_grab_datetime(samples)
        # clean composite dates
        samples = utilities.clean_composite_data_intervals(samples)
        samples = self.combine_site_sample(samples, self.site)
        samples_ts = self.get_samples_timestamp(samples)
        if self.site_measure.empty:
            merged_s_sm = samples_ts
        else:
            site_measure_ts = self.get_site_measure_ts(self.site_measure)
            merged_s_sm = self.combine_site_measure(samples_ts, site_measure_ts)

        merged_s_sm = self.get_polygon_list(merged_s_sm, self.polygon)
        merged_s_sm = utilities.get_polygon_for_cphd(
            merged_s_sm, self.polygon, self.cphd)
        merged_s_sm_p = self.combine_cphd_polygon_sample(
            merged_s_sm, self.polygon)
        merged_s_sm_pp = self.combine_sewershed_polygon_sample(
            merged_s_sm_p, self.polygon)

        cphd_ts = self.get_cphd_ts(self.cphd)
        merged_s_sm_pp_cphd = self.combine_cphd(merged_s_sm_pp, cphd_ts)

        merged_s_sm_pp_cphd.drop_duplicates(keep="first", inplace=True)
        self.combined = merged_s_sm_pp_cphd
        self.combined = self.typecast_combined(self.combined)
        return self.combined


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

