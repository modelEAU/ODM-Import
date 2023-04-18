import itertools
import json
import os
import sqlite3
import warnings
from functools import wraps
from time import time
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import requests

from wbe_odm import utilities
from wbe_odm.odm_mappers import base_mapper


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print(f"{f.__name__}: Elapsed time: { end - start}")
        return result

    return wrapper


# Set pandas to raise en exception when using chained assignment,
# as that may lead to values being set on a view of the data
# instead of on the data itself.
pd.options.mode.chained_assignment = "raise"


class Odm:
    """Data class that holds the contents of the
    tables defined in the Ottawa Data Model (ODM).
    The tables are stored as pandas DataFrames. Utility
    methods are provided to manipulate the data for further analysis.
    """

    def __init__(
        self,
        sample=pd.DataFrame(columns=utilities.get_table_fields("Sample")),  # type: ignore
        ww_measure=pd.DataFrame(columns=utilities.get_table_fields("WWMeasure")),  # type: ignore
        site=pd.DataFrame(columns=utilities.get_table_fields("Site")),  # type: ignore
        site_measure=pd.DataFrame(columns=utilities.get_table_fields("SiteMeasure")),  # type: ignore
        reporter=pd.DataFrame(columns=utilities.get_table_fields("Reporter")),  # type: ignore
        lab=pd.DataFrame(columns=utilities.get_table_fields("Lab")),  # type: ignore
        assay_method=pd.DataFrame(columns=utilities.get_table_fields("AssayMethod")),  # type: ignore
        instrument=pd.DataFrame(columns=utilities.get_table_fields("Instrument")),  # type: ignore
        polygon=pd.DataFrame(columns=utilities.get_table_fields("Polygon")),  # type: ignore
        cphd=pd.DataFrame(columns=utilities.get_table_fields("CPHD")),  # type: ignore
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

    def _default_value_by_dtype(self, dtype: str):
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
            "object": "",
        }
        return null_values.get(dtype, np.nan)

    def combine_table_instances(
        self, table_name: str, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> pd.DataFrame:
        """Merges two instances of an ODM table.

        Args:
            table_name (str): The ODM name of tables you want to combine
            df1 (pd.DataFrame): The first instance of the table
            df2 (pd.DataFrame): The second instance

        Returns:
            pd.DataFrame: The merged table with dupicate rows dropped
        """
        primary_key = utilities.get_primary_key(table_name)
        df = pd.concat([df1, df2])
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
                    combined = self.combine_table_instances(
                        table_name, current_df, new_df  # type: ignore
                    )
                    setattr(self, attr, combined)

                except Exception as e:
                    setattr(self, attr, current_df)
                    raise e from e
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
                    keep="first", ignore_index=True
                )

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

        geo = {"type": "FeatureCollection", "features": []}
        polygon_df = self.polygon.sort_values("polygonID")
        polygon_df["z"] = utilities.rank_polygons_by_desc_area(polygon_df)
        if types is not None:
            if isinstance(types, str):
                types = [types]
            types = [type_.lower() for type_ in types]
            polygon_df = polygon_df.loc[
                polygon_df["type"].str.lower().isin(types)
            ].copy()
        for col in polygon_df.columns:
            is_cat = polygon_df[col].dtype.name == "category"
            polygon_df[col] = (
                polygon_df[col] if is_cat else polygon_df[col].fillna("null")
            )
        for i, row in polygon_df.iterrows():
            if row["wkt"] != "":
                new_feature = {
                    "type": "Feature",
                    "geometry": utilities.convert_wkt_to_geojson(row["wkt"]),
                    "properties": {
                        col: row[col] for col in polygon_df.columns if "wkt" not in col  # type: ignore
                    },
                    "id": i,
                }
                geo["features"].append(new_feature)  # type: ignore
        return geo

    def to_sqlite3(
        self,
        filepath: str,
        attrs_to_save: Optional[list] = None,
    ) -> None:
        """Stores the contents of the ODM object into a SQLite instance.

        Parameters
        ----------
        filepath : [str]
            Path to the SQLite instance
        attrs_to_save : list, optional
            The attributes of the ODM object to save to the database
            (each attribute representing a table).
            If None, all the tables are saved.
        """
        if attrs_to_save is None:
            attrs = self.__dict__
            attrs_to_save = [name for name, value in attrs.items() if not value.empty]
        conversion_dict = base_mapper.BaseMapper.conversion_dict
        if not os.path.exists(filepath):
            create_db(filepath)
        con = sqlite3.connect(filepath)
        for attr in attrs_to_save:
            odm_name = conversion_dict[attr]["odm_name"]
            df = getattr(self, attr)
            if df.empty:
                continue
            df.to_sql(name="myTempTable", con=con, if_exists="replace", index=False)
            cols = df.columns
            cols_str = f"{tuple(cols)}".replace("'", '"')

            sql = f"""REPLACE INTO {odm_name} {cols_str}
                    SELECT * from myTempTable """

            con.execute(sql)
            con.execute("drop table if exists myTempTable")
            con.close()
        return

    def to_csv(
        self,
        path: str,
        file_prefix: Optional[str] = None,
        attrs_to_save: Optional[list[str]] = None,
    ) -> None:
        """Saves the contents of the ODM object to CSV files.

        Parameters
        ----------
        path : str
            The path to the directory where files will be saved.
        file_prefix : str, optional
            The desired prefix that will go in
            front of the Table name in the .csv file name.
        attrs_to_save : list, optional
            The attributes of the ODM object
            to save to file (each attribute representing a table).
            If None, all the tables are saved.
        """
        if attrs_to_save is None:
            attrs = self.__dict__
            attrs_to_save = [
                name for name, df in attrs.items() if df is not None and not df.empty
            ]

        conversion_dict = base_mapper.BaseMapper.conversion_dict
        if not os.path.exists(path):
            os.mkdir(path)
        for attr in attrs_to_save:
            odm_name = conversion_dict[attr]["odm_name"]
            filename = f"{file_prefix}_" + odm_name if file_prefix else odm_name
            df = getattr(self, attr)
            if df is None or df.empty:
                continue
            complete_path = os.path.join(path, filename)
            df.to_csv(complete_path + ".csv", sep=",", index=False)
        return

    def to_parquet(
        self,
        path: str,
        file_prefix: Optional[str] = None,
        attrs_to_save: Optional[list[str]] = None,
    ) -> None:
        """Saves the contents of the ODM object to CSV files.

        Parameters
        ----------
        path : str
            The path to the directory where files will be saved.
        file_prefix : str, optional
            The desired prefix that will go in
            front of the Table name in the .csv file name.
        attrs_to_save : list, optional
            The attributes of the ODM object
            to save to file (each attribute representing a table).
            If None, all the tables are saved.
        """
        if attrs_to_save is None:
            attrs = self.__dict__
            attrs_to_save = [
                name for name, df in attrs.items() if df is not None and not df.empty
            ]

        conversion_dict = base_mapper.BaseMapper.conversion_dict
        if not os.path.exists(path):
            os.mkdir(path)
        for attr in attrs_to_save:
            odm_name = conversion_dict[attr]["odm_name"]
            filename = f"{file_prefix}_" + odm_name if file_prefix else odm_name
            df = getattr(self, attr)
            if df is None or df.empty:
                continue
            complete_path = os.path.join(path, filename)
            df.to_parquet(complete_path + ".parquet")
        return

    def append_odm(self, other_odm) -> None:
        """Joins the data inside another Odm instance to this one.

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
        return TableCombiner(self).combine_per_day_per_site()


class TableWidener:
    wide = None

    def __init__(self, df: pd.DataFrame, features: list[str], qualifiers: list[str]):
        """Creates the widener object and sets which
        columns are qualifiers and which are features

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
                df[qualifier] = (
                    df[qualifier]
                    .str.replace("True", "quality-issue")
                    .replace("False", "no-quality-issue")
                )
        return df

    def widen(self, agg="mean") -> pd.DataFrame:
        """Takes important characteristics inside a table (features) and
        creates new columns to store them based on the value of other columns
        (qualifiers).

        Returns
        -------
        pd.DataFrame
            DataFrame with the original feature and qualifier columns removed
            and the features spread out over new columns named after the values
            of the qualifier columns. The columns have the appropriate data type based on the name of the feature.
        """

        def typecast(df):
            for col_name in df.columns:
                low_col_name = col_name.lower()
                last_part = "_".split(col_name)[-1].lower()  # type: ignore
                if last_part in ["value", "pop", "temp", "size"]:
                    df[col_name] = df[col_name].astype(np.float64)
                elif "timestamp" in last_part or "date" in last_part:
                    df[col_name] = pd.to_datetime(
                        df[col_name], infer_datetime_format=True
                    )
                elif (
                    "flag" in low_col_name or "pooled" in low_col_name or "shippedonice" in low_col_name  # type: ignore
                ):
                    df[col_name] = df[col_name].fillna(False).astype("boolean")
                else:
                    df[col_name] = df[col_name].fillna("").astype(str)
            return df

        df = self.raw_df.copy()
        if df.empty:
            return df
        df = self.clean_qualifier_columns()
        for qualifier in self.qualifiers:
            df[qualifier] = df[qualifier].astype(str)
            df[qualifier] = df[qualifier].str.replace("single", f"single-to-{agg}")
        df["col_qualifiers"] = df[self.qualifiers].agg("_".join, axis=1)
        unique_col_qualifiers = df["col_qualifiers"].unique()
        for col_qualifier, feature in itertools.product(
            unique_col_qualifiers, self.features
        ):
            col_name = "_".join([col_qualifier, feature])
            if "flag" in col_name.lower():
                df[col_name] = pd.Series(dtype=np.bool_)
            else:
                df[col_name] = pd.Series(dtype=np.float64)
            filt = df["col_qualifiers"] == col_qualifier
            df.loc[filt, col_name] = df.loc[filt, feature]  # type: ignore
        df.drop(columns=self.features + self.qualifiers, inplace=True)
        df.drop(columns=["col_qualifiers"], inplace=True)
        df = typecast(df)
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
        for col_name in df.columns.to_list():
            low_col_name = col_name.lower()
            last_part = "_".split(col_name)[-1].lower()  # type: ignore
            if last_part in ["value", "pop", "temp", "size"]:
                df[col_name] = df[col_name].astype(np.float32)
            elif "timestamp" in last_part or "date" in last_part:
                df[col_name] = pd.to_datetime(df[col_name], infer_datetime_format=True)
            elif (
                "flag" in low_col_name or "pooled" in low_col_name or "shippedOnIce" in low_col_name  # type: ignore
            ):
                df[col_name] = df[col_name].fillna(False).astype(np.bool_)
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
        to_remove = [col for col in df.columns if "access" in col.lower()]  # type: ignore
        return df.drop(columns=to_remove)

    # Parsers to go from the standard ODM tables to a unified samples table
    def parse_ww_measure(self, df) -> Optional[pd.DataFrame]:
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
        if wide is None:
            return wide
        wide.drop(columns=["index"], inplace=True)  # type: ignore
        wide = wide.add_prefix("WWMeasure_")
        return wide

    def parse_site_measure(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = self.remove_access(df)
        features = ["value"]
        qualifiers = [
            "type",
            "unit",
            "aggregation",
        ]
        return self.widen(df, features, qualifiers, "SiteMeasure_")

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

    def parse_cphd(self, df) -> Optional[pd.DataFrame]:
        if df.empty:
            return df
        df = self.remove_access(df)
        features = ["value"]
        qualifiers = ["type", "dateType"]
        df["date"] = pd.to_datetime(df["date"])
        return self.widen(df, features, qualifiers, "CPHD_")

    def widen(self, df, features, qualifiers, table_name):
        wide = TableWidener(df, features, qualifiers).widen()
        if wide is None or wide.empty:
            return pd.DataFrame()
        wide = wide.add_prefix(table_name)
        return wide

    def agg_ww_measure_per_sample_id(self, ww: pd.DataFrame) -> pd.DataFrame:
        """Helper function that aggregates the WWMeasure table by sample id.

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
        return ww.groupby("WWMeasure_sampleID").agg(utilities.reduce_by_type)

    @timing
    def combine_sample_table_w_agg_ww_measure(
        self, ww: pd.DataFrame, sample: pd.DataFrame
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
            sample,
            ww,
            how="left",  # samples are the ones with temporal info, so any measure without a sample id is not relevant
            left_on="Sample_sampleID",
            right_on="WWMeasure_sampleID",
        )

    def combine_wide_w_smeasures_based_on_siteid_and_time(
        self, wwmeas_samples_sites: pd.DataFrame, wide_site_measure: pd.DataFrame
    ) -> pd.DataFrame:
        # We want to keep the samples that have no site measure
        # And we want site measures that don't have samples
        # So we use an outer merge
        # The timestamp and the site matter in this merge
        if wwmeas_samples_sites.empty:
            return wide_site_measure
        elif wide_site_measure.empty:
            return wwmeas_samples_sites
        left_columns = wwmeas_samples_sites.columns.tolist()
        left_site = (
            "Sample_siteID" if "Sample_siteID" in left_columns else "Site_siteID"
        )

        return pd.merge(
            wwmeas_samples_sites,
            wide_site_measure,
            how="outer",
            left_on=["Calculated_timestamp", left_site],
            right_on=["Calculated_timestamp", "SiteMeasure_siteID"],
        )

    @timing
    def combine_samples_w_sites(
        self, sample: pd.DataFrame, site: pd.DataFrame
    ) -> pd.DataFrame:
        if site.empty:
            return sample
        elif sample.empty:
            return site

        # Since the samples have the temporal information, the sites without samples are irrelevant
        return pd.merge(
            sample, site, how="left", left_on="Sample_siteID", right_on="Site_siteID"
        )

    @timing
    def select_all_intersected_polygons(self, polygon_lists: npt.NDArray) -> list[str]:
        unique_polygons = set()
        for polygon_list in polygon_lists:
            for polygon in polygon_list:
                unique_polygons.add(polygon)
        if None in unique_polygons:
            unique_polygons.remove(None)
        if "" in unique_polygons:
            unique_polygons.remove("")
        return list(unique_polygons)

    def combine_overlapping_polygons(
        self, merged: pd.DataFrame, polys: pd.DataFrame
    ) -> pd.DataFrame:
        ...

    @timing
    def create_polygon_list_from_intersects(
        self, wide_table: pd.DataFrame, long_polygon_table: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Adds a column called 'polygonIDs' containing a list
        of polygons that pertain to a site
        """
        if wide_table.empty or long_polygon_table.empty:
            return wide_table

        sub_dfs = []

        wide_table["Site_polygonID"] = wide_table["Site_polygonID"].fillna("")
        for site_polygon_id, site_df in wide_table.groupby("Site_polygonID"):
            if site_polygon_id == "":
                site_df["Calculated_polygonList"] = np.nan
            site_df["Calculated_polygonList"] = utilities.get_intersecting_polygons(
                str(site_polygon_id), long_polygon_table
            )
            sub_dfs.append(site_df)
        concatenated_wide_table = pd.concat(sub_dfs, axis=0)

        return concatenated_wide_table

    @timing
    def combine_cphd_polygon_sample(
        self, df: pd.DataFrame, polygon: pd.DataFrame
    ) -> pd.DataFrame:
        if polygon.empty:
            return df
        elif df.empty:
            return polygon
        polygon = polygon.copy()
        polygon = polygon.add_prefix("CPHD-")
        return pd.merge(
            df,
            polygon,
            how="left",
            left_on="Calculated_polygonIDForCPHD",
            right_on="CPHD-Polygon_polygonID",
        )

    @timing
    def combine_polygon_info(
        self, wide_table: pd.DataFrame, polygon_table: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Adds the info of each intersecting polygon to the appropriate rows in the wide table.
        The rows of the wide table each have a Site_siteID.
        The siteID determines what intersecting polygons are relevant to that row.
        For each polygon that intersects with the site, new columns are created for to hold on of the field of the polygon.
        If the polygon is the one mentionned in Site_polygonID, then the prefix of each new column is "SewershedPoly-".
        Otherwise, the prefix is "OverlappingPoly-{polygonID}".
        Once each sub-dataframe has had the polygon info added, they are all combined into one dataframe.
        Because they were split using a groupby, you can concatenate them horizontally.
        The rows that don't have a siteID are dropped because they don't appear in the groupby groups.
        """

        if polygon_table.empty or wide_table.empty:
            return wide_table

        wide_table["Calculated_polygonArr"] = wide_table[
            "Calculated_polygonList"
        ].str.split(";")

        sub_dfs = []
        wide_table["Site_polygonID"] = wide_table["Site_polygonID"].fillna("")
        for poly_id, sub_wide_df in wide_table.groupby("Site_polygonID"):
            if poly_id != "":
                polygon_ids = sub_wide_df["Calculated_polygonArr"].iloc[0]
                for polygon_id in polygon_ids:
                    if polygon_id == "":
                        continue
                    if polygon_id == sub_wide_df["Site_polygonID"].iloc[0]:
                        prefix = "SewershedPoly-"
                    else:
                        prefix = f"OverlappingPoly-{polygon_id}-"
                    for column in polygon_table.columns:
                        polygon_row = polygon_table.loc[
                            polygon_table["Polygon_polygonID"] == polygon_id
                        ].iloc[0]
                        sub_wide_df[f"{prefix}{column}"] = polygon_row[column]
            sub_dfs.append(sub_wide_df)

        full_df = pd.concat(sub_dfs, axis=0)

        del full_df["Calculated_polygonArr"]
        return full_df

    def create_timestamp_from_sm_datetime(self, site_measure):
        if site_measure.empty:
            return site_measure
        site_measure["Calculated_timestamp"] = pd.to_datetime(
            site_measure["SiteMeasure_dateTime"]
        )
        return site_measure

    def create_timestamp_from_sample_dates(self, df):
        # grb -> "dateTime"
        # ps and cp -> if start and end are present: midpoint
        # ps and cp -> if only end is present: end
        df["Calculated_timestamp"] = pd.NaT
        grb_filt = df["Sample_collection"].str.contains("grb")
        s_filt = ~df["Sample_dateTimeStart"].isna()
        e_filt = ~df["Sample_dateTimeEnd"].isna()

        df.loc[grb_filt, "Calculated_timestamp"] = df.loc[grb_filt, "Sample_dateTime"]
        df.loc[grb_filt, "Sample_dateTimeStart"] = df.loc[grb_filt, "Sample_dateTime"]
        df.loc[grb_filt, "Sample_dateTimeEnd"] = df.loc[grb_filt, "Sample_dateTime"]

        df.loc[s_filt & e_filt, "Calculated_timestamp"] = df.apply(
            lambda row: pd.to_datetime(row["Sample_dateTimeEnd"].date()),
            axis=1,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=(
                    ".*will attempt to set the values inplace instead of always setting a new array. "
                    "To retain the old behavior, use either.*"
                ),
            )
            df.loc[e_filt & ~s_filt, "Calculated_timestamp"] = df.loc[
                e_filt & ~s_filt, "Sample_dateTimeEnd"
            ]
        return df

    def create_timestamp_from_cphd_date(self, df):
        df["Calculated_timestamp"] = pd.to_datetime(df["CPHD_date"])
        return df

    @timing
    def combine_cphd_based_on_intersected_polygons(
        self, wide_table: pd.DataFrame, wide_cphd: pd.DataFrame
    ) -> pd.DataFrame:
        def list_overlapping_polygons_from_column_names(
            column_series: pd.Index,
        ) -> list[str]:
            columns = column_series.to_list()
            intersecting_polygons = []
            for col in columns:
                if str(col).startswith("OverlappingPoly"):
                    intersecting_polygons.append(col.split("-")[1])
            return list(set(intersecting_polygons))

        def find_sewershed_polygon_from_column_names(
            merged: pd.DataFrame,
        ) -> str | None:
            sewer_poly_id_col = "SewershedPoly-Polygon_polygonID"
            if sewer_poly_id_col in merged.columns:
                return str(merged[sewer_poly_id_col].unique().item())
            return None

        def rename_wide_cphd_based_on_polygonid(
            wide_cphd: pd.DataFrame, poly_id: str
        ) -> pd.DataFrame:
            to_rename = {}
            for col in wide_cphd.columns:
                if not str(col).startswith("CPHD"):
                    continue
                to_rename[col] = str(col).replace("CPHD", f"CPHD-{poly_id}")
            if to_rename:
                return wide_cphd.rename(columns=to_rename)
            else:
                return wide_cphd

        def merge_wide_table_with_polygon_cphd(
            merged_df: pd.DataFrame, cphd_df: pd.DataFrame, polygon_id: str
        ) -> pd.DataFrame:
            filtered_cphd = cphd_df.loc[cphd_df["CPHD_polygonID"] == polygon_id]
            if filtered_cphd.empty:
                return merged_df
            renamed_cphd = rename_wide_cphd_based_on_polygonid(
                filtered_cphd, polygon_id
            )

            df = pd.merge(
                merged_df,
                renamed_cphd,
                how="outer",
                on="Calculated_timestamp",
            )
            return df

        if wide_cphd.empty:
            return wide_table
        elif wide_table.empty:
            return wide_cphd

        # Here again, we need to group by siteID, because rows from the same site all have the same polygons attached to them.
        # So for a given site, we need to merge the CPHD data for each polygon separately.
        # For the sewershed polygon, we look at the column "SewershedPoly-Polygon_polygonID" and merge the CPHD data for that polygon based on Calculated timestamp.
        # For each of the overlapping polygons, we look at the columns "OverlappingPoly-<polygon_id>-Polygon_polygonID" and merge the CPHD data for that polygon based on Calculated timestamp.
        # Once we have merged the CPHD data for each polygon of a site, we append the merged data to the result_dfs list.
        # Once we've gone through all the sites, we concatenate the result_dfs list into a single dataframe and return it.
        result_dfs = []
        for site_id, site_sub_df in wide_table.groupby("Site_siteID"):
            site_polygons = list_overlapping_polygons_from_column_names(
                site_sub_df.dropna(how="all", axis=1).columns
            )
            site_sewer_polygon = find_sewershed_polygon_from_column_names(site_sub_df)

            site_polygons.append(site_sewer_polygon) if site_sewer_polygon else None

            for polygon in site_polygons:
                site_sub_df = merge_wide_table_with_polygon_cphd(
                    site_sub_df, wide_cphd, polygon
                )

            site_sub_df["Site_siteID"] = site_id

            result_dfs.append(site_sub_df)
        return pd.concat(result_dfs, axis=0)

    @timing
    def agg_site_measure_per_site_and_datetime(self, sm: pd.DataFrame) -> pd.DataFrame:
        if sm.empty:
            return sm
        return sm.groupby(
            ["SiteMeasure_siteID", "SiteMeasure_dateTime"], as_index=False
        ).agg(utilities.reduce_by_type)

    @timing
    def agg_cphd_per_datetime(self, cphd: pd.DataFrame) -> pd.DataFrame:
        if cphd.empty:
            return cphd
        return cphd.groupby(["CPHD_polygonID", "CPHD_date"], as_index=False).agg(
            utilities.reduce_by_type
        )

    def get_sewershed_shape(self, df):
        df["sewershed_shape"] = df["Sewershed-Polygon_wkt"].apply(
            lambda x: utilities.convert_wkt(x), axis=1
        )
        return df

    @timing
    def combine_per_day_per_site(self) -> pd.DataFrame:
        """Combines data from all tables containing data relevant to a given site on a particular day
        Returns
        -------
        pd.DataFrame
            DataFrame with each row representing a day of data for a site.
        """

        # measurements are aligned in time based on the sample date so we align the measurements based on sample id
        ww_measure_per_sample_id = (
            self.agg_ww_measure_per_sample_id(self.ww_measure)
            if self.ww_measure is not None
            else pd.DataFrame()
        )
        if ww_measure_per_sample_id.empty:
            samples_w_wwmeasures = self.sample
        else:
            samples_w_wwmeasures = self.combine_sample_table_w_agg_ww_measure(
                ww_measure_per_sample_id, self.sample
            )

        # clean grab dates
        samples_w_wwmeasures = utilities.clean_grab_datetime(samples_w_wwmeasures)
        # clean composite dates
        samples_w_wwmeasures = utilities.clean_composite_data_intervals(
            samples_w_wwmeasures
        )

        # Add site information to the samples_w_measures table
        samples_w_wwmeasures_w_sites = self.combine_samples_w_sites(
            samples_w_wwmeasures, self.site
        )

        # Add the Calculated_timestamp column to the samples_w_measures_w_sites table
        samples_w_wwmeasures_w_sites = self.create_timestamp_from_sample_dates(
            samples_w_wwmeasures_w_sites
        )

        # With the site information, it's possible to join the site_measure table to the samples_w_measures_w_sites table using the siteID and the timestamp
        if self.site_measure is None or self.site_measure.empty:
            samples_w_wwmeasures_w_sites_w_smeasures = samples_w_wwmeasures_w_sites
        else:
            wide_site_measures = self.agg_site_measure_per_site_and_datetime(
                self.site_measure
            )
            wide_site_measures = self.create_timestamp_from_sm_datetime(
                wide_site_measures
            )

            samples_w_wwmeasures_w_sites_w_smeasures = (
                self.combine_wide_w_smeasures_based_on_siteid_and_time(
                    samples_w_wwmeasures_w_sites, wide_site_measures
                )
            )
        # at this point, 'wide_table' contains all the data from the sample, site, and site_measure tables.

        # The next step is to determine what sewershed and public health regions
        # are relevant to each site and add that information to the table

        wide_table = self.create_polygon_list_from_intersects(
            samples_w_wwmeasures_w_sites_w_smeasures, self.polygon
        )

        # Once we have all the intersecting polygons, we can join the polygon information to the wide_table for each intersecting polygon

        wide_table_w_polys = self.combine_polygon_info(wide_table, self.polygon)

        #
        wide_cphd = (
            self.agg_cphd_per_datetime(self.cphd)
            if self.cphd is not None
            else pd.DataFrame()
        )
        wide_cphd = self.create_timestamp_from_cphd_date(wide_cphd)

        wide_w_polys_w_cphds = self.combine_cphd_based_on_intersected_polygons(
            wide_table_w_polys, wide_cphd
        )

        self.combined = self.typecast_combined(wide_w_polys_w_cphds)
        return self.combined


class OdmEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Odm):
            return {f"__{o.__class__.__name__}__": o.__dict__}
        elif isinstance(o, pd.Timestamp):
            return {"__Timestamp__": str(o)}
        elif isinstance(o, pd.DataFrame):
            return {"__DataFrame__": o.to_json(date_format="iso", orient="split")}
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
