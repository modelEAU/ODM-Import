"""
Description
-----------
General-purpose CsvMapper base class.
"""

# For function annotations
from __future__ import annotations

import os
import re
import warnings
from datetime import datetime

import pandas as pd
import yaml
from easydict import EasyDict
from wbe_odm import utilities
from wbe_odm.odm_mappers import base_mapper, excel_template_mapper


class CsvMapper(base_mapper.BaseMapper):
    # Suffix to add to the attribute names for the ODM tables containing duplicates (that were removed from the actual tables)
    dupes_suffix = "_dupes"

    def __init__(self, processing_functions=None, config_file=None):
        self.start_time = datetime.now()  # Used in format_file_name, to ensure consistent datetime is used
        self.processing_functions = processing_functions

        if config_file:
            with open(config_file, "r") as f:
                self.config = EasyDict(yaml.safe_load(f))
        else:
            self.config = None

    @classmethod
    def remove_null_rows(cls, df, column) -> pd.DataFrame:
        """Remove all rows that have null in the specified column.

        Parameters
        ----------
        df : pd.DataFrame
            Pandas dataframe to remove rows from.
        column : str
            Column to match null values in.

        Returns
        -------
        pd.DataFrame
            The dataframe with all rows with a null in the column removed.
        """
        filt = (~pd.isnull(df[column]))
        return df.loc[filt]

    @classmethod
    def typecast_lab(cls, df, types) -> pd.DataFrame:
        """Typecast all columns in df with the types specified in types.

        Parameters
        ----------
        df : pd.DataFrame
            Pandas DataFrame to typecast.
        types : tuple[str]
            Tuple of string type names (eg. "date", "mixed", "bool", "number", "integer", etc). The order
            corresponds to the order of the columns (df.columns).

        Returns
        -------
        pd.DataFrame
            The DataFrame with all columns typecasted.
        """
        clean_types = []
        for datatype in types:
            if datatype in base_mapper.UNKNOWN_TOKENS:
                datatype = "string"
            datatype = str(datatype)\
                .replace("date", "datetime64[ns]") \
                .replace("mixed", "object") \
                .replace("boolean", "bool") \
                .replace("float", "float64") \
                .replace("integer", "int64") \
                .replace("number", "float64") \
                .replace("text", "string") \
                .replace("blob", "object")
            clean_types.append(datatype)
        for i, col_name in enumerate(df.columns):
            df[col_name] = cls.typecast_column(clean_types[i], df[col_name])
        return df

    @classmethod
    def filter_by_date(cls, df, date_col, start, end) -> pd.DataFrame:
        """Filter all entries in a DataFrame based on a start and end date, in the
        specified column.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to filter.
        date_col : str
            The column, containing dates, to filter by.
        start : int, float, str, datetime
            The start date/time to begin at, exclusive. If empty or None then do not use a lower end.
        end : int, float, str, datetime
            The end date/time to end at, exclusive. If empty or None then do not use an upper end.

        Returns
        -------
        The DataFrame with all rows with dates outside of the exclusive range (start, end) removed, using
        the dates in the column date_col.
        """
        if start is not None and str(start).strip() != "":
            startdate = pd.to_datetime(start, infer_datetime_format=True)
            start_filt = (df[date_col] > startdate)
        else:
            start_filt = None
        if end is not None and str(end).strip() != "":
            enddate = pd.to_datetime(end, infer_datetime_format=True)
            end_filt = (df[date_col] < enddate)
        else:
            end_filt = None
        if start_filt is None and end_filt is None:
            return df
        elif start_filt is None:
            return df[end_filt]
        elif end_filt is None:
            return df[start_filt]
        else:
            return df[start_filt & end_filt]

    @classmethod
    def typecast_column(cls, desired_type, series) -> pd.Series:
        """Typecast all items in a Pandas Series with the specified type.

        Parameters
        ----------
        desired_type : str, type
            The type we want to typecast to. Can be a string (eg. "bool", "string", "int64") or an actual
            type (eg. str, int, bool).
        series : pd.Series
            The series to typecast all elements.

        Returns
        -------
        The series, with all items typecast to desired_type.
        """
        if desired_type == "bool":
            series = series.astype(str)
            series = series.str.strip().str.lower()
            series = series.apply(
                lambda x: base_mapper.replace_unknown_by_default(x, ""))
            series = series.str.replace("oui", "true", case=False)\
                .str.replace("yes", "true", case=False)\
                .str.startswith("true")
        elif desired_type in ["string", "category"]:
            series = series.astype(str)
            series = series.str.lower()
            series = series.str.strip()
            series = series.apply(
                lambda x: base_mapper.replace_unknown_by_default(x, ""))
        elif desired_type in ["int64", "float64"]:
            series = pd.to_numeric(series, errors="coerce")
        elif desired_type == "datetime64[ns]":
            series = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
        series = series.astype(desired_type)
        return series

    @classmethod
    def excel_style(cls, col) -> str:
        """Convert given column number to an Excel-style column name (eg. "A", "BX", etc).

        Parameters
        ----------
        col : int
            The 1-based column number to get the Excel-style column name of.

        Returns
        -------
        str
            The Excel-style column name corresponding to col. (column 1 is "A").
        """
        result = []
        while col:
            col, rem = divmod(col - 1, 26)
            result[:0] = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[rem]
        return "".join(result)

    @classmethod
    def pass_raw(cls, *args) -> object:
        """Return the passed in value, or if multiple values provided combine it into
        one value by joining into a string.

        This is the default function called if a processing_function was specified in the
        CSV mapping file that does not exist, or if no function was identified.

        Parameters
        ----------
        args : tuple
            All arguments to pass.

        Returns
        -------
        varies
            None if no args provided, args[0] if only one arg provided, or all args joined into a
            comma-separated string if multiple args provided.
        """
        if len(args) == 0:
            return None
        elif len(args) == 1:
            return args[0]
        arguments = pd.concat(list(args), axis=1).astype(str)
        return arguments.agg(",".join, axis=1)

    @classmethod
    def str_date_from_timestamp(cls, timestamp_series) -> pd.Series:
        """Convert a date series to a string series.

        Parameters
        ----------
        timestamp_series : pd.Series
            The timestamp series to convert to a string series.

        Returns
        -------
        pd.Series
            timestamp_series convert to a string series, with dates in the format YYYY-mm-dd.
        """
        ts_series = pd.Series(timestamp_series, dtype="datetime64[ns]")
        str_series = ts_series.dt.strftime('%Y-%m-%d')
        return str_series.fillna("")

    @classmethod
    def clean_labels(cls, label) -> str:
        parts = str(label).lower().split("_")
        parts = [part.strip() for part in parts]
        return "_".join(parts)

    @classmethod
    def has_quality_flag(cls, flag) -> bool:
        """Check if quality flag is set.

        Parameters
        ----------
        flag : str
            The quality flag value to test.

        Returns
        -------
        bool
            True if quality flag is set, False otherwise.
        """
        return flag != ""

    @classmethod
    def get_labsheet_inputs(cls, map_row, lab_data, lab_id) -> tuple:
        """Get a tuple of all values specified in the labInputs column of a row in the CSV map file.

        The tuple can be used to pass as parameters to a processing functions.

        Parameters
        ----------
        map_row : pd.Series
            A row from the map CSV file.
        lab_data : pd.DataFrame
            The lab data DataFrame, already loaded and processed, to get the values from.
        lab_id : str
            The lab ID.

        Returns
        -------
        tuple
            A tuple of all parameters specified in map_row's labInputs. The tuple elements
            can be whole columns from lab_data or scalars.
        """
        lab_input = map_row["labInputs"]
        if lab_input == "":
            return None
        var_name = map_row["variableName"]
        raw_inputs = lab_input.split(";")
        final_inputs = []
        for input_ in raw_inputs:
            if re.match(r"__const__.*:.*", input_):
                value, type_ = input_[len("__const__"):].split(":")
                if type_ == "str":
                    value = str(value)
                elif type_ == "int":
                    value = int(value)
            elif input_ == "__labID__":
                value = lab_id
            elif input_ == "__varName__":
                value = var_name
            elif input_ == "__default__":
                value = map_row["defaultValue"]
            else:
                value = lab_data[input_]
            final_inputs.append(value)
        return tuple(final_inputs)

    @classmethod
    def get_all_inputs(cls, row) -> tuple:
        """Get all parameter inputs to the processing function specified in the map CSV row.

        Parameters
        ----------
        row : pd.Series
            The row from the map CSV DataFrame.

        Returns
        -------
        tuple
            A tuple of all parameters to pass to the processing function specified in the row. This will
            include the static data sheet inputs followed by the lab data sheet inputs.
        """
        static_input = row["static"]
        lab_inputs = row["lab_arguments"]
        if static_input is None and lab_inputs is None:
            inputs = None
        elif static_input is None:
            inputs = lab_inputs
        else:
            inputs = (static_input, *lab_inputs)
        if inputs is None:
            inputs = tuple([row["defaultValue"]])
        return inputs

    @classmethod
    def get_static_inputs(cls, map_row, static_data) -> pd.DataFrame:
        """Get the static data sheet (if any) to act as an input to the processing function in the specified map CSV row.

        Static inputs are specified in the "inputSources" column of the map CSV file (eg. "static AssayMethod+lab sheet"
        specifies to use the "AssayMethod" sheet in the static data spreadsheet as a static input).

        Parameters
        ----------
        map_row : pd.Series
            The row from the map CSV file.
        static_data : pd.DataFrame
            The DataFrame of the static data spreadsheet.

        Returns
        -------
        pd.DataFrame
            The static data DataFrame that is specified in the map_row. None if there is no static
            data specified.
        """
        input_sources = map_row["inputSources"]
        if "static" in input_sources:
            static_table = input_sources.split("+")[0]
            static_table = static_table[len("static "):]
            return static_data[static_table]
        else:
            return None

    @classmethod
    def get_processing_function(cls, processing_functions, function_name):
        """Get the processing function with the specified name.

        Parameters
        ----------
        processing_functions : dict | Namespace
            The dictionary or namespace containing all processing functions.
        function_name : str
            The name of the processing function.

        Returns
        -------
        func
            The processing function, or self.raw if none was found.
        """
        if isinstance(processing_functions, dict):
            func = processing_functions.get(function_name, None)
        else:
            func = getattr(processing_functions, function_name, None)
        if func is None:
            func = cls.pass_raw
            if function_name:
                print(f"WARNING: Could not find processing function named {function_name}")
        return func

    @classmethod
    def parse_sheet(cls, mapping, static, lab_data, processing_functions, lab_id) -> dict:
        """Fully parse the lab data and obtain the resulting ODM DataFrames.

        Parameters
        ----------
        mapping : pd.DataFrame
            The mapping DataFrame, obtained from the map CSV file (by calling read_mapping).
        static : dict
            The static data dictionary, with table names as keys and the DataFrames (obtained from the Excel file by
            calling read_static_data)
        lab_data : pd.DataFrame
            The lab data, obtained from loading the lab data spreadsheet, after being fully processed.
        processing_functions : dict | Namespace
            The dictionary or namespace containing all processing functions.
        lab_id : str
            The lab ID.

        Returns
        -------
        dict
            Dictionary of DataFrames obtained from parsing the data. The keys are the ODM table names (str) and the values
            are the DataFrames.
        """
        mapping["lab_arguments"] = mapping.apply(
            lambda row: cls.get_labsheet_inputs(row, lab_data, lab_id), axis=1)
        mapping["static"] = mapping.apply(
            lambda row: cls.get_static_inputs(row, static), axis=1)
        mapping["final_inputs"] = mapping.apply(
            lambda row: cls.get_all_inputs(row), axis=1)
        mapping["func"] = mapping["processingFunction"].apply(lambda x: cls.get_processing_function(processing_functions, x))

        mapping["columnName"] = mapping[
            ["table", "elementName", "variableName"]].agg("_".join, axis=1)
        to_apply = mapping.loc[
            :, ["columnName", "func", "final_inputs"]]
        for _, apply_row in to_apply.iterrows():
            col_name = apply_row["columnName"]
            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore")
                lab_data[col_name] = apply_row["func"](*apply_row["final_inputs"])
        tables = {table: None for table in mapping["table"].unique()}
        for table in tables:
            elements = mapping.loc[
                mapping["table"] == table, "elementName"
            ].unique()
            sub_dfs = []
            for element in elements:
                table_element_filt = (mapping["table"] == table)\
                    & (mapping["elementName"] == element)
                col_names = mapping.loc[table_element_filt, "columnName"]
                var_names = mapping.loc[table_element_filt, "variableName"]
                sub_df = lab_data[col_names]
                sub_df.columns = var_names
                sub_dfs.append(sub_df)
            table_df = pd.concat(sub_dfs, axis=0, ignore_index=True)
            if table in ["WWMeasure", "SiteMeasure"]:
                table_df = table_df.dropna(subset=["value"])
            tables[table] = table_df
        return tables

    def set_empty_odm_tables(self, attr_suffix=None):
        """
        Set all the object's attributes for all the ODM tables, as empty tables.

        Parameters
        ----------
        attr_suffix : str
            Add this suffix to the end of all attribute names. Can be empty.
        """
        attr_suffix = attr_suffix or ""
        for table_name, info in self.conversion_dict.items():
            table_name = f"{table_name}{attr_suffix}"
            setattr(self, table_name, pd.DataFrame(columns=utilities.get_table_fields(info["odm_name"])))

    def set_table_attrs(self, tables, attr_suffix=None):
        """Set all our object's attributes for the ODM DataFrames in the tables dictionary.

        For each item in tables, we convert the table name (eg. "WWMeasure") to the corresponding
        ODM table name (eg. "ww_measure") and use that as the object's attribute name. We optionally
        append attr_suffix to the attribute name when assigning it.

        On return, the tables can be accessed through the attribute accessors (eg. self.ww_measure).

        Parameters
        ----------
        tables : dict
            Dictionary where the keys are are the table names (ODM names) and values are the actual pd.DataFrame tables.
        attr_suffix : str
            An optional suffix to append to all attribute names. If empty then no suffix is added.
        """
        self.set_empty_odm_tables(attr_suffix)

        attr_suffix = attr_suffix or ""
        for table_name, table in tables.items():
            attr = self.get_attr_from_table_name(table_name)
            if attr and table is not None:
                attr = f"{attr}{attr_suffix}"
                setattr(self, attr, table)

    def get_attr_from_table_name(self, table_name) -> str:
        """Get the ODM attribute name from the specified table name.

        Parameters
        ----------
        table_name : str
            The table name (eg. "WWMeasure") to get the ODM attribute name for (eg. "ww_measure").

        Returns
        -------
        str
            The ODM attribute name corresponding to table_name, as specified by the ODM spec.
        """
        for attr, dico in self.conversion_dict.items():
            if dico["odm_name"] == table_name:
                return attr
        print(f"WARNING: Found an unrecognized ODM table name: '{table_name}'")
        return None

    def remove_duplicate_keys(self, tables) -> tuple[dict, dict]:
        """Remove all rows in the parsed ODM tables that have a duplicate primary key.

        Parameters
        ----------
        tables : dict
            Dictionary of all parsed ODM data tables, where they keys are the ODM table names and the
            values are DataFrames.

        Returns
        -------
        new_tables : dict
            The DataFrames with all duplicates (by primary key) removed.
        dupes_tables : dict
            The DataFrames containing the duplicates that were removed from the tables.
        """
        new_tables = {}
        dupes_tables = {}
        primary_keys = utilities.get_primary_key()
        for table_name, df in tables.items():
            if attr := self.get_attr_from_table_name(table_name):
                primary_key = primary_keys.get(table_name, None)
                df_dupes = None
                if primary_key:
                    if primary_key in df.columns:
                        len_a = len(df.index)
                        df_dupes = df
                        df = df.drop_duplicates(subset=[primary_key], keep="last")
                        len_b = len(df.index)
                        df_dupes = df_dupes.drop(df.index, axis="index")
                        print(f"Removed duplicates in {attr}: {len_a} -> {len_b} rows ({len_b-len_a})")
                    else:
                        print(f"WARNING: Primary key '{primary_key}' of table '{attr}' not found when removing duplicates!")
            new_tables[table_name] = df
            dupes_tables[table_name] = df_dupes
        return new_tables, dupes_tables

    def format_file_name(self, file) -> str:
        """Format a file name by replacing tags, surrounded by curly braces, with their
        proper values.

        Parameters
        ----------
        file : str
            The file name, containing the tags to replace.

        Returns
        -------
            The formatted file name.
        """
        if not file:
            return None
        d = self.start_time.strftime("%Y-%m-%d")
        t = self.start_time.strftime("%H-%M-%S")
        dt = f"{d}_{t}"
        lab_id = self.config.lab_id if self.config is not None and "lab_id" in self.config else getattr(self, "lab_id", "unknown_lab")
        return file.format(date=d, time=t, datetime=dt, lab_id=lab_id)

    def save_all(self, output_file, duplicates_file=None):
        """Save all ODM tables that were parsed (and were subsequently set as object attributes).

        This will save both the main DataFrames and the DataFrames containing the removed duplicates (if specified).
        If no tables are available to save, then an Excel file with an empty sheet named "empty" is created.

        Parameters
        ----------
        output_file : str
            The output Excel file to save to. Can contain formatting tags (eg. {date}, see format_file_name)
        duplicates_file : str
            The output Excel file to save the removed duplicates to. If empty then do not save the duplicates.

        Return
        ------
        output_file : str
            The path of the saved output file.
        duplicates_file : str
            The path of the saved duplicates file, or None if none was saved.
        """
        output_file = self.format_file_name(output_file)
        output_file = self.write_tables(output_file)
        if duplicates_file:
            duplicates_file = self.format_file_name(duplicates_file)
            duplicates_file = self.write_tables(duplicates_file, attr_suffix=self.dupes_suffix)

        return output_file, duplicates_file or None

    def write_tables(self, file, attr_suffix=None):
        """Save all processed ODM tables (with the optional suffix) to disk.

        If no tables are available to save, then an Excel file with an empty sheet named "empty" is created.

        This will save one set of tables (each with the same attr_suffix if provided). There is typically
        two sets of tables: the set of main DataFrames, and if duplicates were removed the set of DataFrames
        containing the removed duplicates. This function should be called for each set to save, specifying the
        correct attr_suffix.

        To save all sets, see the save() function.

        Parameters
        ----------
        file : str
            The output Excel file to save to. Can contain formatting tags (eg. {date}, see format_file_name)
        attr_suffix : str
            An optional suffix

        Returns
        -------
        str
            The path of the output file.
        """
        if not file:
            return None
        file = self.format_file_name(file)
        print(f"Saving to '{file}'")
        attr_suffix = attr_suffix or ""
        if os.path.dirname(file):
            os.makedirs(os.path.dirname(file), exist_ok=True)

        tables = [{"table_name": table_name, "table": getattr(self, f"{table_name}{attr_suffix}", None)} for table_name in self.conversion_dict.keys()]
        tables = [t for t in tables if t["table"] is not None]
        if not tables:
            tables = [{"table_name": "empty", "table": pd.DataFrame()}]
        if tables:
            with pd.ExcelWriter(file) as writer:
                for info in tables:
                    table = info["table"]
                    table_name = info["table_name"]
                    table.to_excel(writer, sheet_name=table_name, index=False, freeze_panes=(1, 0))

        return file

    def validates(self):
        """Determine if the ODM data is valid and meets validation requirements

        @TODO: Needs to be implemented.

        Returns
        -------
        bool
            True if the data is valid, false if it is not.
        """
        return True

    def read_static_data(self, staticdata_path) -> dict:
        """Read the static data DataFrames from the Excel file.

        Parameters
        ----------
        staticdata_path : str
            The path to the static data Excel file.

        Returns
        -------
        dict
            Dictionary where all keys are the static data sheet name and values are the DataFrames
            loaded for the sheet. Only the sheets specified in the config file's static_tables array
            are kept.
        """
        if not staticdata_path:
            return None
        staticdata_path = self.format_file_name(staticdata_path)

        # Get the static data
        static_tables = self.config.static_tables
        if static_tables is None:
            return None
        attrs = []
        for table in static_tables:
            attr = self.get_attr_from_table_name(table)
            attrs.append(attr)
        static_data = {}
        excel_mapper = excel_template_mapper.ExcelTemplateMapper()
        excel_mapper.read(staticdata_path, sheet_names=static_tables)
        for table, attr in zip(static_tables, attrs):
            static_data[table] = getattr(excel_mapper, attr)
            setattr(self, attr, static_data[table])
        return static_data

    def read_mapping(self, map_path) -> pd.DataFrame:
        """Read the mapping file from disk.

        Parameters
        ----------
        map_path : str
            The mapping file to load. Can contain formatting tags (eg. {lab_id})

        Returns
        -------
        pd.DataFrame
            The DataFrame of the loaded CSV map file. This is the unprocessed DataFrame.
        """
        map_path = self.format_file_name(map_path)
        mapping = pd.read_csv(map_path, header=0)
        mapping.dropna(axis=0, how="all", inplace=True)
        mapping.fillna("", inplace=True)
        mapping = mapping.astype(str)
        return mapping

    def get_excel_style_columns(self, df) -> list[str]:
        """
        Get list of Excel-style columns for the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to get Excel-style columns for.

        Returns
        -------
        list[str]
            List of Excel-style column names.
        """
        return [
            self.excel_style(i + 1)
            for i, _ in enumerate(df.columns.to_list())
        ]
