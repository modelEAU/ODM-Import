#%%
%load_ext autoreload
%autoreload 2

"""
Description
-----------
Ottawa Lab mapper.
"""

# For function annotations
from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime
import re
import os
from wbe_odm.odm_mappers import base_mapper
from wbe_odm.odm_mappers import excel_template_mapper
from easydict import EasyDict
import argparse
from wbe_odm.odm_mappers.csv_mapper import CsvMapper
from ottawa_cleaner import clean_ottawa_file

def clean_id(id):
    def _clean_id(id):
        return re.sub("[^A-Za-z0-9]", "_", str(id if id not in [None, np.nan] else ""))
    if isinstance(id, str):
        return _clean_id(id)
    else:
        return id.apply(_clean_id)

class MapperFuncs:
    """Namespace for all CSV mapping functions.
    """
    @classmethod
    def get_assay_method_id(cls, lab_id, instrument_id):
        return clean_id(cls.get_instrument_id(lab_id, instrument_id))

    @classmethod
    def get_instrument_description(cls, lab_id, instrument_id):
        return instrument_id.map(lambda x: "")

    @classmethod
    def get_instrument_model(cls, lab_id, instrument_id):
        return instrument_id.map(lambda x: "")

    @classmethod
    def get_instrument_name(cls, lab_id, instrument_id):
        return instrument_id.map(lambda x: str(x).capitalize() if x else "Unknown")

    @classmethod
    def get_assay_method_id(cls, lab_id, instrument_id):
        return clean_id("assaymethod_" + cls.get_instrument_id(lab_id, instrument_id))

    @classmethod
    def get_sample_id(cls, lab_id, sample_date, sample_id):
        sample_id = sample_id.fillna("")
        sample_date = sample_date.fillna("")
        return clean_id(lab_id + "_" + sample_date.astype(str) + "_" + sample_id.astype(str))

    @classmethod
    def get_uwwmeasure_id(cls, lab_id, sample_date, assay_date, instrument_id, sample_id, gene, tag):
        return clean_id(lab_id + "_" + 
            sample_date.astype(str) + "_" + assay_date.astype(str) + "_" + instrument_id.astype(str) + "_" + sample_id.astype(str) + "_" + gene.astype(str) + f"_{tag}")

    @classmethod
    def get_lab_id(cls, lab_id):
        return clean_id(lab_id)

    @classmethod
    def validate_value(cls, value):
        return value

    @classmethod
    def get_instrument_id(cls, lab_id, instrument_id):
        instrument_id = instrument_id.copy()
        instrument_id = instrument_id.fillna("")
        instrument_id[instrument_id == ""] = "unknown"
        instrument_id = lab_id + "_inst_" + instrument_id
        return clean_id(instrument_id)

    @classmethod
    def get_gene_type(cls, txt):
        def _get_type(txt):
            txt = str(txt or "").strip().lower()
            if txt in ["n1", "covn1"]:
                return "covN1"
            elif txt in ["n2", "covn2"]:
                return "covN2"
            elif txt in ["n3", "covn3"]:
                return "covN3"
            return None

        return txt.apply(_get_type)

    @classmethod
    def get_cpc_pmmov(cls, copies, copies_pmmov):
        cpc = copies / copies_pmmov
        return cpc

    @classmethod
    def get_cpl(cls, copies_per_extracted_mass, pellet_weight):
        return copies_per_extracted_mass * pellet_weight / 0.04

    @classmethod
    def get_quality_flag(cls, values):
        return values.isna()

    @classmethod
    def get_notes(cls, values):
        values = values.copy()
        filt = values.isna()
        values.loc[filt] = "Not approved"
        values.loc[~filt] = ""
        return values

class OttawaMapper(CsvMapper):
    def __init__(self, config_file):
        super().__init__(processing_functions=MapperFuncs, config_file=config_file)

    def read(self, labsheet_path, staticdata_path, map_path, clean_first=False, remove_duplicates=False, startdate=None, enddate=None):
        """Read and process all data from disk and convert the data to ODM DataFrames.

        Parameters
        ----------
        labsheet_path : str
            The path to the lab Excel file. The worksheet from the path is the one specified by
            worksheet_name in the config file (and optionally using only the columns specified
            in usecols, also in the config file).
        staticdata_path : str
            The path to the static data Excel file.
        map_path : str
            The mapping file to load.
        clean_first : bool
            If True, then clean the labsheet_path file first.
        remove_duplicates : bool
            If True then remove duplicates from the final ODM data tables based on each table's primary key.
        startdate : int, float, str, datetime
            The start date/time to begin at, exclusive. If empty or None then do not use a
            lower end.
        enddate : int, float, str, datetime
            The end date/time to end at, exclusive. If empty or None then do not use an
            upper end.
        """
        labsheet_path = self.format_file_name(labsheet_path)

        # Clean the file, save cleaned file to temporary file
        if clean_first:
            _, sheets = clean_ottawa_file(labsheet_path)
            lab = sheets[self.config.worksheet_name]
        else:
            try:
                lab = pd.read_excel(labsheet_path,
                                    sheet_name=self.config.worksheet_name,
                                    header=0,
                                    usecols=self.config.usecols or None)
            except Exception as e:
                raise RuntimeError(f"Lab sheet data file does not exist: {labsheet_path}")
            
        # Get all types from type row
        if isinstance(self.config.data_types_row, int):
            lab_datatypes = lab.iloc[self.config.data_types_row].values
        else:
            lab_datatypes = None

        # Convert to Excel-style columns
        lab.columns = self.get_excel_style_columns(lab)

        # Get data section
        lab = lab.iloc[self.config.first_data_row:]

        # Remove null rows
        remove_null_rows_cols = self.config.remove_null_rows_cols
        if not isinstance(remove_null_rows_cols, (list, tuple)):
            remove_null_rows_cols = [remove_null_rows_cols]
        for remove_row in remove_null_rows_cols:
            if remove_row:
                lab = self.remove_null_rows(lab, remove_row)
        
        # Typecast
        if lab_datatypes is not None:
            lab = self.typecast_lab(lab, lab_datatypes)

        lab = lab.dropna(how="all")

        # Filter by date
        if startdate or enddate:
            if self.config.sample_date_col:
                lab = self.filter_by_date(lab, self.config.sample_date_col, startdate, enddate)
            else:
                print("WARNING: sample_date_col was not provided but startdate and/or enddate were.")

        # Fully parse the sheet
        mapping = self.read_mapping(map_path)
        static_data = self.read_static_data(staticdata_path) if staticdata_path else None
        dynamic_tables = self.parse_sheet(
            mapping,
            static_data,
            lab,
            self.processing_functions,
            self.config.lab_id
        )

        # Remove duplicates and save all ODM tables as object attributes.
        dupes_tables = None
        if remove_duplicates:
            dynamic_tables, dupes_tables = self.remove_duplicate_keys(dynamic_tables)

        if dynamic_tables is not None:
            self.set_table_attrs(dynamic_tables)
        if dupes_tables is not None:
            self.set_table_attrs(dupes_tables, attr_suffix=self.dupes_suffix)

if __name__ == "__main__":
    if "get_ipython" in globals():
        opts = EasyDict({
            "config_file" : "ottawa_mapper.yaml",
            "map_path" : "ottawa_map.csv",
            "lab_data" : "/Users/martinwellman/Documents/Health/Wastewater/OttawaLabData/Ottawa - Working Version 2021-05-03.xlsx",
            "clean_first" : True,
            "static_data" : "",
            "start_date" : "",
            "end_date" : "",
            "remove_duplicates" : "/Users/martinwellman/Documents/Health/Wastewater/OttawaLabData/odmdata/{lab_id}/odm_{lab_id}_{datetime}-dupes.xlsx",
            "output" : "/Users/martinwellman/Documents/Health/Wastewater/OttawaLabData/odmdata/{lab_id}/odm_{lab_id}_{datetime}.xlsx",
        })
    else:
        args = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        args.add_argument("--config_file", type=str, help="Path to the mapper YAML config file. (Required)", required=True)
        args.add_argument("--map_path", type=str, help="Path to the mapper CSV file. (Required)", required=True)
        args.add_argument("--lab_data", type=str, help="Path to the lab data Excel file. (Required)", required=True)
        args.add_argument("--clean_first", help="If set then clean the lab_data first. (Optional)", action="store_true")
        args.add_argument("--static_data", type=str, help="Path to the static data Excel file. (Optional)", default=None)
        args.add_argument("--start_date", type=str, help="Filter sample dates starting on this date (exclusive) (yyyy-mm-dd). (Optional)", default=None)
        args.add_argument("--end_date", type=str, help="Filter sample dates ending on this date (exclusive) (yyyy-mm-dd). (Optional)", default=None)
        args.add_argument("--remove_duplicates", type=str, help="If set then remove duplicates from all WW tables based on each table's primary key, and save the duplicates in this additional file. (Optional)", default=None)
        args.add_argument("--output", type=str, help="Path to the Excel output file. (Required)", required=True)
        opts = args.parse_args()

    mapper = OttawaMapper(config_file=opts.config_file)
    mapper.read(opts.lab_data,
                opts.static_data,
                map_path=opts.map_path,
                clean_first=opts.clean_first,
                remove_duplicates=bool(opts.remove_duplicates),
                startdate=opts.start_date, 
                enddate=opts.end_date)
    output_file, duplicates_file = mapper.save_all(opts.output, duplicates_file=opts.remove_duplicates)

