#%%
"""
Description
-----------
This is to be used temporarily until the Ottawa Lab's data sheet format changes.

Cleans the Ottawa Lab QPCR data file, and add sheets for a wide-table version of the QPCR data and QA data sheets.
"""

# For function annotations
from __future__ import annotations

import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import math
from easydict import EasyDict
from collections import OrderedDict

DATE_COL = "Date"

def clean_qa_data(df) -> pd.DataFrame:
    """Clean data from "QA DATA" sheet.

    Columns are renamed, a row specifying column data types is added (at index 0), invalid 
    rows are removed, and other cleanup operations are performed.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame of the "QA DATA" sheet.

    Returns
    -------
    pd.DataFrame
        The cleaned version of the input df.
    """
    date_column = "sample date"
    # The final table column names and corresponding types
    column_names = [date_column, "gene", "stdev/avg", "single 1", "single 2", "single 3", "single 4", "single 5", "single 6", "avg",    "stdev",  "unit"]
    column_types = ["date",      "text", "number",    "number",   "number",   "number",   "number",   "number",   "number",   "number", "number", "text"]
    # Drop any row where all columns in drop_ifna are None
    drop_ifna = ["single 1", "single 2", "single 3", "single 4", "single 5", "single 6"]

    # The lab sheet has 6 tables, arranged in a 2x3 (v x h) grid, split them up and stack them into a single table
    # Each new table begins with the a column named date_column, split into 3 tables
    splits = [idx for idx in range(len(df.columns)) if df.columns[idx].lower().startswith(date_column)]
    splits.append(len(df.columns))
    tables = [df[df.columns[splits[i]:splits[i+1]]].copy() for i in range(len(splits)-1)]

    # Horizontally, the three tables are for "gcPMMoV", "gcGs", and"gcL".
    for table, unit in zip(tables, ["gcPMMoV", "gcGs", "gcL"]):
        table.columns = column_names
        table[date_column] = pd.to_datetime(table[date_column], errors="coerce")
        table.dropna(subset=[date_column], inplace=True)

        table[drop_ifna] = table[drop_ifna].applymap(lambda x: float(x) if (isinstance(x, float) and not math.isnan(x)) else None)
        # applymap (from above) will sometimes force a None object into a float("nan"), so set them as None here
        table[drop_ifna] = table[drop_ifna].where(pd.notnull(table[drop_ifna]), None)

        table.dropna(subset=["single 1", "single 2", "single 3", "single 4", "single 5", "single 6"], how="all", inplace=True)
        table["unit"] = unit

    # Merge tables
    df = pd.concat(tables, axis=0)

    # Add the types column (ie. date, text, number)
    df.loc[-1] = column_types
    df.index = df.index + 1
    df = df.sort_index()
    df.index = list(range(len(df.index)))

    return df

def parse_date(date) -> pd.Timestamp:
    """Parse a datetime or date string to a Pandas timestamp.

    Additional cleaning up of the date is performed specific to the Ottawa Lab datasheet, fixing
    known errors with the dates.

    Parameters
    ----------
        date : datetime, str
            The datetime or str to parse.

    Returns
    -------
        pd.Timestamp
            The date parsed to a Pandas Timestamp.
    """
    if isinstance(date, datetime):
        date = pd.to_datetime(date)
    elif isinstance(date, str):
        s = date.strip()

        # Remove " - REDO" and " Re run", which appear once each
        if " - REDO" in date or " Re run" in date:
            print(f"@TODO: Make sure we properly replace previous measures for assay dates with string \" - REDO\" or \" Re run\". Matching date is: {date}")
        date = date.replace(" - REDO", "").replace(" Re run", "")
        
        # Remove "th" and "nd" suffix (eg. 25th -> 25)
        if date.endswith("th") or date.endswith("nd"):
            date = date[:-2]

        # Fix up "Septembe" typo
        date = date.replace("Septembe ", "September ")
        
        # Add year if not present
        if np.sum([str(y) in date for y in range(2020, 2025)]) == 0:
            date = "{}, 2020".format(date)
        try:
            date = pd.to_datetime(date)
        except:
            date = None
    else:
        date = None

    return date

def get_inst_and_pool(row, prev_inst = None, prev_pool = None) -> tuple[str, str]:
    """Get the instrument name (eg. "Fisher", "Biorad") and the pool from a row within the Ottawa Lab Excel file,
    if they are present.

    @TODO: Not sure what the pool means or if it's important, but it's present in the sheet. The values
    are mostly blank ("") or "New pepper pool"

    Parameters
    ----------
    row : pd.Series
        The row to retrieve the instrument name and pool.
    prev_inst : str
        The instrument that was most recently identified from a non-None return from this function.
    prev_pool : str
        The pool that was most recently identified from a non-None return from this function.

    Returns
    -------
    str
        The retrieved instrument from the row. If this row does not specify an instrument, we instead
        return prev_inst (ie. we keep the previously identified instrument).
    str
        The retrieved pool from the row. If this row does not specify a pool, we instead
        return prev_pool (ie. we keep the previously identified pool).
    """
    row_date = row[DATE_COL]
    # The instrument and pool are specified on rows where the string "qPCR Data" is in the
    # date column. Whenever we meet such a row, we change the instrument and pool and keep
    # those values up until the next "qPCR Data" row.
    if isinstance(row_date, str) and row_date.strip().lower() == "qpcr data":
        row_vals = row[row.index[:10]][~((row == "") | row.isna())]
        inst = row_vals[2].strip().lower() if len(row_vals.index) > 2 else ""
        if "fisher" in inst:
            inst = "Fisher"
        elif "biorad" in inst:
            inst = "Biorad"
        else:
            inst = None
        pool = row_vals[3].strip() if len(row_vals.index) > 3 and isinstance(row_vals[3], str) else ""
        return inst, pool

    return prev_inst, prev_pool

def clean_qpcr_data(df) -> pd.DataFrame:
    """Clean the "Ottawa qPCR Data" sheet from an Ottawa Lab Excel file.

    Parameters
    ----------
    df : pd.DataFrame
        The Pandas DataFrame of the "Ottawa qPCR Data" sheet.

    Returns
    -------
    pd.DataFrame
        The cleaned version of the input df.
    """
    ASSAYDATE_COL = "Assay Date"
    INSTRUMENT_COL = "Instrument"
    POOL_COL = "Pool"

    # Names and types of all our columns. We also add the columns INSTRUMENT_COL and POOL_COL
    original_columns = [ASSAYDATE_COL, DATE_COL, "Sample ID", "GENE", "Ct [1]", "Ct [2]", "Ct [3]", "Ct Avg",  "Ct Stdev", "Copies [1]", "Copies [2]", "Copies [3]", "Copies AVG", "Copies Stdev", "PMMoV Ct [1]", "PMMoV Ct [2]", "PMMoV Ct [3]", "PMMoV Avg", "PMMoV Stdev", "Empty tube weight (g)", "Full tube weight (g)", "Pellet weight (g)", "Extracted Mass (in 100 uL) (g)", "Copies per Extracted Mass (copies/g) [1]", "Copies per Extracted Mass (copies/g) [2]", "Copies per Extracted Mass (copies/g) [3]", "Copies per Extracted Mass (copies/g) Avg", "Copies per Extracted Mass Stdev", "2^Ct",   "2^Ct normalized to a value", "(2^Ct normalized to a value per Extracted Mass)", "PMMoV Copies [1]", "PMMoV Copies [2]", "PMMoV Copies [3]", "PMMoV Copies Avg", "PMMoV Copies Stdev", "Copies per Copies of PMMoV Avg", "Copies per Copies of PMMoV * 10^3 Avg", "Copies per Copies normalized to a value", "Copies per Copies of PMMoV Stdev", "Copies per Copies of PMMoV * 10^3 Stdev", "Copies per Copies normalized to a value Stdev", "Date [2]", "Gene",   "APPROVED: Copies per Copies of PMMoV [1]", "APPROVED: Copies per Copies of PMMoV [2]", "APPROVED: Copies per Copies of PMMoV [3]", "APPROVED: Copies per Copies of PMMoV Avg", "APPROVED: Copies per Copies of PMMoV Stdev", "APPROVED: Copies per Extracted Mass (copies/g) [1]", "APPROVED: Copies per Extracted Mass (copies/g) [2]", "APPROVED: Copies per Extracted Mass (copies/g) [3]", "APPROVED: Copies per Extracted Mass (copies/g) Avg", "APPROVED: Copies/L [1]", "APPROVED: Copies/L [2]", "APPROVED: Copies/L [3]", "APPROVED: Copies/L Avg", "MESP UPLOAD: PMMoV Copies per Extracted Mass (copies/g)", "MESP UPLOAD: PMMoV Copies/L", "INHIBITION CTRL: Date", "INHIBITION CTRL: Sample Name", "INHIBITION CTRL: Pepper 1/10 [1]", "INHIBITION CTRL: Pepper 1/10 [2]", "INHIBITION CTRL: Pepper 1/10 [3]", "INHIBITION CTRL: Pepper 1/10 Avg", "INHIBITION CTRL: Pepper 1/40 [1]", "INHIBITION CTRL: Pepper 1/40 [2]", "INHIBITION CTRL: Pepper 1/40 [3]", "INHIBITION CTRL: Pepper 1/40 Avg", "INHIBITION CTRL: Pepper No Dilution [1]", "INHIBITION CTRL: Pepper No Dilution [2]", "INHIBITION CTRL: Pepper No Dilution [3]", "INHIBITION CTRL: Pepper No Dilution Avg", "Pepper 1/10 ΔCt", "Pepper 1/40 ΔCt"]
    original_types =   ["date",        "date",   "text",      "text", "number", "number", "number", "number",  "number",   "number",     "number",     "number",     "number",     "number",       "number",       "number",       "number",       "number",    "number",      "number",                "number",               "number",            "number",                         "number",                                   "number",                                   "number",                                   "number",                                   "number",                          "number", "number",                     "number",                                          "number",           "number",           "number",           "number",           "number",             "number",                         "number",                                "number",                                  "number",                           "number",                                  "number",                                        "date",     "text",   "number",                                   "number",                                   "number",                                   "number",                                   "number",                                     "number",                                             "number",                                             "number",                                             "number",                                             "number",                 "number",                 "number",                 "number",                 "number",                                                  "number",                      "date",                  "text",                         "number",                           "number",                           "number",                           "number",                           "number",                           "number",                           "number",                           "number",                           "number",                                  "number",                                  "number",                                  "number",                                  "number",          "number"]
    new_columns = original_columns.copy()
    new_columns[2:2] = [INSTRUMENT_COL, POOL_COL]
    new_types = original_types.copy()
    new_types[2:2] = ["text", "text"]

    # Remove unwanted columns at end of dataframe, and add new columns in the correct order
    df = df[df.columns[:len(original_columns)]].copy()
    df.columns = original_columns
    df[list(set(new_columns) - set(original_columns))] = None
    df = df[new_columns]

    # Go through all rows and clean them, assign appropriate values for current instrument, pool, assay date, etc
    cur_instrument, cur_pool = None, None
    cur_assaydate = None
    has_inhibition_controls = False
    has_copies_per_liter = False
    for i,row in df.iterrows():
        cur_instrument, cur_pool = get_inst_and_pool(row, cur_instrument, cur_pool)
        if row[ASSAYDATE_COL]:
            row_date = parse_date(row[ASSAYDATE_COL])
            cur_assaydate = row_date or cur_assaydate
            if row_date is not None:
                has_inhibition_controls = False
                has_copies_per_liter = False

        if np.any(["Pepper 1/10" in c for c in row if isinstance(c, str)]):
            has_inhibition_controls = True
        if np.any(["Copies / VS" in c or "Copies / L" in c for c in row if isinstance(c, str)]):
            has_copies_per_liter = True
        df.loc[i, ASSAYDATE_COL] = cur_assaydate
        df.loc[i, INSTRUMENT_COL] = cur_instrument
        df.loc[i, POOL_COL] = cur_pool
        if not has_inhibition_controls:
            df.loc[i, [c for c in df.columns if c.startswith("INHIBITION CTRL: ")]] = ""
        if not has_copies_per_liter:
            index = [idx for idx, c in enumerate(df.columns) if c.startswith("APPROVED: Copies/L")][0]
            df.loc[i, df.columns[index:]] = ""

    # Remove invalid rows (where no valid "Date" is specified)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.loc[~df[DATE_COL].isna()]

    # 6 rows have a year of 1900 (from "Septembe 27th" assay date), change these to 2020.
    # 12 rows have a year of 2014 (from "Septembe 17th" assay date), change these to 2020.
    f = (df[DATE_COL].dt.year == 1900) | (df[DATE_COL].dt.year == 2014)
    df.loc[f, DATE_COL] = pd.to_datetime(df.loc[f, DATE_COL].dt.strftime("2020-%m-%d"), format="%Y-%m-%d")

    # Rows from assay date "Tuesday, December 22, 2020" and "Wednesday, December 23, 2020" have incorrect
    # year in "Date" column (2021 instead of 2020)
    df[ASSAYDATE_COL] = pd.to_datetime(df[ASSAYDATE_COL], errors="coerce")
    f = (df[ASSAYDATE_COL] == pd.to_datetime("December 22, 2020")) | (df[ASSAYDATE_COL] == pd.to_datetime("December 23, 2020"))
    df.loc[f, DATE_COL] = pd.to_datetime(df.loc[f, DATE_COL].dt.strftime("2020-%m-%d"), format="%Y-%m-%d")

    # Add row at start that specifies the column types (date, text, number)
    df.loc[-1] = new_types
    df.index = df.index + 1
    df = df.sort_index()
    df.index = list(range(len(df.index)))

    return df

def stack_data(qa_data, qpcr_data) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stack the data by creating a wide-table form of the QA and QPCR data sheets.

    The wide table form of qa_data has the matching qpcr_data row

    Parameters
    ----------
    qa_data : pd.DataFrame
        Pandas DataFrame of the "QA DATA" sheet of an Ottawa Lab Excel file.
    qpcr_data : pd.DataFrame
        Pandas DataFrame of the "Ottawa qPCR Data" sheet of an Ottawa Lab Excel file.

    Returns
    -------
    pd.DataFrame
        The wide-table format of qa_data.
    pd.DataFrame
        The wide-table format of qpcr_data.
    """
    qpcr_data_prefix = "QPCR: "
    qa_data_prefix = "QA: "
    num_float_cols = 3
    # In the wide-table version of the QPCR data, we (possibly) have a matching row in the QA data sheet for the 
    # gcPMMoV, gcGs, and gcL values. The dictionary below provides the configuration of these 3 groups, including
    # the new column names in the wide-table for that group (qpcr_qa_cols), the columns in the QPCR sheet that contain
    # the calculated values for the unit (qpcr_float_cols), and a new column name that will include the Excel row number 
    # in the QA data sheet that was copied to the QPCR data sheet (qpcr_qa_matchrow_col)
    qa_groups = {
        "gcPMMoV" : {
            "qpcr_float_cols" : [f"{qpcr_data_prefix}APPROVED: Copies per Copies of PMMoV [{n+1}]" for n in range(num_float_cols)],
            "qpcr_qa_cols" : [],
            "qpcr_qa_matchrow_col" : "",
        }, 
        "gcGs" : {
            "qpcr_float_cols" : [f"{qpcr_data_prefix}APPROVED: Copies per Extracted Mass (copies/g) [{n+1}]" for n in range(num_float_cols)],
            "qpcr_qa_cols" : [],
            "qpcr_qa_matchrow_col" : "",
        }, 
        "gcL" : {
            "qpcr_float_cols" : [f"{qpcr_data_prefix}APPROVED: Copies/L [{n+1}]" for n in range(num_float_cols)],
            "qpcr_qa_cols" : [],
            "qpcr_qa_matchrow_col" : "",
        },
    }
    qa_matched_qpcr_row_col = f"{qa_data_prefix}Matched QPCR Row"
    qpcr_matched_qa_row_col = f"{qpcr_data_prefix}Matched QA Row"

    stacked_qa_data = qa_data.copy()
    stacked_qpcr_data = qpcr_data.copy()

    # Rename QA columns and add the matched row column (in other sheet)
    stacked_qa_data.columns = [f"{qa_data_prefix}{c}" for c in stacked_qa_data.columns]
    stacked_qa_data_new_cols = stacked_qa_data.columns.tolist()
    stacked_qa_data[qa_matched_qpcr_row_col] = ""
    stacked_qa_data.loc[0, qa_matched_qpcr_row_col] = "number"

    # Create the wide-table column names, in stacked_qpcr_data, for each QA group (gcPMMoV, gcGs, and gcL)
    for group, d in qa_groups.items():
        def _fmt(group, c):
            return f"{group} {c}"
        columns = [_fmt(group, c) for c in stacked_qa_data_new_cols]
        d["qpcr_qa_cols"] = columns
        d["qpcr_qa_matchrow_col"] = _fmt(group, qpcr_matched_qa_row_col)

    # Rename QPCR columns
    stacked_qpcr_data.columns = [f"{qpcr_data_prefix}{c}" for c in stacked_qpcr_data.columns]
    stacked_qpcr_data_new_cols = stacked_qpcr_data.columns.tolist()

    # Copy column types row
    stacked_qa_data.loc[0, stacked_qpcr_data_new_cols] = stacked_qpcr_data.loc[0, stacked_qpcr_data_new_cols].tolist()
    for group, d in qa_groups.items():
        stacked_qpcr_data.loc[0, d["qpcr_qa_cols"]] = stacked_qa_data.loc[0, stacked_qa_data_new_cols].tolist()
        stacked_qpcr_data.loc[0, d["qpcr_qa_matchrow_col"]] = "number"

    # Corresponding float values to match in the QA sheet (ie. each column in qa_groups's qpcr_float_cols must match these, in order)
    match_floats_qa_data = [f"{qa_data_prefix}single {n}" for n in range(1, 4)]

    for row_num, row in stacked_qa_data.loc[1:].iterrows():
        # Create the match filter: First match sample date and gene name
        match_filt = (stacked_qpcr_data[f"{qpcr_data_prefix}Date"] == row[f"{qa_data_prefix}sample date"]) & \
            (stacked_qpcr_data[f"{qpcr_data_prefix}GENE"] == row[f"{qa_data_prefix}gene"])
        # Match all floats (from match_floats_qpcr_data)
        group = row[f"{qa_data_prefix}unit"]
        for raw_col, lab_col in zip(qa_groups[group]["qpcr_float_cols"], match_floats_qa_data):
            if row[lab_col] is None or isinstance(row[lab_col], str):
                match_filt = match_filt & (stacked_qpcr_data[raw_col].apply(lambda x: not isinstance(x, float)))
            else:
                match_filt = match_filt & ((stacked_qpcr_data[raw_col].apply(lambda x: x if isinstance(x, float) else float("inf")) - row[lab_col]).abs() < 0.0001)

        num_matches = match_filt.sum()
        if num_matches != 1:
            print(f"No unique match for row {row_num}: Num matches: {num_matches}")
        else:
            match_idx = match_filt.idxmax()
            # Set the matched row numbers
            stacked_qa_data.loc[row_num, qa_matched_qpcr_row_col] = match_idx+2
            stacked_qpcr_data.loc[match_idx, qa_groups[group]["qpcr_qa_matchrow_col"]] = row_num+2
            # Copy the matched rows between the sheets
            stacked_qa_data.loc[row_num, stacked_qpcr_data_new_cols] = stacked_qpcr_data.loc[match_idx, stacked_qpcr_data_new_cols].tolist() #match_row[stacked_qpcr_data.columns].tolist()
            stacked_qpcr_data.loc[match_idx, qa_groups[group]["qpcr_qa_cols"]] = stacked_qa_data.loc[row_num, stacked_qa_data_new_cols].tolist()

    return stacked_qa_data, stacked_qpcr_data

def clean_ottawa_file(input_file, output_file=None) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean the specified Ottawa Lab Excel file.

    Parameters
    ----------
    input_file : str
        Path and filename of the input Excel file.
    output_file : str
        Path and filename of the cleaned output Excel file. If None then do not save to disk,
        instead return the cleaned DataFrames.

    Returns
    -------
    output_file : str
        The path and filename to the cleaned output file, or None if none was saved.
    qpcr_data : pd.DataFrame
        The DataFrame of the cleaned QPCR data
    qa_data : pd.DataFrame
        The DataFrame of the cleaned QA data
    stacked_qpcr_data : pd.DataFrame
        Wide-table form of QPCR data
    stacked_qa_data : pd.DataFrame
        Wide-table form of QA data
    """
    print(f"Loading file '{input_file}'")
    xl = pd.ExcelFile(input_file)

    qpcr_data = xl.parse("Ottawa qPCR Data", header=None, keep_default_na=False)
    qa_data = xl.parse("QA DATA", header=1, keep_default_na=False)

    qpcr_data = clean_qpcr_data(qpcr_data)
    qa_data = clean_qa_data(qa_data)

    stacked_qa_data, stacked_qpcr_data = stack_data(qa_data, qpcr_data)

    sheets = OrderedDict({
        "Ottawa qPCR Data" : qpcr_data,
        "QA DATA" : qa_data,
        "Stacked QPCR Data" : stacked_qpcr_data,
        "Stacked QA Data" : stacked_qa_data,
    })

    if output_file:
        print(f"Saving file '{output_file}'")
        with pd.ExcelWriter(output_file) as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False, freeze_panes=(2, 0))

    return output_file, sheets

if __name__ == "__main__":
    if 'get_ipython' in globals():
        OPTS = EasyDict({
            "input" : "/Users/martinwellman/Documents/Health/Wastewater/OttawaLabData/Ottawa - Working Version 2021-05-03.xlsx",
            "output" : "Cleaned.xlsx",
        })
    else:
        args = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        args.add_argument("--input", type=str, help="Input Excel file to clean", required=True)
        args.add_argument("--output", type=str, help="Output Excel file", required=True)
        OPTS = args.parse_args()

    _ = clean_ottawa_file(OPTS.input, OPTS.output)

    print("Finished!")