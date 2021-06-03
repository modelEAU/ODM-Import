import os
import re
import pandas as pd
import datetime as dt
from wbe_odm.odm_mappers.csv_mapper import CsvMapper


directory = os.path.dirname(__file__)
MODELEAU_MAP_NAME = directory + "/" + "modeleau_map.csv"


class MapperFuncs:
    @classmethod
    def clean_up(cls, df):
        for col in df.columns.to_list():
            df[col] = pd.to_datetime(cls.replace_excel_dates(df[col])) \
                if "date" in col.lower() else df[col]
            df[col] = df[col].apply(lambda x: None if x in ["None", ""] else x)
            if col == "Measurement":
                df[col] = df[col].apply(lambda x: x.replace("*", "").strip())
            if "Unnamed" in col:
                del df[col]
        df.drop_duplicates(keep="first", inplace=True)
        return df

    @classmethod
    def replace_excel_dates(cls, series):
        return series.apply(
            lambda x: pd.to_timedelta(x, unit='d') +
            dt.datetime(1899, 12, 30) if isinstance(x, float) else x
        )

    @classmethod
    def edit_index_in_id(cls, row):
        current_index = row["wwMeasureID"]
        row["wwMeasureID"] = current_index[:-1] + str(row["index"])
        return row

    @classmethod
    def build_missing_indices(cls, df):
        uniques = df["wwMeasureID"].drop_duplicates()
        for _, unique in enumerate(uniques):
            replicates = df.loc[df["wwMeasureID"] == unique]
            indices = [x+1 for x in range(len(replicates))]
            df.loc[df["wwMeasureID"] == unique, ["index"]] = indices
        df = df.apply(lambda x: cls.edit_index_in_id(x), axis=1)
        return df

    @classmethod
    def break_down_labels(cls, label, get):
        city, site_no, collection, type_ = label.split("_")
        if get == "siteID":
            return ("_").join([city, site_no])
        elif get == "collection":
            return collection
        elif get == "type":
            return type_
        raise ValueError(f"Item to get does not exist:{get}")

    @classmethod
    def get_site_id(cls, labels):
        return labels.apply(lambda x: cls.break_down_labels(x, "type"))

    @classmethod
    def is_grab(cls, labels):
        return labels.str.contains("grb")

    @classmethod
    def get_grab_date(cls, labels, raw_dates):
        df = pd.concat([labels, raw_dates], axis=1)
        df.columns = ["labels", "raw_dates"]
        df["grb_dates"] = pd.NaT
        filt = cls.is_grab(df["labels"])
        df.loc[filt, "grb_dates"] = df.loc[filt, "raw_dates"]
        return pd.to_datetime(df["grb_dates"])

    @classmethod
    def get_end_date(cls, labels, raw_dates):
        df = pd.concat([labels, raw_dates], axis=1)
        df.columns = ["labels", "raw_dates"]
        df["end_dates"] = pd.NaT
        filt = cls.is_grab(df["labels"])
        df.loc[~filt, "end_dates"] = df.loc[~filt, "raw_dates"]
        return pd.to_datetime(df["end_dates"])

    @classmethod
    def get_start_date(cls, start_col, end_col, labels):
        def calc_start_date(end_date, type_):
            if pd.isna(end_date) or pd.isna(type_):
                return pd.NaT
            x = type_
            hours = None
            if re.match(r"cp[tf]p[0-9]+h", x, re.IGNORECASE):
                hours = int(x[4:-1])
            elif re.match(r"ps[0-9]+h", x, re.IGNORECASE):
                hours = int(x[2:-1])
            if hours is not None:
                interval = pd.to_timedelta(f"{hours}h")
                return end_date - interval
            return pd.NaT

        df = pd.concat([start_col, end_col, labels], axis=1)
        df.columns = ["start", "end", "labels"]
        df["collection"] = df.apply(
            lambda row: cls.break_down_labels(row["labels"], "collection"), axis=1)
        df["s"] = df.apply(
            lambda row: calc_start_date(row["end"], row["collection"]), axis=1)
        return df["s"]

    @classmethod
    def get_sample_type(cls, labels):
        return labels.apply(lambda x: cls.break_down_labels(x, "type"))

    @classmethod
    def has_quality_flag(cls, flag):
        return flag != ""

    @classmethod
    def get_collection_method(cls, labels):
        return labels.apply(lambda x: cls.break_down_labels(x, "collection"))

    @classmethod
    def get_measure_type(cls, measures):
        measure_dico = {
            "Conductivity": "wqCond",
            "Turbidity": "wqTurb",
            "NH4": "wqNH4N",
            "TS": "wqTS",
            "TSS": "wqTSS",
            "pH": "wqPh"
        }
        return measures.map(measure_dico)

    @classmethod
    def get_wwmeasure_id(
            cls, 
            label,
            end_date,
            sample_index,
            lab_id,
            type_,
            analysis_date,
            meas_index):
        sample_id = cls.get_sample_id(
            label, end_date, sample_index
        )
        ana_date = CsvMapper.str_date_from_timestamp(analysis_date)
        df = pd.concat([sample_id, ana_date], axis=1)
        df["meas_type"] = cls.get_measure_type(type_)
        df["lab_id"] = lab_id
        df["index_no"] = str(meas_index) if not isinstance(meas_index, pd.Series) \
            else meas_index.astype(str)
        return df.agg("_".join, axis=1)

    @classmethod
    def get_sample_id(cls, label, end_date, sample_index):
        # TODO: Deal with index once it's been implemented in McGill sheet
        clean_date = CsvMapper.str_date_from_timestamp(end_date)
        clean_label = label.apply(lambda x: CsvMapper.clean_labels(x))

        df = pd.concat([clean_label, clean_date], axis=1)
        df["index_no"] = str(sample_index) \
            if not isinstance(sample_index, pd.Series) \
            else sample_index.astype(str)
        df.columns = [
            "clean_label", "clean_date", "index_no"
        ]
        df["sample_ids"] = ""

        df["sample_ids"] = df[[
            "clean_label", "clean_date", "index_no"]].agg("_".join, axis=1)

        return df["sample_ids"]

    @classmethod
    def validate_fraction_analyzed(cls, fracs):
        fracs = fracs.str.lower()
        fracs.loc[~fracs.isin(["mixed", "liquid", "solids"])] = ""
        return fracs

    @classmethod
    def validate_value(cls, raw_values):
        return pd.to_numeric(raw_values, errors="coerce")


class ModelEauMapper(CsvMapper):
    def __init__(self, processing_functions=MapperFuncs):
        super().__init__(processing_functions=processing_functions)
    def read(self, filepath, sheet_name,
             modeleau_map=MODELEAU_MAP_NAME, lab_id="modeleau_lab"):
        lab = pd.read_excel(filepath, sheet_name=sheet_name)
        lab = self.processing_functions.clean_up(lab)
        lab.columns = [
            self.excel_style(i+1)
            for i, _ in enumerate(lab.columns.to_list())
        ]
        mapping = pd.read_csv(modeleau_map)
        mapping.fillna("", inplace=True)
        mapping = mapping.astype(str)
        static_data = self.read_static_data(None)
        dynamic_tables = self.parse_sheet(
            mapping,
            static_data,
            lab,
            self.processing_functions,
            lab_id
        )
        for table_name, table in dynamic_tables.items():
            if table_name.lower() == 'wwmeasure':
                table = self.processing_functions.build_missing_indices(table)
            table = table.drop_duplicates(keep="first")
            table = self.type_cast_table(table_name, table)
            attr = self.get_attr_from_table_name(table_name)
            setattr(self, attr, table)
        return


if __name__ == "__main__":
    path = "/Users/jeandavidt/OneDrive - Université Laval/COVID/Latest Data/COVIDProject_Lab Measurements.xlsx"  # noqa
    sheet_name = "Lab analyses"
    mapper = ModelEauMapper()
    mapper.read(path, sheet_name)
    print(mapper.ww_measure)
