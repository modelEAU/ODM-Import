import os
import pandas as pd
from wbe_odm.odm_mappers import base_mapper


class CsvMapper(base_mapper.BaseMapper):
    def is_valid_file_name(self, file_name):
        if "_" in file_name:
            file_name = file_name.partition("_")[-1]
        if ".csv" not in file_name:
            return False
        name = file_name.partition(".")[0]
        acceptable_names = base_mapper.get_odm_names()
        return name in acceptable_names

    def get_odm_name_from_file_name(self, file_name):
        if "_" in file_name:
            file_name = file_name.partition("_")[-1]
        return file_name.partition(".")[0]

    def read(
        self,
        directory,
        table_names=None,
            ) -> bool:
        """Reads an ODM-compatible directory of csv files
        Parameters
        ----------
        directory : str
            Path to directory containing data
        table_names : str, optional
            tables to read, by default None
        """
        if table_names is None:
            table_names = [x for x in self.conversion_dict.keys()]

        dir_files = os.listdir(directory)
        csv_files = [
            file for file in dir_files
            if self.is_valid_file_name(file)]

        for file in csv_files:
            df = pd.read_csv(os.path.join(directory, file))
            odm_name = self.get_odm_name_from_file_name(file)

            df = self.type_cast_table(odm_name, df)
            df = df.drop_duplicates()
            for attribute, dico in self.conversion_dict.items():
                if dico["odm_name"] == odm_name:
                    setattr(self, attribute, df)
                    break
        self.remove_duplicates()
        return

    def validates(self):
        return True
