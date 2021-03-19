import pandas as pd
from sqlalchemy import create_engine
from wbe_odm.odm_mappers import base_mapper


class SQLite3Mapper(base_mapper.BaseMapper):
    def read(
        self,
        cnxn_str: str,
        table_names: list[str] = None
            ) -> None:
        """Loads data from a Ottawa Data Model compatible database into an ODM object

        Parameters
        ----------
        cnxn_str : str
            connextion string to the db
        table_names : list[str], optional
            Names of the tables you want to read in.
            By default None, in which case the function
            collects data from every table.
        """
        if table_names is None:
            table_names = [
                self.conversion_dict[attribute]["odm_name"]
                for attribute in self.conversion_dict.keys()]

        attributes = []
        for table_name in table_names:
            for k, v in self.conversion_dict.items():
                if v["odm_name"] == table_name:
                    attributes.append(k)

        engine = create_engine(cnxn_str)
        for attribute, table_name in zip(
            attributes,
            table_names,
        ):
            df = pd.read_sql(f"select * from {table_name}", engine)
            df = self.type_cast_table(table_name, df)
            setattr(self, attribute, df)
        return

    def validates(table):
        return True
