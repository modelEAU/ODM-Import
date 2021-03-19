import json
import pandas as pd
from wbe_odm.odm_mappers import base_mapper


class SerializedMapper(base_mapper.BaseMapper):
    def read(self, json_str):
        """Reads a JSON-style string and extracts the attributes
         that belong in an Odm object.

        The following format is accepted:
        '{
            '__Odm__': {
                'ww_measure': {
                    '__DataFrame__': ...
                },
                'sample': {
                    '__DataFrame__': ...
                },...
        }'
        ----------
        json_str :
            Serialized Odm object

        Returns
        -------
        None
        """
        json.loads(
            json_str, object_hook=self.decode_object)
        self_attrs = self.__dict__
        for key, df in self_attrs.items():
            odm_table_name = self.conversion_dict[key]['odm_name']
            df = self.type_cast_table(odm_table_name, df)
            setattr(self, key, df)
        return

    def decode_object(self, o):
        if '__Odm__' in o:
            for key, value in o["__Odm__"].items():
                setattr(self, key, value)
            return None

        elif '__DataFrame__' in o:
            a = pd.read_json(o['__DataFrame__'], orient='split')
            return(a)
        elif '__Timestamp__' in o:
            return pd.to_datetime(o['__Timestamp__'], utc=None)
        else:
            return o

    def validates(self):
        return True
