import pandas as pd
from wbe_odm.odm_mappers import base_mapper


class JsonMapper(base_mapper.BaseMapper):
    def read(json):
        keys = json.keys()
        return keys

    def validates():
        return True
