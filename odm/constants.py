from json import load


def get_column_data_types():
    with open('odm/types.json') as f:
        return load(f)


TYPES = get_column_data_types()


UNKNOWN_TOKENS = [
    "nan",
    "na",
    "nd"
    "n.d",
    "none",
    "-",
    "unknown",
    "n/a",
    "n/d"
]

EXCEL_TEST_FILE_PATH = "Data/Ville de Québec 202102.xlsx"
