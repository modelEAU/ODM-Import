import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import dialects
import glob
import random
import odm


def create_dummy_polygons():
    polygons = {}
    default_polygon = {"polygonID": None,
     "name": None,
     "pop": None,
     "type": None,
     "wkt": None,
     "file": None,
     "link": None
     }
    polygon_files = glob.glob("Data/polygons/*.wkt")
    fields = []
    for i, file in enumerate(polygon_files):
        polygon = default_polygon.copy()
        with open(file, "r") as f:
            polygon["wkt"] = f.read()
        polygon["name"] = file.split("/")[-1].split(".")[-2]
        polygon["pop"] = random.randint(250000, 350000)
        polygon["type"] = "swrCat"
        polygons[i] = polygon
    return pd.DataFrame.from_dict(polygons, orient="index")

def read_excel_template(filename):
    xls = pd.read_excel(filename, engine="xlrd", sheet_name=None)
    model = odm.Odm(xls)
    return model.data

def df_to_db_table(df, engine, table_name, id_col):
    pass



if __name__ == "__main__":
    df = create_dummy_polygons()
    excel_filename = "Data/Ville de Québec 202102.xlsx"
    excel_data = read_excel_template(excel_filename)
    path = "Data/WBE.db"
    engine = create_engine(f"sqlite:///{path}")
    table_name = "Polygon"
    engine.execute(f"delete from {table_name}")
    id_col = "polygonID"
    df_to_db_table(df, engine, table_name, id_col)

    print(df)
