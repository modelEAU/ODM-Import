import pandas as pd

def find_time_columns_to_merge(df):
    categories = {}
    for col in df.columns.to_list():
        col = col.lower()
        if "date" not in col: continue
        info = col.split(".")
        for bit in info:
            if "date" in bit:
                if bit in categories:
                    categories[bit].append(col)
                else:
                    categories[bit] = [col]
    return categories

def add_missing_columns(df, needed_names):
    existing_names = df.columns.to_list()
    for name in needed_names:
        if name not in existing_names:
            df[name] = None
    return df

def recombine_times(df):
    categories = find_time_columns_to_merge(df)
    df = add_missing_columns(df)
