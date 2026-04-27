import pandas as pd
import numpy as np

DTYPE_OPTIONS = ["int64", "float64", "str", "bool", "datetime64[ns]", "category"]

DTYPE_DISPLAY = {
    "int64": "Integer (int64)",
    "float64": "Float (float64)",
    "str": "Text (string)",
    "bool": "Boolean",
    "datetime64[ns]": "DateTime",
    "category": "Category",
}

def load_csv(file):
    return pd.read_csv(file)

def dtype_to_str(dtype):
    s = str(dtype)
    if "int" in s:
        return "int64"
    if "float" in s:
        return "float64"
    if "bool" in s:
        return "bool"
    if "datetime" in s:
        return "datetime64[ns]"
    if "category" in s:
        return "category"
    return "str"

def drop_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)

def cast_dtype(df, col, dtype_str):
    df = df.copy()
    try:
        if dtype_str == "datetime64[ns]":
            df[col] = pd.to_datetime(df[col])
        elif dtype_str == "str":
            df[col] = df[col].astype(str)
        elif dtype_str == "category":
            df[col] = df[col].astype("category")
        elif dtype_str == "bool":
            df[col] = df[col].astype(bool)
        else:
            df[col] = df[col].astype(dtype_str)
        return df, None
    except Exception as e:
        return df, str(e)
