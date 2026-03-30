import pandas as pd
from pathlib import Path

def load_raw_data(base_dir):
    path = Path(base_dir) / "data" / "raw" / "base_de_dados_completo_eda.csv"
    return pd.read_csv(path)


def clean_data(df):
    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df