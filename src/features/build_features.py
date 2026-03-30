import numpy as np
import pandas as pd

def create_pivot(df):
    df_pivot = df.pivot_table(
        index="ra",
        columns="ano_referencia",
        values=["ian", "ida", "ieg", "ipv"],
        aggfunc="mean"
    )

    df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    return df_pivot


def create_target(df):
    if "ian_2024" in df.columns and "ian_2022" in df.columns:
        df["risco_defasagem"] = np.where(
            df["ian_2024"] - df["ian_2022"] < 0,
            1,
            0
        )
    return df

def create_features(df):
    df["delta_ian"] = df["ian_2023"] - df["ian_2022"]
    df["delta_ida"] = df["ida_2023"] - df["ida_2022"]
    df["delta_ieg"] = df["ieg_2023"] - df["ieg_2022"]

    return df


def select_model_data(df):
    return df.dropna(subset=[
        "ian_2022", "ian_2023", "ian_2024",
        "ida_2022", "ida_2023",
        "ieg_2022", "ieg_2023",
        "ipv_2022", "ipv_2023"
    ]).copy()