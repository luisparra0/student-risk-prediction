import pandas as pd
import joblib


def load_model(path):
    artifact = joblib.load(path)
    return artifact["model"], artifact["features"], artifact["threshold"]


def create_sample(ian_2022, ian_2023, features):
    delta = ian_2023 - ian_2022
    ratio = ian_2023 / (ian_2022 + 1e-6)

    data = {
        "ian_2022": ian_2022,
        "ian_2023": ian_2023,
        "delta_ian": delta,
        "ratio_ian": ratio
    }

    df = pd.DataFrame([data])
    df = df.reindex(columns=features, fill_value=0)

    return df


def predict_risk(model, sample, threshold):
    proba = model.predict_proba(sample)[0][1]

    if proba < threshold:
        level = "Low"
    elif proba < 0.5:
        level = "Medium"
    else:
        level = "High"

    return proba, level