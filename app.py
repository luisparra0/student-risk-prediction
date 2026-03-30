import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# ==============================
# LOAD MODEL
# ==============================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "modelo_risco.pkl"

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
features = artifact["features"]
threshold = artifact["threshold"]

# ==============================
# FUNÇÃO SAMPLE (ALINHADA)
# ==============================
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

# ==============================
# UI
# ==============================
st.title("Student Risk Prediction")
st.write("Predict risk of academic delay based on performance trend")

ian_2022 = st.slider("IAN 2022", 0.0, 10.0, 7.0)
ian_2023 = st.slider("IAN 2023", 0.0, 10.0, 7.0)

# ==============================
# PREDICT
# ==============================
sample = create_sample(ian_2022, ian_2023, features)

proba = model.predict_proba(sample)[0][1]
risk_score = int(proba * 100)

if proba < threshold:
    risk_level = "Low"
elif proba < 0.5:
    risk_level = "Medium"
else:
    risk_level = "High"

# ==============================
# OUTPUT
# ==============================
st.metric("Risk Score", f"{risk_score}/100")
st.write(f"Risk Level: **{risk_level}**")