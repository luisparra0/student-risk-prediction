def select_features(df):
    df = df.copy()

    required_cols = ["ian_2023", "ian_2022", "risco_defasagem"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas faltando: {missing}")

    if "delta_ian" not in df.columns:
        df["delta_ian"] = df["ian_2023"] - df["ian_2022"]

    df["ratio_ian"] = df["ian_2023"] / (df["ian_2022"] + 1e-6)

    FEATURES = [
        "ian_2023",
        "ian_2022",
        "delta_ian",
        "ratio_ian"
    ]

    TARGET = "risco_defasagem"

    X = df[FEATURES]
    y = df[TARGET]

    return X, y