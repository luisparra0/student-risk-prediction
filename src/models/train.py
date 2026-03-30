from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

def train_model(df):
    X = df.drop(columns=["risco_defasagem", "ra"])
    y = df["risco_defasagem"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    return model, X.columns


def save_model(model, features):
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    MODEL_PATH = BASE_DIR / "models" / "modelo_risco.pkl"

    threshold = 0.2

    joblib.dump({
        "model": model,
        "threshold": threshold,
        "features": list(features)
    }, MODEL_PATH)