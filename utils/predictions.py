"""Model loading and prediction helpers used by the UI and tests."""

import json
from pathlib import Path

import joblib
import pandas as pd

from utils.data_loader import BASE_DIR, FEATURE_COLUMNS

MODELS_DIR = BASE_DIR / "models"
MODEL_FILES = {
    "logistic_regression": "logistic_regression.pkl",
    "knn": "knn.pkl",
    "decision_tree": "decision_tree.pkl",
}
DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "knn": "K-Nearest Neighbors",
    "decision_tree": "Decision Tree",
}


def load_metadata(path: Path | None = None) -> dict:
    metadata_path = path or MODELS_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Trained models not found. Run python train_model.py first.")
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_model(model_key: str, models_dir: Path = MODELS_DIR):
    if model_key not in MODEL_FILES:
        raise ValueError(f"Unknown model: {model_key}")
    path = models_dir / MODEL_FILES[model_key]
    if not path.exists():
        raise FileNotFoundError("Trained models not found. Run python train_model.py first.")
    return joblib.load(path)


def validate_prediction_frame(frame: pd.DataFrame) -> None:
    if len(frame) != 1:
        raise ValueError("Prediction input must contain exactly one session.")
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction input is missing: {', '.join(missing)}")
    if frame[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("All prediction fields must have valid values.")


def predict_session(model, frame: pd.DataFrame) -> tuple[int, float]:
    validate_prediction_frame(frame)
    ordered = frame[FEATURE_COLUMNS].copy()
    prediction = int(model.predict(ordered)[0])
    probabilities = model.predict_proba(ordered)[0]
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Loaded model does not expose the Purchase class.")
    probability = float(probabilities[classes.index(1)])
    return prediction, probability

