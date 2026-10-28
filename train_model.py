"""Train and evaluate all ShopSense AI classification pipelines."""

import json
from datetime import datetime, timezone

import joblib
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from utils.data_loader import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    load_dataset,
    split_dataset,
)
from utils.predictions import DISPLAY_NAMES, MODEL_FILES, MODELS_DIR
from utils.preprocessing import build_model_pipeline


def evaluate_model(model, X_test, y_test) -> dict:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).astype(int).tolist(),
    }


def main() -> None:
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    configurations = {
        "logistic_regression": {
            "classifier": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            "scale_numeric": True,
            "parameters": {"max_iter": 1000, "class_weight": "balanced", "random_state": 42},
        },
        "knn": {
            "classifier": KNeighborsClassifier(n_neighbors=5),
            "scale_numeric": True,
            "parameters": {"n_neighbors": 5},
        },
        "decision_tree": {
            "classifier": DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=10, class_weight="balanced", random_state=42
            ),
            "scale_numeric": False,
            "parameters": {
                "max_depth": 8,
                "min_samples_leaf": 10,
                "class_weight": "balanced",
                "random_state": 42,
            },
        },
    }

    metrics = {}
    for key, configuration in configurations.items():
        print(f"Training {DISPLAY_NAMES[key]}...")
        pipeline = build_model_pipeline(
            configuration["classifier"], configuration["scale_numeric"]
        )
        pipeline.fit(X_train, y_train)
        metrics[key] = evaluate_model(pipeline, X_test, y_test)
        joblib.dump(pipeline, MODELS_DIR / MODEL_FILES[key])

    best_model = max(metrics, key=lambda key: metrics[key]["f1_score"])
    metadata = {
        "project": "ShopSense AI",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "Online Shoppers Purchasing Intention Dataset",
            "file": DATASET_PATH.name,
            "records": int(len(df)),
            "input_features": len(FEATURE_COLUMNS),
            "target": "Revenue",
            "missing_values": int(df.isna().sum().sum()),
            "duplicate_rows_retained": int(df.duplicated().sum()),
        },
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "class_distribution": {
            "full": {str(int(k)): int(v) for k, v in df["Revenue"].astype(int).value_counts().sort_index().items()},
            "training": {str(int(k)): int(v) for k, v in y_train.value_counts().sort_index().items()},
            "testing": {str(int(k)): int(v) for k, v in y_test.value_counts().sort_index().items()},
        },
        "feature_names": FEATURE_COLUMNS,
        "evaluation_metrics": metrics,
        "best_model": best_model,
        "best_model_display_name": DISPLAY_NAMES[best_model],
        "selection_metric": "f1_score",
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "model_configurations": {
            key: value["parameters"] for key, value in configurations.items()
        },
        "library_versions": {"scikit_learn": sklearn.__version__},
    }
    with (MODELS_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nDataset validation")
    print(f"Shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns)}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Duplicate rows retained: {df.duplicated().sum()}")
    print(f"Class distribution: {df['Revenue'].value_counts().to_dict()}")
    print("\nHeld-out test results")
    for key, result in metrics.items():
        print(
            f"{DISPLAY_NAMES[key]:<22} Accuracy={result['accuracy']:.4f} "
            f"Precision={result['precision']:.4f} Recall={result['recall']:.4f} "
            f"F1={result['f1_score']:.4f} ROC-AUC={result['roc_auc']:.4f}"
        )
    print(f"\nSelected model by positive-class F1-score: {DISPLAY_NAMES[best_model]}")
    print(f"Saved artifacts to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
