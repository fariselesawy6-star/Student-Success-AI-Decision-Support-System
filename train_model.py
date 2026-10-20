
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

DATA_PATH = Path("data/student_data.csv")
MODEL_PATH = Path("student_model.joblib")
RESULTS_PATH = Path("data/model_results.txt")


def train_model():
    print("Loading dataset...")

    data = pd.read_csv(DATA_PATH)
    X = data.drop(columns=["Target"])
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    categorical_cols = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical_cols = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numerical_pipeline, numerical_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_,
    )

    joblib.dump(model, MODEL_PATH)

    RESULTS_PATH.write_text(
        f"Test accuracy: {accuracy:.4f}\n\n"
        f"Classification report:\n{report}\n"
        f"Confusion matrix (labels={list(model.classes_)}):\n"
        f"{matrix}\n",
        encoding="utf-8",
    )

    print("\nTraining completed!")
    print(f"Test accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(report)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    train_model()
