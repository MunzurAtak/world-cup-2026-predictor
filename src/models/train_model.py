from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = "data/processed/matches_model.csv"
MODEL_PATH = "models/match_outcome_model.joblib"


def load_model_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the processed dataset created by build_features.py.
    """

    return pd.read_csv(filepath)


def train_match_outcome_model(df: pd.DataFrame) -> Pipeline:
    """
    Train a machine learning model to predict match outcomes.

    The model predicts one of:
    - home_win
    - draw
    - away_win
    """

    features = [
        "year",
        "home_team",
        "away_team",
        "tournament",
        "neutral",
    ]

    target = "result"

    X = df[features]
    y = df[target]

    categorical_features = [
        "home_team",
        "away_team",
        "tournament",
        "neutral",
    ]

    numeric_features = [
        "year",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.3f}")
    print()
    print("Classification report:")
    print(classification_report(y_test, predictions))

    return pipeline


def save_model(model: Pipeline, output_path: str = MODEL_PATH) -> None:
    """
    Save the trained model to disk.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)


if __name__ == "__main__":
    data = load_model_data()

    trained_model = train_match_outcome_model(data)

    save_model(trained_model)

    print()
    print(f"Saved model to: {MODEL_PATH}")
