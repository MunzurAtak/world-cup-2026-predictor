from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = "models/match_outcome_model.joblib"
GROUP_MATCHES_PATH = "data/fixtures/world_cup_2026_group_matches.csv"
OUTPUT_PATH = "data/processed/world_cup_2026_group_predictions.csv"


def load_model(model_path: str = MODEL_PATH):
    """
    Load the trained match outcome model.
    """

    return joblib.load(model_path)


def load_group_matches(filepath: str = GROUP_MATCHES_PATH) -> pd.DataFrame:
    """
    Load generated World Cup 2026 group-stage matches.
    """

    return pd.read_csv(filepath)


def predict_group_matches(matches_df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Predict all group-stage matches and add prediction probabilities.
    """

    feature_columns = [
        "year",
        "home_team",
        "away_team",
        "tournament",
        "neutral",
    ]

    X = matches_df[feature_columns]

    predicted_results = model.predict(X)
    predicted_probabilities = model.predict_proba(X)

    probability_df = pd.DataFrame(
        predicted_probabilities,
        columns=model.classes_,
    )

    predictions_df = matches_df.copy()

    predictions_df["predicted_result"] = predicted_results

    predictions_df["home_win_probability"] = probability_df.get("home_win", 0)
    predictions_df["draw_probability"] = probability_df.get("draw", 0)
    predictions_df["away_win_probability"] = probability_df.get("away_win", 0)

    return predictions_df


def save_predictions(
    predictions_df: pd.DataFrame,
    output_path: str = OUTPUT_PATH,
) -> None:
    """
    Save predicted group-stage matches.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    model = load_model()

    group_matches = load_group_matches()

    predictions = predict_group_matches(group_matches, model)

    save_predictions(predictions)

    print(predictions.head(12))
    print()
    print(f"Predicted matches: {len(predictions)}")
    print(f"Saved to: {OUTPUT_PATH}")
