import joblib
import pandas as pd


MODEL_PATH = "models/match_outcome_model.joblib"


def load_trained_model(model_path: str = MODEL_PATH):
    """
    Load the trained match outcome model from disk.
    """

    return joblib.load(model_path)


def predict_match(
    home_team: str,
    away_team: str,
    tournament: str = "FIFA World Cup",
    neutral: bool = True,
    year: int = 2026,
) -> pd.DataFrame:
    """
    Predict the outcome of one football match.

    Parameters
    ----------
    home_team : str
        Team listed as the home team.
    away_team : str
        Team listed as the away team.
    tournament : str
        Tournament name.
    neutral : bool
        Whether the match is played on neutral ground.
    year : int
        Year of the match.

    Returns
    -------
    pd.DataFrame
        Prediction probabilities for home win, draw, and away win.
    """

    model = load_trained_model()

    match = pd.DataFrame(
        [
            {
                "year": year,
                "home_team": home_team,
                "away_team": away_team,
                "tournament": tournament,
                "neutral": neutral,
            }
        ]
    )

    prediction = model.predict(match)[0]
    probabilities = model.predict_proba(match)[0]
    classes = model.classes_

    result = pd.DataFrame(
        {
            "outcome": classes,
            "probability": probabilities,
        }
    ).sort_values(by="probability", ascending=False)

    result["predicted_result"] = prediction

    return result


if __name__ == "__main__":
    prediction = predict_match(
        home_team="Brazil",
        away_team="Serbia",
    )

    print(prediction)
