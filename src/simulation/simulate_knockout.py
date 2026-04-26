from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = "models/match_outcome_model.joblib"
KNOCKOUT_TEAMS_PATH = "data/processed/world_cup_2026_knockout_teams.csv"
OUTPUT_PATH = "data/processed/world_cup_2026_knockout_bracket.csv"


def load_model(model_path: str = MODEL_PATH):
    return joblib.load(model_path)


def load_knockout_teams(filepath: str = KNOCKOUT_TEAMS_PATH) -> pd.DataFrame:
    return pd.read_csv(filepath)


def predict_knockout_match(model, home_team: str, away_team: str) -> dict:
    match = pd.DataFrame(
        [
            {
                "year": 2026,
                "home_team": home_team,
                "away_team": away_team,
                "tournament": "FIFA World Cup",
                "neutral": True,
            }
        ]
    )

    probabilities = model.predict_proba(match)[0]
    classes = model.classes_

    probability_map = dict(zip(classes, probabilities))

    home_win_probability = probability_map.get("home_win", 0)
    away_win_probability = probability_map.get("away_win", 0)

    # Knockout matches cannot end in draw, so ignore draw probability.
    if home_win_probability >= away_win_probability:
        winner = home_team
        predicted_result = "home_win"
    else:
        winner = away_team
        predicted_result = "away_win"

    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_result": predicted_result,
        "home_win_probability": home_win_probability,
        "draw_probability": probability_map.get("draw", 0),
        "away_win_probability": away_win_probability,
        "winner": winner,
    }


def create_seeded_matchups(teams: list[str]) -> list[tuple[str, str]]:
    matchups = []

    for i in range(len(teams) // 2):
        home_team = teams[i]
        away_team = teams[-(i + 1)]
        matchups.append((home_team, away_team))

    return matchups


def simulate_round(
    model, teams: list[str], round_name: str
) -> tuple[list[dict], list[str]]:
    matchups = create_seeded_matchups(teams)

    round_results = []
    winners = []

    for match_number, (home_team, away_team) in enumerate(matchups, start=1):
        prediction = predict_knockout_match(model, home_team, away_team)

        prediction["round"] = round_name
        prediction["match_number"] = match_number

        round_results.append(prediction)
        winners.append(prediction["winner"])

    return round_results, winners


def simulate_knockout_bracket(knockout_teams: pd.DataFrame, model) -> pd.DataFrame:
    teams = knockout_teams.sort_values(
        by=["points", "goal_difference", "goals_for"],
        ascending=[False, False, False],
    )["team"].tolist()

    all_results = []

    rounds = [
        ("Round of 32", 32),
        ("Round of 16", 16),
        ("Quarter-finals", 8),
        ("Semi-finals", 4),
        ("Final", 2),
    ]

    current_teams = teams

    for round_name, expected_teams in rounds:
        if len(current_teams) != expected_teams:
            raise ValueError(
                f"{round_name} expected {expected_teams} teams, got {len(current_teams)}"
            )

        round_results, winners = simulate_round(model, current_teams, round_name)

        all_results.extend(round_results)
        current_teams = winners

    bracket_df = pd.DataFrame(all_results)

    columns = [
        "round",
        "match_number",
        "home_team",
        "away_team",
        "predicted_result",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
        "winner",
    ]

    return bracket_df[columns]


def save_bracket(df: pd.DataFrame, output_path: str = OUTPUT_PATH) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    model = load_model()
    knockout_teams = load_knockout_teams()

    bracket = simulate_knockout_bracket(knockout_teams, model)

    save_bracket(bracket)

    print(bracket)
    print()
    print(f"Predicted winner: {bracket.iloc[-1]['winner']}")
    print(f"Saved to: {OUTPUT_PATH}")
