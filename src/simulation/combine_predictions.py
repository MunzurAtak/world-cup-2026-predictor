from pathlib import Path

import pandas as pd


GROUP_PREDICTIONS_PATH = "data/processed/world_cup_2026_group_predictions.csv"
KNOCKOUT_BRACKET_PATH = "data/processed/world_cup_2026_knockout_bracket.csv"
OUTPUT_PATH = "data/processed/world_cup_2026_all_predictions.csv"


def load_group_predictions(filepath: str = GROUP_PREDICTIONS_PATH) -> pd.DataFrame:
    return pd.read_csv(filepath)


def load_knockout_bracket(filepath: str = KNOCKOUT_BRACKET_PATH) -> pd.DataFrame:
    return pd.read_csv(filepath)


def combine_predictions(
    group_predictions: pd.DataFrame, knockout_bracket: pd.DataFrame
) -> pd.DataFrame:
    group_matches = group_predictions.copy()

    group_matches["stage"] = "Group Stage"
    group_matches["round"] = group_matches["group"]
    group_matches["winner"] = group_matches.apply(
        lambda row: (
            row["home_team"]
            if row["predicted_result"] == "home_win"
            else row["away_team"] if row["predicted_result"] == "away_win" else "Draw"
        ),
        axis=1,
    )

    group_matches = group_matches[
        [
            "stage",
            "round",
            "home_team",
            "away_team",
            "predicted_result",
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
            "winner",
        ]
    ]

    knockout_matches = knockout_bracket.copy()
    knockout_matches["stage"] = "Knockout Stage"

    knockout_matches = knockout_matches[
        [
            "stage",
            "round",
            "home_team",
            "away_team",
            "predicted_result",
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
            "winner",
        ]
    ]

    return pd.concat([group_matches, knockout_matches], ignore_index=True)


def save_all_predictions(df: pd.DataFrame, output_path: str = OUTPUT_PATH) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    group_predictions = load_group_predictions()
    knockout_bracket = load_knockout_bracket()

    all_predictions = combine_predictions(group_predictions, knockout_bracket)

    save_all_predictions(all_predictions)

    print(all_predictions.head())
    print()
    print(f"Total predicted matches: {len(all_predictions)}")
    print(f"Saved to: {OUTPUT_PATH}")
