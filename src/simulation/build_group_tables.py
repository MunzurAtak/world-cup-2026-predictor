from pathlib import Path

import pandas as pd


PREDICTIONS_PATH = "data/processed/world_cup_2026_group_predictions.csv"
OUTPUT_PATH = "data/processed/world_cup_2026_group_tables.csv"


def load_predictions(filepath: str = PREDICTIONS_PATH) -> pd.DataFrame:
    """
    Load predicted World Cup 2026 group-stage matches.
    """

    return pd.read_csv(filepath)


def assign_predicted_scores(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple predicted scores based on predicted match result.

    This is a temporary baseline:
    - home_win = 2-1
    - draw = 1-1
    - away_win = 1-2
    """

    predictions_df = predictions_df.copy()

    score_map = {
        "home_win": (2, 1),
        "draw": (1, 1),
        "away_win": (1, 2),
    }

    predictions_df["predicted_home_score"] = predictions_df["predicted_result"].map(
        lambda result: score_map[result][0]
    )

    predictions_df["predicted_away_score"] = predictions_df["predicted_result"].map(
        lambda result: score_map[result][1]
    )

    return predictions_df


def initialize_table_row(group: str, team: str) -> dict:
    """
    Create an empty group-table row for one team.
    """

    return {
        "group": group,
        "team": team,
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_difference": 0,
        "points": 0,
    }


def build_group_tables(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build group-stage ranking tables from predicted match results.
    """

    predictions_df = assign_predicted_scores(predictions_df)

    table = {}

    for _, match in predictions_df.iterrows():
        group = match["group"]
        home_team = match["home_team"]
        away_team = match["away_team"]

        home_score = match["predicted_home_score"]
        away_score = match["predicted_away_score"]

        for team in [home_team, away_team]:
            key = (group, team)

            if key not in table:
                table[key] = initialize_table_row(group, team)

        home_key = (group, home_team)
        away_key = (group, away_team)

        table[home_key]["played"] += 1
        table[away_key]["played"] += 1

        table[home_key]["goals_for"] += home_score
        table[home_key]["goals_against"] += away_score

        table[away_key]["goals_for"] += away_score
        table[away_key]["goals_against"] += home_score

        if home_score > away_score:
            table[home_key]["wins"] += 1
            table[away_key]["losses"] += 1
            table[home_key]["points"] += 3

        elif home_score < away_score:
            table[away_key]["wins"] += 1
            table[home_key]["losses"] += 1
            table[away_key]["points"] += 3

        else:
            table[home_key]["draws"] += 1
            table[away_key]["draws"] += 1
            table[home_key]["points"] += 1
            table[away_key]["points"] += 1

    table_df = pd.DataFrame(table.values())

    table_df["goal_difference"] = table_df["goals_for"] - table_df["goals_against"]

    table_df = table_df.sort_values(
        by=[
            "group",
            "points",
            "goal_difference",
            "goals_for",
        ],
        ascending=[
            True,
            False,
            False,
            False,
        ],
    )

    table_df["rank"] = table_df.groupby("group").cumcount() + 1

    columns = [
        "group",
        "rank",
        "team",
        "played",
        "wins",
        "draws",
        "losses",
        "goals_for",
        "goals_against",
        "goal_difference",
        "points",
    ]

    return table_df[columns]


def save_group_tables(
    group_tables_df: pd.DataFrame,
    output_path: str = OUTPUT_PATH,
) -> None:
    """
    Save predicted group-stage tables.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    group_tables_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    predictions = load_predictions()

    group_tables = build_group_tables(predictions)

    save_group_tables(group_tables)

    print(group_tables)
    print()
    print(f"Saved group tables to: {OUTPUT_PATH}")
