from pathlib import Path

import pandas as pd


GROUP_TABLES_PATH = "data/processed/world_cup_2026_group_tables.csv"
OUTPUT_PATH = "data/processed/world_cup_2026_knockout_teams.csv"


def load_group_tables(filepath: str = GROUP_TABLES_PATH) -> pd.DataFrame:
    return pd.read_csv(filepath)


def select_knockout_teams(group_tables: pd.DataFrame) -> pd.DataFrame:
    top_two = group_tables[group_tables["rank"] <= 2].copy()
    top_two["qualification_type"] = "top_two"

    third_place = group_tables[group_tables["rank"] == 3].copy()

    best_thirds = third_place.sort_values(
        by=["points", "goal_difference", "goals_for"],
        ascending=[False, False, False],
    ).head(8)

    best_thirds["qualification_type"] = "best_third"

    qualified = pd.concat([top_two, best_thirds], ignore_index=True)

    qualified = qualified.sort_values(
        by=["group", "rank", "points", "goal_difference"],
        ascending=[True, True, False, False],
    )

    qualified["seed"] = range(1, len(qualified) + 1)

    return qualified


def save_knockout_teams(df: pd.DataFrame, output_path: str = OUTPUT_PATH) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    group_tables = load_group_tables()
    knockout_teams = select_knockout_teams(group_tables)

    save_knockout_teams(knockout_teams)

    print(
        knockout_teams[
            [
                "seed",
                "group",
                "rank",
                "team",
                "points",
                "goal_difference",
                "qualification_type",
            ]
        ]
    )
    print()
    print(f"Qualified teams: {len(knockout_teams)}")
    print(f"Saved to: {OUTPUT_PATH}")
