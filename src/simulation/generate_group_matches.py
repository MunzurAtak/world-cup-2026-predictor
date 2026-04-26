from itertools import combinations
from pathlib import Path

import pandas as pd


GROUPS_PATH = "data/fixtures/world_cup_2026_groups.csv"
OUTPUT_PATH = "data/fixtures/world_cup_2026_group_matches.csv"


def load_groups(filepath: str = GROUPS_PATH) -> pd.DataFrame:
    """
    Load the World Cup 2026 group file.
    """

    return pd.read_csv(filepath)


def generate_group_matches(groups_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all group-stage matches.

    Each team in a group plays every other team once.
    """

    matches = []

    for group_name, group_data in groups_df.groupby("group"):
        teams = group_data["team"].tolist()

        for home_team, away_team in combinations(teams, 2):
            matches.append(
                {
                    "group": group_name,
                    "home_team": home_team,
                    "away_team": away_team,
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                    "year": 2026,
                }
            )

    return pd.DataFrame(matches)


def save_group_matches(
    matches_df: pd.DataFrame,
    output_path: str = OUTPUT_PATH,
) -> None:
    """
    Save generated group-stage matches.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matches_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    groups = load_groups()

    group_matches = generate_group_matches(groups)

    save_group_matches(group_matches)

    print(group_matches.head(12))
    print()
    print(f"Generated matches: {len(group_matches)}")
    print(f"Saved to: {OUTPUT_PATH}")
