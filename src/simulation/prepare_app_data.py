from pathlib import Path
import shutil


FILES_TO_COPY = {
    "data/fixtures/world_cup_2026_group_matches.csv": "data/app/world_cup_2026_group_matches.csv",
    "data/processed/world_cup_2026_group_predictions.csv": "data/app/world_cup_2026_group_predictions.csv",
    "data/processed/world_cup_2026_group_tables.csv": "data/app/world_cup_2026_group_tables.csv",
    "data/processed/world_cup_2026_knockout_teams.csv": "data/app/world_cup_2026_knockout_teams.csv",
    "data/processed/world_cup_2026_knockout_bracket.csv": "data/app/world_cup_2026_knockout_bracket.csv",
}


def prepare_app_data() -> None:
    Path("data/app").mkdir(parents=True, exist_ok=True)

    for source, destination in FILES_TO_COPY.items():
        source_path = Path(source)
        destination_path = Path(destination)

        if not source_path.exists():
            raise FileNotFoundError(f"Missing file: {source}")

        shutil.copy(source_path, destination_path)
        print(f"Copied {source} -> {destination}")


if __name__ == "__main__":
    prepare_app_data()
