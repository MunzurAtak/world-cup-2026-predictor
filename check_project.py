from pathlib import Path


REQUIRED_PATHS = [
    "README.md",
    "requirements.txt",
    "run_pipeline.py",
    "app/streamlit_app.py",
    "src/data/load_data.py",
    "src/features/build_features.py",
    "src/models/train_model.py",
    "src/models/predict_model.py",
    "src/simulation/generate_group_matches.py",
    "src/simulation/predict_group_matches.py",
    "src/simulation/build_group_tables.py",
    "src/simulation/select_knockout_teams.py",
    "src/simulation/simulate_knockout.py",
    "src/simulation/combine_predictions.py",
    "src/simulation/prepare_app_data.py",
    "data/fixtures/world_cup_2026_groups.csv",
    "data/fixtures/world_cup_2026_group_matches.csv",
    "data/app/world_cup_2026_all_predictions.csv",
    "data/app/world_cup_2026_group_predictions.csv",
    "data/app/world_cup_2026_group_tables.csv",
    "data/app/world_cup_2026_knockout_teams.csv",
    "data/app/world_cup_2026_knockout_bracket.csv",
]


def main():
    missing_paths = []

    for path in REQUIRED_PATHS:
        if not Path(path).exists():
            missing_paths.append(path)

    if missing_paths:
        print("Missing files:")
        for path in missing_paths:
            print(f"- {path}")
    else:
        print("Project check passed. All required files exist.")


if __name__ == "__main__":
    main()
