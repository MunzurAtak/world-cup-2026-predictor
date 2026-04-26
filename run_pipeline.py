from src.features.build_features import build_features, save_processed_data
from src.data.load_data import load_results_data
from src.models.train_model import train_match_outcome_model, save_model
from src.simulation.generate_group_matches import (
    load_groups,
    generate_group_matches,
    save_group_matches,
)
from src.simulation.predict_group_matches import (
    load_model,
    load_group_matches,
    predict_group_matches,
    save_predictions,
)
from src.simulation.build_group_tables import (
    load_predictions,
    build_group_tables,
    save_group_tables,
)
from src.simulation.select_knockout_teams import (
    load_group_tables,
    select_knockout_teams,
    save_knockout_teams,
)
from src.simulation.simulate_knockout import (
    load_knockout_teams,
    simulate_knockout_bracket,
    save_bracket,
)
from src.simulation.prepare_app_data import prepare_app_data
from src.simulation.combine_predictions import (
    load_group_predictions,
    load_knockout_bracket,
    combine_predictions,
    save_all_predictions,
)


def main():
    print("1. Loading and processing historical results")
    results = load_results_data()
    model_data = build_features(results)
    save_processed_data(model_data)

    print("2. Training model")
    model = train_match_outcome_model(model_data)
    save_model(model)

    print("3. Generating group-stage matches")
    groups = load_groups()
    group_matches = generate_group_matches(groups)
    save_group_matches(group_matches)

    print("4. Predicting group-stage matches")
    model = load_model()
    group_matches = load_group_matches()
    predictions = predict_group_matches(group_matches, model)
    save_predictions(predictions)

    print("5. Building group tables")
    predictions = load_predictions()
    group_tables = build_group_tables(predictions)
    save_group_tables(group_tables)

    print("6. Selecting knockout teams")
    group_tables = load_group_tables()
    knockout_teams = select_knockout_teams(group_tables)
    save_knockout_teams(knockout_teams)

    print("7. Simulating knockout bracket")
    knockout_teams = load_knockout_teams()
    bracket = simulate_knockout_bracket(knockout_teams, model)
    save_bracket(bracket)

    print("8. Combining all match predictions")
    group_predictions = load_group_predictions()
    knockout_bracket = load_knockout_bracket()
    all_predictions = combine_predictions(group_predictions, knockout_bracket)
    save_all_predictions(all_predictions)

    print("9. Preparing app data")
    prepare_app_data()

    print("Pipeline finished.")


if __name__ == "__main__":
    main()
