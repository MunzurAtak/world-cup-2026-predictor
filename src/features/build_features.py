from pathlib import Path

import pandas as pd

from src.data.load_data import load_results_data


RAW_DATA_PATH = "data/raw/results.csv"
PROCESSED_DATA_PATH = "data/processed/matches_model.csv"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a machine learning-ready dataset from historical match results.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned match results dataframe.

    Returns
    -------
    pd.DataFrame
        Dataset with selected features and target column.
    """

    # Use only modern football matches
    df = df[df["date"].dt.year >= 1990].copy()

    # Create year feature
    df["year"] = df["date"].dt.year

    # Select features we want to use for the first model
    columns = [
        "date",
        "year",
        "home_team",
        "away_team",
        "tournament",
        "neutral",
        "result",
    ]

    model_df = df[columns].copy()

    # Remove rows with missing values in important columns
    model_df = model_df.dropna()

    return model_df


def save_processed_data(
    df: pd.DataFrame, output_path: str = PROCESSED_DATA_PATH
) -> None:
    """
    Save the processed dataset to the data/processed folder.
    """

    output_path = Path(output_path)

    # Make sure data/processed exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    results = load_results_data(RAW_DATA_PATH)

    model_data = build_features(results)

    save_processed_data(model_data)

    print(model_data.head())
    print()
    print(f"Processed dataset shape: {model_data.shape}")
    print()
    print(model_data["result"].value_counts())
    print()
    print(f"Saved processed data to: {PROCESSED_DATA_PATH}")
