import pandas as pd


def load_results_data(filepath: str = "data/raw/results.csv") -> pd.DataFrame:
    """
    Load historical international football match results.

    Parameters
    ----------
    filepath : str
        Path to the raw results.csv file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with parsed dates and a match outcome column.
    """

    df = pd.read_csv(filepath)

    # Convert date column from text to real datetime format
    df["date"] = pd.to_datetime(df["date"])

    # Remove rows where scores are missing
    df = df.dropna(subset=["home_score", "away_score"])

    # Convert scores from float to integer
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Create target column for machine learning
    df["result"] = df.apply(get_match_result, axis=1)

    return df


def get_match_result(row: pd.Series) -> str:
    """
    Convert match scores into a result label.

    Returns:
    - home_win
    - draw
    - away_win
    """

    if row["home_score"] > row["away_score"]:
        return "home_win"
    elif row["home_score"] < row["away_score"]:
        return "away_win"
    else:
        return "draw"


if __name__ == "__main__":
    data = load_results_data()

    print(data.head())
    print()
    print(data.info())
    print()
    print(data["result"].value_counts())
