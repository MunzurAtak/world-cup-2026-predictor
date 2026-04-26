import pandas as pd
import plotly.express as px
import streamlit as st


ALL_PREDICTIONS_PATH = "data/app/world_cup_2026_all_predictions.csv"
GROUP_PREDICTIONS_PATH = "data/app/world_cup_2026_group_predictions.csv"
GROUP_TABLES_PATH = "data/app/world_cup_2026_group_tables.csv"
KNOCKOUT_TEAMS_PATH = "data/app/world_cup_2026_knockout_teams.csv"
KNOCKOUT_BRACKET_PATH = "data/app/world_cup_2026_knockout_bracket.csv"


st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)


@st.cache_data
def load_data():
    all_predictions = pd.read_csv(ALL_PREDICTIONS_PATH)
    group_predictions = pd.read_csv(GROUP_PREDICTIONS_PATH)
    group_tables = pd.read_csv(GROUP_TABLES_PATH)
    knockout_teams = pd.read_csv(KNOCKOUT_TEAMS_PATH)
    knockout_bracket = pd.read_csv(KNOCKOUT_BRACKET_PATH)

    return (
        all_predictions,
        group_predictions,
        group_tables,
        knockout_teams,
        knockout_bracket,
    )


def format_probability(value):
    return f"{value:.1%}"


def format_probability_columns(df):
    df = df.copy()

    for column in [
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
    ]:
        if column in df.columns:
            df[column] = df[column].apply(format_probability)

    return df


def show_overview(all_predictions, group_predictions, group_tables, knockout_bracket):
    st.title("World Cup 2026 Match Outcome Predictor")

    predicted_winner = knockout_bracket.iloc[-1]["winner"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted matches", len(all_predictions))
    col2.metric("Teams", group_tables["team"].nunique())
    col3.metric("Knockout matches", len(knockout_bracket))
    col4.metric("Predicted winner", predicted_winner)

    st.subheader("Predicted outcome distribution")

    outcome_counts = all_predictions["predicted_result"].value_counts().reset_index()
    outcome_counts.columns = ["Predicted result", "Count"]

    fig = px.bar(
        outcome_counts,
        x="Predicted result",
        y="Count",
        text="Count",
        title="All predicted match outcomes",
    )

    st.plotly_chart(fig, use_container_width=True)


def show_all_matches(all_predictions):
    st.header("All Predicted Matches")

    stage_filter = st.multiselect(
        "Stage",
        sorted(all_predictions["stage"].unique()),
        default=sorted(all_predictions["stage"].unique()),
    )

    filtered = all_predictions[all_predictions["stage"].isin(stage_filter)].copy()

    display_df = filtered[
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

    display_df = format_probability_columns(display_df)

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def show_group_tables(group_tables):
    st.header("Predicted Group Tables")

    groups = sorted(group_tables["group"].unique())

    for i in range(0, len(groups), 3):
        cols = st.columns(3)

        for col, group in zip(cols, groups[i : i + 3]):
            with col:
                st.subheader(f"Group {group}")
                filtered = group_tables[group_tables["group"] == group]
                st.dataframe(filtered, use_container_width=True, hide_index=True)


def show_knockout_teams(knockout_teams):
    st.header("Qualified Knockout Teams")

    display_df = knockout_teams[
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

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def show_bracket_visual(knockout_bracket):
    st.header("Predicted Tournament Bracket")

    rounds = knockout_bracket["round"].drop_duplicates().tolist()
    columns = st.columns(len(rounds))

    for col, round_name in zip(columns, rounds):
        with col:
            st.subheader(round_name)

            round_matches = knockout_bracket[knockout_bracket["round"] == round_name]

            for _, match in round_matches.iterrows():
                st.markdown(
                    f"""
                    **Match {match['match_number']}**  
                    {match['home_team']} vs {match['away_team']}  
                    Winner: **{match['winner']}**
                    """
                )


def show_knockout_details(knockout_bracket):
    st.header("Knockout Match Details")

    selected_round = st.selectbox(
        "Select knockout round",
        knockout_bracket["round"].drop_duplicates().tolist(),
    )

    filtered = knockout_bracket[knockout_bracket["round"] == selected_round].copy()

    display_df = filtered[
        [
            "round",
            "match_number",
            "home_team",
            "away_team",
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
            "winner",
        ]
    ]

    display_df = format_probability_columns(display_df)

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def main():
    try:
        (
            all_predictions,
            group_predictions,
            group_tables,
            knockout_teams,
            knockout_bracket,
        ) = load_data()

    except FileNotFoundError:
        st.error("App data is missing. Run: python run_pipeline.py")
        return

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "All Matches",
            "Group Tables",
            "Knockout Teams",
            "Bracket Visual",
            "Knockout Details",
        ],
    )

    if page == "Overview":
        show_overview(
            all_predictions, group_predictions, group_tables, knockout_bracket
        )

    elif page == "All Matches":
        show_all_matches(all_predictions)

    elif page == "Group Tables":
        show_group_tables(group_tables)

    elif page == "Knockout Teams":
        show_knockout_teams(knockout_teams)

    elif page == "Bracket Visual":
        show_bracket_visual(knockout_bracket)

    elif page == "Knockout Details":
        show_knockout_details(knockout_bracket)


if __name__ == "__main__":
    main()
