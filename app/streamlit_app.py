import pandas as pd
import plotly.express as px
import streamlit as st


GROUP_MATCHES_PATH = "data/app/world_cup_2026_group_matches.csv"
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
    group_matches = pd.read_csv(GROUP_MATCHES_PATH)
    group_predictions = pd.read_csv(GROUP_PREDICTIONS_PATH)
    group_tables = pd.read_csv(GROUP_TABLES_PATH)
    knockout_teams = pd.read_csv(KNOCKOUT_TEAMS_PATH)
    knockout_bracket = pd.read_csv(KNOCKOUT_BRACKET_PATH)

    return (
        group_matches,
        group_predictions,
        group_tables,
        knockout_teams,
        knockout_bracket,
    )


def format_probability(value):
    return f"{value:.1%}"


def show_overview(group_predictions, group_tables, knockout_bracket):
    st.title("World Cup 2026 Match Outcome Predictor")

    predicted_winner = knockout_bracket.iloc[-1]["winner"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Group-stage matches", len(group_predictions))
    col2.metric("Teams", group_tables["team"].nunique())
    col3.metric("Knockout matches", len(knockout_bracket))
    col4.metric("Predicted winner", predicted_winner)

    st.subheader("Predicted outcome distribution")

    outcome_counts = group_predictions["predicted_result"].value_counts().reset_index()
    outcome_counts.columns = ["Predicted result", "Count"]

    fig = px.bar(
        outcome_counts,
        x="Predicted result",
        y="Count",
        text="Count",
        title="Predicted group-stage outcomes",
    )

    st.plotly_chart(fig, use_container_width=True)


def show_group_matches(group_predictions):
    st.header("Predicted Group-Stage Matches")

    selected_group = st.selectbox(
        "Select group",
        sorted(group_predictions["group"].unique()),
    )

    filtered = group_predictions[group_predictions["group"] == selected_group].copy()

    display_df = filtered[
        [
            "group",
            "home_team",
            "away_team",
            "predicted_result",
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
        ]
    ].copy()

    for column in [
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
    ]:
        display_df[column] = display_df[column].apply(format_probability)

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def show_group_tables(group_tables):
    st.header("Predicted Group Tables")

    selected_group = st.selectbox(
        "Select group table",
        sorted(group_tables["group"].unique()),
    )

    filtered = group_tables[group_tables["group"] == selected_group]

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


def show_knockout_bracket(knockout_bracket):
    st.header("Predicted Knockout Bracket")

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
    ].copy()

    for column in [
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
    ]:
        display_df[column] = display_df[column].apply(format_probability)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("Full knockout path")
    st.dataframe(knockout_bracket, use_container_width=True, hide_index=True)


def main():
    try:
        (
            group_matches,
            group_predictions,
            group_tables,
            knockout_teams,
            knockout_bracket,
        ) = load_data()

    except FileNotFoundError:
        st.error("App data is missing. Run: python -m src.simulation.prepare_app_data")
        return

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Group Matches",
            "Group Tables",
            "Knockout Teams",
            "Knockout Bracket",
        ],
    )

    if page == "Overview":
        show_overview(group_predictions, group_tables, knockout_bracket)

    elif page == "Group Matches":
        show_group_matches(group_predictions)

    elif page == "Group Tables":
        show_group_tables(group_tables)

    elif page == "Knockout Teams":
        show_knockout_teams(knockout_teams)

    elif page == "Knockout Bracket":
        show_knockout_bracket(knockout_bracket)


if __name__ == "__main__":
    main()
