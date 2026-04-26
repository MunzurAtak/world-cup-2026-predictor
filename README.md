# World Cup 2026 Match Outcome Predictor

A data science project that predicts match outcomes for the FIFA World Cup 2026 using historical international football match data.

The project includes a full machine learning pipeline, group-stage simulation, knockout qualification logic, tournament bracket generation, and a Streamlit dashboard.

---

## Project Overview

This project predicts:

- All World Cup 2026 group-stage matches
- Group-stage rankings
- Knockout qualification
- Knockout bracket results
- Final predicted tournament winner

The system uses historical international football results to train a baseline machine learning model that predicts one of three possible outcomes:

```text
home_win / draw / away_win
```

The final results are displayed in a Streamlit web app.

---

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- Streamlit
- Plotly
- joblib
- GitHub

---

## Main Features

The Streamlit dashboard includes:

- Overview page with key tournament metrics
- All predicted matches
- Predicted group-stage results
- Group-stage tables
- Qualified knockout teams
- Knockout bracket visual
- Knockout match details
- Predicted tournament winner

---

## Project Structure

```text
world-cup-2026-predictor/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── app/
│   │   ├── world_cup_2026_all_predictions.csv
│   │   ├── world_cup_2026_group_predictions.csv
│   │   ├── world_cup_2026_group_tables.csv
│   │   ├── world_cup_2026_knockout_teams.csv
│   │   └── world_cup_2026_knockout_bracket.csv
│   │
│   ├── fixtures/
│   │   ├── world_cup_2026_groups.csv
│   │   └── world_cup_2026_group_matches.csv
│   │
│   ├── raw/
│   │   └── results.csv
│   │
│   └── processed/
│       ├── matches_model.csv
│       ├── world_cup_2026_group_predictions.csv
│       ├── world_cup_2026_group_tables.csv
│       ├── world_cup_2026_knockout_teams.csv
│       ├── world_cup_2026_knockout_bracket.csv
│       └── world_cup_2026_all_predictions.csv
│
├── models/
│   └── match_outcome_model.joblib
│
├── notebooks/
│
├── reports/
│   └── figures/
│
├── src/
│   ├── data/
│   │   └── load_data.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── train_model.py
│   │   └── predict_model.py
│   │
│   └── simulation/
│       ├── generate_group_matches.py
│       ├── predict_group_matches.py
│       ├── build_group_tables.py
│       ├── select_knockout_teams.py
│       ├── simulate_knockout.py
│       ├── combine_predictions.py
│       └── prepare_app_data.py
│
├── .gitignore
├── check_project.py
├── requirements.txt
├── run_pipeline.py
└── README.md
```

---

## Dataset

This project uses historical international football match results.

The main required file is:

```text
data/raw/results.csv
```

Expected columns:

```text
date
home_team
away_team
home_score
away_score
tournament
city
country
neutral
```

The raw data is not tracked in GitHub because it can be large. It should be downloaded manually and placed inside:

```text
data/raw/
```

---

## Model

The current version uses a baseline `RandomForestClassifier`.

### Input features

```text
year
home_team
away_team
tournament
neutral
```

### Target variable

```text
result
```

The target is created from the match score:

```text
home_score > away_score  → home_win
home_score = away_score  → draw
home_score < away_score  → away_win
```

---

## Current Model Performance

The first baseline model achieved approximately:

```text
Accuracy: 0.538
```

The model performs best on `home_win`, reasonably on `away_win`, and struggles with `draw`, which is common in football prediction.

This is a baseline model. The goal of the project is first to build the full end-to-end system, then improve the model with stronger features.

---

## Pipeline

The full pipeline does the following:

```text
1. Load historical match results
2. Clean the raw data
3. Build a modelling dataset
4. Train a baseline machine learning model
5. Generate all 2026 group-stage matches
6. Predict group-stage matches
7. Build predicted group tables
8. Select knockout teams
9. Simulate the knockout bracket
10. Combine all predictions
11. Prepare app-ready data
```

The full pipeline can be run with:

```bash
python run_pipeline.py
```

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/world-cup-2026-predictor.git
cd world-cup-2026-predictor
```

Replace `YOUR_USERNAME` with your own GitHub username.

---

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```bash
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```bash
.\.venv\Scripts\Activate.ps1
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Add the raw dataset

Place the historical match dataset here:

```text
data/raw/results.csv
```

---

### 5. Run the full pipeline

```bash
python run_pipeline.py
```

This creates all processed prediction files and prepares the app data.

---

### 6. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

The app will open in the browser.

---

## Individual Scripts

### Load data

```bash
python -m src.data.load_data
```

Loads and cleans the raw historical match data.

---

### Build features

```bash
python -m src.features.build_features
```

Creates a machine learning-ready dataset.

---

### Train model

```bash
python -m src.models.train_model
```

Trains and saves the baseline match outcome prediction model.

---

### Predict one match

```bash
python -m src.models.predict_model
```

Tests the trained model on one example match.

---

### Generate group matches

```bash
python -m src.simulation.generate_group_matches
```

Creates all 72 group-stage matches from the 2026 group file.

---

### Predict group matches

```bash
python -m src.simulation.predict_group_matches
```

Predicts all group-stage match outcomes.

---

### Build group tables

```bash
python -m src.simulation.build_group_tables
```

Creates predicted group rankings.

---

### Select knockout teams

```bash
python -m src.simulation.select_knockout_teams
```

Selects the top two teams from each group plus the best third-place teams.

---

### Simulate knockout bracket

```bash
python -m src.simulation.simulate_knockout
```

Creates a predicted knockout bracket and tournament winner.

---

### Combine predictions

```bash
python -m src.simulation.combine_predictions
```

Combines group-stage and knockout-stage predictions into one file.

---

### Prepare app data

```bash
python -m src.simulation.prepare_app_data
```

Copies generated prediction files into `data/app/` for the Streamlit app.

---

## Streamlit Dashboard Pages

### Overview

Shows:

- Total predicted matches
- Number of teams
- Number of knockout matches
- Predicted winner
- Outcome distribution chart

### All Matches

Shows all predicted matches with:

- Stage
- Round
- Home team
- Away team
- Predicted result
- Home win probability
- Draw probability
- Away win probability
- Winner

### Group Tables

Shows predicted standings for all groups.

### Knockout Teams

Shows teams that qualified for the knockout stage.

### Bracket Visual

Shows the predicted knockout path round by round.

### Knockout Details

Shows detailed knockout match probabilities.

---

## World Cup 2026 Format Used

This project uses the 48-team World Cup format:

```text
12 groups of 4 teams
Top 2 teams from each group qualify
8 best third-place teams qualify
32 teams enter the knockout stage
```

The current group-stage simulation generates:

```text
12 groups × 6 matches = 72 group-stage matches
```

The knockout simulation generates:

```text
Round of 32      16 matches
Round of 16       8 matches
Quarter-finals    4 matches
Semi-finals       2 matches
Final             1 match
```

Total predicted matches in this version:

```text
72 group matches + 31 knockout matches = 103 predicted matches
```

The real tournament includes 104 matches because it also includes a third-place match. This can be added in a future version.

---

## Current Limitations

This is a baseline version.

The model currently does not use:

- FIFA rankings
- Elo ratings
- recent team form
- player quality
- squad selection
- injuries
- manager strength
- betting odds
- expected goals
- exact score prediction
- official Round of 32 bracket slot rules

The knockout bracket is currently a simplified seeded bracket.

---

## Future Improvements

Planned improvements:

- Add Elo rating features
- Add recent form features
- Add rolling average goals scored
- Add rolling average goals conceded
- Add team win rate over last 5, 10, and 20 matches
- Add FIFA ranking data
- Add confederation strength
- Improve draw prediction
- Predict expected goals
- Predict exact scores
- Add third-place match
- Add official FIFA knockout bracket logic
- Add model comparison page
- Add confusion matrix and feature importance
- Deploy with Streamlit Cloud

---

## Example Output

Example prediction format:

```text
Brazil vs Morocco
Predicted result: home_win
Home win probability: 58%
Draw probability: 22%
Away win probability: 20%
```

Example group table format:

```text
Group C

Rank  Team      Played  Wins  Draws  Losses  GF  GA  GD  Points
1     Brazil    3       3     0      0       6   3   +3  9
2     Morocco   3       2     0      1       5   4   +1  6
3     Scotland  3       1     0      2       4   5   -1  3
4     Haiti     3       0     0      3       3   6   -3  0
```

---

## Git Ignore Policy

The repository tracks code and small app-ready files.

Ignored files/folders include:

```text
.venv/
data/raw/
data/processed/
models/
__pycache__/
*.joblib
*.pkl
```

This keeps the GitHub repo clean and avoids uploading large generated files.

---

## Project Status

Current status:

```text
Working baseline system completed
```

Completed:

- Project setup
- Data loading
- Feature engineering
- Model training
- Single-match prediction
- Group-stage match generation
- Group-stage prediction
- Group-table creation
- Knockout qualification
- Knockout bracket simulation
- Combined prediction file
- Streamlit dashboard

Next focus:

```text
Improve model quality with better football-specific features
```

---

## Author

Munzur Atak

AI student building a data science portfolio project focused on machine learning, football prediction, and interactive deployment.
