import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# ==============================
# CONFIG
# ==============================
DELIVERIES_PATH = "deliveries.csv"
MATCHES_PATH = "matches.csv"
FORM_WINDOW = 30

st.set_page_config(page_title="IPL Predictor", layout="wide")

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    deliveries = pd.read_csv(DELIVERIES_PATH)
    matches = pd.read_csv(MATCHES_PATH)

    deliveries.rename(columns={"batter": "batsman"}, inplace=True)

    df = deliveries.merge(matches, left_on="match_id", right_on="id")
    df = df.dropna(subset=["batsman", "bowler"])

    df["match_order"] = df["match_id"].rank(method="dense").astype(int)
    return df


# ==============================
# FEATURE ENGINEERING
# ==============================
@st.cache_data
def create_features(df):
    df = df.sort_values(["batsman", "match_order", "over", "ball"])

    df["recent_runs"] = (
        df.groupby("batsman")["batsman_runs"]
        .rolling(FORM_WINDOW, min_periods=1)
        .sum()
        .reset_index(0, drop=True)
    )

    df["recent_balls"] = (
        df.groupby("batsman")["batsman_runs"]
        .rolling(FORM_WINDOW, min_periods=1)
        .count()
        .reset_index(0, drop=True)
    )

    df["recent_sr"] = (df["recent_runs"] / df["recent_balls"]) * 100

    df["cumulative_runs"] = df.groupby("match_id")["total_runs"].cumsum()
    df["wickets_fallen"] = df.groupby("match_id")["is_wicket"].cumsum()

    df["current_run_rate"] = df["cumulative_runs"] / (df["over"] + 1)
    df["pressure_index"] = df["current_run_rate"] * (1 + df["wickets_fallen"] * 0.1)

    df["is_powerplay"] = (df["over"] <= 6).astype(int)
    df["is_death"] = (df["over"] >= 16).astype(int)

    venue_stats = df.groupby("venue")["total_runs"].mean().to_dict()
    df["venue_avg_runs"] = df["venue"].map(venue_stats)
    df["venue_avg_runs"] = df["venue_avg_runs"] / df["venue_avg_runs"].max()

    df["batsman_enc"] = df["batsman"].astype("category").cat.codes
    df["bowler_enc"] = df["bowler"].astype("category").cat.codes
    df["batting_team_enc"] = df["batting_team"].astype("category").cat.codes
    df["bowling_team_enc"] = df["bowling_team"].astype("category").cat.codes

    df["temp"] = 30
    df["humidity"] = 60

    return df.fillna(0)


# ==============================
# TRAIN MODEL
# ==============================
@st.cache_resource
def train_model(df):
    features = [
        "batsman_enc","bowler_enc","recent_sr",
        "venue_avg_runs","batting_team_enc","bowling_team_enc",
        "is_powerplay","is_death",
        "current_run_rate","pressure_index",
        "temp","humidity"
    ]

    target = "batsman_runs"

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(df[features], df[target])

    return model, features


# ==============================
# PREDICTION
# ==============================
def predict_runs(df, model, features, batsman, bowler, city, balls):
    data = df[
        (df["batsman"] == batsman) &
        (df["bowler"] == bowler) &
        (df["city"].astype(str) == city)
    ]

    if len(data) == 0:
        return None

    latest = data.iloc[-1]
    X = latest[features].values.reshape(1, -1)

    total = 0
    for _ in range(balls):
        total += model.predict(X)[0]

    return round(total, 2)


# ==============================
# UI
# ==============================
st.title("🏏 IPL Player vs Player Predictor")

df = load_data()
df = create_features(df)
model, features = train_model(df)

# MODE
mode = st.radio("Select Mode", ["Batsman First", "Bowler First"])

if mode == "Batsman First":
    batsman = st.selectbox("Select Batsman", sorted(df["batsman"].astype(str).unique()))
    bowler_list = df[df["batsman"] == batsman]["bowler"].astype(str).unique()
    bowler = st.selectbox("Select Bowler", sorted(bowler_list))
else:
    bowler = st.selectbox("Select Bowler", sorted(df["bowler"].astype(str).unique()))
    batsman_list = df[df["bowler"] == bowler]["batsman"].astype(str).unique()
    batsman = st.selectbox("Select Batsman", sorted(batsman_list))

# CITY FIX (IMPORTANT)
matchup = df[(df["batsman"] == batsman) & (df["bowler"] == bowler)]

cities = matchup["city"].dropna().astype(str).unique()

if len(cities) == 0:
    st.error("No city data available")
    st.stop()

city = st.selectbox("Select City", sorted(cities))

# PREDICT BUTTON
if st.button("Predict"):
    st.subheader("📊 Predictions")

    for balls in [30, 40, 50]:
        result = predict_runs(df, model, features, batsman, bowler, city, balls)

        if result is None:
            st.write(f"{balls} balls → No data")
        else:
            st.success(f"{balls} balls → {result} runs")