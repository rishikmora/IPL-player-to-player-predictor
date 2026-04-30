import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

DELIVERIES_PATH = "deliveries.csv"
MATCHES_PATH = "matches.csv"

st.set_page_config(page_title="IPL Predictor", layout="wide")

@st.cache_data
def load_data():
    deliveries = pd.read_csv(DELIVERIES_PATH)
    matches = pd.read_csv(MATCHES_PATH)

    deliveries.rename(columns={"batter": "batsman"}, inplace=True)

    df = deliveries.merge(matches, left_on="match_id", right_on="id")
    df = df.dropna(subset=["batsman", "bowler", "city"])

    return df


@st.cache_data
def create_dataset(df):

    # ==========================
    # AGGREGATE MATCHUP DATA
    # ==========================
    grouped = df.groupby(["batsman", "bowler", "city"]).agg({
        "batsman_runs": ["sum", "count"],
        "is_wicket": "sum",
        "total_runs": "mean"
    }).reset_index()

    grouped.columns = [
        "batsman", "bowler", "city",
        "total_runs", "balls",
        "wickets",
        "avg_total_runs"
    ]

    # target: runs per ball
    grouped["runs_per_ball"] = grouped["total_runs"] / grouped["balls"]

    # batsman overall avg
    bat_avg = df.groupby("batsman")["batsman_runs"].mean()
    grouped["bat_avg"] = grouped["batsman"].map(bat_avg)

    # bowler economy
    bowl_eco = df.groupby("bowler")["total_runs"].mean()
    grouped["bowl_eco"] = grouped["bowler"].map(bowl_eco)

    # venue scoring
    venue_avg = df.groupby("city")["total_runs"].mean()
    grouped["venue_avg"] = grouped["city"].map(venue_avg)

    # encoding
    grouped["bat_enc"] = grouped["batsman"].astype("category").cat.codes
    grouped["bowl_enc"] = grouped["bowler"].astype("category").cat.codes
    grouped["city_enc"] = grouped["city"].astype("category").cat.codes

    return grouped.fillna(0)


@st.cache_resource
def train_model(df):

    features = [
        "bat_enc", "bowl_enc", "city_enc",
        "bat_avg", "bowl_eco", "venue_avg",
        "wickets", "balls"
    ]

    target = "runs_per_ball"

    split = int(len(df) * 0.8)

    train = df.iloc[:split]
    test = df.iloc[split:]

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    score = r2_score(y_test, preds)

    return model, features, score


def predict_runs(df, model, features, batsman, bowler, city, balls):

    data = df[
        (df["batsman"] == batsman) &
        (df["bowler"] == bowler) &
        (df["city"] == city)
    ]

    if len(data) == 0:
        return None

    row = data.iloc[0]
    X = row[features].values.reshape(1, -1)

    runs_per_ball = model.predict(X)[0]

    return round(runs_per_ball * balls, 2)


# ==============================
# UI
# ==============================
st.title("🏏 IPL Predictor (Stable Model)")

df = load_data()
data = create_dataset(df)
model, features, score = train_model(data)

st.sidebar.write("Model R²:", round(score, 3))

mode = st.radio("Mode", ["Batsman First", "Bowler First"])

if mode == "Batsman First":
    batsman = st.selectbox("Batsman", sorted(data["batsman"].unique()))
    bowlers = data[data["batsman"] == batsman]["bowler"].unique()
    bowler = st.selectbox("Bowler", sorted(bowlers))
else:
    bowler = st.selectbox("Bowler", sorted(data["bowler"].unique()))
    batsmen = data[data["bowler"] == bowler]["batsman"].unique()
    batsman = st.selectbox("Batsman", sorted(batsmen))

cities = data[
    (data["batsman"] == batsman) &
    (data["bowler"] == bowler)
]["city"].unique()

if len(cities) == 0:
    st.error("No city data available")
    st.stop()

city = st.selectbox("City", sorted(cities))

if st.button("Predict"):
    st.subheader("📊 Predictions")

    for balls in [30, 40, 50]:
        result = predict_runs(data, model, features, batsman, bowler, city, balls)

        if result:
            st.success(f"{balls} balls → {result} runs")
        else:
            st.write(f"{balls} balls → No data")
