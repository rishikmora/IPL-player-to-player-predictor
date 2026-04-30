import pandas as pd
import numpy as np
import requests
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import difflib

# ==============================
# CONFIG
# ==============================
DELIVERIES_PATH = "deliveries.csv"
MATCHES_PATH = "matches.csv"
API_KEY = "YOUR_OPENWEATHER_API_KEY"
FORM_WINDOW = 30


# ==============================
# LOAD DATA
# ==============================
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

    df = df.fillna(0)

    return df


# ==============================
# TRAIN MODEL
# ==============================
def train_model(df):
    df = df.sort_values("match_order")

    features = [
        "batsman_enc",
        "bowler_enc",
        "recent_sr",
        "venue_avg_runs",
        "batting_team_enc",
        "bowling_team_enc",
        "is_powerplay",
        "is_death",
        "current_run_rate",
        "pressure_index",
        "temp",
        "humidity"
    ]

    target = "batsman_runs"

    split = int(len(df) * 0.8)

    train = df.iloc[:split]
    test = df.iloc[split:]

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("XGBoost R2 Score:", round(r2_score(y_test, preds), 3))

    return model, features


# ==============================
# WEATHER
# ==============================
def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        data = requests.get(url).json()
        return data["main"]["temp"], data["main"]["humidity"]
    except:
        return 30, 60


# ==============================
# SELECT FUNCTIONS
# ==============================
def select_item(items, label):
    items = sorted([str(x) for x in items])

    while True:
        search = input(f"\nSearch {label}: ").strip().lower()
        filtered = [x for x in items if search in x.lower()]

        if not filtered:
            print("❌ No results, try again")
            continue

        if len(filtered) == 1:
            print(f"✅ Auto-selected: {filtered[0]}")
            return filtered[0]

        print(f"\n--- {label} Options ---")
        for i, item in enumerate(filtered):
            print(f"{i+1}. {item}")

        choice = input("Select number: ").strip()

        if not choice.isdigit():
            print("❌ Enter valid number")
            continue

        choice = int(choice)

        if 1 <= choice <= len(filtered):
            return filtered[choice - 1]


def select_city(cities):
    cities = sorted([str(x) for x in cities])

    if len(cities) == 1:
        print(f"🏙️ Auto-selected city: {cities[0]}")
        return cities[0]

    while True:
        search = input("\nSearch City: ").lower()
        filtered = [c for c in cities if search in c.lower()]

        if len(filtered) == 1:
            print(f"🏙️ Auto-selected: {filtered[0]}")
            return filtered[0]

        if not filtered:
            match = difflib.get_close_matches(search, cities, n=1, cutoff=0.6)
            if match:
                print(f"🤖 Closest match: {match[0]}")
                return match[0]
            print("❌ No results")
            continue

        for i, c in enumerate(filtered):
            print(f"{i+1}. {c}")

        choice = input("Select: ")
        if choice.isdigit():
            return filtered[int(choice)-1]


# ==============================
# PREDICT
# ==============================
def predict_runs(df, model, features, batsman, bowler, city, balls):
    data = df[
        (df["batsman"] == batsman) &
        (df["bowler"] == bowler) &
        (df["city"] == city)
    ]

    if len(data) == 0:
        return "❌ No data"

    latest = data.iloc[-1]

    temp, humidity = get_weather(city)

    X = latest[features].copy()
    X["temp"] = temp
    X["humidity"] = humidity

    X = X.values.reshape(1, -1)

    total = 0
    for _ in range(balls):
        total += model.predict(X)[0]

    return round(total, 2)


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    df = load_data()
    df = create_features(df)
    model, features = train_model(df)

    print("\n===== IPL PREDICTOR =====")

    mode = input("1. Batsman First\n2. Bowler First\nChoose: ")

    if mode == "1":
        batsman = select_item(df["batsman"].unique(), "Batsman")
        bowler = select_item(df[df["batsman"] == batsman]["bowler"].unique(), "Bowler")
    else:
        bowler = select_item(df["bowler"].unique(), "Bowler")
        batsman = select_item(df[df["bowler"] == bowler]["batsman"].unique(), "Batsman")

    matchup = df[(df["batsman"] == batsman) & (df["bowler"] == bowler)]
    cities = matchup["city"].dropna().unique()

    city = select_city(cities)

    print("\n===== RESULT =====")
    print("Batsman:", batsman)
    print("Bowler:", bowler)
    print("City:", city)

    # 🔥 AUTO PREDICTION FOR MULTIPLE BALLS
    for balls in [30, 40, 50]:
        result = predict_runs(df, model, features, batsman, bowler, city, balls)
        print(f"Predicted Runs ({balls} balls): {result}")