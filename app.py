import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = "new_ipl.csv"

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="IPL Predictor",
    layout="wide",
    page_icon="🏏"
)

# ==============================
# MODERN CSS 🔥
# ==============================
st.markdown("""
<style>

body {
    background-color: #0e1117;
}

.main {
    background: linear-gradient(135deg, #0e1117, #1c1f2b);
}

/* Cards */
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 30px rgba(0,0,0,0.3);
    transition: 0.3s;
}
.card:hover {
    transform: scale(1.02);
}

/* Title */
.title {
    font-size: 40px;
    font-weight: bold;
    color: white;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #ff4b2b, #ff416c);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
.stButton>button:hover {
    transform: scale(1.05);
}

/* Sidebar */
.css-1d391kg {
    background-color: #111827;
}

/* Metrics */
.metric-card {
    text-align: center;
    padding: 15px;
    border-radius: 12px;
    background: rgba(255,255,255,0.05);
}

</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    df.rename(columns={
        "batter": "batsman",
        "runs_batter": "batsman_runs"
    }, inplace=True)

    df["over"] = pd.to_numeric(df["over"], errors="coerce")
    df = df.dropna(subset=["batsman", "bowler", "city", "over"])

    return df

# ==============================
# CREATE DATASET
# ==============================
@st.cache_data
def create_dataset(df):

    phase = pd.cut(df["over"], bins=[0,6,15,20], labels=[0,1,2])
    df["phase"] = phase.cat.codes.replace(-1,1)

    df["recent_form"] = df.groupby("batsman")["batsman_runs"].transform(
        lambda x: x.rolling(30, min_periods=10).mean()
    )

    if "runs_target" in df.columns:
        df["cum_runs"] = df.groupby("match_id")["batsman_runs"].cumsum()
        df["pressure"] = df["runs_target"] - df["cum_runs"]
    else:
        df["pressure"] = 0

    global_avg = df["batsman_runs"].mean()

    grouped = df.groupby(["batsman","bowler","city"]).agg({
        "batsman_runs": ["sum","count"],
        "phase": "mean",
        "bat_pos": "mean",
        "pressure": "mean",
        "innings": "mean"
    }).reset_index()

    grouped.columns = [
        "batsman","bowler","city",
        "runs_sum","balls",
        "avg_phase","avg_position",
        "avg_pressure","avg_innings"
    ]

    k = np.maximum(10, 50 - grouped["balls"])
    grouped["runs_per_ball"] = (
        grouped["runs_sum"] + global_avg * k
    ) / (grouped["balls"] + k)

    grouped["bat_avg"] = grouped["batsman"].map(
        df.groupby("batsman")["batsman_runs"].mean()
    )

    grouped["bowl_impact"] = grouped["bowler"].map(
        df.groupby("bowler")["batsman_runs"].mean()
    )

    grouped["venue_avg"] = grouped["city"].map(
        df.groupby("city")["batsman_runs"].mean()
    )

    grouped["recent_form"] = grouped["batsman"].map(
        df.groupby("batsman")["recent_form"].last()
    )

    grouped["confidence"] = np.log1p(grouped["balls"])
    grouped["bat_vs_bowl"] = grouped["bat_avg"] - grouped["bowl_impact"]

    grouped["bat_enc"] = grouped["batsman"].astype("category").cat.codes
    grouped["bowl_enc"] = grouped["bowler"].astype("category").cat.codes
    grouped["city_enc"] = grouped["city"].astype("category").cat.codes

    grouped = grouped.replace([np.inf, -np.inf], 0).fillna(0)

    return grouped

# ==============================
# TRAIN
# ==============================
@st.cache_resource
def train_model(df):

    features = [
        "bat_enc","bowl_enc","city_enc",
        "bat_avg","recent_form",
        "bowl_impact","venue_avg",
        "avg_phase","avg_position",
        "avg_pressure","avg_innings",
        "balls","confidence",
        "bat_vs_bowl"
    ]

    X = df[features]
    y = df["runs_per_ball"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    score = r2_score(y_test, preds)

    return model, features, score

# ==============================
# PREDICT
# ==============================
def predict_runs(df, model, features, batsman, bowler, city, balls):

    data = df[
        (df["batsman"]==batsman)&
        (df["bowler"]==bowler)&
        (df["city"]==city)
    ]

    if len(data)==0:
        return None

    row = data.iloc[0]
    X = row[features].values.reshape(1,-1)

    rpb = model.predict(X)[0]
    rpb = np.clip(rpb, 0.2, 3.0)
    rpb = 0.7 * rpb + 0.3 * row["bat_avg"]

    runs = rpb * balls

    base = 1.5
    reliability = 1 / (1 + np.log1p(row["balls"]))
    horizon = np.sqrt(balls)

    unc = base * (1 + 2 * reliability) * horizon
    unc = np.clip(unc, 2, 8)

    return int(round(runs)), int(round(unc))

# ==============================
# UI
# ==============================

st.markdown('<p class="title">🏏 IPL AI Predictor</p>', unsafe_allow_html=True)

df = load_data()
data = create_dataset(df)
model,features,score = train_model(data)

st.sidebar.metric("Model Accuracy (R²)", round(score,3))

# INPUTS
st.markdown("### 🎯 Match Context")

c1, c2, c3 = st.columns(3)

with c1:
    batsman = st.selectbox("Batsman", sorted(data["batsman"].unique()))

with c2:
    bowler = st.selectbox(
        "Bowler",
        sorted(data[data["batsman"] == batsman]["bowler"].unique())
    )

with c3:
    cities = data[
        (data["batsman"]==batsman)&
        (data["bowler"]==bowler)
    ]["city"].unique()

    if len(cities)==0:
        st.error("No city data available")
        st.stop()

    city = st.selectbox("City", sorted(cities))

# PLAYER INSIGHTS
row = data[
    (data["batsman"]==batsman)&
    (data["bowler"]==bowler)&
    (data["city"]==city)
]

if len(row) > 0:
    row = row.iloc[0]

    st.markdown("### 📊 Player Insights")

    i1, i2, i3, i4 = st.columns(4)

    i1.markdown(f'<div class="metric-card">🔥 Bat Avg<br><h2>{round(row["bat_avg"],2)}</h2></div>', unsafe_allow_html=True)
    i2.markdown(f'<div class="metric-card">📈 Form<br><h2>{round(row["recent_form"],2)}</h2></div>', unsafe_allow_html=True)
    i3.markdown(f'<div class="metric-card">🎯 Bowler Impact<br><h2>{round(row["bowl_impact"],2)}</h2></div>', unsafe_allow_html=True)
    i4.markdown(f'<div class="metric-card">⚡ Confidence<br><h2>{round(row["confidence"],2)}</h2></div>', unsafe_allow_html=True)

# PREDICTION
st.markdown("### 🚀 Run Prediction")

if st.button("Predict Now"):

    balls_list = [10,20,30]
    runs_list, lower, upper = [], [], []

    for balls in balls_list:
        result = predict_runs(data,model,features,batsman,bowler,city,balls)

        if result:
            r,u = result
            runs_list.append(r)
            lower.append(r-u)
            upper.append(r+u)

            st.success(f"{balls} balls → {r} runs (±{u})")

    st.markdown("### 📈 Performance Trends")

    colA, colB = st.columns(2)

    with colA:
        trend_df = pd.DataFrame({
            "Balls": balls_list,
            "Runs": runs_list
        })
        st.line_chart(trend_df.set_index("Balls"))

    with colB:
        band_df = pd.DataFrame({
            "Balls": balls_list,
            "Lower": lower,
            "Upper": upper
        })
        st.area_chart(band_df.set_index("Balls"))
