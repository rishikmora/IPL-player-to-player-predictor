# 🏏 IPL AI Predictor  
### ⚡ Context-Aware Player vs Player Performance Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b?logo=streamlit)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Live Idea

Predict how a batsman performs against a bowler **before the match even starts**.

This is an AI-powered cricket analytics system that simulates realistic match scenarios and predicts:

- 🔥 Runs in next 10 balls  
- ⚡ Runs in next 20 balls  
- 🚀 Runs in next 30 balls  
- 📊 Prediction range with uncertainty (± runs)  

---

## 🧠 Why This Project?

Traditional cricket stats show:

- Batting average ❌  
- Strike rate ❌  

But they don’t answer:

> What will happen *right now* in this matchup?

This project solves that using:

- Context-aware machine learning  
- Player vs player intelligence  
- Simulation-based prediction  
- Uncertainty-aware outputs  

---

## 🎯 Features

- ✨ Modern responsive UI (Streamlit dashboard)  
- 🎯 Player vs Player selection  
- 🌍 Smart city-based filtering  
- ⚡ Fast predictions (cached model)  
- 📊 Visual analytics (trend + uncertainty charts)  
- 🧠 Smart uncertainty engine  
- 📈 Player insights (form, impact, confidence)  

---

## 🧬 Tech Stack

| Layer | Technology |
|------|-----------|
| Frontend | Streamlit |
| Backend | Python |
| ML Model | XGBoost |
| Data | IPL Ball-by-Ball Dataset |
| Processing | Pandas, NumPy |

---

## 🧠 Model Intelligence

The model captures multiple real-world cricket factors:

### 🔹 Player Form
- Batting average  
- Rolling recent performance  

### 🔹 Match Context
- Pressure (target-based)  
- Match situation awareness  

### 🔹 Player Matchup
- Batsman vs Bowler interaction  
- Historical performance patterns  

### 🔹 Venue Effect
- Stadium scoring behavior  

### 🔹 Stability Layer
- Bayesian smoothing to reduce noise  
- Prediction clipping to avoid unrealistic outputs  

---

## 📊 Example Output

```
10 balls → 13 runs (±2)
20 balls → 26 runs (±4)
30 balls → 39 runs (±6)
```

---

## 📁 Project Structure

```
project/
│
├── app.py
├── new_ipl.csv
├── README.md
```

---

## ⚠️ Dataset Setup

Dataset is not included due to GitHub size limits.

Download from:  
https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025  

Place file in project folder:
- new_ipl.csv  

---

## ▶️ Run Locally

### Install dependencies
```
pip install streamlit pandas numpy xgboost scikit-learn
```

### Run app
```
streamlit run app.py
```

### Open browser
```
http://localhost:8501
```

---

## ⚠️ Current Limitations

- Moderate model accuracy (~0.1–0.2 R²)  
- No live match inputs (wickets, required rate)  
- Limited contextual depth  

---

## 🚀 Future Roadmap

- Live match simulation (overs, wickets, target)  
- Ball-by-ball prediction engine  
- Player role classification  
- Deep learning models (LSTM)  
- Improve accuracy  
- Deploy online  

---

## 🧠 Key Learnings

- Feature engineering matters more than model choice  
- Sports data is highly contextual  
- Stability improves predictions  
- UI is critical in ML systems  

---

## 🏆 Highlights

✔ Real IPL dataset  
✔ End-to-end ML pipeline  
✔ Smart uncertainty modeling  
✔ Interactive dashboard  
✔ Player vs player analytics  

---

## 👨‍💻 Author

**Rishik Mora**  
Machine Learning Enthusiast  

---

## 🔥 Final Note

This is not just a model — it is a cricket analytics decision system.

---

⭐ If you like this project, consider giving it a star!!
