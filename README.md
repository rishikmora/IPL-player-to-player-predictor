# 🏏 IPL AI Predictor  
### ⚡ Player vs Player Performance Simulation Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b?logo=streamlit)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Live Idea

Predict how a batsman performs against a bowler **before the match even starts**.

This project is an AI-powered cricket analytics system that simulates real match scenarios and predicts:

- 🔥 Runs in next 30 balls  
- ⚡ Runs in next 40 balls  
- 🚀 Runs in next 50 balls  

---

## 🧠 Why This Project?

Traditional cricket stats show:
- Average ❌  
- Strike rate ❌  

But they don’t answer:

> What will happen *now* in this matchup under pressure?

This project solves that using:
- Context-aware machine learning  
- Simulation-based prediction  
- Player vs player intelligence  

---

## 🎯 Features

- ✨ Modern interactive UI (Streamlit)  
- 🎯 Player vs Player selection  
- 🌍 Smart city filtering  
- ⚡ Fast predictions (cached model)  
- 📊 Multi-scenario simulation  
- 🧠 Context-aware modeling  

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

The model captures multiple dimensions:

### 🔹 Player Form
- Recent runs  
- Strike rate trend  

### 🔹 Match Context
- Current run rate  
- Wickets fallen  
- Pressure index  

### 🔹 Game Phases
- Powerplay behavior  
- Death overs strategy  

### 🔹 Venue Effect
- Stadium scoring patterns  

---

## 📊 Example Output

30 balls → 38 runs  
40 balls → 52 runs  
50 balls → 65 runs  

---

## 📁 Project Structure

project/  
│  
├── app.py  
├── deliveries.csv (not included)  
├── matches.csv (not included)  
├── README.md  

---

## ⚠️ Dataset Setup

Dataset is not included due to GitHub size limits.

Download from:  
https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020  

Place in project folder:  
- deliveries.csv  
- matches.csv  

---

## ▶️ Run Locally

### Install dependencies
pip install streamlit pandas numpy xgboost scikit-learn  

### Run app
streamlit run app.py  

### Open browser
http://localhost:8501  

---

## ⚠️ Current Limitations

- Low model accuracy (~0.05 R²)  
- Uses latest matchup snapshot only  
- No confidence intervals  
- No explainability layer  

---

## 🚀 Future Roadmap

- Improve model accuracy  
- Add confidence intervals  
- Add “upper hand” prediction  
- Add visual charts/dashboard  
- Deploy online  
- Add feature importance  

---

## 🧠 Key Learnings

- Feature engineering matters more than model choice  
- Sports data is highly contextual  
- Simulation improves realism  
- UI is critical in ML systems  

---

## 🏆 Highlights

✔ Real IPL dataset  
✔ End-to-end ML pipeline  
✔ Interactive web app  
✔ Player vs player analytics  
✔ Scalable design  

---

## 👨‍💻 Author

Built as a portfolio-grade ML project for high-impact roles.

---

## 🔥 Final Note

This is not just a model — it is a cricket analytics decision system.

---

⭐ If you like this project, consider giving it a star!
