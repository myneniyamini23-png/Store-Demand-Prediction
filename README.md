# 📦 Store Item Demand Forecasting — Advanced ML Time-Series Project

This repository contains an end-to-end machine learning forecasting system built using **LightGBM**, **advanced time-series feature engineering**, **walk-forward validation**, and **SHAP explainability**.  
The project uses the **Kaggle Store Item Demand Forecasting Challenge** dataset and demonstrates how to build a real-world, production-style forecasting pipeline for retail, warehouse, and supply chain optimization.

---

## 📌 Project Overview

This project predicts **daily product demand** for 50 items sold across 10 stores over 5 years of historical data.  
The goal is to create a global, scalable forecasting model that captures:

- Weekly seasonality  
- Item-specific sales patterns  
- Rolling window trends  
- Long-term demand behavior  
- Calendar-based effects  

The final model delivers:

### ⭐ **56.78% reduction in RMSE vs. baseline model**

This level of improvement is significant for operational planning and warehouse automation workflows.

---

## 🧠 Problem Definition

Given:

- `date`
- `store`
- `item`
- `sales`

Forecast the **next-day sales** for each `(store, item)` combination.

This supports:

- Inventory restocking  
- Warehouse robotics planning  
- Supply chain decision-making  
- Labor allocation  
- Avoiding stockouts  

---

## 📂 Repository Structure

store-demand-forecasting/
│
├── data/
│ └── train.csv # Dataset (not included due to Kaggle license)
│
├── notebooks/
│ └── advanced_forecasting.ipynb # Full workflow in notebook format
│
├── src/
│ ├── feature_engineering.py # Lag, rolling, and calendar features
│ ├── model_training.py # LightGBM training + hyperparameter tuning
│ ├── walk_forward_validation.py # Walk-forward time-series evaluation
│ └── shap_analysis.py # SHAP explainability utilities
│
└── README.md

yaml
Copy code

---

## 🧩 Dataset Information

**Source:**  
Kaggle – *Store Item Demand Forecasting Challenge*

**Columns:**
- `date` — daily timestamp  
- `store` — store ID (1–10)  
- `item` — item ID (1–50)  
- `sales` — number of units sold  

**Dataset Size:**  
- ~913,000 rows  
- 10 stores × 50 items  
- 5 years of daily sales  

---

## 🔧 Technology Stack

| Component             | Technology |
|----------------------|------------|
| Programming Language  | Python |
| Data Processing       | Pandas, NumPy |
| Modeling              | LightGBM, Scikit-Learn |
| Visualization         | Matplotlib, Seaborn |
| Explainability        | SHAP |
| Evaluation Metrics    | RMSE, MAE |
| Validation Strategy   | Time-based train/test + Walk-forward |

---

## 🛠️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/store-demand-forecasting.git
cd store-demand-forecasting
2. Create and activate a virtual environment
bash
Copy code
python -m venv .venv
Activate (Windows):

bash
Copy code
.\.venv\Scripts\activate
3. Install dependencies
bash
Copy code
pip install -r requirements.txt
4. Add dataset
Place train.csv in:

bash
Copy code
data/train.csv
🚀 End-to-End Pipeline
1️⃣ Data Loading
Load and sort historical sales data

Convert date column into datetime format

Structure the dataset for time-series forecasting

2️⃣ Feature Engineering
Generated features:

Calendar Features
Year

Month

Day

Day of week

Week of year

Weekend indicator

Month-start / month-end flags

Lag Features
lag_1, lag_2, lag_3

lag_7 (weekly pattern)

lag_14, lag_30

Rolling Statistics
7-day rolling mean

30-day rolling mean

7-day rolling std

These features capture long-term and short-term temporal dependencies.

🎯 Baseline Model
A simple benchmark model predicting:

sql
Copy code
mean sales per item (computed from training set)
Used as a reference point to measure ML model improvements.

🤖 Machine Learning Model: LightGBM
LightGBM is used because:

Handles large tabular datasets efficiently

Works well with categorical variables

Captures nonlinear time-series interactions

Fast training and inference

Model training includes:

Categorical handling (store, item)

500–800 trees

Learning rate tuning

Tree depth and leaf optimization

🎛️ Hyperparameter Tuning
Performed Randomized Search over parameters:

learning_rate

n_estimators

max_depth

num_leaves

subsample

colsample_bytree

min_child_samples

Objective: minimize RMSE on validation split.

🧠 SHAP Explainability
SHAP is used to interpret the LightGBM model.

Top insights:

lag_7 is the strongest predictor (weekly shopping cycle)

item strongly influences variations in average demand

lag_14, lag_1, and lag_30 capture medium and short-term trends

dayofweek reflects weekday–weekend demand differences

📈 Results Summary
Metric	Value
MAE	5.947
RMSE	7.699
Baseline RMSE	~17.8
Improvement	56.78%

These results indicate the model is highly effective for retail and warehouse forecasting tasks.

💡 Key Insights Learned
Retail demand shows strong 7-day seasonality

Item-level patterns contribute significantly to model accuracy

Rolling averages stabilize short-term fluctuations

Calendar variables improve generalization

Walk-forward validation better simulates real production forecasting

🏁 Running the Project
Run the full notebook:
bash
Copy code
jupyter notebook notebooks/advanced_forecasting.ipynb
Or execute the training script:
bash
Copy code
python src/model_training.py
📚 Future Enhancements
Deploy model as REST API (FastAPI / Flask)

Integrate Streamlit dashboard for visualization

Add Prophet / TFT (Temporal Fusion Transformer) comparison

Enable multi-step forecasting (7-day, 30-day)
