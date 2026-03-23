# NBA Upset Prediction Model  

Compare whether NBA game upsets are better predicted using recent performance (last 10 games) or full historical data using gradient boosting models and engineered features.  

**Course:** DS 4002  
**Group Name:** DLC  
**Group Leader:** Caroline Lingle  
**Group Members:** Lauren Medica, Dev Patel, Caroline Lingle  

---

## Overview  

This project builds and compares two machine learning models to predict whether an NBA game will result in an upset (when the underdog wins):  

- Recent-10 games performance  
- Full historical performance  

Both models use engineered features based on team performance, Elo ratings, and game context.  

We evaluate performance primarily using **F1 score**, with threshold tuning to account for class imbalance.  

**Key Idea:**  
Rather than using a fixed threshold, we optimize the classification threshold using precision-recall curves to maximize F1 score.  

---

## Installation  

Clone the repository:  

```bash
git clone <your-repo-link>
cd <repo-name>

pip install -r requirements.txt

Software & Platform

Language: Python 3

Libraries Used:

pandas
numpy
scikit-learn
matplotlib
seaborn

PROJECT_ROOT/
│
├── README.md
├── requirements.txt
│
├── DATA/
│   └── nbaallelo.csv
│
├── SCRIPTS/
│   ├── 01_load_clean.py
│   ├── 02_feature_engineering.py
│   ├── 03_model_recent10.py
│   ├── 04_model_full_history.py
│   └── 05_comparison.py
│
└── P2Project.ipynb

Reproducing Results
Step 1 — Download Data

Download the dataset:
https://github.com/fivethirtyeight/data/blob/master/nba-elo/nbaallelo.csv

Place file in:

DATA/  

Required file:

nbaallelo.csv
Step 2 — Install Dependencies
pip install -r requirements.txt  
Step 3 — Run Notebook or Scripts

Run the notebook:

P2Project.ipynb  

OR execute scripts:

python SCRIPTS/01_load_clean.py  
python SCRIPTS/02_feature_engineering.py  
python SCRIPTS/03_model_recent10.py  
python SCRIPTS/04_model_full_history.py  
python SCRIPTS/05_comparison.py  
Modeling Approach
Data Preparation
Chronologically sorted game data
Removed unnecessary columns
Created binary target: upset
Feature Engineering

Key features include:

Elo rating (elo_i, opp_elo_i, elo_gap)
Game context (home, neutral, playoffs)
Season progression (seasongame)
Rolling performance metrics (win rates, stats)

Two feature sets:

Recent-10 model: rolling averages over last 10 games
Full-history model: cumulative/long-term performance
Train-Test Split
Chronological 80/20 split
Prevents data leakage from future games
Models Used
HistGradientBoostingClassifier (sklearn)
Evaluation Strategy
Predicted probabilities → threshold tuning
Selected threshold that maximizes F1 score

Metrics:

F1 Score (primary)
Precision
Recall
Results
Model	Description
Recent-10 Model	Short-term team performance
Full History Model	Long-term performance trends

Main Insight:
Recent performance captures short-term momentum, while full history provides more stable trends.

Notes
Chronological split is critical for realistic sports prediction
Upsets are relatively rare → class imbalance handled with threshold tuning
Rolling features help capture momentum and recent form
Future Work
Try Random Forest and Logistic Regression
Incorporate player-level data
Add betting odds
Use time-series models (LSTM)
Evaluate performance by season
Acknowledgements
Dataset: FiveThirtyEight NBA Elo dataset (GitHub)
Instructor: Karsten Siller
TA: Cole Whittington
