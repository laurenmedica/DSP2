# NBA Upset Prediction Model

Compare whether NBA game upsets are better predicted using recent performance (last 10 games) or full historical data using gradient boosting models and engineered features.

**Course:** DS 4002  
**Group Name:** DLC  
**Group Leader:** Caroline Lingle  
**Group Members:** Lauren Medica, Dev Patel, Caroline Lingle  

---

## Overview

This project compares two machine learning approaches:

- Recent-10 games performance  
- Full historical performance  

The goal is to predict **NBA game upsets**, defined as games where a team with a win probability ≤ 0.40 wins.

We evaluate performance using both **F1 score** and **ROC-AUC**.

Because upsets are relatively rare events, F1 score alone does not fully capture model performance. We initially used F1 score, but incorporated **ROC-AUC as a primary metric** since it better measures how well the model distinguishes between upset and non-upset games across all thresholds. We also evaluate **average precision (AP)** and use a **precision-recall curve to select an optimal classification threshold**, rather than relying on a fixed cutoff like 0.5.

**Key Result:**  
Both models perform similarly, but the full-history model slightly outperforms in ROC-AUC, suggesting stronger overall ranking ability, while the recent-performance model better captures short-term trends.

---

## Software & Platform

**Language:** Python 3  

**Libraries Used:**

- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  

---

## Repository Structure


PROJECT_ROOT/
│
├── README.md
├── requirements.txt
│
├── data/
│ └── nbaallelo.csv
│
├── scripts/
│ ├── P2Project.ipynb
│ └── P2EDATake2.ipynb
│
├── output/
│ └── summary_results.csv
│ └── *.pdf (plots)
│
├── LICENSE.md
└── .gitignore


---

## Reproducing Results

### Step 1 — Download Data

Download dataset from GitHub:  
https://github.com/fivethirtyeight/data/blob/master/nba-elo/nbaallelo.csv  

Place the file into:


data/nbaallelo.csv


---

### Step 2 — Run Code

All analysis is contained in the notebook:


scripts/P2Project.ipynb


Open the notebook and run all cells sequentially to reproduce results.
---

### Step 3 — View Output

Results will appear in:


output/


Including:

- Model comparison table (`summary_results.csv`)  
- Precision-recall curves  
- Model comparison bar charts  
- Feature importance plots  

---

## Modeling Approach

- Data cleaning and chronological sorting of NBA game data  
- Creation of an **upset label** based on forecast probability and game outcome  
- Feature engineering using:
  - Elo ratings and Elo differences  
  - Game location (home/away/neutral)  
  - Rolling averages (last 10 games)  
  - Expanding averages (full history)  
  - Win streaks  
  - Elo momentum (short-term vs long-term trends)  
  - Underdog-related features (underdog margin, upset pressure, upset zone)  
  - Variability and consistency measures  

- Two model setups:
  - **Model A (Recent-10):** uses rolling 10-game features  
  - **Model B (Full History):** uses expanding full-history features  

- Chronological 80/20 train-test split to prevent data leakage  

- Model:
  - **HistGradientBoostingClassifier** (scikit-learn)  
  - Class-balanced to account for rare upset events  

- Evaluation:
  - ROC-AUC (primary metric)  
  - F1 score (secondary)  
  - Average Precision (AP)  
  - Precision-recall curve with optimized threshold  

---

## Results

| Model              | F1 Score | ROC-AUC |
|--------------------|---------|--------|
| Recent-10 Model    | 0.468   | 0.73   |
| Full History Model | 0.469   | 0.74   |

---

## Notes

- The same train/test split is used for both models.  
- Upsets are rare, which creates class imbalance in the dataset.  
- ROC-AUC is emphasized because it better evaluates model performance on rare-event classification.  
- Threshold tuning using precision-recall curves improves upset detection compared to a fixed threshold.  
- Recent performance captures short-term momentum, while full historical data provides more stable trends.  

---

## Future Work

- Compare with Logistic Regression and Random Forest baselines  
- Incorporate player-level or injury data  
- Add betting odds or external features  
- Further tune model hyperparameters  

---

## Acknowledgements

- Dataset: FiveThirtyEight NBA Elo dataset (GitHub)  
- Instructor: Karsten Siller  
- TA: Cole Whittington   
