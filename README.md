# NBA Upset Prediction Model

> Compare whether NBA game upsets are better predicted using recent performance (last 10 games) or full historical data using gradient boosting models and engineered features.

**Course:** DS 4002  
**Group Name:** DLC  
**Group Leader:** Caroline Lingle  
**Group Members:** Lauren Medica, Dev Patel, Caroline Lingle  

---

## Overview

This project compares two machine learning approaches:

- Recent-10 games performance
- Full historical performance

We evaluate performance primarily using **F1 score**.

**Key Result:**  
The recent-performance and full-history models capture different aspects of team behavior, highlighting the tradeoff between short-term momentum and long-term trends.

---

## Installation

Clone the repository:


git clone <your-repo-link>
cd <repo-name>


Install required packages:


pip install -r requirements.txt


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
├── DATA/
│ └── nbaallelo.csv
│
├── SCRIPTS/
│ ├── 01_load_clean.py
│ ├── 02_feature_engineering.py
│ ├── 03_model_recent10.py
│ ├── 04_model_full_history.py
│ └── 05_model_comparison.py
│
├── OUTPUT/
│ └── model_results.csv
│
└── P2Project.ipynb


---

## Reproducing Results

### Step 1 — Download Data

Download dataset from GitHub:

https://github.com/fivethirtyeight/data/blob/master/nba-elo/nbaallelo.csv

Place the following file into:


DATA/


- nbaallelo.csv

---

### Step 2 — Install Dependencies

From the project root directory, run:


pip install -r requirements.txt


---

### Step 3 — Run Scripts (in order)

From the project root directory, execute:


python SCRIPTS/01_load_clean.py
python SCRIPTS/02_feature_engineering.py
python SCRIPTS/03_model_recent10.py
python SCRIPTS/04_model_full_history.py
python SCRIPTS/05_model_comparison.py


---

### Step 4 — View Output

Results will appear in:


OUTPUT/


Including:
- F1 scores
- Precision / Recall
- Model comparisons

---

## Modeling Approach

- Feature engineering using Elo ratings and game context
- Rolling averages for recent performance
- 80/20 chronological train-test split
- Gradient boosting classifier
- Primary evaluation metric: **F1 score**

---

## Results

| Model | F1 Score |
|-------|----------|
| Recent-10 Model | .468 |
| Full History Model | .469 |

---

## Notes

- The same train/test split is used for both models.
- A key design decision was whether to prioritize recent vs long-term performance.
- Recent data captures momentum, while full history provides stability.

---

## Future Work

- Logistic Regression comparison
- Random Forest model
- Player-level analysis
- Betting odds integration

---

## Acknowledgements

Dataset: FiveThirtyEight NBA Elo dataset (GitHub)  
Instructor: Karsten Siller  
TA: Cole Whittington  
