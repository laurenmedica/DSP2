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
python SCRIPTS/05_comparison.py


---

### Step 4 — View Output

Results will appear in the notebook or output files.

Including:
- F1 score
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

| Model | Description |
|-------|----------|
| Recent-10 Model | Short-term performance |
| Full History Model | Long-term performance |

---

## Notes

- The same chronological split is used for both models
- A key design decision was whether to prioritize recent vs long-term performance
- Recent data captures momentum, while full history provides stability

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
