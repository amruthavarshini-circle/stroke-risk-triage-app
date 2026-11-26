# Stroke Risk Factors & Clinical Triage App 🩺

**Project #49** — Introduction to Data Science (`25A12104`)  
**Student:** Vaddi Varshitha (`S230617`) — AIML-2A  
**Course Instructor:** Satish C  
**Submission:** November 2026

---

## What This Project Is About

> *Impute BMI and smoking attributes, test hypertension/glucose risk via Chi-Square & Logistic Regression, and deploy a Streamlit stroke risk screening app.*

Basically, I wanted to see which health factors actually matter for predicting strokes, and then build a small web app that could flag high-risk patients. The dataset is super imbalanced (only ~5% stroke cases), so a lot of my work went into handling that properly.

## Dataset

| Field | Detail |
| :--- | :--- |
| **Source** | [Kaggle — Stroke Prediction Dataset (fedesoriano)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) |
| **File** | `healthcare-dataset-stroke-data.csv` |
| **Dimensions** | 5,110 rows × 12 columns |
| **Target** | `stroke` (binary) — 249 positive (4.87%) vs. 4,861 negative (95.13%) |

> Download the CSV from Kaggle and place it in `data/raw/`. Raw data is gitignored.

Check [data/README.md](data/README.md) for the full data dictionary.

## Key Findings

My analysis found some interesting stuff:

- **Age** is by far the biggest risk factor — OR = 2.02 per 10 years (p < 1e-38). Not surprising, but the effect size is huge.
- **Missing BMI** turned out to be a really strong signal: patients with no BMI recorded had a 19.9% stroke rate vs 4.3% otherwise. I think this is because those patients were probably emergency admissions where nobody measured their BMI.
- **Hypertension** (OR = 1.39, p = 0.049) and **avg glucose** (OR = 1.04 per 10 mg/dL, p = 0.002) are both independently significant after adjusting for other variables.
- I tried both SMOTE and `class_weight='balanced'` for the imbalance — balanced class weights actually worked better (AUC 0.849 vs 0.842).
- **Logistic Regression** (Test ROC-AUC: 0.831) was surprisingly competitive with Random Forest (0.833) and XGBoost (0.841). I went with LR for the app since it gives you interpretable coefficients.
- At threshold ~0.54, the model gets **80% Recall** with **75% Specificity** on the test set.

## Repo Structure

```text
ds-stroke-risk-triage-app/
├── data/
│   ├── README.md                        # Data dictionary & download steps
│   ├── raw/                             # CSV goes here (gitignored)
│   └── processed/                       # Cleaned CSVs (gitignored)
├── notebooks/
│   ├── 01_data_audit_and_cleaning.ipynb
│   ├── 02_eda_and_visualizations.ipynb
│   ├── 03_statistical_hypothesis_testing.ipynb
│   └── 04_modeling_and_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py                        # Paths, seeds, thresholds
│   ├── data_cleaning.py                 # Audit + stratified imputation
│   ├── statistical_tests.py             # Chi-Square, Mann-Whitney, ORs
│   ├── feature_engineering.py           # Transformers & ColumnTransformer
│   ├── train_models.py                  # Model benchmark & tuning
│   └── app.py                           # Streamlit triage app
├── models/
│   ├── logistic_regression_pipeline.joblib
│   ├── best_model_pipeline.joblib
│   └── model_metrics.json
├── docs/
│   ├── figures/                         # Charts (Figures 1–10)
│   └── VIVA_PREP.md                     # Viva Q&A notes
├── .gitignore
├── requirements.txt
└── README.md
```

## Tech Stack

| Category | Libraries |
| :--- | :--- |
| Data & Statistics | pandas, NumPy, SciPy, Statsmodels |
| Machine Learning | scikit-learn, imbalanced-learn, XGBoost |
| Visualization | Matplotlib, Seaborn, Plotly |
| Web App | Streamlit |
| Version Control | Git / GitHub |

## How to Run

```bash
# 1. Clone
git clone https://github.com/RajuGudem26/ds-stroke-risk-triage-app.git
cd ds-stroke-risk-triage-app

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

# 3. Install dependencies (on macOS you might need: brew install libomp for XGBoost)
pip install -r requirements.txt

# 4. Download the dataset from Kaggle and put it in data/raw/

# 5. Run data cleaning
python -m src.data_cleaning
jupyter nbconvert --execute --inplace notebooks/01_data_audit_and_cleaning.ipynb

# 6. Run statistical tests
python -m src.statistical_tests

# 7. Train models
python -m src.train_models

# 8. Launch the app
streamlit run src/app.py
```

## License

Academic capstone project — for educational purposes only.
