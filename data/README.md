# Dataset — Stroke Prediction

## Source

- **Dataset:** Stroke Prediction Dataset by fedesoriano on Kaggle
- **Link:** https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- **File:** `data/raw/healthcare-dataset-stroke-data.csv`
- **Size:** 5,110 rows × 12 columns (4.87% stroke cases)

### How to Download

1. Go to the [Kaggle page](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) and sign in.
2. Click **Download** to get the zip.
3. Extract `healthcare-dataset-stroke-data.csv` into `data/raw/`.
4. Don't commit CSVs or zip files — they're gitignored.

---

## Data Dictionary

| # | Column | Type | Values | Description | Quality Notes |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | `id` | Integer | Unique ID | Patient identifier | Drop before modeling — not predictive |
| 2 | `gender` | Categorical | Male, Female, Other | Patient's gender | 1 record is "Other" (0.02%) |
| 3 | `age` | Float | 0.08 – 82.0 years | Age in years | Values < 1.0 are infants |
| 4 | `hypertension` | Binary | 0 = No, 1 = Yes | Has high blood pressure | ~9.7% positive |
| 5 | `heart_disease` | Binary | 0 = No, 1 = Yes | Has heart disease | ~5.4% positive |
| 6 | `ever_married` | Categorical | No, Yes | Has been married | Correlated with age |
| 7 | `work_type` | Categorical | Private, Self-employed, Govt_job, children, Never_worked | Type of work | "children" = pediatric patients |
| 8 | `Residence_type` | Categorical | Urban, Rural | Where they live | Roughly 50/50 split |
| 9 | `avg_glucose_level` | Float | ~55 – 272 mg/dL | Average blood glucose | Right-skewed, bimodal around diabetic range (≥126 mg/dL) |
| 10 | `bmi` | Float | ~10.3 – 97.6 kg/m² | Body mass index | 201 missing values (~3.9%), some extreme outliers |
| 11 | `smoking_status` | Categorical | formerly smoked, never smoked, smokes, Unknown | Smoking history | ~30% are "Unknown", mostly kids under 12 |
| 12 | `stroke` | Binary | 0 = No, 1 = Yes | **Target** — had a stroke | Very imbalanced: only 4.87% positive |

---

## Reference Thresholds I Used

- **BMI (WHO):** Underweight < 18.5, Normal 18.5–24.9, Overweight 25–29.9, Obese ≥ 30
- **Glucose (ADA):** Normal < 100, Prediabetes 100–125, Diabetic ≥ 126, Severe ≥ 200 mg/dL
- **Cardio risk (AHA):** Hypertension = 1, Heart disease = 1, elevated metabolic burden
