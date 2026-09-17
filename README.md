# Hospital Readmission Prediction — Logistic Regression Case Study

## Overview
This project applies **Logistic Regression with L2 regularization** to predict whether a patient will be readmitted within **30 days** of hospital discharge, using patient records (diagnosis codes, vitals, and prior visit history). This is the Unit 1 case study for the Supervised Learning – Regression and Classification lab.

## Problem Statement
Hospitals want to identify high-risk patients likely to be readmitted within 30 days so they can provide targeted follow-up care and reduce preventable readmissions.

## Dataset
- **File:** `data/raw.csv`
- **Type:** Synthetic dataset (1000 patient records) generated to mimic real EHR data, since real hospital data is privacy-protected.
- **Features:** age, gender, diagnosis_code, systolic_bp, heart_rate, glucose, bmi, prior_visits, prior_admissions
- **Target:** `readmitted_30_days` (0 = No, 1 = Yes)
- *(Alternative: the public [UCI Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) can be substituted for real-world data.)*

## Methodology
1. **Preprocessing** (`src/preprocess.py`): clean missing values, one-hot encode categorical features, scale numeric features, train/test split (80/20, stratified).
2. **Model Training** (`src/train.py`): Logistic Regression with **L2 regularization**, tuned via `GridSearchCV` + 5-fold cross-validation on the regularization strength `C`. An L1-regularized model is also trained for comparison.
3. **Evaluation** (`src/evaluate.py`): Confusion Matrix, Precision/Recall/F1, ROC-AUC score and curve.

## Results
| Model | ROC-AUC (cross-val) |
|---|---|
| L2 Logistic Regression | ~0.66 |
| L1 Logistic Regression | ~0.66 |

*(Exact numbers vary slightly by random seed — see console output when you run `train.py`.)*

**L1 vs L2:** L1 regularization drives some feature coefficients to exactly zero (built-in feature selection), while L2 shrinks all coefficients smoothly without eliminating any — useful when most features are believed to carry some signal, as is the case with general patient vitals and history.

## Clinical Cost Discussion: False Positives vs. False Negatives
In this context:
- **False Negative** (model predicts "no readmission" but patient IS readmitted) is the more costly error — it means a high-risk patient misses out on preventive follow-up care, increasing the risk of complications or harm.
- **False Positive** (model predicts "readmission" but patient is NOT readmitted) simply results in some unnecessary follow-up (a phone call, extra check-in) — costly in staff time, but not dangerous.

**Conclusion:** Given this asymmetry, the model should be tuned to favor **higher Recall (sensitivity)** over Precision — even at the cost of more false alarms — by lowering the classification threshold below the default 0.5. This ensures fewer at-risk patients slip through undetected.

## How to Run
```bash
pip install -r requirements.txt
cd src
python preprocess.py
python train.py
python evaluate.py
```

## Project Structure
```
hospital-readmission-logistic-regression/
├── data/
│   ├── raw.csv              # original synthetic dataset
│   └── processed.csv        # cleaned + encoded dataset
├── notebooks/
│   └── exploration.ipynb    # EDA and step-by-step walkthrough
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

## Tools Used
NumPy · Pandas · scikit-learn · Matplotlib · Google Colab
