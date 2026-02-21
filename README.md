# Credit Risk Predictor

An end-to-end Machine Learning project to predict credit risk (good/bad customers) using Logistic Regression with a complete preprocessing and modeling pipeline.

## 🔍 Problem Statement
Credit risk assessment is a critical task in the financial industry. The goal of this project is to predict whether a loan applicant is a **high-risk** or **low-risk** customer based on financial and personal attributes.

## 📊 Dataset
- German Credit Risk Dataset (UCI Machine Learning Repository)
- 1000 records with mixed numerical and categorical features

## ⚙️ Features Used
- Age
- Sex
- Job
- Housing
- Saving accounts
- Checking account
- Credit amount
- Loan duration
- Purpose

## 🧠 Model
- Logistic Regression
- Full preprocessing + model pipeline using `scikit-learn`
- Threshold tuning to handle class imbalance

## 📈 Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrix
- Business-focused threshold optimization

## 🚀 Key Highlights
- End-to-end ML workflow
- Pipeline-based training (production-ready)
- Threshold tuning for high-risk recall
- Saved model for deployment

## 🛠 Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn

## 📦 Saved Artifacts
- `credit_risk_pipeline.pkl`
- `threshold.pkl`
- `feature_columns.pkl`

## 📌 Future Work
- Streamlit web application
- Model comparison (XGBoost, LightGBM)
- Deployment to cloud
---

⭐ If you find this project useful, consider starring the repo!
