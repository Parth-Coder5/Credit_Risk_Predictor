# 🚀 Credit Risk Predictor (Machine Learning + Streamlit)

An end-to-end **Credit Risk Prediction System** that classifies loan applicants as **Low Risk** or **High Risk** using Machine Learning and a business-aware decision threshold.

---

## 📌 Project Overview

Banks and financial institutions must minimize **false negatives** (high-risk customers wrongly approved).  
This project focuses on **probability-based decision making** rather than raw accuracy.

The system predicts:
- 📉 **Risk Probability**
- ⚖️ **Decision using tuned threshold**
- ✅ Final classification: **Low Risk / High Risk**

---

## 🧠 Machine Learning Approach

- **Model**: Logistic Regression  
- **Pipeline**:  
  - Missing value handling  
  - One-Hot Encoding  
  - Feature scaling  
  - Classification  
- **Threshold Tuning**:  
  - Default: `0.5`
  - Final: `0.4` (chosen to improve high-risk recall)

---

## 🧪 Model Performance (Test Set)

| Metric | Value |
|------|------|
| ROC-AUC | ~0.66 |
| High-Risk Recall | **92%** |
| Accuracy | ~62% |

> ⚠️ Accuracy drops after threshold tuning — **this is expected and acceptable** in credit risk problems.

---

## 🖥️ Web Application (Streamlit)

### Features:
- Interactive UI for applicant details
- Real-time risk probability
- Business-driven decision explanation
- Clean, modern interface

📸 **Preview:**
![App Screenshot](https://github.com/Parth-Coder5/Credit_Risk_Predictor/blob/main/App%20Screenshot.png)

---

## 🧰 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib
- Git & GitHub

---

## 📂 Project Structure

Credit_Risk_Predictor/
├── app.py
├── requirements.txt
├── README.md
├── Data/
├── Model/
└── Notebooks/


---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
