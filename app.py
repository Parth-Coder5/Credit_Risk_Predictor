import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title='Credit Risk Predictor',
    page_icon= "💳",
    layout='centered'
)
st.title("💳 Credit Risk Predictor")
st.write(
    "Predict whether a loan applicant is **Low Risk** or **High Risk** "
    "using a Machine Learning model."
)

@st.cache_resource
def load_artifacts():
    model = joblib.load('Model/credit_risk_model.pkl')
    threshold = joblib.load('Model/threshold.pkl')
    features = joblib.load('Model/feature_columns.pkl')
    return model,threshold,features

model, FINAL_THRESHOLD, feature_cols = load_artifacts()

st.subheader("🧾 Applicant Information")

age = st.slider('Age', 18,75,30)
sex = st.selectbox('Sex',['Male','Female'])
job = st.selectbox('Job Level (0=unskilled,3=skilled)', [0,1,2,3])
housing = st.selectbox('Housing', ['own','rent','free'])

savings_account = st.selectbox(
    "Saving Accounts",
    ["little", "moderate", "quite rich", "rich", "unknown"]
)

checking_account = st.selectbox(
    "Checking Account",
    ["little", "moderate", "rich", "unknown"]
)

credit_amount = st.number_input('Credit Amount',min_value=100, step=100)
duration = st.slider('Loan Duration (months)', 4,72,24)
purpose = st.selectbox(
    "Purpose",
    ["car", "radio/TV", "education", "furniture/equipment", "business", "repairs"]
)

input_dict = {
    'Age':age,
    'Sex':sex,
    'Job':job,
    'Housing':housing,
    "Saving accounts": savings_account if savings_account != "unknown" else np.nan,
    "Checking account": checking_account if checking_account != "unknown" else np.nan,
    'Credit amount':credit_amount,
    'Duration':duration,
    'Purpose':purpose,
}

input_df = pd.DataFrame([input_dict])

if st.button("🔍 Predict Credit Risk"):
    prob = model.predict_proba(input_df)[0][1]
    prediction = int(prob >= FINAL_THRESHOLD)

    st.divider()
    
    st.subheader("📊 Prediction Result")

    st.write(f"**Risk Probability:** `{prob:.2%}`")
    st.write(f"**Decision Threshold:** `{FINAL_THRESHOLD}`")

    if prediction == 1:
        st.error("🚨 **HIGH CREDIT RISK**")
        st.write(
            "This applicant has a **high risk of default**. "
            "Loan approval should be done cautiously."
        )
    else:
        st.success("✅ **LOW CREDIT RISK**")
        st.write(
            "This applicant is considered **low risk** and is "
            "more likely to repay the loan."
        )
    st.caption(
        "⚠️ Model prioritizes **high-risk recall** to reduce false negatives, "
        "which is critical in lending decisions."
    )
