import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -----------------------------------------
# Streamlit Page Settings
# -----------------------------------------
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

# Load Model + Scaler + Feature Names
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
feature_names = joblib.load("features.joblib")

st.title("🏦 Loan Approval Prediction App")

st.markdown("""
This application predicts whether a **loan will be approved or rejected**  
based on applicant information.  
Please enter the applicant details below.
""")

# -----------------------------------------
# Define Feature Types
# -----------------------------------------
int_features = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "Loan_Amount_Term", "Credit_History", "Property_Area"
]

float_features = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]

# -----------------------------------------
# Input Form
# -----------------------------------------
user_input = {}

for i in range(0, len(feature_names), 2):
    cols = st.columns(2)
    for j in range(2):
        if i + j < len(feature_names):
            feature = feature_names[i + j]
            with cols[j]:

                # Integer Inputs
                if feature in int_features:
                    value = st.number_input(
                        f"{feature}",
                        value=None,
                        step=1,
                        format="%d"
                    )

                # Float Inputs
                elif feature in float_features:
                    value = st.number_input(
                        f"{feature}",
                        value=None,
                        step=0.01,
                        format="%.2f"
                    )

                user_input[feature] = value

# -----------------------------------------
# Prediction
# -----------------------------------------
if st.button("Predict"):
    try:
        df_input = pd.DataFrame([user_input])

        # Scale only float features
        continuous = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
        df_input[continuous] = scaler.transform(df_input[continuous])

        prediction = model.predict(df_input)[0]

        st.subheader("🔍 Prediction Result")

        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

    except Exception as e:
        st.error("⚠️ Error: Please check your inputs.")
        st.write(e)