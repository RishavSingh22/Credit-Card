
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("Credit Card Fraud Detection App")

st.write("""
This app predicts whether a credit card transaction is **fraudulent** or **legitimate** 
based on user-provided input.
""")

# Load dataset and train model
@st.cache_data
def load_and_train():
    df = pd.read_csv("creditcard.zip")
    X = df.drop(columns="Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X.columns.tolist()

model, feature_names = load_and_train()

st.sidebar.header("Enter Transaction Details")

# Collect user input for each feature
user_input = []
for feature in feature_names:
    val = st.sidebar.number_input(f"{feature}", value=0.0, format="%.5f")
    user_input.append(val)

input_df = pd.DataFrame([user_input], columns=feature_names)

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction_prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This transaction is likely FRAUDULENT with a probability of {prediction_prob:.2f}")
    else:
        st.success(f"✅ This transaction is LEGITIMATE with a probability of {1 - prediction_prob:.2f}")
