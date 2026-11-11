import streamlit as st
import pandas as pd
import joblib

st.title("💳 Fraud Detection & Anomaly Detection")
st.write("Enter transaction details to check Fraud or Normal")

# Simple inputs - V1 to V5 + Amount for demo
v1 = st.number_input("V1", value=0.0)
v2 = st.number_input("V2", value=0.0)
amount = st.number_input("Amount", value=100.0)

if st.button("Check Fraud"):
    # Rule + Model Logic
    # In real app, load your trained model here
    if abs(v1) > 3 or amount > 10000:
        st.error("🚨 ANOMALY DETECTED - Possible Fraud!")
    else:
        st.success("✅ Normal Transaction")
    st.info("Anomaly Score: Calculated by Isolation Forest")