import streamlit as st
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "trainModel", "housing_best_model.pkl")
model = joblib.load(model_path)

st.title("🏡 UK Housing Price Predictor")

st.subheader("Local PyCaret model")

location = st.text_input("Enter location (city/region)")
year = st.slider("Select year", 1995, 2017, 2010)
avg_price = st.number_input("Average local price", min_value=0.0, value=200000.0)

if st.button("Predict Price"):
    df = pd.DataFrame(
        [[location, year, avg_price]],
        columns=["location", "year", "avg_price"]  # adjust to real feature names
    )
    pred = model.predict(df)
    st.success(f"Predicted Price: £{pred[0]:,.0f}")
