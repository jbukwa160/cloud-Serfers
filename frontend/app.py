import streamlit as st
import requests

# If you deploy, change this to the deployed backend URL
BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Cloud-Serfers ML Demo", layout="centered")

st.title("Cloud-Serfers – UK Housing & Electricity Predictions")

st.sidebar.header("Configuration")
dataset = st.sidebar.selectbox("Choose dataset", ["Housing", "Electricity"])

st.write("Fill in the inputs and click **Predict**.")

if dataset == "Housing":
    st.subheader("Housing Price Prediction")

    region = st.text_input("Region", "East Midlands")
    property_type = st.selectbox("Property Type", ["D", "S", "T", "F", "O"])
    tenure = st.text_input("Tenure", "F")
    year = st.number_input("Year of sale", min_value=1995, max_value=2050, value=2015, step=1)
    month = st.number_input("Month of sale", min_value=1, max_value=12, value=7, step=1)
    is_new_build = st.checkbox("New build", value=False)

    if st.button("Predict housing price"):
        payload = {
            "region": region,
            "property_type": property_type,
            "tenure": tenure,
            "year": int(year),
            "month": int(month),
            "is_new_build": bool(is_new_build),
        }

        try:
            resp = requests.post(f"{BACKEND_URL}/predict/housing", json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            price = float(data["predicted_price"])
            formatted_price = f"£{price:,.0f}" 
            st.success(f"Predicted price: {formatted_price}")
        except Exception as e:
            st.error(f"Error while predicting: {e}")

else:
    st.subheader("Electricity Demand Prediction")

    year = st.number_input("Year", min_value=2000, max_value=2100, value=2023, step=1)
    month = st.number_input("Month", min_value=1, max_value=12, value=11, step=1)
    day = st.number_input("Day", min_value=1, max_value=31, value=24, step=1)
    hour = st.number_input("Hour", min_value=0, max_value=23, value=18, step=1)
    is_weekend = st.selectbox("Is weekend?", options=[0, 1], format_func=lambda x: "Weekend (1)" if x == 1 else "Weekday (0)")

    if st.button("Predict electricity demand"):
        payload = {
            "year": int(year),
            "month": int(month),
            "day": int(day),
            "hour": int(hour),
            "is_weekend": int(is_weekend),
        }

        try:
            resp = requests.post(f"{BACKEND_URL}/predict/electricity", json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            demand = float(data["predicted_demand"])
            formatted_demand = f"{demand:,.0f} MW" 
            st.success(f"Predicted demand: {formatted_demand}")
        except Exception as e:
            st.error(f"Error while predicting: {e}")
