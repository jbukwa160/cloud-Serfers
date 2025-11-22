import streamlit as st
import requests
import os

# If you deploy, change this to the deployed backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# -------------------------
# Page config + tiny styling
# -------------------------
st.set_page_config(
    page_title="Cloud-Serfers ML Demo",
    page_icon="☁️",
    layout="centered"
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: .2rem;
        }
        .sub-title {
            color: #6b7280;
            margin-top: 0;
            margin-bottom: 1.2rem;
        }
        .card {
            padding: 1.2rem;
            border-radius: 14px;
            border: 1px solid #eee;
            background: #fafafa;
            margin-bottom: 1rem;
        }
        .result-card {
            padding: 1rem;
            border-radius: 12px;
            background: #ecfdf3;
            border: 1px solid #a7f3d0;
            font-weight: 700;
        }
        .small-note {
            font-size: .9rem;
            color: #6b7280;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --------
# Sidebar
# --------
st.sidebar.header("⚙️ Configuration")
dataset = st.sidebar.selectbox("Choose dataset", ["Housing", "Electricity"])
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Cloud-Serfers Demo App**  
    - FastAPI backend  
    - Streamlit frontend  
    - ML model served locally or deployed  

    _Tip:_ if deployed, set **BACKEND_URL** in your environment.
    """,
)

# -----------
# Main header
# -----------
st.markdown('<div class="main-title">Cloud-Serfers – UK Housing & Electricity Predictions</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Fill in the inputs below and click Predict.</div>', unsafe_allow_html=True)

# ----------------
# Housing Section
# ----------------
if dataset == "Housing":
    st.subheader("🏡 Housing Price Prediction")

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            region = st.text_input("Region", "East Midlands", help="Example: East Midlands, London, North West, etc.")
            property_type = st.selectbox(
                "Property Type",
                ["D", "S", "T", "F", "O"],
                help="D=Detached, S=Semi, T=Terraced, F=Flat, O=Other"
            )
            duration = st.text_input(
                "Duration / Tenure",
                "F",
                help="Example: F=Freehold, L=Leasehold, U=Unknown"
            )

        with col2:
            year = st.number_input("Year of sale", min_value=1995, max_value=2050, value=2015, step=1)
            month = st.number_input("Month of sale", min_value=1, max_value=12, value=7, step=1)
            is_new_build = st.checkbox("New build", value=False)

        st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("✨ Predict housing price", use_container_width=True)

    if predict_btn:
        payload = {
            "region": region,
            "property_type": property_type,
            "duration": duration,
            "year": int(year),
            "month": int(month),
            "is_new_build": bool(is_new_build),
        }

        try:
            with st.spinner("Running model..."):
                resp = requests.post(f"{BACKEND_URL}/predict/housing", json=payload, timeout=10)
                resp.raise_for_status()

            data = resp.json()
            price = float(data["predicted_price"])
            formatted_price = f"£{price:,.0f}"

            st.markdown(
                f'<div class="result-card">✅ Predicted price: {formatted_price}</div>',
                unsafe_allow_html=True
            )
            st.markdown('<div class="small-note">Prediction is based on the trained PyCaret pipeline.</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error while predicting: {e}")

# ---------------------
# Electricity Section
# ---------------------
else:
    st.subheader("⚡ Electricity Demand Prediction")

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            year = st.number_input("Year", min_value=2000, max_value=2100, value=2023, step=1)
            month = st.number_input("Month", min_value=1, max_value=12, value=11, step=1)
            day = st.number_input("Day", min_value=1, max_value=31, value=24, step=1)

        with col2:
            hour = st.number_input("Hour", min_value=0, max_value=23, value=18, step=1)
            is_weekend = st.selectbox(
                "Weekend or weekday?",
                options=[0, 1],
                format_func=lambda x: "Weekend" if x == 1 else "Weekday"
            )

        st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("✨ Predict electricity demand", use_container_width=True)

    if predict_btn:
        payload = {
            "year": int(year),
            "month": int(month),
            "day": int(day),
            "hour": int(hour),
            "is_weekend": int(is_weekend),
        }

        try:
            with st.spinner("Running model..."):
                resp = requests.post(f"{BACKEND_URL}/predict/electricity", json=payload, timeout=10)
                resp.raise_for_status()

            data = resp.json()
            demand = float(data["predicted_demand"])
            formatted_demand = f"{demand:,.0f} MW"

            st.markdown(
                f'<div class="result-card">✅ Predicted demand: {formatted_demand}</div>',
                unsafe_allow_html=True
            )
            st.markdown('<div class="small-note">Electricity prediction currently reuses the housing model pipeline.</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error while predicting: {e}")

# Footer
st.markdown("---")
st.markdown(
    "<div class='small-note'>Cloud-Serfers ML Demo • FastAPI + Streamlit</div>",
    unsafe_allow_html=True
)
