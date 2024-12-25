import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load models and encoders
xgb_model = joblib.load("models/xgb_model.joblib")
lr_model = joblib.load("models/linear_regression_model.joblib")
mn_encoder = joblib.load("models/mn_encoder.joblib")
m_encoder = joblib.load("models/m_encoder.joblib")
f_encoder = joblib.load("models/f_encoder.joblib")
g_encoder = joblib.load("models/g_encoder.joblib")

# Streamlit app
st.title("Used Car Price Predictor")

# User input form
st.sidebar.header("User Input Features")
levy = st.sidebar.number_input("Levy (in numeric value)", min_value=0, value=0)
manufacturer = st.sidebar.text_input("Manufacturer (e.g., TOYOTA ->Capital)")
model = st.sidebar.text_input("Model (e.g., Corolla)")
prod_year = st.sidebar.number_input("Production Year", min_value=1900, max_value=2024, value=2015)
fuel_type = st.sidebar.text_input("Fuel Type (e.g., Petrol)")
engine_volume = st.sidebar.text_input("Engine Volume (e.g., 1.6 or 1.6 Turbo)")
mileage = st.sidebar.text_input("Mileage (e.g., 50000 km)")
gear_box = st.sidebar.text_input("Gear Box Type (e.g., Automatic)")

# Model selection
model_choice = st.radio("Select a model for prediction:", ("XGBoost", "Linear Regression"))

# Process input
try:
    engine_volume = float(engine_volume.replace(" Turbo", ""))
    mileage = int(mileage.replace(" km", ""))
except ValueError:
    st.error("Please enter valid numeric inputs for Engine Volume and Mileage.")
    st.stop()

manufacturer_encoded = mn_encoder.transform([manufacturer])[0] if manufacturer in mn_encoder.classes_ else -1
model_encoded = m_encoder.transform([model])[0] if model in m_encoder.classes_ else -1
fuel_type_encoded = f_encoder.transform([fuel_type])[0] if fuel_type in f_encoder.classes_ else -1
gear_box_encoded = g_encoder.transform([gear_box])[0] if gear_box in g_encoder.classes_ else -1

# Check if encoders are valid
if -1 in [manufacturer_encoded, model_encoded, fuel_type_encoded, gear_box_encoded]:
    st.error("One or more input values are invalid. Please check your inputs.")
    st.stop()

# Prepare input for prediction
features = np.array([[levy, manufacturer_encoded, model_encoded, prod_year, fuel_type_encoded,
                      engine_volume, mileage, gear_box_encoded]])

# Predict
if st.button("Predict Price"):
    if model_choice == "XGBoost":
        predicted_price = xgb_model.predict(features)[0]
    else:
        predicted_price = lr_model.predict(features)[0]
    
    # Format harga menjadi Rupiah dengan pemisah titik
    predicted_price_formatted = f"Rp. {predicted_price:,.0f}"
    st.success(predicted_price_formatted)
