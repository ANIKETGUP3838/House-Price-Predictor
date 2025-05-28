
import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("ready_to_move_model.pkl")
le_posted_by = joblib.load("posted_by_encoder.pkl")
le_bhk_or_rk = joblib.load("bhk_or_rk_encoder.pkl")

st.title("🏠 Ready to Move Property Predictor")

# User input
posted_by = st.selectbox("Posted By", le_posted_by.classes_)
under_construction = st.selectbox("Under Construction", [0, 1])
rera = st.selectbox("RERA Registered", [0, 1])
bhk_no = st.slider("Number of BHK", 1, 10, 2)
bhk_or_rk = st.selectbox("BHK or RK", le_bhk_or_rk.classes_)
square_ft = st.number_input("Square Footage", min_value=100.0, max_value=10000.0, value=800.0)
resale = st.selectbox("Resale", [0, 1])
longitude = st.number_input("Longitude", value=77.0)
latitude = st.number_input("Latitude", value=28.0)

# Prediction
if st.button("Predict"):
    features = np.array([[
        le_posted_by.transform([posted_by])[0],
        under_construction,
        rera,
        bhk_no,
        le_bhk_or_rk.transform([bhk_or_rk])[0],
        square_ft,
        resale,
        longitude,
        latitude
    ]])

    prediction = model.predict(features)[0]
    result = "✅ Ready to Move" if prediction == 1 else "🚧 Not Ready Yet"
    st.success(f"Prediction: {result}")
