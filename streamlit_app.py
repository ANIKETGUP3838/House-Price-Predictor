# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("🏠 House Price Prediction App")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Test.csv")
    # Simulate price for demo purposes: PRICE = base + (sqft * rate) + bhk influence
    np.random.seed(42)
    df["PRICE"] = 200000 + (df["SQUARE_FT"] * 4000) + (df["BHK_NO."] * 100000) + np.random.randint(-100000, 100000, size=len(df))
    return df

data = load_data()

# Feature selection
features = ["UNDER_CONSTRUCTION", "RERA", "BHK_NO.", "SQUARE_FT", "READY_TO_MOVE", "RESALE"]
target = "PRICE"

# Sidebar: input fields for user prediction
st.sidebar.header("Input House Features")
def user_input():
    under_construction = st.sidebar.selectbox("Under Construction", [0, 1])
    rera = st.sidebar.selectbox("RERA Approved", [0, 1])
    bhk = st.sidebar.slider("Number of BHK", 1, 5, 2)
    sqft = st.sidebar.slider("Square Feet", 300, 5000, 1200)
    ready = st.sidebar.selectbox("Ready to Move", [0, 1])
    resale = st.sidebar.selectbox("Is Resale", [0, 1])

    return pd.DataFrame([[under_construction, rera, bhk, sqft, ready, resale]],
                        columns=features)

input_df = user_input()

# Train-test split
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Predictions
prediction = model.predict(input_df)[0]

# Display predictions
st.subheader("Predicted House Price 💰")
st.success(f"₹ {prediction:,.0f}")

st.markdown("---")
st.subheader("Model Evaluation")
st.write(f"**RMSE:** ₹ {rmse:,.0f}")
st.write(f"**R² Score:** {r2:.2f}")

st.markdown("---")
st.subheader("Sample Data")
st.dataframe(data.head())
