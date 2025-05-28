# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="🏠 House Price Predictor", layout="wide")

st.title("🏠 House Price Prediction App")

# Sidebar: Upload CSV
st.sidebar.header("📁 Upload Your CSV")
uploaded_file = st.sidebar.file_uploader("Upload 'Test.csv'", type=["csv"])

# Function to load and simulate price
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    np.random.seed(42)
    # Simulate price for demonstration
    df["PRICE"] = 200000 + (df["SQUARE_FT"] * 4000) + (df["BHK_NO."] * 100000) + np.random.randint(-100000, 100000, size=len(df))
    return df

# Stop if no file uploaded
if uploaded_file is not None:
    data = load_data(uploaded_file)
else:
    st.warning("Please upload a `Test.csv` file to proceed.")
    st.stop()

# Feature selection
features = ["UNDER_CONSTRUCTION", "RERA", "BHK_NO.", "SQUARE_FT", "READY_TO_MOVE", "RESALE"]
target = "PRICE"

# Sidebar: user input for prediction
st.sidebar.header("🏗️ House Features for Prediction")
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

# Train model
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Predict user input
prediction = model.predict(input_df)[0]

# Output results
st.subheader("💰 Predicted House Price")
st.success(f"₹ {prediction:,.0f}")

st.markdown("---")
st.subheader("📊 Model Evaluation")
st.write(f"**RMSE:** ₹ {rmse:,.0f}")
st.write(f"**R² Score:** {r2:.2f}")

st.markdown("---")
st.subheader("🔍 Sample Data")
st.dataframe(data.head())
