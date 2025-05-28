import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="🏠 House Price Predictor", layout="wide")

st.title("🏠 House Price Prediction App")

# Sidebar: Upload CSV
st.sidebar.header("📁 Upload Your CSV")
uploaded_file = st.sidebar.file_uploader("Upload 'Test.csv'", type=["csv"])

# Load and simulate price only if PRICE column missing
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    if "PRICE" not in df.columns:
        np.random.seed(42)
        df["PRICE"] = 200000 + (df["SQUARE_FT"] * 4000) + (df["BHK_NO."] * 100000) + np.random.randint(-100000, 100000, size=len(df))
    return df

if uploaded_file is not None:
    data = load_data(uploaded_file)
else:
    st.warning("Please upload a `Test.csv` file to proceed.")
    st.stop()

# Features & target
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
    return pd.DataFrame([[under_construction, rera, bhk, sqft, ready, resale]], columns=features)

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

# Predict for user input
prediction = model.predict(input_df)[0]

# Display prediction
st.subheader("💰 Predicted House Price")
st.success(f"₹ {prediction:,.0f}")

# Display evaluation
st.markdown("---")
st.subheader("📊 Model Evaluation")
st.write(f"**RMSE:** ₹ {rmse:,.0f}")
st.write(f"**R² Score:** {r2:.2f}")

# Confidence interval
st.markdown("### 🔍 Prediction Confidence Interval")
residual_std = np.std(y_test - y_pred)
lower = prediction - residual_std
upper = prediction + residual_std
st.info(f"Estimated range: ₹ {lower:,.0f} to ₹ {upper:,.0f}")

# Tabs for Prediction and Visualization
tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Visualizations"])

with tab1:
    st.header("🔮 House Price Prediction")

    # Show input features from sidebar as static
    st.subheader("🏗️ Selected House Features")
    st.write(input_df)

    st.subheader("💰 Predicted Price")
    st.success(f"₹ {prediction:,.0f}")

    st.info(f"Estimated range: ₹ {lower:,.0f} to ₹ {upper:,.0f}")

    st.markdown("---")
    st.subheader("📈 Price Distribution")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.histplot(data["PRICE"], bins=30, kde=True, ax=ax1, color="skyblue", label="Prices")
    ax1.axvline(prediction, color='red', linestyle='--', label=f'Predicted: ₹{prediction:,.0f}')
    ax1.set_xlabel("Price (₹)")
    ax1.set_ylabel("Count")
    ax1.legend()
    st.pyplot(fig1)

    st.subheader("📊 Model Evaluation")
    st.write(f"**RMSE:** ₹ {rmse:,.0f}")
    st.write(f"**R² Score:** {r2:.2f}")

with tab2:
    st.header("📊 Data Visualizations for Price Estimation")

    st.subheader("🏗️ Square Foot vs Price")
    fig2, ax2 = plt.subplots()
    sns.regplot(x="SQUARE_FT", y="PRICE", data=data, ax=ax2, line_kws={"color": "red"})
    st.pyplot(fig2)

    st.subheader("🛏️ Average Price by BHK")
    avg_price_bhk = data.groupby("BHK_NO.")["PRICE"].mean().reset_index()
    fig3, ax3 = plt.subplots()
    sns.barplot(data=avg_price_bhk, x="BHK_NO.", y="PRICE", palette="Blues", ax=ax3)
    ax3.set_ylabel("Avg Price (₹)")
    st.pyplot(fig3)

    st.subheader("📦 Ready to Move vs Price")
    fig4, ax4 = plt.subplots()
    sns.boxplot(data=data, x="READY_TO_MOVE", y="PRICE", palette="Set2", ax=ax4)
    ax4.set_xticklabels(["No", "Yes"])
    st.pyplot(fig4)

    st.subheader("📉 Correlation Heatmap")
    fig5, ax5 = plt.subplots()
    corr = data[features + [target]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax5)
    st.pyplot(fig5)

    st.subheader("📐 Pairplot (sampled)")
    sampled = data.sample(min(200, len(data)))
    pairplot_fig = sns.pairplot(sampled[["PRICE", "SQUARE_FT", "BHK_NO.", "RESALE"]], diag_kind='kde')
    st.pyplot(pairplot_fig.fig)

# Sample data preview
st.markdown("---")
st.subheader("🔍 Sample Data")
st.dataframe(data.head())
