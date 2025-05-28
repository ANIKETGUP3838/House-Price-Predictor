import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
import io

st.set_page_config(page_title="🏠 House Price Predictor", layout="wide")

st.title("🏠 House Price Prediction App")

# Sidebar upload
st.sidebar.header("📁 Upload Your CSV")
uploaded_file = st.sidebar.file_uploader("Upload 'Test.csv'", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    # Just a simple fallback to add PRICE if missing
    if "PRICE" not in df.columns:
        np.random.seed(42)
        df["PRICE"] = (
            200000 + (df["SQUARE_FT"] * 4000) + (df["BHK_NO."] * 100000) +
            np.random.randint(-100000, 100000, size=len(df))
        )
    return df

if uploaded_file is not None:
    data = load_data(uploaded_file)
else:
    st.warning("Please upload a `Test.csv` file to proceed.")
    st.stop()

# Features and target
features = ["UNDER_CONSTRUCTION", "RERA", "BHK_NO.", "SQUARE_FT", "READY_TO_MOVE", "RESALE"]
target = "PRICE"

# Sidebar inputs for prediction
st.sidebar.header("🏗️ House Features for Prediction")

def user_input():
    inputs = {}
    for feat in features:
        if feat in ["UNDER_CONSTRUCTION", "RERA", "READY_TO_MOVE", "RESALE"]:
            inputs[feat] = st.sidebar.selectbox(f"{feat.replace('_', ' ').title()}", [0, 1])
        elif feat == "BHK_NO.":
            inputs[feat] = st.sidebar.slider("Number of BHK", 1, 10, 2)
        elif feat == "SQUARE_FT":
            inputs[feat] = st.sidebar.slider("Square Feet", 300, 5000, 1200)
    return pd.DataFrame([inputs])

input_df = user_input()

# Validate inputs vs data ranges
if input_df["SQUARE_FT"].iloc[0] > data["SQUARE_FT"].max():
    st.sidebar.warning("Entered square footage is unusually high compared to dataset.")
if input_df["BHK_NO."].iloc[0] > data["BHK_NO."].max():
    st.sidebar.warning("Entered BHK number is unusually high compared to dataset.")

# Model selection
st.sidebar.header("⚙️ Select Model")
model_choice = st.sidebar.radio("Choose regression model", ("Linear Regression", "Random Forest"))

X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if model_choice == "Linear Regression":
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    prediction = model.predict(input_df)[0]
    fi_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.coef_
    }).sort_values(by="Importance", key=abs, ascending=False)
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    prediction = model.predict(input_df)[0]
    fi_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Display metrics in columns
col1, col2, col3 = st.columns(3)
col1.metric("🏠 Predicted Price", f"₹ {prediction:,.0f}")
col2.metric("📉 RMSE", f"₹ {rmse:,.0f}")
col3.metric("🎯 R² Score", f"{r2:.2f}")

st.markdown("---")
st.subheader("📈 Actual vs Predicted Prices")
fig_avp, ax_avp = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax_avp)
ax_avp.set_xlabel("Actual Price")
ax_avp.set_ylabel("Predicted Price")
ax_avp.set_title("Actual vs Predicted House Prices")
st.pyplot(fig_avp)

# Confidence interval
if model_choice == "Random Forest":
    preds_per_tree = np.array([t.predict(input_df)[0] for t in model.estimators_])
    lower = np.percentile(preds_per_tree, 5)
    upper = np.percentile(preds_per_tree, 95)
else:
    residual_std = np.std(y_test - y_pred)
    lower = prediction - residual_std
    upper = prediction + residual_std

st.markdown("### 🔍 Prediction Confidence Interval")
st.info(f"Estimated range: ₹ {lower:,.0f} to ₹ {upper:,.0f}")

# Feature importance plot
st.subheader("🔑 Feature Importance")
fig_fi, ax_fi = plt.subplots()
sns.barplot(data=fi_df, x="Importance", y="Feature", palette="viridis", ax=ax_fi)
ax_fi.set_title("Feature Importance")
st.pyplot(fig_fi)

# SHAP explainability for Random Forest only
if model_choice == "Random Forest":
    st.subheader("🧠 Model Explanation (SHAP values)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    st.write("SHAP Summary Plot (global feature impact)")
    fig_shap, ax_shap = plt.subplots()
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False, max_display=10)
    st.pyplot(fig_shap)

# Download prediction result
st.markdown("---")
st.subheader("💾 Download Your Prediction")
result_df = input_df.copy()
result_df[target] = prediction
csv = result_df.to_csv(index=False)
st.download_button(label="Download prediction as CSV", data=csv, file_name="house_price_prediction.csv", mime="text/csv")

# Sample CSV download
st.sidebar.markdown("---")
st.sidebar.markdown("📥 Need a sample?")
sample_csv = data[features].head(10).to_csv(index=False)
st.sidebar.download_button("Download Sample CSV", sample_csv, "sample_input.csv", "text/csv")

# Data visualizations and preview tabs
tab1, tab2 = st.tabs(["📊 Visualizations", "🔍 Data Preview"])

with tab1:
    st.header("📊 Data Visualizations")

    st.subheader("💹 Price Distribution")
    fig_hist, ax_hist = plt.subplots()
    sns.histplot(data["PRICE"], kde=True, ax=ax_hist, color="purple")
    st.pyplot(fig_hist)

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

    # Pie charts for categorical variables
    st.subheader("🏘️ Property Characteristics Distribution (Pie Charts)")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # Ready to Move pie chart
    with col1:
        st.markdown("**🏗️ Ready to Move vs Under Construction**")
        rt_counts = data["READY_TO_MOVE"].value_counts().sort_index()
        labels = ["Under Construction", "Ready to Move"]
        fig1, ax1 = plt.subplots()
        ax1.pie(rt_counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#FF9999", "#99FF99"])
        ax1.axis("equal")
        st.pyplot(fig1)

    # Resale pie chart
    with col2:
        st.markdown("**🔄 New vs Resale Property**")
        resale_counts = data["RESALE"].value_counts().sort_index()
        labels = ["New Property", "Resale Property"]
        fig2, ax2 = plt.subplots()
        ax2.pie(resale_counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#66B2FF", "#FFCC99"])
        ax2.axis("equal")
        st.pyplot(fig2)

    # Under Construction pie chart
    with col3:
        st.markdown("**🚧 Under Construction Status**")
        uc_counts = data["UNDER_CONSTRUCTION"].value_counts().sort_index()
        labels = ["No", "Yes"]
        fig3, ax3 = plt.subplots()
        ax3.pie(uc_counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#FFD700", "#FFA07A"])
        ax3.axis("equal")
        st.pyplot(fig3)

    # RERA pie chart
    with col4:
        st.markdown("**✅ RERA Approved**")
        rera_counts = data["RERA"].value_counts().sort_index()
        labels = ["No", "Yes"]
        fig4, ax4 = plt.subplots()
        ax4.pie(rera_counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#B0C4DE", "#4682B4"])
        ax4.axis("equal")
        st.pyplot(fig4)

with tab2:
    st.header("🔍 Data Preview")
    st.dataframe(data.head(10))
