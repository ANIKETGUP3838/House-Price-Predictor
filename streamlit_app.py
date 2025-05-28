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
import streamlit.components.v1 as components  # For embedding HTML (SHAP force plot)
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="🏠 House Price Predictor", layout="wide")

st.title("🏠 House Price Prediction App")

# Sidebar: Upload CSV
st.sidebar.header("📁 Upload Your CSV")
uploaded_file = st.sidebar.file_uploader("Upload 'Test.csv'", type=["csv"])

# Use @st.cache_data if supported, else fallback to @st.cache
try:
    cache_decorator = st.cache_data
except AttributeError:
    cache_decorator = st.cache

@cache_decorator
def load_data(file):
    df = pd.read_csv(file)
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

features = ["UNDER_CONSTRUCTION", "RERA", "BHK_NO.", "SQUARE_FT", "READY_TO_MOVE", "RESALE"]
target = "PRICE"

st.sidebar.header("🏗️ House Features for Prediction")

feature_tooltips = {
    "UNDER_CONSTRUCTION": "Is the property currently under construction? (0 = No, 1 = Yes)",
    "RERA": "Is the property RERA approved? (0 = No, 1 = Yes)",
    "BHK_NO.": "Number of bedrooms, halls, kitchens (BHK)",
    "SQUARE_FT": "Total square feet area",
    "READY_TO_MOVE": "Is the property ready to move in? (0 = No, 1 = Yes)",
    "RESALE": "Is this a resale property? (0 = No, 1 = Yes)"
}

def user_input():
    inputs = {}
    for feat in features:
        label = feat.replace('_', ' ').title()
        if feat in ["UNDER_CONSTRUCTION", "RERA", "READY_TO_MOVE", "RESALE"]:
            inputs[feat] = st.sidebar.selectbox(label, [0, 1], help=feature_tooltips[feat])
        elif feat == "BHK_NO.":
            inputs[feat] = st.sidebar.slider(label, 1, 5, 2, help=feature_tooltips[feat])
        elif feat == "SQUARE_FT":
            inputs[feat] = st.sidebar.slider(label, 300, 5000, 1200, help=feature_tooltips[feat])
    return pd.DataFrame([inputs])

input_df = user_input()

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

st.subheader("💰 Predicted House Price")
st.success(f"₹ {prediction:,.0f}")

st.markdown("---")
st.subheader("📊 Model Evaluation")
st.write(f"**Model:** {model_choice}")
st.write(f"**RMSE:** ₹ {rmse:,.0f}")
st.write(f"**R² Score:** {r2:.2f}")

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

# SHAP explanation only for Random Forest to save compute
if model_choice == "Random Forest":
    st.subheader("🧠 Model Explanation (SHAP values)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # SHAP summary plot (bar)
    st.write("SHAP Summary Plot (global feature impact)")
    fig_shap, ax_shap = plt.subplots()
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False, max_display=10)
    st.pyplot(fig_shap)

    # SHAP force plot for user input (local explanation)
    st.write("SHAP Force Plot for your input (local explanation)")
    shap.initjs()
    force_plot = shap.force_plot(
        explainer.expected_value,
        explainer.shap_values(input_df),
        input_df,
        matplotlib=False
    )
    # Render the JS force plot as HTML in Streamlit
    components.html(force_plot.data, height=300)

# Download prediction result
st.markdown("---")
st.subheader("💾 Download Your Prediction")
result_df = input_df.copy()
result_df[target] = prediction
csv = result_df.to_csv(index=False)
st.download_button(label="Download prediction as CSV", data=csv, file_name="house_price_prediction.csv", mime="text/csv")

# Simulate lat/lon if not present
if "LATITUDE" not in data.columns or "LONGITUDE" not in data.columns:
    np.random.seed(42)
    data["LATITUDE"] = 28.61 + np.random.normal(0, 0.02, size=len(data))
    data["LONGITUDE"] = 77.23 + np.random.normal(0, 0.02, size=len(data))

tab1, tab2, tab3 = st.tabs(["📊 Visualizations", "🔍 Data Preview", "🗺️ Map Visualization"])

with tab1:
    st.header("📊 Data Visualizations")

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

with tab2:
    st.header("🔍 Sample Data")
    st.dataframe(data.head())

with tab3:
    st.header("🗺️ House Locations & Prices")
    center = [data["LATITUDE"].mean(), data["LONGITUDE"].mean()]
    m = folium.Map(location=center, zoom_start=12)
    for _, row in data.iterrows():
        folium.CircleMarker(
            location=[row["LATITUDE"], row["LONGITUDE"]],
            radius=6,
            popup=(f"Price: ₹{row['PRICE']:,.0f}<br>"
                   f"BHK: {row['BHK_NO.']}<br>"
                   f"SqFt: {row['SQUARE_FT']}"),
            color='crimson',
            fill=True,
            fill_color='crimson',
            fill_opacity=0.6,
        ).add_to(m)
    st_folium(m, width=700, height=450)
