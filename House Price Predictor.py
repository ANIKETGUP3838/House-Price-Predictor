import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
import io
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="🏠 House Price Predictor", layout="wide")
st.title("🏠 House Price Prediction App")

# Sidebar: Upload CSV
st.sidebar.header("📁 Upload Your CSV")
uploaded_file = st.sidebar.file_uploader("Upload 'Test.csv'", type=["csv"])

@st.cache_data
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
    st.warning("Please upload a Test.csv file to proceed.")
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
        if feat in ["UNDER_CONSTRUCTION", "RERA", "READY_TO_MOVE", "RESALE"]:
            inputs[feat] = st.sidebar.selectbox(f"{feat.replace('_', ' ').title()}",
                                                [0, 1],
                                                help=feature_tooltips[feat])
        elif feat == "BHK_NO.":
            inputs[feat] = st.sidebar.slider("Number of BHK", 1, 5, 2, help=feature_tooltips[feat])
        elif feat == "SQUARE_FT":
            inputs[feat] = st.sidebar.slider("Square Feet", 300, 5000, 1200, help=feature_tooltips[feat])
    return pd.DataFrame([inputs])

input_df = user_input()

# Model selection
st.sidebar.header("⚙️ Select Model")
model_choice = st.sidebar.radio("Choose regression model", ("Linear Regression", "Random Forest"))

X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_random_forest(X_train, y_train):
    rf = RandomForestRegressor(random_state=42)
    param_dist = {
        "n_estimators": [100],  # reduced for speed
        "max_depth": [None, 10],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
        "max_features": ["sqrt"]
    }
    rf_cv = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=5,
        cv=3,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )
    rf_cv.fit(X_train, y_train)
    return rf_cv.best_estimator_

with st.spinner("Training model..."):
    if model_choice == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        prediction = model.predict(input_df)[0]
        fi_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.coef_
        }).sort_values(by="Importance", key=abs, ascending=False)
        residual_std = np.std(y_test - y_pred)
        lower = prediction - residual_std
        upper = prediction + residual_std

    else:
        model = train_random_forest(X_train, y_train)
        y_pred = model.predict(X_test)
        prediction = model.predict(input_df)[0]
        fi_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        preds_per_tree = np.array([tree.predict(input_df)[0] for tree in model.estimators_])
        lower = np.percentile(preds_per_tree, 5)
        upper = np.percentile(preds_per_tree, 95)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("💰 Predicted House Price")
st.success(f"₹ {prediction:,.0f}")

st.markdown("---")
st.subheader("📊 Model Evaluation")
st.write(f"**Model:** {model_choice}")
st.write(f"**RMSE:** ₹ {rmse:,.0f}")
st.write(f"**R² Score:** {r2:.2f}")

st.markdown("### 🔍 Prediction Confidence Interval")
st.info(f"Estimated range: ₹ {lower:,.0f} to ₹ {upper:,.0f}")

# Feature importance plot
st.subheader("🔑 Feature Importance")
fig_fi, ax_fi = plt.subplots()
sns.barplot(data=fi_df, x="Importance", y="Feature", palette="viridis", ax=ax_fi)
ax_fi.set_title("Feature Importance")
st.pyplot(fig_fi)

# SHAP explainability for Random Forest
if model_choice == "Random Forest":
    st.subheader("🧠 Model Explanation (SHAP values)")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    st.write("SHAP Summary Plot (global feature impact)")
    fig_shap = shap.plots.bar(shap_values, max_display=10, show=False)
    st.pyplot(fig_shap)

    st.write("SHAP Waterfall Plot for Your Input (Local Explanation)")
    shap_input_values = explainer(input_df)
    fig_waterfall = shap.plots.waterfall(shap_input_values[0], show=False)
    st.pyplot(fig_waterfall)

# Download prediction result
st.markdown("---")
st.subheader("💾 Download Your Prediction")
result_df = input_df.copy()
result_df[target] = prediction
csv = result_df.to_csv(index=False)
st.download_button(label="Download prediction as CSV", data=csv, file_name="house_price_prediction.csv", mime="text/csv")

# Visualizations and data preview
tab1, tab2 = st.tabs(["📊 Visualizations", "🔍 Data Preview"])

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

    st.subheader("🏘️ Property Characteristics Distribution")
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        st.markdown("**🏗️ Ready to Move vs Under Construction**")
        rt_counts = data["READY_TO_MOVE"].value_counts().sort_index()
        labels = ["Under Construction", "Ready to Move"]
        fig1, ax1 = plt.subplots()
        ax1.pie(rt_counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#FF9999", "#99FF99"])
        ax1.axis("equal")
        st.pyplot(fig1)

    with col2:
        st.markdown("**🔄 New vs Resale Property**")
        resale_counts = data["RESALE"].value_counts().sort_index()
        labels = ["New Property", "Resale Property"]
        fig2, ax2 = plt.subplots()
        ax2.pie(resale_counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#66B2FF", "#FFCC99"])
        ax2.axis("equal")
        st.pyplot(fig2)

    with col3:
        st.markdown("**📋 RERA Approval**")
        rera_counts = data["RERA"].value_counts().sort_index()
        labels = ["Not Approved", "RERA Approved"]
        fig3, ax3 = plt.subplots()
        ax3.pie(rera_counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#FFB266", "#99FFCC"])
        ax3.axis("equal")
        st.pyplot(fig3)

    with col4:
        st.markdown("**🛏️ BHK Configuration**")
        bhk_counts = data["BHK_NO."].value_counts().sort_index()
        labels = [f"{int(i)} BHK" for i in bhk_counts.index]
        fig4, ax4 = plt.subplots()
        ax4.pie(bhk_counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=plt.cm.viridis(np.linspace(0, 1, len(bhk_counts))))
        ax4.axis("equal")
        st.pyplot(fig4)

with tab2:
    st.header("🔍 Sample Data")
    st.dataframe(data.head())
