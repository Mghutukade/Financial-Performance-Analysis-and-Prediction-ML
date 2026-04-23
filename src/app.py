import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Financial Dashboard", layout="wide")

# ---------------- LOAD MODEL ----------------
model = joblib.load("../model.pkl")

# ---------------- LOAD DATA ----------------
data = pd.read_csv("../data/Financials.csv")
data.columns = data.columns.str.strip()

cols = ['Units Sold','Manufacturing Price','Sale Price','Gross Sales',
        'Discounts','Sales','COGS','Profit']

for col in cols:
    data[col] = data[col].replace('[\$,]', '', regex=True)
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna()

# ---------------- TITLE ----------------
st.title("📊 Financial Performance Analysis & Prediction System")

# ---------------- KPIs ----------------
st.subheader("📌 Key Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"{data['Sales'].sum():,.0f}")
col2.metric("Total Profit", f"{data['Profit'].sum():,.0f}")
col3.metric("Average Profit", f"{data['Profit'].mean():,.2f}")

# ---------------- FILTER ----------------
st.subheader("🔍 Filter Data")

country = st.selectbox("Select Country", ["All"] + list(data['Country'].unique()))

if country != "All":
    filtered_data = data[data['Country'] == country]
else:
    filtered_data = data

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📈 Insights", "🤖 Models"])

# =========================================================
# 🔮 TAB 1: PREDICTION
# =========================================================
with tab1:
    st.subheader("Profit Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        units = st.number_input("Units Sold", value=1000.0)
    with col2:
        cogs = st.number_input("COGS", value=5000.0)
    with col3:
        sales = st.number_input("Sales", value=8000.0)

    if st.button("Predict Profit"):
        input_data = np.array([[units, cogs, sales]])
        prediction = model.predict(input_data)

        st.success(f"Predicted Profit: {prediction[0]:,.2f}")

        st.write("### Input vs Prediction")
        st.write({
            "Units Sold": units,
            "COGS": cogs,
            "Sales": sales,
            "Predicted Profit": prediction[0]
        })

# =========================================================
# 📈 TAB 2: INSIGHTS
# =========================================================
with tab2:
    st.subheader("📊 Data Insights")

    # 🔥 Fix index issue
    filtered_data = filtered_data.reset_index(drop=True)

    col1, col2 = st.columns(2)

    # Scatter
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.scatter(filtered_data['Sales'], filtered_data['Profit'])
        ax1.set_xlabel("Sales")
        ax1.set_ylabel("Profit")
        ax1.set_title("Sales vs Profit")
        st.pyplot(fig1)

    # Line Chart
    with col2:
        st.line_chart(filtered_data[['Sales', 'Profit']])

    # Top Products
    st.subheader("🏆 Top Performing Products")
    top_products = filtered_data.groupby('Product')['Profit'].sum().sort_values(ascending=False)
    st.bar_chart(top_products.head(5))

    # Country Analysis (FIXED)
    st.subheader("🌍 Profit by Country")
    country_profit = filtered_data.groupby('Country')['Profit'].sum()
    st.bar_chart(country_profit)

    # Data Table (CLEAN)
    st.subheader("📄 Dataset Preview")
    st.dataframe(filtered_data.head(50), use_container_width=True)

# =========================================================
# 🤖 TAB 3: MODELS (FIXED UI)
# =========================================================
with tab3:
    st.subheader("🤖 Model Analysis")

    col1, col2 = st.columns(2)

    # LEFT → Model Performance
    with col1:
        st.write("### 📊 Model Performance")

        with open("../scores.json") as f:
            scores_data = json.load(f)

        models = list(scores_data.keys())
        scores = list(scores_data.values())

        fig2, ax2 = plt.subplots()
        ax2.bar(models, scores)
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1)
        ax2.set_title("Model Comparison")

        st.pyplot(fig2)

        st.write("### 🧠 Interpretation")
        st.info("""
        Linear Regression performs best due to linear patterns in data.
        Random Forest and XGBoost handle complex relationships.
        """)

    # RIGHT → Feature Importance
    with col2:
        st.write("### 🔥 Feature Importance")

        features = ['Units Sold', 'COGS', 'Sales']
        importance = [0.0760, 0.0391, 0.8849]

        fig3, ax3 = plt.subplots()
        ax3.bar(features, importance)
        ax3.set_title("Impact on Profit")

        st.pyplot(fig3)

        st.write("### 💡 Insight")
        st.success("Sales is the most important factor influencing profit.")

    # FINAL SUMMARY
    st.markdown("---")
    st.subheader("📌 Final Conclusion")

    st.write("""
    ✔ Random Forest selected as final model  
    ✔ Sales is the dominant factor affecting profit  
    ✔ Dashboard enables real-time financial decision-making  
    """)

# ---------------- ABOUT ----------------
st.markdown("---")
st.subheader("📘 About Project")

st.write("""
This project analyzes financial data and predicts profit using machine learning models.
It integrates data preprocessing, model evaluation, and an interactive dashboard 
for business decision-making.
""")