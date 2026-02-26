import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Trend Analysis Dashboard", layout="wide")

st.title("📊 Interactive Trend Analysis & Forecasting Dashboard")
st.markdown("Final Year Project - Time Series Forecasting with Machine Learning & User Input System")

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_trend_data.csv")
    df['Month'] = pd.to_datetime(df['Month'])
    return df

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = joblib.load("revenue_forecast_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features

df = load_data()
model, features = load_model()

# ================= SIDEBAR =================
st.sidebar.header("🔍 Dashboard Controls")
industry = st.sidebar.selectbox("Select Industry", df["Industry"].unique())

filtered_df = df[df["Industry"] == industry].sort_values("Month")

# ================= KPI SECTION =================
st.subheader(f"📌 Key Performance Indicators - {industry}")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Revenue", f"${filtered_df['Revenue'].sum():,.0f}")
col2.metric("Total Customers", f"{int(filtered_df['Customers'].sum()):,}")
col3.metric("Total Orders", f"{int(filtered_df['Orders'].sum()):,}")
col4.metric("Total Profit", f"${filtered_df['Profit'].sum():,.0f}")

# ================= TREND CHART =================
st.subheader("📈 Monthly Revenue Trend")
fig1, ax1 = plt.subplots(figsize=(10,5))
sns.lineplot(data=filtered_df, x="Month", y="Revenue", marker="o", ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

# ================= INDUSTRY COMPARISON =================
st.subheader("🏭 Industry Comparison")
fig2, ax2 = plt.subplots(figsize=(12,6))
sns.lineplot(data=df, x="Month", y="Revenue", hue="Industry", ax=ax2)
plt.xticks(rotation=45)
st.pyplot(fig2)

# ================= CORRELATION =================
st.subheader("🔥 Correlation Heatmap")
fig3, ax3 = plt.subplots()
sns.heatmap(
    filtered_df[['Revenue','Customers','Orders','Profit']].corr(),
    annot=True, cmap="coolwarm", ax=ax3
)
st.pyplot(fig3)

# =========================================================
# 🧠 REAL USER INPUT SECTION (MOST IMPORTANT - INTERACTIVE)
# =========================================================
st.markdown("---")
st.header("🧮 Real-Time Revenue Prediction (User Input)")

st.markdown("""
Enter your real business data for the current month.  
The machine learning model will predict the **next month's revenue**.
""")

# Get latest historical values (for smart defaults)
latest_row = filtered_df.iloc[-1]

col1, col2 = st.columns(2)

with col1:
    customers_input = st.number_input(
        "👥 Number of Customers",
        min_value=0,
        value=int(latest_row["Customers"]),
        help="Enter expected customers for current month"
    )

    orders_input = st.number_input(
        "📦 Number of Orders",
        min_value=0,
        value=int(latest_row["Orders"]),
        help="Enter expected orders"
    )

with col2:
    profit_input = st.number_input(
        "💰 Profit ($)",
        min_value=0.0,
        value=float(latest_row["Profit"]),
        help="Enter estimated profit"
    )

    revenue_input = st.number_input(
        "💵 Current Month Revenue ($)",
        min_value=0.0,
        value=float(latest_row["Revenue"]),
        help="Used as lag feature for time-series prediction"
    )

# Predict Button
if st.button("🔮 Predict Next Month Revenue", use_container_width=True):

    try:
        # Create input dictionary based on trained features
        input_data = {
            "Customers": customers_input,
            "Orders": orders_input,
            "Profit": profit_input,
            "Revenue_Lag_1": revenue_input,
            "Revenue_Lag_2": revenue_input,
            "Revenue_Lag_3": revenue_input
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure correct feature order
        input_df = input_df[features]

        # Make prediction
        prediction = model.predict(input_df)[0]

        # ================= OUTPUT DISPLAY =================
        st.success(f"📈 Predicted Next Month Revenue: ${prediction:,.2f}")

        # Business Insight Logic (VERY GOOD FOR VIVA)
        historical_avg = filtered_df["Revenue"].mean()

        if prediction > historical_avg:
            st.info("📊 Insight: Expected growth compared to historical average.")
        elif prediction < historical_avg:
            st.warning("📉 Insight: Possible decline in revenue trend.")
        else:
            st.info("📊 Insight: Stable revenue trend expected.")

        # Comparison Chart
        st.subheader("📊 Prediction vs Historical Average")
        compare_df = pd.DataFrame({
            "Category": ["Historical Average", "Predicted Revenue"],
            "Revenue": [historical_avg, prediction]
        })

        fig4, ax4 = plt.subplots()
        sns.barplot(data=compare_df, x="Category", y="Revenue", ax=ax4)
        st.pyplot(fig4)

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# =========================================================
# 🔮 AUTOMATIC 6-MONTH FORECAST
# =========================================================
st.markdown("---")
st.header("🔮 6-Month Automatic Revenue Forecast")

industry_df = filtered_df.copy()

for lag in [1, 2, 3]:
    industry_df[f"Revenue_Lag_{lag}"] = industry_df["Revenue"].shift(lag)

industry_df = industry_df.dropna()

if len(industry_df) > 0:
    last_row = industry_df.iloc[-1:].copy()
    future_predictions = []
    future_months = []

    for i in range(6):
        X_future = last_row[features]
        next_revenue = model.predict(X_future)[0]

        next_month = pd.to_datetime(last_row["Month"].values[0]) + pd.DateOffset(months=1)

        future_predictions.append(next_revenue)
        future_months.append(next_month)

        new_row = last_row.copy()
        new_row["Month"] = next_month
        new_row["Revenue_Lag_3"] = new_row["Revenue_Lag_2"]
        new_row["Revenue_Lag_2"] = new_row["Revenue_Lag_1"]
        new_row["Revenue_Lag_1"] = next_revenue
        new_row["Revenue"] = next_revenue

        last_row = new_row

    forecast_df = pd.DataFrame({
        "Month": future_months,
        "Forecasted Revenue": future_predictions
    })

    st.dataframe(forecast_df, use_container_width=True)

    fig5, ax5 = plt.subplots(figsize=(10,5))
    ax5.plot(filtered_df["Month"], filtered_df["Revenue"], label="Historical", marker="o")
    ax5.plot(forecast_df["Month"], forecast_df["Forecasted Revenue"],
             label="Forecast", linestyle="--", marker="o")
    ax5.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig5)

else:
    st.error("Not enough data for forecasting.")