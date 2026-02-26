# 📊 Trend Dashboard – Industry Trend Analysis & Revenue Forecasting

## 🎓 Final Year Project

This project is an interactive **Trend Analysis Dashboard** developed using Machine Learning and Streamlit to analyze time-based industry trends, compare performance across sectors, and forecast future revenue.

The system provides clear visualization of 48 months of time-series data and predicts the next 6 months of revenue using a regression-based forecasting model.

---

## 🚀 Live Features

* 📈 Time-Series Trend Analysis (48 Months)
* 🏭 Industry Comparison Dashboard
* 🔮 6-Month Revenue Forecast (Machine Learning)
* 📊 KPI Metrics (Revenue, Customers, Orders, Profit)
* 🔥 Correlation Analysis Heatmap
* 🎯 Interactive Filters by Industry

---

## 🏭 Industries Included

* Retail (Seasonal Growth)
* Healthcare (Stable Growth)
* Technology (Rapid Growth)
* Transportation (Fluctuating Growth)
* Finance (Steady Linear Growth)

---

## 🧠 Machine Learning Model

* Model Type: Random Forest Regressor
* Forecasting Target: Revenue
* Evaluation Metrics:

  * RMSE (Root Mean Squared Error)
  * R² Score (Coefficient of Determination)

The trained model is saved using `joblib` and integrated into the Streamlit dashboard for real-time forecasting.

---

## 📂 Project Folder Structure

```
trend dashboard/
│── app.py
│── revenue_forecast_model.pkl
│── model_features.pkl
│── synthetic_trend_data.csv
│── requirements.txt
│── README.md
```

---

## ⚙️ Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* GitHub (Version Control)
* Streamlit Cloud (Deployment)

---

## 📊 Dataset Description

A synthetic time-series dataset was generated to simulate real-world industry performance over 48 months.

### Dataset Columns:

* Industry
* Month
* Revenue
* Customers
* Orders
* Profit

Each industry follows a different growth pattern to support realistic trend analysis and forecasting.

---

## 🧪 How to Run the Project Locally (Step-by-Step)

### Step 1: Clone the Repository

```
git clone https://github.com/your-username/trend-dashboard.git
```

### Step 2: Navigate to Project Folder

```
cd "trend dashboard"
```

### Step 3: Install Required Libraries

```
pip install -r requirements.txt
```

### Step 4: Run Streamlit App

```
streamlit run app.py
```

The dashboard will open in your browser at:

```
http://localhost:8501
```

---

## 🌍 Streamlit Cloud Deployment

This project is deployed using Streamlit Cloud for live dashboard access.
The deployment is connected to the GitHub repository for automatic updates and reproducibility.

---

## 📈 Dashboard Functional Modules

### System Components:

1. Data Loading Module
2. Exploratory Data Analysis (EDA)
3. Time-Series Feature Engineering (Lag Features)
4. Machine Learning Forecasting Model
5. Interactive Visualization Dashboard

---

## 🎓 Academic Significance (For Viva)

This project demonstrates the integration of:

* Time-Series Analytics
* Machine Learning Forecasting
* Business Intelligence Visualization
* Cloud Deployment
* Version Control using GitHub

The dashboard supports data-driven decision-making by providing trend insights, industry benchmarking, and predictive analytics.

---

## 🔮 Forecasting Capability

The system predicts future revenue for the next 6 months using historical lag-based features, enabling:

* Strategic Planning
* Budget Forecasting
* Industry Performance Analysis
* Trend Monitoring

---

## 👩‍💻 Author

Final Year Undergraduate Project
Trend Analysis & Forecasting Dashboard using Machine Learning and Streamlit

---

## 📜 License

This project is developed for academic and educational purposes.
