import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# --- Page Config ---
st.set_page_config(page_title="FairSplit AI", layout="wide")

# --- App Header ---
st.markdown("""
<style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: #f5b700;
        text-align: center;
        margin-bottom: 0px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #dddddd;
        margin-top: 0px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚡ FairSplit AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">By Students, For Students | Powered by Real Data</div>', unsafe_allow_html=True)
st.markdown("---")

# --- Load Data ---
st.header("📂 Data Preview")
df = pd.read_csv("6-Month_Updated_Room_Energy_Usage_Data.csv")
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
st.dataframe(df.head())

# --- Summary Statistics ---
st.header("📊 Summary Statistics")
total_kwh = df['kwh_used'].sum()
st.metric("Total Energy Used (kWh)", f"{total_kwh:.2f}")
st.write(df.describe())

# --- Correlation Heatmap ---
st.header("📈 Correlation Heatmap")
corr = df[['room_size', 'occupancy_hours', 'device_count', 'avg_temp', 'kwh_used']].corr()

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# --- Regression Model ---
st.header("🧮 OLS Regression: What Drives Energy Use?")
with st.expander("Show regression output"):
    X = df[['room_size', 'occupancy_hours', 'device_count', 'avg_temp']]
    y = df['kwh_used']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    st.text(model.summary())

# --- Forecasting ---
st.header("🔮 15-Day Forecast for Room A")

room_a = df[df['room_id'] == 'Room A'].copy()
room_a['date'] = pd.to_datetime(room_a['date'], dayfirst=True)

X = room_a[['room_size', 'occupancy_hours', 'device_count', 'avg_temp']]
y = room_a['kwh_used']
X_train, X_test = X.iloc[:-15], X.iloc[-15:]
y_train, y_test = y.iloc[:-15], y.iloc[-15:]

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

st.metric("R² Score", f"{r2:.2f}")
st.metric("Mean Squared Error", f"{mse:.2f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(room_a['date'].iloc[-15:], y_test.values, label='Actual', marker='o')
ax.plot(room_a['date'].iloc[-15:], predictions, label='Forecast', marker='x')
ax.set_title("Room A: Forecasted vs Actual Energy Usage")
ax.set_ylabel("kWh Used")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.caption("⚠️ Forecast is simulated using room-level features and linear regression.")
