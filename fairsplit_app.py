import streamlit as st
import pandas as pd

st.title("FairSplit AI Demo")
st.caption("📊 This app shows how to fairly split energy bills.")

# Load the data
df = pd.read_csv("6-Month_Updated_Room_Energy_Usage_Data.csv")
st.write("Preview of data:")
st.dataframe(df.head())
# --- Summary Stats ---
st.subheader("📊 Summary Statistics")
total_kwh = df['kwh_used'].sum()
st.write(f"**Total Energy Used (kWh):** {total_kwh:.2f}")
st.write(df.describe())

import matplotlib.pyplot as plt
import seaborn as sns

st.subheader("📈 Correlation Heatmap")
corr = df[['room_size', 'occupancy_hours', 'device_count', 'avg_temp', 'kwh_used']].corr()

fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

import statsmodels.api as sm

st.subheader("📉 OLS Regression Model")

# Define features (X) and target (y)
X = df[['room_size', 'occupancy_hours', 'device_count', 'avg_temp']]
y = df['kwh_used']

# Add constant to X
X = sm.add_constant(X)

# Fit model
model = sm.OLS(y, X).fit()

# Display summary
st.text(model.summary())

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

st.subheader("📈 15-Day Energy Forecast for Room A")

# Filter Room A only
room_a = df[df['room_id'] == 'Room A'].copy()

# Convert date
room_a['date'] = pd.to_datetime(room_a['date'], dayfirst=True)

# Prepare features and target
X = room_a[['room_size', 'occupancy_hours', 'device_count', 'avg_temp']]
y = room_a['kwh_used']

# Train/test split (last 15 days as test set)
X_train, X_test = X.iloc[:-15], X.iloc[-15:]
y_train, y_test = y.iloc[:-15], y.iloc[-15:]

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

st.write(f"**R² Score:** {r2:.2f}")
st.write(f"**MSE:** {mse:.2f}")

# Plot predictions vs actual
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(room_a['date'].iloc[-15:], y_test.values, label='Actual', marker='o')
ax.plot(room_a['date'].iloc[-15:], predictions, label='Predicted', marker='x')
ax.set_title("15-Day Forecast vs Actual (Room A)")
ax.set_ylabel("Energy (kWh)")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True)

st.pyplot(fig)

st.caption("⚠️ Forecast is simulated based on room features and past data.")
