import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- Page Settings ---
st.set_page_config(page_title="FairSplit AI Demo")

# --- Title ---
st.title("⚡ FairSplit AI Demo")
st.caption("By Students, For Students | Powered by Real Data")

# --- Load Data ---
df = pd.read_csv("6-Month_Updated_Room_Energy_Usage_Data.csv")
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

st.header("📊 How It Works")
st.markdown("""
**FairSplit AI** uses real or simulated energy usage (in kWh) to:

1. ⚡ **Measure** each room’s energy use  
2. 💰 **Calculate** their share of the total bill  
3. 📊 **Compare** to an equal split  
4. 💡 **Show** who overpays or saves

👉 Use the slider to simulate different bill totals!
""")

# --- Sidebar Slider for Bill Total ---
bill_total = st.sidebar.slider("Select the total monthly electricity bill (€):", 80, 200, 120)

# --- Fair Split Calculation ---
st.subheader("🏠 FairSplit vs Equal Split")
df_totals = df.groupby("room_id")["kwh_used"].sum().reset_index()
total_kwh = df_totals["kwh_used"].sum()
df_totals["fair_split"] = (df_totals["kwh_used"] / total_kwh) * bill_total
df_totals["equal_split"] = bill_total / df_totals.shape[0]
df_totals["difference"] = df_totals["equal_split"] - df_totals["fair_split"]

st.dataframe(df_totals[['room_id', 'kwh_used', 'fair_split', 'equal_split', 'difference']])

fig1, ax1 = plt.subplots()
bar_width = 0.35
x = np.arange(len(df_totals["room_id"]))
ax1.bar(x, df_totals["equal_split"], width=bar_width, label="Equal Split")
ax1.bar(x + bar_width, df_totals["fair_split"], width=bar_width, label="Fair Split")
ax1.set_xticks(x + bar_width / 2)
ax1.set_xticklabels(df_totals["room_id"])
ax1.set_ylabel("Amount (€)")
ax1.set_title("FairSplit vs Equal Split per Room")
ax1.legend()
st.pyplot(fig1)

# --- Forecasting Section ---
st.subheader("📈 15-Day Energy Forecast for Room A")
room_a = df[df['room_id'] == 'Room A']
x = room_a[['room_size', 'occupancy_hours', 'device_count', 'avg_temp']]
y = room_a['kwh_used']
X_train, X_test = x[:-15], x[-15:]
y_train, y_test = y[:-15], y[-15:]
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

fig2, ax2 = plt.subplots()
ax2.plot(room_a['date'].iloc[-15:], y_test.values, label='Actual', marker='o')
ax2.plot(room_a['date'].iloc[-15:], predictions, label='Forecast', marker='x')
ax2.set_title("Room A: Forecasted vs Actual Energy Usage")
ax2.set_ylabel("kWh Used")
ax2.set_xlabel("Date")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

st.caption("⚠️ Forecast is simulated using room-level features and linear regression.")

