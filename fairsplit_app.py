import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# --- Page Settings ---
st.set_page_config(page_title="FairSplit AI")

# --- Sidebar ---
with st.sidebar:
    st.header("📘 How It Works")
    st.markdown("""
    **FairSplit AI** uses real or simulated energy usage (in kWh) to:

    1. ⚡ **Measure** each room’s energy use
    2. 💶 **Calculate** their share of the total bill
    3. 📊 **Compare** to an equal split
    4. 💡 **Show** who overpays or saves

    👉 Use the slider to simulate different bill totals!
    """)

# --- Main Page ---
st.title("⚡ FairSplit AI Demo")
st.caption("By Students, For Students | Powered by Real Data")

st.markdown("""
<div style='background-color:#f9f2ec;padding:20px;border-radius:10px'>
<h3 style='color:#e69f00;'>🎓 Welcome to FairSplit AI</h3>
<p>This prototype demonstrates how electricity bills can be split more fairly in shared accommodation using real-time or simulated room-level energy usage data.</p>

<p>The data used here is <strong>sample data</strong> collected over a 6-month period, showing electricity usage across four rooms in a shared student house. While not real-time, the dataset closely mimics realistic consumption patterns and provides the foundation for testing our approach.</p>

<p style='color:#333;'>From our analysis, we’ve found:</p>
<ul>
<li>💸 Students often operate on very limited budgets</li>
<li>😤 Equal bill splitting can lead to disputes and unnecessary costs</li>
<li>🔍 FairSplit AI brings transparency, fairness, and savings to shared living</li>
</ul>

<p>Use the interactive elements below to explore how FairSplit AI makes billing transparent, personalized, and fair.</p>
</div>
""", unsafe_allow_html=True)

# --- Load Data ---
df = pd.read_csv("6-Month_Updated_Room_Energy_Usage_Data.csv")
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
room_totals = df.groupby('room_id')['kwh_used'].sum().reset_index()

# --- Fair Split Calculation ---
total_bill = st.slider("Select Monthly Bill (€):", min_value=80, max_value=200, value=120)
total_kwh = room_totals['kwh_used'].sum()
room_totals['fair_bill'] = (room_totals['kwh_used'] / total_kwh) * total_bill
room_totals['equal_split'] = total_bill / len(room_totals)

# --- Comparison Chart ---
st.subheader("💸 FairSplit vs Equal Split")
fig, ax = plt.subplots()
bar_width = 0.35
x = range(len(room_totals))
ax.bar(x, room_totals['equal_split'], bar_width, label='Equal Split (€)')
ax.bar([p + bar_width for p in x], room_totals['fair_bill'], bar_width, label='FairSplit (€)')
ax.set_xticks([p + bar_width / 2 for p in x])
ax.set_xticklabels(room_totals['room_id'])
ax.set_ylabel('Bill Amount (€)')
ax.set_title('Monthly Bill Comparison by Room')
ax.legend()
st.pyplot(fig)

# --- Forecast Section ---
st.subheader("📈 15-Day Energy Forecast")
selected_room = st.selectbox("Choose a room for forecast:", df['room_id'].unique())
room_data = df[df['room_id'] == selected_room].copy()
X = room_data[['room_size', 'occupancy_hours', 'device_count', 'avg_temp']]
y = room_data['kwh_used']
X_train, X_test = X[:-15], X[-15:]
y_train, y_test = y[:-15], y[-15:]

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

fig, ax = plt.subplots()
ax.plot(room_data['date'].iloc[-15:], y_test, label='Actual')
ax.plot(room_data['date'].iloc[-15:], predictions, label='Forecast', marker='x')
ax.set_title(f"{selected_room}: Forecasted vs Actual Energy Usage")
ax.set_ylabel("kWh Used")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.caption(f"⚠️ Forecast is simulated using room-level features and linear regression. R² = {r2:.2f}, MSE = {mse:.2f}")

# --- Testimonial ---
st.info("“I didn’t even realise how much extra I was paying until we used FairSplit. Love it!” – Sam, Student in Cork")


