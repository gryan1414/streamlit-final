import streamlit as st
import pandas as pd

st.title("FairSplit AI Demo")
st.caption("📊 This app shows how to fairly split energy bills.")

# Load the data
df = pd.read_csv("6-Month_Updated_Room_Energy_Usage_Data.csv")
st.write("Preview of data:")
st.dataframe(df.head())
