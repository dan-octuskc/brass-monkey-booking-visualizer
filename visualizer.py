import streamlit as st
import pandas as pd
import plotly.express as px

daily = pd.read_csv("booking_outputs/daily_snapshot.csv")
cum = pd.read_csv("booking_outputs/cumulative_asof_snapshot.csv")

st.title("Booking Lead-Time Dashboard")

# Lead-time curve for a chosen date
target = st.selectbox("Select booking date", sorted(cum['booking_date'].unique()))
subset = cum[cum['booking_date'] == target]
fig = px.line(subset, x="as_of_date", y="cum_guests", title=f"Cumulative Guests for {target}")
st.plotly_chart(fig)

# Week-over-week summary
wow = pd.read_csv("booking_outputs/week_over_week_latest.csv")
st.subheader("Week-over-Week Comparison")
st.dataframe(wow)
