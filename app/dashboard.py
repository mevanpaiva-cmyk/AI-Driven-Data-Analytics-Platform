import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
import plotly.express as px

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="AI-Driven Retail Analytics",
    layout="wide"
)

st.title("AI-Driven Retail Data Analytics Platform")

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "retail_analytics.db"

# ----------------------------
# Load data
# ----------------------------
conn = sqlite3.connect(DB_PATH)

raw_df = pd.read_sql("SELECT * FROM raw_sales_data", conn)
cleaned_df = pd.read_sql("SELECT * FROM cleaned_sales_data", conn)
pred_df = pd.read_sql("SELECT * FROM sales_predictions", conn)
insights_df = pd.read_sql("SELECT * FROM generated_insights", conn)

conn.close()

# ----------------------------
# Overview section
# ----------------------------
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(raw_df))
col2.metric("Cleaned Records", len(cleaned_df))
col3.metric("Detected Anomalies", int(cleaned_df["anomaly_flag"].sum()))

# ----------------------------
# Anomaly distribution
# ----------------------------
st.subheader("Anomaly Distribution")

anomaly_counts = cleaned_df["anomaly_flag"].value_counts().rename(
    index={0: "Normal", 1: "Anomaly"}
)

st.bar_chart(anomaly_counts)

# ----------------------------
# Sales prediction plot
# ----------------------------
st.subheader("Sales Prediction vs Actual")

fig = px.line(
    pred_df,
    x="date",
    y=["actual_sales", "predicted_sales"],
    labels={"value": "Sales Amount", "date": "Date"},
    title="Actual vs Predicted Sales"
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# AI Insights
# ----------------------------
st.subheader("AI-Generated Insights")

for _, row in insights_df.iterrows():
    st.markdown(f"- {row['insight_text']}")
