import pandas as pd
import sqlite3
from pathlib import Path

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "retail_analytics.db"

# ----------------------------
# Connect to database
# ----------------------------
conn = sqlite3.connect(DB_PATH)

# ----------------------------
# Load cleaned data (for anomalies)
# ----------------------------
cleaned_df = pd.read_sql(
    "SELECT * FROM cleaned_sales_data",
    conn
)

# ----------------------------
# Load predictions
# ----------------------------
pred_df = pd.read_sql(
    "SELECT * FROM sales_predictions",
    conn
)

insights = []

# ----------------------------
# Insight 1: Anomaly summary
# ----------------------------
anomaly_count = cleaned_df["anomaly_flag"].sum()
total_records = len(cleaned_df)

if anomaly_count > 0:
    insights.append(
        f"{anomaly_count} unusual transactions were detected "
        f"out of {total_records} records, indicating potential anomalies "
        f"such as sudden spikes or drops in sales."
    )

# ----------------------------
# Insight 2: Sales trend (prediction-based)
# ----------------------------
avg_actual = pred_df["actual_sales"].mean()
avg_predicted = pred_df["predicted_sales"].mean()

if avg_predicted > avg_actual:
    insights.append(
        "Sales are predicted to show an upward trend in the near future "
        "based on recent historical patterns."
    )
else:
    insights.append(
        "Sales are predicted to experience a slight decline or stagnation "
        "in the near future, indicating possible seasonal or demand-related effects."
    )

# ----------------------------
# Insight 3: Prediction error awareness
# ----------------------------
error = abs(avg_predicted - avg_actual)

insights.append(
    f"The forecasting model shows an average deviation of approximately "
    f"{error:.2f} units, which is acceptable for medium-scale retail prediction tasks."
)

# ----------------------------
# Store insights
# ----------------------------
insight_df = pd.DataFrame({
    "insight_id": range(1, len(insights) + 1),
    "insight_text": insights
})

insight_df.to_sql(
    name="generated_insights",
    con=conn,
    if_exists="replace",
    index=False
)

conn.close()

print("AI-generated insights stored successfully.")
