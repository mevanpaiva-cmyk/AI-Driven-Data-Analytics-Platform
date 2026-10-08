import pandas as pd
import sqlite3
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "retail_analytics.db"

# ----------------------------
# Load raw data from database
# ----------------------------
print("Loading raw data from database...")

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM raw_sales_data", conn)

print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

# ----------------------------
# Identify numeric columns
# ----------------------------
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("Numeric columns detected:")
print(numeric_cols)

# ----------------------------
# Handle missing values using KNN Imputer
# ----------------------------
print("Applying KNN Imputation...")

imputer = KNNImputer(n_neighbors=5)

df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

print("Missing values handled.")

# ----------------------------
# Anomaly detection using Isolation Forest
# ----------------------------
print("Detecting anomalies...")

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
)

df["anomaly_flag"] = iso_forest.fit_predict(df[numeric_cols])
df["anomaly_flag"] = df["anomaly_flag"].map({1: 0, -1: 1})

print(f"Anomalies detected: {df['anomaly_flag'].sum()}")

# ----------------------------
# Store cleaned data
# ----------------------------
df.to_sql(
    name="cleaned_sales_data",
    con=conn,
    if_exists="replace",
    index=False
)

conn.close()

print("Cleaned data successfully stored in database.")
