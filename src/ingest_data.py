import pandas as pd
import sqlite3
from pathlib import Path

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "retail_store_sales.csv"
DB_PATH = BASE_DIR / "database" / "retail_analytics.db"

# ----------------------------
# Load CSV
# ----------------------------
print("Loading raw dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# ----------------------------
# Basic schema validation
# ----------------------------
required_columns = [
    "Transaction ID",
    "Category",
    "Item",
    "Price Per Unit",
    "Quantity",
    "Total Spent",
    "Transaction Date"
]

missing_cols = [col for col in required_columns if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

print("Schema validation passed.")

# ----------------------------
# Date parsing (NO cleaning yet)
# ----------------------------
df["Transaction Date"] = pd.to_datetime(
    df["Transaction Date"],
    errors="coerce"
)

print("Date parsing complete.")

# ----------------------------
# Ensure database directory exists
# ----------------------------
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Store raw data in SQLite
# ----------------------------
conn = sqlite3.connect(DB_PATH)

df.to_sql(
    name="raw_sales_data",
    con=conn,
    if_exists="replace",
    index=False
)

conn.close()

print("Raw data successfully stored in database.")
print(f"Database location: {DB_PATH}")
