import pandas as pd
import sqlite3
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "retail_analytics.db"

# ----------------------------
# Load cleaned data
# ----------------------------
print("Loading cleaned data...")

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM cleaned_sales_data", conn)
conn.close()

# ----------------------------
# Aggregate daily sales
# ----------------------------
df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])

daily_sales = (
    df.groupby("Transaction Date")["Total Spent"]
    .sum()
    .reset_index()
    .sort_values("Transaction Date")
)

print(f"Daily records: {len(daily_sales)}")

# ----------------------------
# Prepare time series data
# ----------------------------
sales_values = daily_sales["Total Spent"].values.reshape(-1, 1)

scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(sales_values)

# ----------------------------
# Create sequences
# ----------------------------
def create_sequences(data, window=30):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

WINDOW_SIZE = 30
X, y = create_sequences(sales_scaled, WINDOW_SIZE)

# ----------------------------
# Train-test split
# ----------------------------
split_index = int(len(X) * 0.8)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# ----------------------------
# Build LSTM model
# ----------------------------
model = Sequential([
    LSTM(50, activation="relu", input_shape=(WINDOW_SIZE, 1)),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

print("Training LSTM model...")
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# ----------------------------
# Prediction
# ----------------------------
y_pred = model.predict(X_test)

# Inverse scaling
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(f"RMSE: {rmse:.2f}")

# ----------------------------
# Store predictions
# ----------------------------
prediction_dates = daily_sales["Transaction Date"].iloc[-len(y_test_inv):].reset_index(drop=True)

pred_df = pd.DataFrame({
    "date": prediction_dates,
    "actual_sales": y_test_inv.flatten(),
    "predicted_sales": y_pred_inv.flatten()
})

conn = sqlite3.connect(DB_PATH)
pred_df.to_sql(
    name="sales_predictions",
    con=conn,
    if_exists="replace",
    index=False
)
conn.close()

print("Predictions stored in database.")
