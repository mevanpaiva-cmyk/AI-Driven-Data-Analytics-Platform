 
#==================================================
#OPIK CONFIG (MUST BE FIRST NO EXCEPTIONS)
#==================================================
import os
os.environ["OPIK_URL_OVERRIDE"] = "http://localhost:5173/api"

import opik
opik.Opik()
from opik import track

#==================================================
#IMPORTS
#==================================================
import streamlit as st
import pandas as pd
import numpy as np
import requests
import region_sales_analysis as rsa

from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import plotly.express as px

#==================================================
#PAGE CONFIG
#==================================================
st.set_page_config(
    page_title="AI-Driven Data Analytics Platform",
    layout="wide"
)
st.markdown(
    """
<style>
.metric-card {
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.45);
    background: #020617;
    border-left: 6px solid transparent;
}

.metric-title {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 600;
    color: #e5e7eb;
}

/* Accent themes */
.accent-blue {
    border-left-color: #3b82f6;
}

.accent-green {
    border-left-color: #22c55e;
}

.accent-red {
    border-left-color: #ef4444;
}

.accent-purple {
    border-left-color: #a855f7;
}

.accent-amber {
    border-left-color: #f59e0b;
}
</style>
""",
    unsafe_allow_html=True
)


st.markdown(
    """
<style>
/* ===== AI Interpretation – Top Divider Style ===== */

.ai-container {
    margin-top: 36px;
}

.ai-header {
    font-size: 1.35rem;
    font-weight: 600;
    color: #e5e7eb;
    margin-bottom: 12px;
}

.ai-panel {
    position: relative;
    background: #020617;
    border-radius: 18px;
    padding: 26px 30px;
    box-shadow: 0 14px 36px rgba(0,0,0,0.45);
}

/* Top gradient divider */
.ai-panel::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    height: 5px;
    width: 100%;
    background: linear-gradient(90deg, #6366f1, #22c55e);
    border-top-left-radius: 18px;
    border-top-right-radius: 18px;
}

.ai-panel h3 {
    font-size: 1.15rem;
    margin-top: 1.6rem;
    margin-bottom: 0.6rem;
    color: #e5e7eb;
}

.ai-panel p,
.ai-panel li {
    color: #d1d5db;
    line-height: 1.7;
    font-size: 0.95rem;
}

.ai-panel ul {
    padding-left: 1.4rem;
    margin-top: 0.5rem;
}

.ai-panel li {
    margin-bottom: 0.7rem;
}
/* ===== AI Insight Tiles ===== */

.ai-section {
    border-radius: 14px;
    padding: 18px 20px;
    margin-top: 18px;
    box-shadow: inset 0 0 0 1px rgba(148,163,184,0.08);
}

/* Individual section themes */
.ai-trend {
    background: rgba(59,130,246,0.08);
    border-left: 4px solid #3b82f6;
    border-right: 4px solid #3b82f6;
}

.ai-risk {
    background: rgba(239,68,68,0.08);
    border-left: 4px solid #ef4444;
    border-right: 4px solid #ef4444;
}

.ai-seasonal {
    background: rgba(34,197,94,0.08);
    border-left: 4px solid #22c55e;
    border-right: 4px solid #22c55e;
}

.ai-recommendation {
    background: rgba(168,85,247,0.08);
    border-left: 4px solid #a855f7;
    border-right: 4px solid #a855f7;
}


.ai-section h3 {
    margin-top: 0;
    margin-bottom: 8px;
    font-size: 1.1rem;
    color: #e5e7eb;
}
.metric-outline {
    border-radius: 14px;
    padding: 18px 22px;
    background: #020617;
    border: 2px solid;
    box-shadow: none;
}

.metric-outline .metric-title {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-bottom: 6px;
}

.metric-outline .metric-value {
    font-size: 1.6rem;
    font-weight: 600;
    color: #e5e7eb;
}

/* Outline color variants */
.outline-blue {
    border-color: #3b82f6;
}

.outline-purple {
    border-color: #a855f7;
}

/* ===== Forecast Control Box (Enhanced Typography) ===== */
/* Style the container that holds Future Forecast Controls */
.forecast-box {
    background: #020617;
    border: 2px solid #3b82f6;
    border-radius: 18px;
    padding: 26px 30px;
    margin-top: 28px;
    margin-bottom: 20px;
}


/* Reduce vertical gap between stacked success/info alerts */
/* Tabs container */
div[data-testid="stTabs"] {
    margin-top: 0.5rem;
}

/* Tab button container */
button[data-baseweb="tab"] {
    padding: 1rem 1.8rem !important;
}

/* THIS is the actual tab text */
button[data-baseweb="tab"] span {
    font-size: 3.6rem !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
}

/* Active tab text – bigger */
button[data-baseweb="tab"][aria-selected="true"] span {
    font-size: 3.2rem !important;
    font-weight: 800 !important;
}

/* ===============================
   NUCLEAR OPTION: FILE UPLOADER
   =============================== */

/* Entire uploader block */
section[data-testid="stFileUploader"] {
    margin-bottom: 0rem !important;
    padding-bottom: 0rem !important;
}

/* Kill all children margins */
section[data-testid="stFileUploader"] * {
    margin-bottom: 0rem !important;
}

/* Kill the spacer Streamlit inserts after uploader */
section[data-testid="stFileUploader"] + div {
    height: 0px !important;
    margin: 0px !important;
    padding: 0px !important;
}

/* Tabs immediately after uploader */
div[data-testid="stTabs"] {
    margin-top: 0rem !important;
    padding-top: 0rem !important;
}

.limitations-box {
    background: #020617;
    border-radius: 18px;
    padding: 26px 30px;
    margin-top: 20px;
    box-shadow: 0 14px 36px rgba(0,0,0,0.45);
    border-left: 6px solid #ef4444;
}

.limitations-box h3 {
    color: #e5e7eb;
    margin-bottom: 14px;
    font-size: 1.25rem;
}

.limitations-box p {
    color: #d1d5db;
    font-size: 0.95rem;
    line-height: 1.7;
}

.limitations-box li {
    margin-bottom: 12px;
    color: #e5e7eb;
    line-height: 1.6;
}

</style>
""",
    unsafe_allow_html=True
)


st.title("AI-Driven Data Analytics Platform")
st.caption("Upload a time-based sales CSV. The system analyzes monthly trends and generates business-grade insights.")

#==================================================
#USER GUIDANCE + CONTEXT INPUT
#==================================================
st.info(
    "📌 CSV Requirements\n\n"
    "- Must contain a date / time column\n"
    "- Must contain numeric sales or volume columns\n"
    "- Data should be ordered or orderable by time\n"
)

dataset_description = st.text_area(
    "Dataset Description (business context)",
    placeholder="Example: This dataset contains monthly avocado sales volumes across US regions.",
    height=120,
    key="dataset_description_input"
)

#==================================================
#FILE UPLOAD
#==================================================
uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"],
    key="csv_uploader_main"
)

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
st.session_state.setdefault("analysis_mode", "Time & Sales")


#==================================================
#STEP 2: ANALYSIS MODE SELECTOR
#==================================================
analysis_mode = st.selectbox(
    "Select Analysis Type",
    ["Time & Sales", "Region & Sales"],
    key="analysis_mode"
)

styled_ai_text = None
styled_region_ai_text = None


def normalize_headings(text):
    headings = [
    "Trend Explanation",
    "Risk Interpretation",
    "Seasonal Behavior",
    "Regional Behavior",
    "Business Recommendation"
    ]

    for h in headings:
        text = text.replace(h + ":", h)
        text = text.replace(h, f"\n\n{h}\n")

    return text

def style_region_sections(text):
    replacements = {
        "\nTrend Explanation\n":
            "<div class='ai-section ai-trend'><h3>Trend Explanation</h3>",
        "\nRisk Interpretation\n":
            "</div><div class='ai-section ai-risk'><h3>Risk Interpretation</h3>",
        "\nSeasonal Behavior\n":
            "</div><div class='ai-section ai-seasonal'><h3>Seasonal Behavior</h3>",
        "\nRegional Behavior\n":
            "</div><div class='ai-section ai-seasonal'><h3>Regional Behavior</h3>",
        "\nBusiness Recommendation\n":
            "</div><div class='ai-section ai-recommendation'><h3>Business Recommendation</h3>",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)
    return text + "</div>"
    
    
##################################################
# 🔵 STEP 3: TIME & SALES WALL (BIG BLOCK)
##################################################
if analysis_mode == "Time & Sales":

    # --------------------------------------------------
    # 3A: Date + Numeric checks
    # --------------------------------------------------
    def detect_date_column(df):
        keywords = ["date", "time", "year", "month"]
        for col in df.columns:
            if any(k in col.lower() for k in keywords):
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.3:
                    df[col] = parsed
                    return col
        return None

    date_col = detect_date_column(df)

    if date_col is None:
        st.error("No valid date column detected.")
        st.stop()

    df = df.dropna(subset=[date_col])
    df["date"] = pd.to_datetime(df[date_col])

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found.")
        st.stop()

    target_col = df[numeric_cols].var().idxmax()

    # --------------------------------------------------
    # 3B: Core Logic Processing
    # --------------------------------------------------
    
    # MISSING VALUE IMPUTATION
    imputer = KNNImputer(n_neighbors=5)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # ANOMALY DETECTION
    iso = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    df["anomaly_flag"] = iso.fit_predict(df[numeric_cols])
    df["anomaly_flag"] = df["anomaly_flag"].map({1: 0, -1: 1})

    # ANOMALY DISTRIBUTION GRAPH
    anomaly_counts = (
        df["anomaly_flag"]
        .value_counts()
        .rename(index={0: "Normal", 1: "Anomaly"})
        .reset_index()
    )
    anomaly_counts.columns = ["Type", "Count"]

    fig_anomaly = px.bar(
        anomaly_counts,
        x="Type",
        y="Count",
        text="Count",
        title="Distribution of Normal vs Anomalous Records"
    )
    fig_anomaly.update_layout(title_x=0.5)
    fig_anomaly.update_traces(textposition="outside")

    # MONTHLY LABEL FORMATTER
    def format_month_label(period):
        return period.strftime("%b %Y")

    # MONTHLY AGGREGATION
    df["month"] = df["date"].dt.to_period("M")
    monthly = (
        df.groupby("month")[target_col]
        .sum()
        .reset_index()
    )
    monthly["month_label"] = monthly["month"].apply(format_month_label)
    monthly["month_num"] = monthly["month"].dt.month

    # MONTHLY TIME-SERIES TREND
    fig_trend = px.line(
        monthly,
        x="month_label",
        y=target_col,
        markers=True,
        title="Monthly Performance Trend"
    )
    fig_trend.update_layout(
        title_x=0.5,
        xaxis=dict(tickangle=-45, tickfont=dict(size=11))
    )

    # HEAT MAP
    monthly["year"] = monthly["month"].dt.year
    monthly["month_name"] = monthly["month"].dt.strftime("%b")
    heatmap_df = monthly.pivot_table(
        index="year",
        columns="month_name",
        values=target_col,
        aggfunc="mean"
    )
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    heatmap_df = heatmap_df.reindex(columns=month_order)
    fig_heatmap = px.imshow(
        heatmap_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Seasonal Heatmap: Month vs Year"
    )
    fig_heatmap.update_layout(title_x=0.5)

    # LSTM MODEL
    values = monthly[target_col].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0.1, 0.9))
    scaled = scaler.fit_transform(values)

    def create_sequences(data, window=6):
        X, y = [], []
        for i in range(len(data) - window):
            X.append(data[i:i+window])
            y.append(data[i+window])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled,window=6)
    if X.ndim != 3 or X.shape[0] < 10:
        st.warning(
        "Not enough historical data to train LSTM.\n\n"
        "At least 7–10 time periods are required."
        )
        st.stop()

    split = len(X) // 2
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    timesteps = X_train.shape[1]
    features = X_train.shape[2]
    model = Sequential([
        LSTM(64, activation="relu", return_sequences=True,
             input_shape=(timesteps, features)),
        LSTM(32, activation="relu"),
        Dense(1)
    ])


    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=50, verbose=0)
    y_pred = model.predict(X_test)

    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)

    lstm_pred_df = pd.DataFrame({
        "month": monthly["month_label"].iloc[-len(y_test_inv):],
        "actual": y_test_inv.flatten(),
        "predicted": y_pred_inv.flatten()
    })

    # SARIMA MODEL
    series = monthly[target_col].astype(float)
    forecast_steps = len(lstm_pred_df)

    sarima_model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarima_result = sarima_model.fit(disp=False)
    sarima_pred = sarima_result.predict(
        start=len(series) - forecast_steps,
        end=len(series) - 1,
        dynamic=False
    )
    sarima_pred_df = pd.DataFrame({
        "month": monthly["month_label"].iloc[-forecast_steps:],
        "actual": series.iloc[-forecast_steps:].values,
        "predicted": sarima_pred.values
    })

    # METRICS CALCULATION
    lstm_rmse = np.sqrt(mean_squared_error(lstm_pred_df["actual"], lstm_pred_df["predicted"]))
    sarima_rmse = np.sqrt(mean_squared_error(sarima_pred_df["actual"], sarima_pred_df["predicted"]))
    lstm_mape = mean_absolute_percentage_error(lstm_pred_df["actual"], lstm_pred_df["predicted"]) * 100
    sarima_mape = mean_absolute_percentage_error(sarima_pred_df["actual"], sarima_pred_df["predicted"]) * 100

    # KEY MONTHLY METRICS
    mom = monthly[target_col].pct_change() * 100
    metrics = {
        "analysis_period": f"{monthly['month_label'].iloc[0]} to {monthly['month_label'].iloc[-1]}",
        "latest_month": monthly["month_label"].iloc[-1],
        "mom_change": round(mom.iloc[-1], 2),
        "highest_growth": f"{monthly['month_label'].iloc[mom.idxmax()]} ({mom.max():.2f}%)",
        "largest_decline": f"{monthly['month_label'].iloc[mom.idxmin()]} ({abs(mom.min()):.2f}%)"
    }

    # DATA STATS FOR DYNAMIC LOGIC (Limitations)
    row_count = df.shape[0]
    num_features = len(numeric_cols)
    # Recalculate missing percentage on the original df before imputation if possible, or use current logic
    missing_pct = df.isnull().mean().mean() * 100 
    date_range = (df["date"].max() - df["date"].min()).days
    volatility = monthly[target_col].pct_change().std()
    num_periods = monthly.shape[0]

    # OLLAMA INSIGHTS + OPIK TRACE
    def build_prompt(metrics, description):
        return f"""
    Dataset Context:
    {description}

    Key Monthly Metrics:
    Analysis Period: {metrics['analysis_period']}
    Latest Month: {metrics['latest_month']}
    MoM Change: {metrics['mom_change']}%
    Highest Growth Month: {metrics['highest_growth']}
    Largest Decline Month: {metrics['largest_decline']}

    Provide:
    Trend explanation
    Risk interpretation
    Seasonal behavior
    Business recommendation
    """

    def style_ai_sections(text, analysis_type="time"):
        replacements = {
            "Trend Explanation":
                "<div class='ai-section ai-trend'><h3>Trend Explanation</h3>",
            "Risk Interpretation":
                "</div><div class='ai-section ai-risk'><h3>Risk Interpretation</h3>",
            "Seasonal Behavior":
                "</div><div class='ai-section ai-seasonal'><h3>Seasonal Behavior</h3>",
            "Business Recommendation":
                "</div><div class='ai-section ai-recommendation'><h3>Business Recommendation</h3>"
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text + "</div>"

    @track(name="ollama_monthly_insights")
    def generate_insights(prompt):
        payload = {
            "model": "llama3.2:1b",
            "prompt": prompt,
            "stream": False
        }
        res = requests.post(
            "http://localhost:11434/api/generate",
            json=payload
        )
        return res.json()["response"]
        
    with st.spinner("Generating insights..."):
        ai_text = generate_insights(build_prompt(metrics, dataset_description))
        styled_ai_text = style_ai_sections(ai_text)  # analysis_type defaults to "time"

    # --------------------------------------------------
    # 3C: Time & Sales Tabs
    # --------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Trends & Seasonality",
        "Forecasting",
        "AI Insights",
        "Project Limitations"
    ])

    with tab1:
        with st.container():
            st.success("CSV uploaded successfully")
            st.success(f"Selected target column: **{target_col}**")

        st.subheader("Data Preview")
        st.dataframe(df.head())

        st.subheader("Anomaly Distribution")
        st.plotly_chart(fig_anomaly, use_container_width=True)

        st.subheader("Key Monthly Metrics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div class="metric-card accent-blue">
                    <div class="metric-title">Analysis Period</div>
                    <div class="metric-value">{metrics['analysis_period']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="metric-card accent-green">
                    <div class="metric-title">Highest Growth Month</div>
                    <div class="metric-value">{metrics['highest_growth']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card accent-purple">
                    <div class="metric-title">Latest Month</div>
                    <div class="metric-value">{metrics['latest_month']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="metric-card accent-amber">
                    <div class="metric-title">Largest Decline Month</div>
                    <div class="metric-value">{metrics['largest_decline']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            mom_class = "accent-green" if metrics["mom_change"] >= 0 else "accent-red"
            st.markdown(
                f"""
                <div class="metric-card {mom_class}">
                    <div class="metric-title">MoM Change</div>
                    <div class="metric-value">{metrics['mom_change']}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("<br><br>", unsafe_allow_html=True)

    with tab2:
        st.subheader("Monthly Time-Series Trend")
        st.plotly_chart(fig_trend, use_container_width=True)

        st.subheader("Seasonal Pattern Heatmap (Month vs Year)")
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with tab3:
        model_choice = st.selectbox(
            "Select Prediction Model",
            ["LSTM", "SARIMA"]
        )
        pred_df = lstm_pred_df if model_choice == "LSTM" else sarima_pred_df

        fig_pred = px.bar(
            pred_df.melt(id_vars="month", value_vars=["actual", "predicted"]),
            x="month",
            y="value",
            color="variable",
            barmode="group",
            title=f"Actual vs Predicted (Monthly) – {model_choice}"
        )
        fig_pred.update_layout(title_x=0.5)

        st.subheader("Model Performance")
        rmse_value = lstm_rmse if model_choice == "LSTM" else sarima_rmse
        mape_value = lstm_mape if model_choice == "LSTM" else sarima_mape
        outline_class = "outline-blue" if model_choice == "LSTM" else "outline-purple"

        col_rmse, col_mape = st.columns(2)

        with col_rmse:
            st.markdown(
                f"""
                <div class="metric-outline {outline_class}">
                    <div class="metric-title">Model RMSE</div>
                    <div class="metric-value">{rmse_value:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_mape:
            st.markdown(
                f"""
                <div class="metric-outline {outline_class}">
                    <div class="metric-title">Model MAPE</div>
                    <div class="metric-value">{mape_value:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.subheader("Actual vs Predicted")
        st.plotly_chart(
            fig_pred,
            use_container_width=True,
            key=f"actual_vs_predicted_{model_choice}"
        )
        st.subheader("Future Forecast Controls")

        forecast_label = st.radio(
            "Select Forecast Horizon",
            ["3 Months", "6 Months"],
            horizontal=True,
            key="forecast_horizon_tab"
        )

        # Convert label → number
        forecast_horizon = int(forecast_label.split()[0])

        # Recompute SARIMA forecast EVERY time
        future_forecast = sarima_result.forecast(steps=forecast_horizon)

        last_month = monthly["month"].iloc[-1].to_timestamp()

        future_dates = pd.date_range(
            start=last_month + pd.offsets.MonthBegin(),
            periods=forecast_horizon,
            freq="MS"
        )

        forecast_df = pd.DataFrame({
            "date": future_dates,
            "value": future_forecast.values,
            "type": "Forecast"
        })

        historical_df = monthly.copy()
        historical_df["date"] = historical_df["month"].dt.to_timestamp()
        historical_df["value"] = historical_df[target_col]
        historical_df["type"] = "Actual"

        combined_df = pd.concat([
            historical_df[["date", "value", "type"]],
            forecast_df
        ])

        fig_forecast = px.line(
            combined_df,
            x="date",
            y="value",
            color="type",
            line_dash="type",
            title=f"SARIMA Forecast – Next {forecast_horizon} Months"
        )

        st.plotly_chart(fig_forecast, use_container_width=True)

    with tab4:
        st.subheader("AI-Generated Interpretation")

        if styled_ai_text:
            st.markdown(
                "<div class='ai-panel'>" + styled_ai_text + "</div>",
                unsafe_allow_html=True
            )
        else:
            st.warning("AI insights unavailable.")


    with tab5:
        limitations = []
        st.markdown(
            """
            <div class="limitations-box">
                <h2>Project Limitations</h2>
                <p>Displayed limitations are dynamically determined based on the uploaded dataset.</p>
            """,
            unsafe_allow_html=True
        )

        # Dataset size
        if row_count < 1000:
            limitations.append(
                "• **Limited Data Volume**  \n"
                "The dataset contains a relatively small number of records, which may reduce the robustness of trend and forecast results."
            )
        elif row_count > 50000:
            limitations.append(
                "• **Scalability Constraints**  \n"
                "Large dataset size can increase model training and inference time, affecting real-time dashboard responsiveness."
            )

        # Missing values
        if missing_pct > 0:
            limitations.append(
                f"• **Missing Data Presence**  \n"
                f"Approximately {missing_pct:.2f}% of the dataset contains missing values, which may impact model accuracy despite preprocessing."
            )

        # Date range
        if date_range < 365:
            limitations.append(
                "• **Short Temporal Coverage**  \n"
                "The dataset spans less than one year, limiting the reliability of long-term seasonal pattern detection."
            )

        # Feature limitation
        if num_features <= 2:
            limitations.append(
                "• **Limited Feature Diversity**  \n"
                "The analysis relies on a small number of variables, reducing the ability to capture complex influencing factors."
            )

        # Always include uncertainty
        limitations.append(
            "• **Prediction Uncertainty**  \n"
            "Forecast outputs represent estimated trends and should be interpreted as decision support rather than exact outcomes."
        )
        limitations.append(
            "• **External Factor Exclusion**  \n"
            "The analysis does not incorporate external variables such as economic conditions, weather patterns, or policy changes that may influence trends."
        )
        limitations.append(
            "• **Model Assumption Dependency**  \n"
            "Forecasting models rely on historical patterns and assume continuity in trends, which may not hold during abrupt market shifts."
        )

        if volatility > 0.5:
            limitations.append(
                "• **High Variability in Observations**  \n"
                "The dataset shows significant fluctuations over time, which can increase uncertainty in trend analysis and long-term forecasting."
            )
            
        if num_periods < 24:
            limitations.append(
                "• **Limited Historical Depth**  \n"
                "The dataset contains a limited number of time periods, which may restrict the model’s ability to learn long-term trends and seasonality."
            )

        for lim in limitations:
            st.markdown(lim)
        
        st.markdown("</div>", unsafe_allow_html=True)


##################################################
# 🟣 STEP 4: REGION & SALES WALL (SECOND BIG BLOCK)
##################################################
elif analysis_mode == "Region & Sales":

    # --------------------------------------------------
    # 4A: Region column detection (FIRST thing inside)
    # --------------------------------------------------
    region_col = rsa.detect_region_column(df)
    sales_col = rsa.detect_sales_column(df)

    if region_col is None or sales_col is None:
        st.error("Could not detect required Region or Sales columns.")
        st.stop()
    
    # --------------------------------------------------
    # 4B: Cleaning + aggregation
    # --------------------------------------------------
    def clean_region_sales_data(df, region_col, sales_col):
        df = df.copy()
        # Remove rows with missing region or sales
        df = df.dropna(subset=[region_col, sales_col])
        # Standardize region names
        df[region_col] = (
            df[region_col]
            .astype(str)
            .str.strip()
            .str.title()
        )
        # Convert sales to numeric
        df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
        # Drop invalid sales values
        df = df.dropna(subset=[sales_col])
        df = df[df[sales_col] > 0]
        # Remove duplicate region-sales rows
        df = df.drop_duplicates(subset=[region_col, sales_col])
        return df

    clean_df = clean_region_sales_data(df, region_col, sales_col)

    # CORE AGGREGATION (ONCE)
    overview_df = rsa.region_overview(clean_df, region_col, sales_col)
    overview_df = rsa.region_trends(overview_df, sales_col)

    # Calculate metrics for display
    top_region = overview_df.iloc[0][region_col]
    least_region = overview_df.iloc[-1][region_col]
    total_sales = overview_df[sales_col].sum()
    total_regions = overview_df.shape[0]

    # --------------------------------------------------
    # 4C: Region Tabs
    # --------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Regional Analysis",
        "AI Insights",
        "Model Limitations"
    ])

    # =========================
    # TAB 1: OVERVIEW
    # =========================
    with tab1:
        st.success("CSV uploaded successfully")
        st.success(f"Detected Region Column: **{region_col}**")
        st.success(f"Detected Sales Column: **{sales_col}**")
        
        st.subheader("Regional Anomaly Distribution (IQR-based)")
        Q1 = overview_df[sales_col].quantile(0.25)
        Q3 = overview_df[sales_col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        overview_df["anomaly_flag"] = np.where(
            (overview_df[sales_col] < lower_bound) |
            (overview_df[sales_col] > upper_bound),
            "Anomaly",
            "Normal"
        )
        
        anomaly_counts = (
            overview_df["anomaly_flag"]
            .value_counts()
            .reset_index()
        )
        anomaly_counts.columns = ["Type", "Count"]

        fig_region_anomaly_dist = px.bar(
            anomaly_counts,
            x="Type",
            y="Count",
            text="Count",
            title="Distribution of Normal vs Anomalous Regions"
        )
        fig_region_anomaly_dist.update_layout(
            title_x=0.5,
            xaxis_title="Type",
            yaxis_title="Count",
            template="plotly_dark"
        )
        fig_region_anomaly_dist.update_traces(textposition="outside")

        st.plotly_chart(fig_region_anomaly_dist, use_container_width=True)
        st.markdown("<br><br>", unsafe_allow_html=True)

        st.subheader("Key Regional Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
                <div class="metric-card accent-blue">
                    <div class="metric-title">Total Regions</div>
                    <div class="metric-value">{total_regions}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card accent-purple">
                    <div class="metric-title">Total Sales</div>
                    <div class="metric-value">{total_sales:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"""
                <div class="metric-card accent-green">
                    <div class="metric-title">Top Sold Region</div>
                    <div class="metric-value">{top_region}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col4:
            st.markdown(
                f"""
                <div class="metric-card accent-amber">
                    <div class="metric-title">Least Sold Region</div>
                    <div class="metric-value">{least_region}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("<br><br>", unsafe_allow_html=True)    
        st.subheader(
            f"Rows before cleaning: {df.shape[0]} | Rows after cleaning: {clean_df.shape[0]}"
        )
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.subheader("Top Regions by Sales")
        st.dataframe(overview_df.head(10))

    # =========================
    # TAB 2: REGIONAL ANALYSIS
    # =========================
    with tab2:
        st.subheader("Sales Distribution by Region")

        metric_choice = st.selectbox(
            "Select Metric",
            ["Total Sales", "Contribution (%)"],
            key="region_metric_selector"
        )

        if metric_choice == "Contribution (%)":
            heatmap_df = rsa.region_contribution_heatmap(
                overview_df,
                region_col,
                sales_col
            )
            title = "Regional Contribution (%) Heatmap"
        else:
            heatmap_df = rsa.region_heatmap_data(
                overview_df,
                region_col,
                sales_col
            )
            title = "Regional Sales Heatmap"

        fig_heatmap = px.imshow(
            heatmap_df,
            aspect="auto",
            color_continuous_scale="Inferno",
            title=title
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.subheader("Regional Sales Ranking")
        fig_bar = px.bar(
            overview_df,
            x=region_col,
            y=sales_col,
            title="Total Sales by Region",
            text_auto=True
        )
        fig_bar.update_layout(
            xaxis_title="Region",
            yaxis_title="Total Sales",
            xaxis_tickangle=-30,
            template="plotly_dark"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # =========================
    # TAB 3: AI INSIGHTS
    # =========================
     
    with tab3:
        st.subheader("AI-Generated Regional Insights")

        summary = rsa.prepare_region_summary(
            overview_df,
            region_col,
            sales_col
        )

        def region_llm_client(prompt: str) -> str:
            payload = {
                "model": "llama3.2:1b",
                "prompt": prompt,
                "stream": False
            }
            res = requests.post(
                "http://localhost:11434/api/generate",
                json=payload
            )
            return res.json()["response"]

        try:
            with st.spinner("Generating insights..."):
                ai_text = rsa.generate_region_ai_insights(
                    region_llm_client,
                    summary,
                    dataset_description
                )

            # ✅ EXACTLY HERE
            ai_text = normalize_headings(ai_text)
            styled_ai_text = style_region_sections(ai_text)

            # ✅ AND ONLY THEN render
            st.markdown(
                "<div class='ai-panel'>" + styled_ai_text + "</div>",
                unsafe_allow_html=True
            )

        except Exception as e:
            st.warning(
                f"AI service unavailable: {e}. Showing static limitations instead."
            )
            for lim in rsa.region_limitations_static():
                st.markdown(f"- {lim}")

    # =========================
    # TAB 4: MODEL LIMITATIONS
    # =========================
    with tab4:
        st.markdown(
            """
            <div class="limitations-box">
                <h3>Analysis Limitations</h3>
            """,
            unsafe_allow_html=True
        )

        limitations = rsa.region_limitations_static()

        for lim in limitations:
            st.markdown(f"- {lim}")
        
        st.markdown("</div>", unsafe_allow_html=True)
 