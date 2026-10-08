# ==================================================
# REGION vs SALES ANALYSIS MODULE
# ==================================================
# Purpose:
# - Detect region & sales columns (heuristic-based)
# - Compute overview and contribution metrics
# - Prepare heatmap-ready data
# - Prepare structured summaries for LLM insights
# - Generate AI-driven insights & limitations
#
# NOTE:
# - NO Streamlit imports here
# - NO plotting here
# - This file does logic only
# ==================================================  

import pandas as pd

# --------------------------------------------------
# COLUMN DETECTION
# --------------------------------------------------

def detect_sales_column(df: pd.DataFrame):
    """
    Detects the primary sales/volume column.
    Uses keyword matching with variance fallback.
    """
    keywords = ["sales", "revenue", "volume", "total", "units", "quantity"]

    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) == 0:
        return None

    # Keyword-based detection
    for col in numeric_cols:
        col_lower = col.lower()
        if any(k in col_lower for k in keywords):
            return col

    # Fallback: highest variance numeric column
    return df[numeric_cols].var().idxmax()


def detect_region_column(df: pd.DataFrame):
    """
    Detects a region-like categorical column.
    Uses keyword priority with cardinality fallback.
    """
    priority_keywords = ["region", "state", "zone", "area", "market"]
    object_cols = df.select_dtypes(include=["object"]).columns

    # Keyword-first detection
    for col in object_cols:
        col_lower = col.lower()
        if any(k in col_lower for k in priority_keywords):
            return col

    # Fallback: reasonable categorical cardinality
    for col in object_cols:
        nunique = df[col].nunique()
        if 2 < nunique < max(100, int(len(df) * 0.5)):
            return col

    return None

# ==================================================
# REGION vs SALES — DATA CLEANING MODULE
# ==================================================
# Purpose:
# - Clean region & sales columns for non-temporal analysis
# - Ensure categorical consistency and numeric validity
# - Remove invalid, empty, or misleading records
#
# NOTE:
# - NO Streamlit imports
# - NO plotting
# - NO time-based assumptions
# ==================================================

import pandas as pd
# --------------------------------------------------
# CORE COMPUTATIONS
# --------------------------------------------------

def region_overview(df: pd.DataFrame, region_col: str, sales_col: str):
    """
    Aggregates total sales per region.
    """
    overview_df = (
        df.groupby(region_col)[sales_col]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    return overview_df


def region_trends(overview_df: pd.DataFrame, sales_col: str):
    """
    Computes contribution percentage per region.
    """
    df = overview_df.copy()
    total_sales = df[sales_col].sum()

    if total_sales == 0:
        df["Contribution (%)"] = 0.0
    else:
        df["Contribution (%)"] = (df[sales_col] / total_sales) * 100

    return df


# --------------------------------------------------
# HEATMAP DATA PREPARATION
# --------------------------------------------------

def region_heatmap_data(
    overview_df: pd.DataFrame,
    region_col: str,
    sales_col: str
):
    """
    Prepares heatmap-ready matrix using raw sales.
    Rows = regions, Column = sales
    """
    return (
        overview_df
        .sort_values(sales_col, ascending=False)
        .set_index(region_col)[[sales_col]]
    )


def region_contribution_heatmap(
    overview_df: pd.DataFrame,
    region_col: str,
    sales_col: str
):
    """
    Prepares heatmap-ready matrix using contribution percentage.
    """
    df = overview_df.copy()
    total_sales = df[sales_col].sum()

    if total_sales == 0:
        df["Contribution (%)"] = 0.0
    else:
        df["Contribution (%)"] = (df[sales_col] / total_sales) * 100

    return (
        df
        .sort_values(sales_col, ascending=False)
        .set_index(region_col)[["Contribution (%)"]]
    )


# --------------------------------------------------
# LLM PREPARATION
# --------------------------------------------------

def prepare_region_summary(
    overview_df: pd.DataFrame,
    region_col: str,
    sales_col: str,
    top_n: int = 5
):
    """
    Creates a compact, structured summary for LLM interpretation.
    """
    summary = {
        "analysis_type": "Region vs Sales",
        "region_column": region_col,
        "sales_column": sales_col,
        "total_sales": float(overview_df[sales_col].sum()),
        "number_of_regions": int(overview_df.shape[0]),
        "top_regions": overview_df.head(top_n).to_dict(orient="records")
    }

    return summary


# --------------------------------------------------
# AI INSIGHTS (LLM-DRIVEN)
# --------------------------------------------------

def generate_region_ai_insights(
    llm_client,
    summary: dict,
    dataset_description: str
):
    """
    Uses an LLM to generate region-based insights.
    The output MUST follow strict section headers
    to enable UI styling.
    """

    prompt = f"""
You are a professional business data analyst.

Analyze the following regional sales information and respond using
EXACTLY the following section headers on separate lines,
with NO formatting:

Trend Explanation
Risk Interpretation
Seasonal Behavior
Business Recommendation

Rules:
- The dataset contains NO time, date, or seasonal information
- The business context below is descriptive only, not temporal
- Do NOT mention months, seasons, quarters, or time periods
- Do NOT use words like winter, summer, annual, monthly, or yearly
- Do NOT infer trends, growth, or changes over time
- Describe differences ONLY across regions
- Each section MUST contain at least 2 complete sentences
- Do NOT leave any section empty
- Do NOT create sub-headings, lists, or extra titles inside sections
- Do NOT repeat headers inside sections
- Do NOT use markdown (no **, no bullets)
- Use plain text only
- Base insights strictly on the provided data and context
- Do NOT assume external market factors
- Keep the tone analytical and professional

Business Context (User-Provided, Non-Temporal):
{dataset_description}

Regional Sales Summary:
- Total Sales: {summary['total_sales']}
- Number of Regions: {summary['number_of_regions']}
- Top Regions by Sales: {summary['top_regions']}
"""

    return llm_client(prompt)




# --------------------------------------------------
# FALLBACK LIMITATIONS (NON-LLM)
# --------------------------------------------------

def region_limitations_static():
    """
    Used only if LLM is unavailable.
    """
    return [
        "Region identification is based on heuristic detection and may vary across datasets.",
        "Regional aggregation hides intra-region temporal variations.",
        "Categorical analysis does not capture seasonality or time-based trends.",
        "Forecasting is not performed unless sufficient time-series data exists per region."
    ]
