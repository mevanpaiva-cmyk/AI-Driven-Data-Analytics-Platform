def build_insight_prompt(metrics: dict) -> str:
    return f"""
You are a data analytics expert.

You are given PRE-COMPUTED statistics.
Do NOT perform calculations.
Do NOT invent numbers.

Metrics:
- Total records: {metrics['total_records']}
- Anomalous records: {metrics['anomaly_count']} ({metrics['anomaly_pct']}%)
- Overall trend change: {metrics['trend_pct']}%
- Forecast error (MAPE): {metrics['mape']}%
- Recent improvement: {metrics['recent_improvement']}%

Write 4 professional bullet-point insights.
Be cautious when error is high.
"""
