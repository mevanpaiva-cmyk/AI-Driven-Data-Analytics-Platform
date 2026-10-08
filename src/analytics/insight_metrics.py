def compute_insight_metrics(df, anomaly_flags, forecast, actual):
    total_records = len(df)
    anomaly_count = int(anomaly_flags.sum())
    anomaly_pct = (anomaly_count / total_records) * 100

    trend_pct = (
        (df.iloc[-1].mean() - df.iloc[0].mean()) /
        abs(df.iloc[0].mean())
    ) * 100

    mape = (abs((actual - forecast) / actual).mean()) * 100

    recent_improvement = (
        (df.tail(10).mean().mean() - df.head(10).mean().mean()) /
        abs(df.head(10).mean().mean())
    ) * 100

    return {
        "total_records": total_records,
        "anomaly_count": anomaly_count,
        "anomaly_pct": round(anomaly_pct, 2),
        "trend_pct": round(trend_pct, 2),
        "mape": round(mape, 2),
        "recent_improvement": round(recent_improvement, 2),
    }
