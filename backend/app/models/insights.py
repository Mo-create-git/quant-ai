"""
insights.py
───────────
Calls Claude API to generate real, intelligent business insights
from ML model results. Replaces hardcoded template strings.
"""
import os
import json
import urllib.request
import urllib.error

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

def generate_insights(model_type: str, context: dict) -> list[str]:
    """
    Generate 3 actionable business insights using Claude.
    Falls back to template insights if API key not set or call fails.
    """
    if not ANTHROPIC_API_KEY:
        return _fallback_insights(model_type, context)

    prompt = _build_prompt(model_type, context)

    try:
        payload = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}]
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01"
            },
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = data["content"][0]["text"].strip()

        # Parse numbered list from Claude response
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        insights = []
        for line in lines:
            # Remove numbering like "1.", "1)", "-", "•"
            clean = line.lstrip("0123456789.-•) ").strip()
            if clean and len(clean) > 20:
                insights.append(clean)
        return insights[:4] if insights else _fallback_insights(model_type, context)

    except Exception:
        return _fallback_insights(model_type, context)


def _build_prompt(model_type: str, ctx: dict) -> str:
    if model_type == "churn":
        return f"""You are a business analyst AI. A RandomForest churn prediction model just ran on a customer dataset.

Results:
- Total customers: {ctx.get('total_rows')}
- High churn risk: {ctx.get('high_risk_count')} ({ctx.get('high_risk_pct')}%)
- Medium churn risk: {ctx.get('medium_risk_count')}
- Low churn risk: {ctx.get('low_risk_count')}
- Model accuracy: {ctx.get('accuracy')}
- AUC-ROC score: {ctx.get('auc_roc')}
- Top churn drivers: {', '.join([f['feature'] for f in ctx.get('feature_importances', [])[:5]])}

Write exactly 3 concise, actionable business insights (1-2 sentences each). Be specific, use the numbers, and give real business advice — not generic statements. Format as a numbered list."""

    elif model_type == "sales":
        return f"""You are a business analyst AI. A GradientBoosting sales forecast model just ran on a dataset.

Results:
- Total records: {ctx.get('total_rows')}
- R² score: {ctx.get('r2_score')} (1.0 = perfect)
- Mean Absolute Error: {ctx.get('mae')}
- RMSE: {ctx.get('rmse')}
- Top revenue drivers: {', '.join([f['feature'] for f in ctx.get('feature_importances', [])[:5]])}
- 6-period forecast: {[f['forecast'] for f in ctx.get('forecast_next_6', [])]}

Write exactly 3 concise, actionable business insights (1-2 sentences each). Be specific with the numbers and forecast trend. Give real revenue growth advice. Format as a numbered list."""

    elif model_type == "anomaly":
        return f"""You are a business analyst AI. An IsolationForest anomaly detection model just ran on a dataset.

Results:
- Total records: {ctx.get('total_rows')}
- Anomalies detected: {ctx.get('anomaly_count')} ({ctx.get('anomaly_rate_pct')}% of data)
- Normal records: {ctx.get('normal_count')}
- Columns with highest anomaly contribution: {', '.join([c['column'] for c in ctx.get('column_contributions', [])[:4]])}
- Top anomaly scores: {[a['anomaly_score'] for a in ctx.get('top_anomalies', [])[:5]]}

Write exactly 3 concise, actionable business insights (1-2 sentences each). Be specific about what the anomalies might mean for the business and what to do next. Format as a numbered list."""

    return "Provide 3 business insights from this AI analysis result."


def _fallback_insights(model_type: str, ctx: dict) -> list[str]:
    """Used when API key is missing or call fails."""
    if model_type == "churn":
        high = ctx.get('high_risk_count', 0)
        total = ctx.get('total_rows', 1)
        pct = round(high / total * 100, 1)
        top = ctx.get('feature_importances', [{}])[0].get('feature', 'unknown')
        auc = ctx.get('auc_roc', 0)
        return [
            f"{pct}% of customers ({high}) are at high churn risk — prioritise immediate retention outreach for this segment.",
            f"'{top}' is the strongest churn predictor — focus product and support improvements here to reduce churn.",
            f"Model AUC of {auc} indicates {'strong' if auc and auc >= 0.8 else 'moderate'} predictive performance — {'reliable' if auc and auc >= 0.8 else 'consider adding more features'} for business decisions.",
        ]
    elif model_type == "sales":
        r2 = ctx.get('r2_score', 0)
        mae = ctx.get('mae', 0)
        top = ctx.get('feature_importances', [{}])[0].get('feature', 'unknown')
        forecast = ctx.get('forecast_next_6', [])
        trend = "upward" if forecast and forecast[-1]['forecast'] > forecast[0]['forecast'] else "downward"
        return [
            f"Sales forecast shows a {trend} trend over the next 6 periods — plan inventory and resources accordingly.",
            f"'{top}' is the top revenue driver — optimising this variable could yield the highest sales impact.",
            f"Model R² of {r2} with MAE of {mae} — {'high confidence' if r2 >= 0.75 else 'moderate confidence'} in these forecasts.",
        ]
    elif model_type == "anomaly":
        count = ctx.get('anomaly_count', 0)
        rate = ctx.get('anomaly_rate_pct', 0)
        top_col = ctx.get('column_contributions', [{}])[0].get('column', 'unknown')
        return [
            f"{count} anomalous records ({rate}%) detected — review these rows for data errors, fraud, or unusual activity.",
            f"Column '{top_col}' shows the highest deviation in anomalous records — investigate this field first.",
            f"{'Anomaly rate exceeds 10% — this is unusually high and warrants a full data quality review.' if rate > 10 else 'Anomaly rate is within a normal range — focus review efforts on the top-scored flagged rows.'}",
        ]
    return ["Analysis complete. Review the charts above for detailed results."]
