"""
Anomaly Detection
Uses Isolation Forest to detect anomalous rows in any numeric dataset.
No label column needed — fully unsupervised.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from app.models.insights import generate_insights


def run_anomaly_detection(df: pd.DataFrame, contamination: float = 0.05) -> dict:
    """
    contamination: expected fraction of anomalies (0.01–0.5). Default 5%.
    """
    df = df.copy()
    X = _preprocess(df)

    if X.shape[1] == 0:
        raise ValueError("No usable numeric columns found after preprocessing.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    labels = model.fit_predict(X_scaled)           # 1 = normal, -1 = anomaly
    scores = model.score_samples(X_scaled)         # lower = more anomalous

    is_anomaly = (labels == -1)
    anomaly_indices = np.where(is_anomaly)[0].tolist()
    anomaly_count   = int(is_anomaly.sum())
    normal_count    = int((~is_anomaly).sum())

    # Normalise anomaly score 0–1 (1 = most anomalous)
    norm_scores = _normalize_scores(scores)

    # Top 20 most anomalous rows with details
    top_anomalies = sorted(
        [{"row": int(i), "anomaly_score": round(float(norm_scores[i]), 4),
          "values": {c: _safe_val(df.iloc[i][c]) for c in df.columns[:8]}}
         for i in anomaly_indices],
        key=lambda x: x["anomaly_score"], reverse=True
    )[:20]

    # Column-level anomaly contribution (mean absolute z-score for anomaly rows vs normal)
    contributions = _column_contributions(X, is_anomaly)

    return {
        "type": "anomaly_detection",
        "total_rows": len(df),
        "anomaly_count": anomaly_count,
        "normal_count": normal_count,
        "anomaly_rate_pct": round(anomaly_count / len(df) * 100, 2),
        "top_anomalies": top_anomalies,
        "column_contributions": contributions,
        "all_scores": [
            {"row": int(i), "anomaly_score": round(float(norm_scores[i]), 4), "is_anomaly": bool(is_anomaly[i])}
            for i in range(len(df))
        ][:500],
        "insights": generate_insights("anomaly", {"total_rows": len(df), "anomaly_count": anomaly_count, "normal_count": normal_count, "anomaly_rate_pct": round(anomaly_count/len(df)*100,2), "column_contributions": contributions, "top_anomalies": sorted([{"row": int(i), "anomaly_score": round(float(norm_scores[i]),4)} for i in anomaly_indices], key=lambda x: x["anomaly_score"], reverse=True)[:5]}),
    }


def _normalize_scores(scores):
    mn, mx = scores.min(), scores.max()
    if mx == mn:
        return np.zeros_like(scores)
    return 1 - (scores - mn) / (mx - mn)   # invert: high = more anomalous


def _column_contributions(X: pd.DataFrame, is_anomaly: np.ndarray):
    if is_anomaly.sum() == 0:
        return []
    scaler = StandardScaler()
    Zs = pd.DataFrame(np.abs(scaler.fit_transform(X)), columns=X.columns)
    anomaly_means = Zs[is_anomaly].mean()
    normal_means  = Zs[~is_anomaly].mean()
    diff = (anomaly_means - normal_means).sort_values(ascending=False)
    return [{"column": col, "contribution": round(float(val), 4)} for col, val in diff.head(8).items()]


def _anomaly_insights(anomaly_count, total, contributions):
    rate = round(anomaly_count / total * 100, 1)
    top_col = contributions[0]["column"] if contributions else "unknown"
    insights = [
        f"{anomaly_count} anomalous records detected ({rate}% of dataset).",
        f"Column '{top_col}' shows the highest deviation in anomalous rows.",
    ]
    if rate > 10:
        insights.append("Anomaly rate exceeds 10% — consider reviewing data quality or adjusting contamination threshold.")
    else:
        insights.append("Anomaly rate is within a normal range. Focus review on the top flagged rows.")
    return insights


def _preprocess(X: pd.DataFrame) -> pd.DataFrame:
    X = X.loc[:, X.isnull().mean() < 0.6]
    for col in X.select_dtypes(include="object").columns:
        if X[col].nunique() <= 20:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        else:
            X = X.drop(columns=[col])
    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)
    X = X.select_dtypes(include=[np.number])
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return X


def _safe_val(v):
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 4)
    return str(v)