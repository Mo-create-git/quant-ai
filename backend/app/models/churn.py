import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, f1_score as sk_f1,
                             precision_score as sk_precision,
                             recall_score as sk_recall)
from sklearn.impute import SimpleImputer
from app.models.insights import generate_insights

def run_churn_prediction(df: pd.DataFrame, target_col: str) -> dict:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    df = df.copy()
    y = df[target_col].copy()
    if y.dtype == object or str(y.dtype) == "bool":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), name=target_col)
    X = df.drop(columns=[target_col])
    X = _preprocess(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() == 2 else None
    )
    model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    try:
        auc = round(roc_auc_score(y_test, y_pred_proba), 4)
    except Exception:
        auc = None
    report   = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    f1_val   = round(float(sk_f1(y_test, y_pred, average="weighted", zero_division=0)), 4)
    prec_val = round(float(sk_precision(y_test, y_pred, average="weighted", zero_division=0)), 4)
    rec_val  = round(float(sk_recall(y_test, y_pred, average="weighted", zero_division=0)), 4)
    cm       = confusion_matrix(y_test, y_pred).tolist()
    all_proba = model.predict_proba(X)[:, 1]
    feat_imp = sorted(
        zip(X.columns.tolist(), model.feature_importances_.tolist()),
        key=lambda x: x[1], reverse=True
    )[:10]
    high_risk   = int((all_proba >= 0.70).sum())
    medium_risk = int(((all_proba >= 0.40) & (all_proba < 0.70)).sum())
    low_risk    = int((all_proba < 0.40).sum())
    return {
        "type": "churn",
        "accuracy":  round(report.get("accuracy", 0), 4),
        "auc_roc":   auc,
        "f1_score":  f1_val,
        "precision": prec_val,
        "recall":    rec_val,
        "confusion_matrix": cm,
        "high_risk_count":   high_risk,
        "medium_risk_count": medium_risk,
        "low_risk_count":    low_risk,
        "total_rows": len(df),
        "feature_importances": [{"feature": f, "importance": round(i, 4)} for f, i in feat_imp],
        "predictions": [
            {"row": int(i), "churn_probability": round(float(p), 4),
             "risk_level": "High" if p >= 0.70 else "Medium" if p >= 0.40 else "Low"}
            for i, p in enumerate(all_proba)
        ][:500],
        "insights": generate_insights("churn", {
            "total_rows": len(df), "high_risk_count": high_risk,
            "high_risk_pct": round(high_risk/len(df)*100, 1),
            "medium_risk_count": medium_risk, "low_risk_count": low_risk,
            "accuracy": round(report.get("accuracy", 0), 4), "auc_roc": auc,
            "feature_importances": [{"feature": f, "importance": round(i,4)} for f,i in feat_imp]
        }),
    }

def _preprocess(X: pd.DataFrame) -> pd.DataFrame:
    X = X.loc[:, X.isnull().mean() < 0.6]
    for col in X.select_dtypes(include="object").columns:
        if X[col].nunique() <= 20:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        else:
            X = X.drop(columns=[col])
    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return X
