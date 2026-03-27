"""
Sales Forecasting
Uses a GradientBoosting regressor to predict a numeric sales/revenue column.
Also performs a simple time-series extrapolation if a date column is present.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from app.models.insights import generate_insights


def run_sales_forecast(df: pd.DataFrame, target_col: str, date_col: str = None) -> dict:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    df = df.copy()

    # Extract time features from date column if provided
    if date_col and date_col in df.columns:
        df = _extract_date_features(df, date_col)
        df = df.drop(columns=[date_col])

    y = pd.to_numeric(df[target_col], errors="coerce")
    X = df.drop(columns=[target_col])
    X = _preprocess(X)

    # Drop rows where target is NaN
    mask = y.notna()
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae   = round(float(mean_absolute_error(y_test, y_pred)), 4)
    rmse  = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
    r2    = round(float(r2_score(y_test, y_pred)), 4)
    mape  = round(float(np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1, y_test))) * 100), 2)
    train_score = round(float(model.score(X_train, y_train)), 4)
    test_score  = round(float(model.score(X_test, y_test)), 4)

    all_preds = model.predict(X)

    # Feature importances
    feat_imp = sorted(
        zip(X.columns.tolist(), model.feature_importances_.tolist()),
        key=lambda x: x[1], reverse=True
    )[:10]

    # Forecast next 6 periods (simple linear extrapolation on predictions)
    forecast = _forecast_next_periods(all_preds, periods=6)

    return {
        "type": "sales_forecast",
        "mae": mae,
        "rmse": rmse,
        "r2_score": r2,
        "mape": mape,
        "train_score": train_score,
        "test_score": test_score,
        "mape": mape,
        "train_score": train_score,
        "test_score": test_score,
        "total_rows": len(df),
        "feature_importances": [{"feature": f, "importance": round(i, 4)} for f, i in feat_imp],
        "predictions": [
            {"row": int(i), "predicted_value": round(float(v), 2), "actual_value": round(float(y.iloc[i]), 2)}
            for i, v in enumerate(all_preds)
        ][:500],
        "forecast_next_6": [
            {"period": f"Period +{i+1}", "forecast": round(float(v), 2)}
            for i, v in enumerate(forecast)
        ],
        "insights": generate_insights("sales", {"total_rows": len(df), "mae": mae, "rmse": rmse, "r2_score": r2, "feature_importances": [{"feature": f, "importance": round(i,4)} for f,i in feat_imp], "forecast_next_6": [{"period": f"Period +{i+1}", "forecast": round(float(v),2)} for i,v in enumerate(_forecast_next_periods(all_preds))]}),
    }


def _forecast_next_periods(preds, periods=6):
    """Simple trend extrapolation using the last 20% of predictions."""
    tail = preds[int(len(preds) * 0.8):]
    trend = np.polyfit(range(len(tail)), tail, 1)
    forecasts = []
    for i in range(1, periods + 1):
        val = np.polyval(trend, len(tail) + i)
        forecasts.append(max(val, 0))  # no negative sales
    return forecasts


def _extract_date_features(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["_year"]    = df[date_col].dt.year
    df["_month"]   = df[date_col].dt.month
    df["_quarter"] = df[date_col].dt.quarter
    df["_dayofweek"] = df[date_col].dt.dayofweek
    return df


def _sales_insights(mae, r2, feat_imp, preds, actual):
    top = feat_imp[0][0] if feat_imp else "unknown"
    avg_pred = round(float(np.mean(preds)), 2)
    trend = "upward 📈" if preds[-1] > preds[0] else "downward 📉"
    insights = [
        f"Overall sales trend is {trend} across the dataset.",
        f"Average predicted value: {avg_pred}. Mean Absolute Error: {mae}.",
        f"Top sales driver: '{top}'.",
    ]
    if r2 >= 0.75:
        insights.append(f"R² score of {r2} — the model explains most sales variance well.")
    else:
        insights.append(f"R² of {r2} suggests other factors may influence sales not captured in this data.")
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
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return X