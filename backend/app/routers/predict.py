import os, json
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.models.churn   import run_churn_prediction
from app.models.sales   import run_sales_forecast
from app.models.anomaly import run_anomaly_detection
from app.database import get_connection

router = APIRouter()
UPLOAD_DIR  = "uploads"
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".json"}

class ChurnRequest(BaseModel):
    session_id: str
    target_col: str = Field(..., description="Column with 0/1 churn labels")

class SalesRequest(BaseModel):
    session_id: str
    target_col: str
    date_col: Optional[str] = None

class AnomalyRequest(BaseModel):
    session_id: str
    contamination: float = Field(0.05, ge=0.01, le=0.5)

@router.post("/churn")
def predict_churn(req: ChurnRequest):
    df = _load_session(req.session_id)
    try:
        result = run_churn_prediction(df, req.target_col)
    except ValueError as e:
        raise HTTPException(400, str(e))
    _save_result(req.session_id, "churn", result)
    return result

@router.post("/sales")
def predict_sales(req: SalesRequest):
    df = _load_session(req.session_id)
    try:
        result = run_sales_forecast(df, req.target_col, req.date_col)
    except ValueError as e:
        raise HTTPException(400, str(e))
    _save_result(req.session_id, "sales", result)
    return result

@router.post("/anomaly")
def predict_anomaly(req: AnomalyRequest):
    df = _load_session(req.session_id)
    try:
        result = run_anomaly_detection(df, req.contamination)
    except ValueError as e:
        raise HTTPException(400, str(e))
    _save_result(req.session_id, "anomaly", result)
    return result

@router.post("/auto")
def predict_auto(session_id: str, model_type: str = "auto", target: str = None):
    df = _load_session(session_id)
    results = {}

    binary_cols = [c for c in df.columns if df[c].nunique() == 2]
    if binary_cols:
        try: results["churn"] = run_churn_prediction(df, binary_cols[0])
        except Exception as e: results["churn"] = {"error": str(e)}

    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in binary_cols]
    if numeric_cols:
        try: results["sales"] = run_sales_forecast(df, numeric_cols[0])
        except Exception as e: results["sales"] = {"error": str(e)}

    try: results["anomaly"] = run_anomaly_detection(df)
    except Exception as e: results["anomaly"] = {"error": str(e)}

    _save_result(session_id, "auto", results)
    return results

def _load_session(session_id: str) -> pd.DataFrame:
    for ext in ALLOWED_EXTENSIONS:
        p = os.path.join(UPLOAD_DIR, f"{session_id}{ext}")
        if os.path.exists(p):
            if ext == ".csv": return pd.read_csv(p)
            elif ext == ".xlsx": return pd.read_excel(p)
            elif ext == ".json": return pd.read_json(p)
    raise HTTPException(404, "Session not found.")

def _save_result(session_id: str, kind: str, data: dict):
    """Save prediction results to SQLite."""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO predictions (session_id, type, results_json) VALUES (?, ?, ?)",
            (session_id, kind, json.dumps(data, default=str))
        )
        conn.commit()
    finally:
        conn.close()
