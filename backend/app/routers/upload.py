"""
upload.py - Handles file uploads and saves metadata to SQLite uploads table.
"""

import uuid, os, shutil
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.database import get_connection

router = APIRouter()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".json"}

@router.post("/")
async def upload_dataset(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use CSV, XLSX, or JSON.")

    session_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{session_id}{ext}")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        df = _load_file(save_path, ext)
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(400, f"Could not parse file: {e}")

    # Save to SQLite
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO uploads (session_id, filename, file_rows, file_cols) VALUES (?, ?, ?, ?)",
            (session_id, file.filename, len(df), len(df.columns))
        )
        conn.commit()
    finally:
        conn.close()

    return JSONResponse({
        "session_id": session_id,
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(5).fillna("").to_dict(orient="records"),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    })

@router.get("/history")
def upload_history():
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM uploads ORDER BY uploaded_at DESC").fetchall()
        return {"uploads": [dict(r) for r in rows], "total": len(rows)}
    finally:
        conn.close()

@router.get("/{session_id}/columns")
def get_columns(session_id: str):
    path = _find_file(session_id)
    ext  = os.path.splitext(path)[1].lower()
    df   = _load_file(path, ext)
    return {"columns": list(df.columns), "dtypes": {c: str(t) for c, t in df.dtypes.items()}}


@router.post("/{session_id}/clean")
def clean_dataset(session_id: str):
    """Run automated cleaning pipeline on uploaded dataset."""
    from app.models.cleaner import run_cleaning
    path = _find_file(session_id)
    ext  = os.path.splitext(path)[1].lower()
    df   = _load_file(path, ext)

    report, cleaned_df = run_cleaning(df)

    # Save cleaned file (overwrite original)
    if ext == ".csv":
        cleaned_df.to_csv(path, index=False)
    elif ext == ".xlsx":
        cleaned_df.to_excel(path, index=False)
    elif ext == ".json":
        cleaned_df.to_json(path, orient="records")

    # Update DB with new row/col counts
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE uploads SET file_rows=?, file_cols=? WHERE session_id=?",
            (len(cleaned_df), len(cleaned_df.columns), session_id)
        )
        conn.commit()
    finally:
        conn.close()

    return JSONResponse({"session_id": session_id, "report": report})

def _find_file(session_id: str) -> str:
    for ext in ALLOWED_EXTENSIONS:
        p = os.path.join(UPLOAD_DIR, f"{session_id}{ext}")
        if os.path.exists(p):
            return p
    raise HTTPException(404, "Session not found.")

def _load_file(path: str, ext: str) -> pd.DataFrame:
    if ext == ".csv": return pd.read_csv(path)
    elif ext == ".xlsx": return pd.read_excel(path)
    elif ext == ".json": return pd.read_json(path)
    raise ValueError(f"Unknown extension: {ext}")
