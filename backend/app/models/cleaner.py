"""
cleaner.py
──────────
Automated data cleaning pipeline.
Handles: missing values, duplicates, type fixes, outliers, column name cleanup.
Returns a detailed cleaning report.
"""
import pandas as pd
import numpy as np

def run_cleaning(df: pd.DataFrame) -> dict:
    report = {
        "original_rows": len(df),
        "original_cols": len(df.columns),
        "steps": [],
        "cleaned_rows": 0,
        "cleaned_cols": 0,
        "issues_fixed": 0,
    }

    df = df.copy()

    # ── 1. Fix column names ────────────────────────
    old_cols = list(df.columns)
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r'[^a-z0-9_]', '_', regex=True)
                  .str.replace(r'_+', '_', regex=True)
                  .str.strip('_')
    )
    renamed = [(o, n) for o, n in zip(old_cols, df.columns) if o != n]
    if renamed:
        report["steps"].append({
            "step": "Column Name Cleanup",
            "icon": "edit",
            "severity": "info",
            "detail": f"Standardised {len(renamed)} column name(s) — removed spaces, special characters, converted to lowercase.",
            "count": len(renamed),
            "examples": [f'"{o}" → "{n}"' for o, n in renamed[:3]]
        })
        report["issues_fixed"] += len(renamed)

    # ── 2. Remove fully empty columns ─────────────
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df.drop(columns=empty_cols, inplace=True)
        report["steps"].append({
            "step": "Removed Empty Columns",
            "icon": "delete_sweep",
            "severity": "warning",
            "detail": f"Dropped {len(empty_cols)} column(s) that were entirely empty and contained no useful data.",
            "count": len(empty_cols),
            "examples": empty_cols[:3]
        })
        report["issues_fixed"] += len(empty_cols)

    # ── 3. Remove duplicate rows ───────────────────
    dupes = df.duplicated().sum()
    if dupes > 0:
        df.drop_duplicates(inplace=True)
        report["steps"].append({
            "step": "Removed Duplicate Rows",
            "icon": "content_copy",
            "severity": "warning",
            "detail": f"Found and removed {dupes} exact duplicate row(s). Duplicates can skew model training and inflate results.",
            "count": int(dupes),
            "examples": []
        })
        report["issues_fixed"] += int(dupes)

    # ── 4. Fix data types ──────────────────────────
    type_fixes = []
    for col in df.columns:
        if df[col].dtype == object:
            # Try numeric
            converted = pd.to_numeric(df[col].str.replace(',', '', regex=False), errors='coerce')
            if converted.notna().sum() / max(len(df), 1) > 0.8:
                df[col] = converted
                type_fixes.append(f'"{col}": text → number')
                continue
            # Try datetime
            try:
                converted_dt = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                if converted_dt.notna().sum() / max(len(df), 1) > 0.8:
                    df[col] = converted_dt
                    type_fixes.append(f'"{col}": text → date')
            except Exception:
                pass

    if type_fixes:
        report["steps"].append({
            "step": "Fixed Data Types",
            "icon": "transform",
            "severity": "info",
            "detail": f"Automatically converted {len(type_fixes)} column(s) from text to their correct numeric or date format.",
            "count": len(type_fixes),
            "examples": type_fixes[:3]
        })
        report["issues_fixed"] += len(type_fixes)

    # ── 5. Handle missing values ───────────────────
    missing_before = df.isna().sum()
    missing_cols = missing_before[missing_before > 0]
    filled = []

    for col in missing_cols.index:
        missing_count = int(missing_before[col])
        pct = missing_count / len(df) * 100

        if pct > 60:
            # Too many missing — drop column
            df.drop(columns=[col], inplace=True)
            filled.append(f'"{col}" dropped ({pct:.0f}% missing)')
        elif df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            filled.append(f'"{col}": {missing_count} values filled with median ({median_val:.2f})')
        else:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col].fillna(mode_val, inplace=True)
            filled.append(f'"{col}": {missing_count} values filled with most common ("{mode_val}")')

    if filled:
        report["steps"].append({
            "step": "Handled Missing Values",
            "icon": "healing",
            "severity": "warning",
            "detail": f"Detected and filled missing data in {len(filled)} column(s). Numeric columns filled with median, text columns with most frequent value.",
            "count": int(missing_cols.sum()),
            "examples": filled[:3]
        })
        report["issues_fixed"] += int(missing_cols.sum())

    # ── 6. Remove outliers (numeric cols) ─────────
    outlier_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        outliers = ((df[col] < Q1 - 3*IQR) | (df[col] > Q3 + 3*IQR)).sum()
        if outliers > 0:
            df[col] = df[col].clip(lower=Q1 - 3*IQR, upper=Q3 + 3*IQR)
            outlier_cols.append(f'"{col}": {outliers} extreme value(s) clamped')

    if outlier_cols:
        report["steps"].append({
            "step": "Treated Outliers",
            "icon": "filter_alt",
            "severity": "info",
            "detail": f"Detected extreme outliers in {len(outlier_cols)} numeric column(s) using the IQR method. Values clamped to 3× IQR boundaries to prevent model distortion.",
            "count": len(outlier_cols),
            "examples": outlier_cols[:3]
        })
        report["issues_fixed"] += len(outlier_cols)

    # ── Final summary ──────────────────────────────
    report["cleaned_rows"] = len(df)
    report["cleaned_cols"] = len(df.columns)
    report["rows_removed"] = report["original_rows"] - len(df)
    report["cols_removed"] = report["original_cols"] - len(df.columns)
    report["clean_score"] = min(100, max(0, 100 - (report["issues_fixed"] * 2)))

    return report, df
