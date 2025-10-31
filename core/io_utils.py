# core/io_utils.py
from __future__ import annotations
import io
import pandas as pd

def try_read_csv(uploaded_file: io.BytesIO) -> pd.DataFrame:
    """CSV 읽기 시도: UTF-8 → CP949 순서로 시도, ; 또는 , 구분자 자동 처리."""
    content = uploaded_file.read()
    for enc in ("utf-8", "cp949"):
        for sep in (",", ";"):
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=enc, sep=sep)
                if df.shape[1] > 1:
                    return df
            except Exception:
                pass
    uploaded_file.seek(0)
    return pd.read_csv(io.BytesIO(content), engine="python")

def detect_time_columns(df: pd.DataFrame) -> list[str]:
    name_hits = []
    KEYWORDS = ["date", "time", "datetime", "timestamp", "ts", "일", "시", "분", "초", "날짜", "시간"]
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in KEYWORDS):
            name_hits.append(c)
    candidates = list(dict.fromkeys(name_hits + df.columns.tolist()))
    found: list[str] = []
    for c in candidates:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            if s.notna().mean() >= 0.5:
                found.append(c)
                continue
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
            if parsed.notna().mean() >= 0.5:
                found.append(c)
    return list(dict.fromkeys(found))
