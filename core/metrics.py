# core/metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd

def shannon_entropy_from_counts(counts: np.ndarray, base: float = 2.0, normalize: bool = True) -> float:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return np.nan
    p = counts[counts > 0] / total
    H = -np.sum(np.log(p) / np.log(base) * p)
    if not normalize:
        return float(H)
    k = (counts > 0).sum()
    if k <= 1:
        return 0.0
    return float(H / (np.log(k) / np.log(base)))

def gini_impurity_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return np.nan
    p = counts / total
    return float(1.0 - np.sum(p ** 2))

def gini_coefficient_from_values(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return np.nan
    min_x = np.min(x)
    if min_x < 0:
        x = x - min_x
    S = np.sum(x)
    if S <= 0:
        return 0.0
    x_sorted = np.sort(x)
    i = np.arange(1, n + 1)
    G = (2.0 * np.sum(i * x_sorted)) / (n * S) - (n + 1.0) / n
    return float(G)

def numeric_counts(series: pd.Series, binning: str = "auto", bins_int: int = 30) -> np.ndarray:
    x = series.dropna().values
    if x.size == 0:
        return np.array([0])
    bins = bins_int if isinstance(binning, int) else binning
    counts, _ = np.histogram(x, bins=bins)
    return counts

def compute_metrics_numeric(
    df: pd.DataFrame,
    binning: str = "auto",
    bins_int: int = 30,
    gini_mode: str = "Gini impurity (확률기반)",
    entropy_normalize: bool = True,
) -> pd.DataFrame:
    records = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        s = df[col]
        counts = numeric_counts(s, binning=binning, bins_int=bins_int)
        H = shannon_entropy_from_counts(counts, base=2.0, normalize=entropy_normalize)
        g_imp = gini_impurity_from_counts(counts)
        g_coef = gini_coefficient_from_values(s.values)
        g = g_imp if gini_mode.startswith("Gini impurity") else g_coef
        records.append(dict(column=col, n=int(s.notna().sum()), missing=int(s.isna().sum()), entropy=float(H), gini=float(g)))
    out = pd.DataFrame.from_records(records)
    return out.sort_values(["column"]).reset_index(drop=True)
