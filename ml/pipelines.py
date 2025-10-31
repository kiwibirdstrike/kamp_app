# ml/pipelines.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def infer_task_type(y: pd.Series, uniq_threshold: int = 20) -> str:
    y_nonnull = y.dropna()
    if y_nonnull.empty:
        return "classification"
    if not pd.api.types.is_numeric_dtype(y_nonnull):
        return "classification"
    nunique = y_nonnull.nunique()
    if nunique <= uniq_threshold and np.all(np.floor(y_nonnull.values) == y_nonnull.values):
        return "classification"
    return "regression"

def auto_feature_recommendations(
    df: pd.DataFrame,
    target_col: str,
    time_cols: list[str] | None = None,
    max_missing_ratio: float = 0.5,
    drop_constant: bool = True,
    drop_id_like: bool = True,
) -> Tuple[List[str], Dict[str, str]]:
    time_cols = set(time_cols or [])
    candidates = [c for c in df.columns if c not in set([target_col]) | time_cols | {"__time_dt__", "__row__"}]
    reasons = {}
    keep = []
    for c in candidates:
        s = df[c]
        miss_ratio = 1.0 - s.notna().mean()
        if miss_ratio > max_missing_ratio:
            reasons[c] = f"결측률 {miss_ratio:.0%} > {max_missing_ratio:.0%}"
            continue
        if drop_constant:
            try:
                if s.dropna().nunique() <= 1:
                    reasons[c] = "상수 또는 유효 고유값 1"
                    continue
            except Exception:
                pass
        if drop_id_like and s.dtype == object:
            nn = s.dropna().shape[0]
            if nn > 0:
                uniq_ratio = s.dropna().nunique() / nn
                if uniq_ratio >= 0.9:
                    reasons[c] = f"ID 유사 컬럼(고유비율 {uniq_ratio:.0%})"
                    continue
        keep.append(c)
    return keep, reasons

def build_tree_pipeline(task: str, numeric_features: list[str], categorical_features: list[str], complexity: int, random_state: int) -> Pipeline:
    # 복잡도(1~10) → 트리의 깊이/리프에 매핑
    complexity = int(np.clip(complexity, 1, 10))
    max_depth = int(np.interp(complexity, [1, 10], [3, 30]))
    min_samples_leaf = int(np.round(np.interp(complexity, [1, 10], [20, 1])))
    min_samples_leaf = max(1, min_samples_leaf)

    num_pipe = SimpleImputer(strategy="median")
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preproc = ColumnTransformer([
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, categorical_features),
    ])

    if task == "classification":
        model = DecisionTreeClassifier(
            criterion="gini",
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    else:
        model = DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    return Pipeline([("preprocess", preproc), ("model", model)])

def get_feature_names_from_preprocessor(preproc: ColumnTransformer) -> list[str]:
    names: list[str] = []
    for name, trans, cols in preproc.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if hasattr(trans, "get_feature_names_out"):
            try:
                fn = trans.get_feature_names_out(cols)
                names.extend(fn)
            except Exception:
                names.extend(list(cols))
        elif hasattr(trans, "_final_estimator") and hasattr(trans._final_estimator, "get_feature_names_out"):
            fn = trans._final_estimator.get_feature_names_out(cols)
            names.extend(fn)
        else:
            names.extend(list(cols))
    return list(names)

def extract_numeric_split_thresholds(model, feature_names: list[str], numeric_feature_names: list[str]) -> dict:
    """트리에서 수치형 피처의 분기 임계값을 추출하여 {feature: sorted unique thresholds} 반환"""
    thresholds_by_feat = {f: [] for f in numeric_feature_names}
    if not hasattr(model, "tree_"):
        return thresholds_by_feat
    t = model.tree_
    for node in range(t.node_count):
        fid = t.feature[node]
        if fid >= 0:  # split node
            fname = feature_names[fid] if fid < len(feature_names) else None
            if fname in thresholds_by_feat:
                thr = float(t.threshold[node])
                thresholds_by_feat[fname].append(thr)
    for k in thresholds_by_feat:
        if thresholds_by_feat[k]:
            thresholds_by_feat[k] = sorted(set(np.round(thresholds_by_feat[k], 6)))
        else:
            thresholds_by_feat[k] = []
    return thresholds_by_feat
