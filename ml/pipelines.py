from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# ------------------------------------------------------------
# 1) 타스크 유형 자동 판정 (분류/회귀)
# ------------------------------------------------------------
def infer_task_type(y: pd.Series, uniq_threshold: int = 20) -> str:
    """y가 이산(정수/범주형, 고유값 적음)이면 classification, 그 외 regression.
    - uniq_threshold: 정수형이면서 고유값 수가 임계 이하일 때 분류로 간주
    """
    y_nonnull = y.dropna()
    if y_nonnull.empty:
        return "classification"
    if not pd.api.types.is_numeric_dtype(y_nonnull):
        return "classification"
    nunique = y_nonnull.nunique()
    # 정수형 + 고유값이 적으면 분류로 처리
    if np.all(np.floor(y_nonnull.values) == y_nonnull.values) and nunique <= uniq_threshold:
        return "classification"
    return "regression"

# ------------------------------------------------------------
# 2) 자동 피처 추천 (결측률/상수/ID성 컬럼 제외)
# ------------------------------------------------------------
def auto_feature_recommendations(
    df: pd.DataFrame,
    target_col: str,
    time_cols: List[str] | None = None,
    max_missing_ratio: float = 0.5,
    drop_constant: bool = True,
    drop_id_like: bool = True,
) -> Tuple[List[str], Dict[str, str]]:
    time_cols = set(time_cols or [])
    candidates = [c for c in df.columns if c not in set([target_col]) | time_cols | {"__time_dt__", "__row__"}]
    reasons: Dict[str, str] = {}
    keep: List[str] = []

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

# ------------------------------------------------------------
# 3) 트리 파이프라인 (분류/회귀 모두 지원)
# ------------------------------------------------------------
def build_tree_pipeline(
    task: str,
    numeric_features: List[str],
    categorical_features: List[str],
    complexity: int,
    random_state: int,
) -> Pipeline:
    """분류/회귀 공용 트리 파이프라인.
    - complexity(1~10)를 트리의 깊이/leaf 샘플 수에 매핑
    """
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

    task = (task or "classification").lower()
    if task == "regression":
        model = DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    else:
        model = DecisionTreeClassifier(
            criterion="gini",
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    return Pipeline([("preprocess", preproc), ("model", model)])

# ------------------------------------------------------------
# 4) 전처리 후 피처 이름 추출 (OHE 컬럼명 포함)
# ------------------------------------------------------------
def get_feature_names_from_preprocessor(preproc: ColumnTransformer) -> List[str]:
    names: List[str] = []
    for name, trans, cols in preproc.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        # ColumnTransformer 내부에 Pipeline(onehot 등)이 들어간 경우 처리
        if hasattr(trans, "get_feature_names_out"):
            try:
                fn = trans.get_feature_names_out(cols)
                names.extend(list(fn))
                continue
            except Exception:
                pass
        if hasattr(trans, "_final_estimator") and hasattr(trans._final_estimator, "get_feature_names_out"):
            try:
                fn = trans._final_estimator.get_feature_names_out(cols)
                names.extend(list(fn))
                continue
            except Exception:
                pass
        # fallback
        names.extend(list(cols))
    return list(names)

# ------------------------------------------------------------
# 5) 트리에서 수치형 스플릿 임계값 추출
# ------------------------------------------------------------
def extract_numeric_split_thresholds(model, feature_names: List[str], numeric_feature_names: List[str]) -> Dict[str, List[float]]:
    """트리에서 수치형 피처의 분기 임계값을 추출하여 {feature: sorted unique thresholds} 반환"""
    thresholds_by_feat: Dict[str, List[float]] = {f: [] for f in numeric_feature_names}
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
