# ml/train.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split

from .pipelines import build_tree_pipeline

def build_tree_pipeline_and_train(
    df: pd.DataFrame,
    target_col: str,
    features_selected: list[str],
    task: str,
    complexity: int,
    test_size: float,
    random_state: int,
) -> Dict[str, Any]:
    """파이프라인 구성+학습+평가 결과를 dict로 반환."""
    out: Dict[str, Any] = {}

    mask = df[target_col].notna()
    df_train = df.loc[mask].copy()
    y = df_train[target_col]
    X = df_train[features_selected]

    if task == "regression" and not pd.api.types.is_numeric_dtype(y):
        y = pd.to_numeric(y, errors="coerce")
        mask2 = y.notna()
        X, y = X.loc[mask2], y.loc[mask2]

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    pipe = build_tree_pipeline(task, numeric_features, categorical_features, complexity, int(random_state))

    strat = y if task == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(random_state), stratify=strat
    )

    if task == "classification" and (y_train.nunique() < 2 or y_test.nunique() < 2):
        out["warning"] = "분류에서 클래스가 하나만 남았습니다. 타깃/피처/검증 비율을 조정하세요."
        return out

    pipe.fit(X_train, y_train)

    metrics_dict = {}
    y_pred = pipe.predict(X_test)
    if task == "classification":
        metrics_dict = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        }
        try:
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in np.unique(y_test)], columns=[f"pred_{c}" for c in np.unique(y_pred)])
        except Exception:
            cm_df = None
    else:
        metrics_dict = {
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "R2": float(r2_score(y_test, y_pred)),
        }
        cm_df = None

    df_train_sample = df_train[numeric_features].copy()
    if df_train_sample.shape[0] > 200000:
        df_train_sample = df_train_sample.sample(200000, random_state=int(random_state))

    out.update(
        pipe=pipe,
        metrics_dict=metrics_dict,
        cm_df=cm_df,
        numeric_features=numeric_features,
        df_train_sample=df_train_sample,
    )
    return out
