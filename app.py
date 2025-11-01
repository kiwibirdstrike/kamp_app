# app.py
from __future__ import annotations
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import tree as sktree

from core.io_utils import try_read_csv, detect_time_columns
from core.metrics import compute_metrics_numeric
from core.plotting import plot_numeric_feature_with_thresholds
from ml.pipelines import (
    auto_feature_recommendations,
    get_feature_names_from_preprocessor,
    extract_numeric_split_thresholds,
)
from ml.train import build_tree_pipeline_and_train

# OpenAI API 사용
from llm.openai_client import (
    OPENAI_MODEL,
    is_openai_configured,
    call_llm_openai,
)
# 프롬프트 빌더는 기존 것 재사용(같은 함수명 유지)
from llm.ollama_client import (
    build_tree_summary_for_llm,
    build_kor_explanation_prompt,
)

# ============================
# 기본 설정
# ============================
st.set_page_config(page_title="소성가공데이터분석", layout="wide")
st.title("데이터 시각화 · 모델 현장 적용 학습 ")

with st.expander("ℹ️ 사용법", expanded=True):
    st.write(
        """
        - CSV 업로드 → 시간열 자동 탐지 → 공정 시간 흐름에 따른 변수 분포도 시각화
        - 시각화 화면에서 각 변수 플롯 옆에 해당 변수의 ‘무질서도(분포의 넓이)’와 ‘불균형도(분포의 치우침)’ 지표가 함께 제시합니다.
        - 모델 학습(분류; y=passorfail 내부 고정): 의사결정나무로 학습하고, 결과는 **분석 해설** 탭에서 확인합니다.
        - 분석 해설: 중요 수치형 변수 TOP N의 임계값 히스토그램과 전문 한국어 해설(LLM)이 제공됩니다.
        """
    )

# ============================
# 상단 CSV 업로더
# ============================
uploaded = st.file_uploader("CSV 업로드", type=["csv"])

def _uploaded_sig(u) -> str | None:
    if u is None:
        return None
    try:
        return f"{u.name}:{u.size}"
    except Exception:
        return u.name

new_sig = _uploaded_sig(uploaded)
if "uploaded_sig" not in st.session_state:
    st.session_state.uploaded_sig = None
if new_sig != st.session_state.uploaded_sig:
    st.session_state.uploaded_sig = new_sig
    st.session_state.shape_shown_once = False
    st.session_state.trained = False
    st.session_state.pipe = None
    st.session_state.feature_importance_df = None
    st.session_state.rules_text = ""
    st.session_state.viz_page = 1  # 시각화 페이지네이션 초기화
    st.session_state.lag_info_note = "0 (미적용)"

# ============================
# NAV
# ============================
PAGES = ["대시보드", "시각화", "모델 학습", "분석 해설"]
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]

nav_cols = st.columns(len(PAGES))
for i, name in enumerate(PAGES):
    if nav_cols[i].button(name, use_container_width=True, key=f"nav_{name}"):
        st.session_state.page = name
        st.rerun()

st.markdown("---")

# ============================
# 데이터 로드
# ============================
if uploaded is None:
    st.info("⬆️ 상단에서 CSV 파일을 업로드해 주세요.")
    st.stop()

with st.spinner("CSV 로딩 중…"):
    df = try_read_csv(uploaded)

if st.session_state.page == "대시보드" and not st.session_state.get("shape_shown_once", False):
    st.success(f"로드 완료! shape = {df.shape}")
    st.session_state.shape_shown_once = True

# 시각화 샘플 제한
if df.shape[0] > 100_000:
    st.warning(f"행이 {df.shape[0]:,}개로 큽니다. 시각화는 성능 보호를 위해 상위 {100_000:,}행만 사용합니다.")
    df_viz = df.head(100_000).copy()
else:
    df_viz = df.copy()

# ============================
# 시간열 처리
# ============================
found_time_cols = detect_time_columns(df_viz)
if "selected_time_col" not in st.session_state:
    st.session_state.selected_time_col = (found_time_cols[0] if found_time_cols else None)

def _apply_time(df_src: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if st.session_state.selected_time_col:
        tcol = st.session_state.selected_time_col
        time_dt = pd.to_datetime(df_src[tcol], errors="coerce", infer_datetime_format=True)
        df2 = df_src.loc[time_dt.notna()].copy()
        df2["__time_dt__"] = time_dt.loc[time_dt.notna()]
        return df2, f"🕒 사용 중인 시간열: {tcol} (유효 행 {df2.shape[0]:,}개)"
    else:
        df2 = df_src.reset_index(drop=False).rename(columns={"index": "__row__"})
        df2["__time_dt__"] = pd.to_datetime(df2["__row__"], unit="s", origin="unix")
        return df2, "🕒 감지된 시간열이 없어 행 순서를 시간축으로 사용합니다."

df_viz_time, time_caption = _apply_time(df_viz)

# ============================
# 대시보드 — 구조 + N/결측
# ============================
if st.session_state.page == "대시보드":
    st.subheader("데이터 구조(상위 50행)")
    if found_time_cols:
        st.selectbox(
            "시간열 선택",
            options=found_time_cols,
            index=(found_time_cols.index(st.session_state.selected_time_col)
                   if st.session_state.selected_time_col in found_time_cols else 0),
            key="selected_time_col"
        )
    st.caption(time_caption)
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("열별 유효개수(N) · 결측 수")
    counts = [{"열": c, "유효개수(N)": int(df[c].notna().sum()), "결측": int(df[c].isna().sum())}
              for c in df.columns]
    counts_df = pd.DataFrame(counts).sort_values("열").reset_index(drop=True)
    st.dataframe(counts_df, use_container_width=True)

# ============================
# 유틸: Feature Importance 집계
# ============================
def aggregate_feature_importances(
    feature_names: list[str],
    importances: np.ndarray,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    agg: dict[str, float] = {}
    for name, imp in zip(feature_names, importances):
        raw = None
        if name in numeric_features:
            raw = name
        else:
            for c in categorical_features:
                if name == c or name.startswith(c + "_"):
                    raw = c
                    break
        if raw is None:
            raw = name
        agg[raw] = agg.get(raw, 0.0) + float(imp)

    df_imp = pd.DataFrame({"feature": list(agg.keys()), "importance": list(agg.values())})
    df_imp.sort_values("importance", ascending=False, inplace=True, key=lambda s: np.round(s, 12))
    df_imp.reset_index(drop=True, inplace=True)
    return df_imp

# ============================
# 시각화 — 4개씩, 버튼 페이지네이션
# ============================
if st.session_state.page == "시각화":
    st.subheader("공정 시간 흐름에 따른 변수 분포도")
    st.caption("각 변수 플롯 오른쪽에 ‘무질서도(분포의 넓이) / 불균형도(값의 치우침)’ 지표가 함께 표시됩니다.")

    num_cols_all = df_viz_time.select_dtypes(include=[np.number]).columns.tolist()
    num_cols_all = [c for c in num_cols_all if c != "__row__"]

    if not num_cols_all:
        st.info("연속형(수치형) 변수가 없습니다.")
    else:
        met_df = compute_metrics_numeric(df=df_viz_time)[["column", "entropy", "gini"]]
        metrics_all = {row["column"]: (row["entropy"], row["gini"]) for _, row in met_df.iterrows()}

        page_size = 4
        total_pages = (len(num_cols_all) + page_size - 1) // page_size
        if "viz_page" not in st.session_state:
            st.session_state.viz_page = 1

        def _render_pagination(key_prefix: str):
            curr = st.session_state.viz_page
            window = 10
            start_p = max(1, curr - window // 2)
            end_p = min(total_pages, start_p + window - 1)
            cols = st.columns(2 + (end_p - start_p + 1))
            if cols[0].button("◀ Prev", use_container_width=True, disabled=(curr == 1), key=f"{key_prefix}_prev"):
                st.session_state.viz_page = max(1, curr - 1); st.rerun()
            for i, p in enumerate(range(start_p, end_p + 1), start=1):
                label = f"[{p}]" if p == curr else f"{p}"
                if cols[i].button(label, use_container_width=True, key=f"{key_prefix}_page_{p}"):
                    st.session_state.viz_page = p; st.rerun()
            if cols[-1].button("Next ▶", use_container_width=True, disabled=(curr == total_pages), key=f"{key_prefix}_next"):
                st.session_state.viz_page = min(total_pages, curr + 1); st.rerun()

        _render_pagination("viz_top")

        start = (st.session_state.viz_page - 1) * page_size
        batch_cols = num_cols_all[start:start + page_size]

        for col in batch_cols:
            left, right = st.columns([3, 1], vertical_alignment="center")
            with left:
                fig, ax = plt.subplots(figsize=(10, 3))
                y = pd.to_numeric(df_viz_time[col], errors="coerce")
                m = df_viz_time["__time_dt__"].notna() & y.notna()
                ax.plot(df_viz_time.loc[m, "__time_dt__"], y.loc[m], linewidth=1)
                ax.set_title(col, fontsize=11)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
            with right:
                entropy, gini = metrics_all.get(col, (np.nan, np.nan))
                st.markdown(f"**무질서도**: `{entropy:.4f}`  \n**불균형도**: `{gini:.4f}`")

        _render_pagination("viz_bottom")

# ============================
# 모델 학습 — y=passorfail 고정 + X 시차(행=5초)
# ============================
if st.session_state.page == "모델 학습":
    st.header("모델 학습")

    TARGET_COL = "passorfail"
    if TARGET_COL not in df.columns:
        st.error("데이터에 'passorfail' 타깃 열이 없습니다. 불량여부를 나타내는 열명을 'passorfail'로 맞춰주세요.")
        st.stop()
    target_col = TARGET_COL

    # X 후보 (타깃/시간/보조 제외)
    exclude_cols_for_target = set(found_time_cols or []) | {"__time_dt__", "__row__", target_col}
    all_candidates = [c for c in df.columns if c not in exclude_cols_for_target]

    # 자동 추천 & 사용자 선택
    recommended, drop_reasons = auto_feature_recommendations(df, target_col, found_time_cols)
    with st.expander("학습에 사용할 변수 선택", expanded=True):
        if drop_reasons:
            dr = pd.DataFrame({"column": list(drop_reasons.keys()), "reason": list(drop_reasons.values())})
            st.caption("자동 제외 사유")
            st.dataframe(dr, use_container_width=True)
        features_selected = st.multiselect(
            "입력 변수(X) 선택",
            options=all_candidates,
            default=[c for c in recommended if c in all_candidates]
        )
        if not features_selected:
            st.warning("최소 1개 이상의 피처를 선택하세요.")

    # ---- 현장 반응시간(시차) 적용 ----
    st.subheader("현장 반응시간(시차) 적용")
    st.caption("품질 결과가 일정 시간 후 반영된다고 가정하여, 입력 변수(X)를 과거로 이동시켜 학습합니다.")
    lag_mode = st.radio(
        "시차 단위 선택",
        ["없음(실시간 가정)", "분 단위(시간열 기준)", "행 수(샘플 수 기준)"],
        index=0, horizontal=True
    )

    secs_per_row = 5.0  # 한 행 = 5초 (고정)
    lag_minutes = 0
    lag_rows_input = 0
    if lag_mode == "분 단위(시간열 기준)":
        lag_minutes = int(st.number_input("지연 시간(분)", min_value=0, value=0, step=1,
                        help="한 행당 5초로 환산하여 분을 행 수로 변환합니다."))
        approx_rows = int(round((lag_minutes * 60) / secs_per_row)) if lag_minutes > 0 else 0
        st.caption(f"기준 간격 = {secs_per_row:.2f}초/행 → {lag_minutes}분 ≈ {approx_rows}행 이동")
    elif lag_mode == "행 수(샘플 수 기준)":
        lag_rows_input = int(st.number_input("지연 행 수(샘플 수)", min_value=0, value=0, step=1,
                           help="시간열이 불명확하거나 균등 간격이 아닐 때 직접 행 수로 시차를 지정합니다."))

    # ---- 모델 설정 ----
    st.subheader("모델 설정")
    complexity = st.slider("모델 복잡도 (1=단순, 10=복잡)", 1, 10, 6)
    test_size = st.slider("검증 비율(test_size)", 0.05, 0.5, 0.2, step=0.05)
    random_state = st.number_input("random_state", value=42)

    do_train = st.button("학습 실행", type="primary")
    if "trained" not in st.session_state:
        st.session_state.trained = False

    if do_train:
        if not features_selected:
            st.error("학습할 피처를 선택하세요.")
            st.stop()

        # ---- 시차 적용: Y(t) ~ X(t - lag)
        lag_rows = 0
        lag_info_note = "0 (미적용)"
        if lag_mode == "분 단위(시간열 기준)" and lag_minutes > 0:
            lag_rows = int(round((lag_minutes * 60) / secs_per_row))
            lag_info_note = f"{lag_minutes}분 ≈ {lag_rows}행 (간격={secs_per_row:.2f}s/행)"
        elif lag_mode == "행 수(샘플 수 기준)" and lag_rows_input > 0:
            lag_rows = int(lag_rows_input)
            lag_info_note = f"{lag_rows}행"

        df_for_train = df.copy()
        if lag_rows > 0:
            for c in features_selected:
                try:
                    df_for_train[c] = df_for_train[c].shift(lag_rows)
                except Exception:
                    df_for_train[c] = pd.Series(df_for_train[c]).shift(lag_rows)
            df_for_train = df_for_train.dropna(subset=features_selected + [target_col])
        else:
            if df_for_train[target_col].isna().any():
                df_for_train = df_for_train.dropna(subset=[target_col])

        with st.spinner("모델 학습 중…"):
            result = build_tree_pipeline_and_train(
                df=df_for_train,
                target_col=target_col,
                features_selected=features_selected,
                task="classification",             # 고정(이진 불량여부)
                complexity=int(complexity),
                test_size=float(test_size),
                random_state=int(random_state),
            )

        if result.get("warning"):
            st.warning(result["warning"])
            st.stop()

        st.session_state.trained = True
        st.session_state.pipe = result["pipe"]
        st.session_state.task = "classification"
        st.session_state.metrics_dict = result["metrics_dict"]
        st.session_state.cm_df = result["cm_df"]
        st.session_state.feat_names = get_feature_names_from_preprocessor(
            st.session_state.pipe.named_steps["preprocess"]
        )
        st.session_state.numeric_features = result["numeric_features"]
        st.session_state.categorical_features = [
            c for c in features_selected if c not in st.session_state.numeric_features
        ]
        st.session_state.thresholds_map = extract_numeric_split_thresholds(
            st.session_state.pipe.named_steps["model"],
            st.session_state.feat_names,
            st.session_state.numeric_features,
        )
        st.session_state.df_train_sample = result["df_train_sample"]
        st.session_state.lag_info_note = lag_info_note  # 분석 해설 탭에서 안내용

        try:
            st.session_state.rules_text = sktree.export_text(
                st.session_state.pipe.named_steps["model"],
                feature_names=st.session_state.feat_names,
                max_depth=4,
            )
        except Exception:
            st.session_state.rules_text = ""

        try:
            importances = st.session_state.pipe.named_steps["model"].feature_importances_
            fi_df = aggregate_feature_importances(
                feature_names=st.session_state.feat_names,
                importances=importances,
                numeric_features=st.session_state.numeric_features,
                categorical_features=st.session_state.categorical_features,
            )
            st.session_state.feature_importance_df = fi_df
        except Exception:
            st.session_state.feature_importance_df = pd.DataFrame(columns=["feature", "importance"])

        st.success("학습이 완료되었습니다. 👉 ‘분석 해설’ 탭에서 결과와 해설을 확인하세요.")

# ============================
# 분석 해설 — 결과 모아보기 + TOP N 임계값 + 한국어 해설
# ============================
if st.session_state.page == "분석 해설":
    st.subheader("분석 해설")

    if not st.session_state.get("trained", False):
        st.warning("먼저 ‘모델 학습’ 페이지에서 학습을 완료해 주세요.")
    else:
        # ---- 시차 적용 정보 안내 ----
        st.caption(f"🕒 학습 시 적용된 시차: {st.session_state.get('lag_info_note', '0 (미적용)')}")
        st.caption("※ 입력 변수(X)는 위 시차만큼 과거로 이동하여 학습되었습니다.")

        # ---- (1) 평가 결과 & 혼동행렬 ----
        st.subheader("평가 결과")
        st.write(st.session_state.metrics_dict)
        if st.session_state.cm_df is not None:
            st.write("혼동행렬")
            st.dataframe(st.session_state.cm_df, use_container_width=True)

        # ---- (2) 트리 구조 시각화 ----
        st.subheader("트리 구조 시각화")
        plot_depth = st.slider("표시할 트리 깊이", 1, 10, 4, key="tree_plot_depth")
        fig, ax = plt.subplots(figsize=(24, 12))
        model = st.session_state.pipe.named_steps["model"]
        sktree.plot_tree(
            model,
            feature_names=st.session_state.feat_names,
            class_names=None,
            filled=True,
            rounded=True,
            max_depth=int(st.session_state.tree_plot_depth),
            fontsize=6,
            ax=ax,
        )
        st.pyplot(fig)
        plt.close(fig)

        # ---- (3) 영향력 (Feature Importance) ----
        st.subheader("영향력 (Feature Importance)")
        if st.session_state.feature_importance_df is not None and not st.session_state.feature_importance_df.empty:
            st.dataframe(st.session_state.feature_importance_df, use_container_width=True)
        else:
            st.info("영향력을 계산할 수 없습니다.")

        # ---- (4) 수치형 피처 분기 임계값 (TOP N) ----
        top_n = st.slider("상위 수치형 변수 개수(임계값 시각화)", 3, 12, 6, 1)

        st.subheader("수치형 피처 분기 임계값 (TOP N)")
        thresholds_map = st.session_state.thresholds_map
        df_train_sample = st.session_state.df_train_sample
        num_feats = st.session_state.numeric_features or []

        top_numeric = []
        if st.session_state.feature_importance_df is not None and not st.session_state.feature_importance_df.empty:
            fi = st.session_state.feature_importance_df
            fi_num = fi[fi["feature"].isin(num_feats)].head(int(top_n))
            top_numeric = fi_num["feature"].tolist()

        if not top_numeric:
            st.info("중요도가 높은 수치형 피처가 없습니다.")
        else:
            cols = st.columns(3)
            for i, f in enumerate(top_numeric):
                ths = thresholds_map.get(f, [])
                with cols[i % 3]:
                    if ths:
                        plot_numeric_feature_with_thresholds(df_train_sample, f, ths, bins=40)
                    else:
                        st.caption(f"- {f}: 트리 분기 임계값이 없습니다.")
            st.caption("세로선은 의사결정나무가 실제로 사용한 분기 임계값입니다.")

        # ---- (5) LLM 한국어 해설 ----
        if not is_openai_configured():
            st.info(
                "LLM 해설을 보려면 OpenAI API 키가 필요합니다.\n"
                "로컬: `.streamlit/secrets.toml` 또는 환경변수 OPENAI_API_KEY 설정\n"
                "배포: Streamlit Community Cloud의 Settings → Secrets에 키를 추가하세요."
            )
        else:
            if st.button("해설 보기"):
                model = st.session_state.pipe.named_steps["model"]
                base_summary = build_tree_summary_for_llm(
                    model=model,
                    feat_names=st.session_state.feat_names,
                    thresholds_map=st.session_state.thresholds_map,
                    task="classification",
                    metrics_dict=st.session_state.metrics_dict,
                )
                if st.session_state.get("rules_text"):
                    base_summary += "\n\nRules (depth<=4):\n" + st.session_state.rules_text
                if st.session_state.feature_importance_df is not None and not st.session_state.feature_importance_df.empty:
                    top15 = st.session_state.feature_importance_df.head(15)
                    base_summary += "\n\nTop-15 Feature Importance:\n" + "\n".join(
                        f"- {row.feature}: {row.importance:.4f}" for _, row in top15.iterrows()
                    )

                prompt = build_kor_explanation_prompt(base_summary)
                with st.spinner("해설 생성 중…"):
                    resp = call_llm_openai(
                        prompt,
                        options={
                            "model": OPENAI_MODEL,
                            "temperature": 0.2,
                            "top_p": 0.9,
                            "max_tokens": 2000,
                        },
                    )
                st.markdown(resp)
