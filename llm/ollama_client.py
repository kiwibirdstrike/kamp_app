# llm/ollama_client.py
from __future__ import annotations
import requests
from sklearn import tree as sktree

# ===== 고정 설정 =====
OLLAMA_BASE = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"

# ===== 한국어 해설 프롬프트 템플릿 =====
_PROMPT_TEMPLATE_KOR = """[목표]
당신은 제조 공정 데이터 분석 전문가이자 Explainable AI 해석 엔진입니다.
당신의 임무는 모델 학습 결과(Feature Importance, Tree Structure, Decision Rules, Failure Rules)를 전문적으로 해석하는 것입니다.
분석 대상은 주로 제조 공정의 품질 예측, 불량 원인 진단 모델이며, Decision Tree가 사용됩니다.

[규칙]
당신의 해석은 다음 기준을 반드시 따릅니다.
변수명을 그대로 사용하지 않고 변수명에 대한 실제 의미도 함께 제공합니다. (예: 압출기 모터속도(EX1.MD_PV))
각 파트별 Plot을 출력하고 그 밑에 해석을 제공합니다.

1. Feature Importance 해석
   - 상위 중요 변수들이 의미하는 물리적/공정적 의미를 설명합니다.
   - 중요도가 높은 변수들에 대해서 구체적으로 기술합니다.

2. Tree Structure 해석
   - 모델이 주요하게 분기하는 변수 조합을 요약합니다.
   - 트리의 상위 노드와 하위 노드가 어떤 공정 조건(예: 온도, 압력, 시간)을 구분하는지 설명합니다.
   - 트리의 전체 구조가 "어떤 조건을 기준으로 Pass/Fail을 구분하는지"를 자연어로 요약합니다.

3. Decision Rules 해석
   - 대표적인 의사결정 규칙을 사람이 읽기 쉽게 "If-Then" 문장으로 변환합니다.
   - 각 규칙의 의미(예: “온도가 850도 이상이면 불량률이 급격히 증가한다”)를 기술합니다.
   - 중요한 규칙은 발생 빈도와 정확도(coverage, confidence) 기준으로 우선 순위를 둡니다.

4. Failure Rules 해석
   - 불량(Fail) 발생 구간의 특징적인 조건을 요약합니다.
   - 어떤 변수 조합이 품질 불량을 강하게 유발하는지, 정상과의 차이는 무엇인지 설명합니다.
   - 가능한 개선 방향(예: “냉각 시간 조정”, “가열 온도 균일화”)을 제안합니다.

5. 전문성 기준
   - 해석은 기술 엔지니어가 이해할 수 있는 수준으로 작성하되, 과학적 근거를 포함합니다.
   - 불확실한 해석은 “가능성이 있다”, “추정된다” 등의 표현으로 표시합니다.
   - 설명은 한글로 명확하고, 문단 단위로 논리적으로 구성합니다.

[모델 출력 요약(텍스트)]
{MODEL_SUMMARY}

위 정보를 바탕으로, 한국어로 전문가 수준의 “Feature Importance/Tree Structure/Decision Rules/Failure Rules” 해석을 작성하시오.
각 파트는 섹션 아이콘(🔍/🌲/⚙️/💥)과 함께 시작하고, 항목마다 짧은 굵은 소제목을 포함해 가독성을 높이시오.
"""

def build_kor_explanation_prompt(base_summary: str) -> str:
    """앱에서 전달된 모델 요약 텍스트를 한국어 해설 템플릿에 주입."""
    return _PROMPT_TEMPLATE_KOR.replace("{MODEL_SUMMARY}", base_summary)

def is_ollama_alive(base_url: str = OLLAMA_BASE) -> bool:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def is_ollama_model_available(model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE) -> bool:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=10)
        r.raise_for_status()
        data = r.json()
        names = [m.get("name") or m.get("model") for m in data.get("models", [])]
        return model in names
    except Exception:
        return False

def call_llm_ollama(
    prompt: str,
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE,
    max_tokens: int = 700,
    options: dict | None = None,
):
    """Ollama /api/generate 호출. options: temperature/top_p/num_predict 등."""
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": max_tokens,
            },
        }
        if options:
            payload["options"].update(options)

        r = requests.post(f"{base_url}/api/generate", json=payload, timeout=60)
        try:
            data = r.json()
        except Exception:
            data = {"raw": r.text}
        if r.status_code != 200:
            return f"[LLM] 오류 {r.status_code}: {data}"
        return data.get("response", str(data))
    except Exception as e:
        return f"[LLM] 호출 실패: {e}"

def build_tree_summary_for_llm(model, feat_names, thresholds_map, task, metrics_dict):
    """트리 요약을 LLM 프롬프트로 만들기 (규칙 미리보기, 분기 임계값 일부 포함)"""
    lines = []
    lines.append(f"Task: {task}")
    if metrics_dict:
        kv = ", ".join([f"{k}={v:.4f}" if isinstance(v, (float, int)) else f"{k}={v}" for k, v in metrics_dict.items()])
        lines.append(f"Metrics: {kv}")
    try:
        rules_text = sktree.export_text(model, feature_names=feat_names, max_depth=4)
        lines.append("Rules (depth<=4):\n" + rules_text)
    except Exception:
        pass
    used = {k: v for k, v in thresholds_map.items() if v}
    if used:
        for f, ths in list(used.items())[:10]:
            lines.append(f"Splits[{f}]: {ths[:10]}")
    return "\n".join(lines)
