# llm/openai_client.py
from __future__ import annotations
import os
from typing import Optional, Dict, Any
import streamlit as st
from openai import OpenAI
from openai import APIConnectionError, RateLimitError, BadRequestError

# 기본 모델은 시크릿/환경변수 우선, 없으면 합리적 디폴트
OPENAI_MODEL = (
    st.secrets.get("OPENAI_MODEL")
    or os.getenv("OPENAI_MODEL")
    or "gpt-4o-mini"
)

def _get_api_key() -> Optional[str]:
    return st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

def is_openai_configured() -> bool:
    return bool(_get_api_key())

def call_llm_openai(
    prompt: str,
    system: str = "당신은 제조 데이터 분석 전문가입니다. 간결하고 명확하게 한국어로 설명하세요.",
    options: Optional[Dict[str, Any]] = None,
) -> str:
    api_key = _get_api_key()
    if not api_key:
        return "OPENAI_API_KEY가 설정되어 있지 않습니다. Streamlit Secrets 또는 환경변수를 확인하세요."

    client = OpenAI(api_key=api_key)
    model = (options or {}).get("model") or OPENAI_MODEL
    temperature = (options or {}).get("temperature", 0.2)
    top_p = (options or {}).get("top_p", 0.9)
    max_tokens = (options or {}).get("max_tokens", 900)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except (APIConnectionError, RateLimitError, BadRequestError) as e:
        return f"[LLM 오류] {type(e).__name__}: {e}"
