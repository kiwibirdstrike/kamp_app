# main.py
# 목적: PyInstaller onefile로 빌드한 EXE를 더블클릭하면
#       1) 번들된 app.py를 절대경로로 실행
#       2) Streamlit 서버가 뜬 뒤 브라우저 자동 오픈
#       3) 오프라인/내부망 기본 설정 및 LLM(OLLAMA) 기본값 지정

from __future__ import annotations
import os
import sys
import time
import webbrowser
import threading
from streamlit.web import cli as stcli


def resource_path(rel_path: str) -> str:
    """
    PyInstaller onefile 모드에서 번들 리소스의 실제 경로를 반환.
    개발환경(소스 실행)에서는 현재 파일 기준, 배포 EXE에서는 sys._MEIPASS 기준.
    """
    base = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
    return os.path.join(base, rel_path)


def open_browser_after_delay(url: str, delay_sec: float = 3.0) -> None:
    """
    서버가 기동되기 전에 브라우저를 먼저 열면 연결 거부가 날 수 있어
    약간 지연 후 오픈.
    """
    time.sleep(delay_sec)
    try:
        webbrowser.open_new(url)
    except Exception:
        pass

if __name__ == "__main__":
    # ✅ 개발모드 OFF (정확한 ENV 키: STREAMLIT_GLOBAL_DEVELOPMENT_MODE)
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"

    # 번들된 config.toml을 확실히 사용하게 강제
    bundled_cfg = resource_path(".streamlit/config.toml")
    if os.path.exists(bundled_cfg):
        os.environ["STREAMLIT_CONFIG"] = bundled_cfg

    # 기본값들
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_SERVER_ENABLECORS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_PORT", "8501")

    os.environ.setdefault("LLM_BACKEND", "ollama")
    os.environ.setdefault("OLLAMA_BASE", "http://127.0.0.1:11434")
    os.environ.setdefault("OLLAMA_MODEL", "llama3")

    app_script = resource_path("app.py")
    port = os.environ.get("STREAMLIT_SERVER_PORT", "8501")
    url = f"http://127.0.0.1:{port}"

    threading.Thread(target=open_browser_after_delay, args=(url, 3.0), daemon=True).start()

    # ✅ CLI 플래그로도 재확정(최종 우선권)
    sys.argv = [
        "streamlit", "run", app_script,
        "--global.developmentMode", "false",
        "--server.address", "127.0.0.1",
        "--server.port", port,
    ]
    sys.exit(stcli.main())