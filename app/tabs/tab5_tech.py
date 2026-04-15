import streamlit as st
import pandas as pd

def render():
    st.header("System Architecture & Tech Stack")

    st.markdown("### 데이터 파이프라인 흐름도")
    st.markdown("""
    ```mermaid
    graph LR
        A[현장 작물 이미지] --> B(PyTorch: DANN Vision Model)
        C[ESA Sentinel-2 위성] --> D(TensorFlow: LSTM Time-Series)
        B --> E{멀티모달 융합 로직}
        D --> E
        E --> F[LangChain/Ollama: RAG Engine]
        F --> G[ROI & 처방 대시보드]
    ```
    """)
    st.caption("Streamlit 환경에서 Mermaid.js 미지원 시 텍스트 파이프라인으로 대체됩니다.")

    st.markdown("---")

    st.markdown("### 단계별 추론 지연 시간 (Latency 모니터링)")
    latency = st.session_state.get('latency', {})

    df_latency = pd.DataFrame({
        "파이프라인 모듈": ["Vision AI (DANN)", "위성 API (GEE)", "처방 생성 (RAG)", "재무 시나리오 (LLM)"],
        "소요 시간 (초)": [
            round(latency.get('vision', 0), 2),
            round(latency.get('gee', 0), 2),
            round(latency.get('rag', 0), 2),
            round(latency.get('roi', 0), 2)
        ]
    })

    st.dataframe(df_latency, use_container_width=True)
    st.info("빠르고 정확한 의사결정을 위해 각 AI 모델의 경량화 및 최적화를 적용했습니다.")
