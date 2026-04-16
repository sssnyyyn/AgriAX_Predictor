import streamlit as st
import pandas as pd
import plotly.express as px
import os

def render():
    st.header("시스템 아키텍처 및 기술 스택")

    st.markdown("### 데이터 파이프라인 흐름도 (Local AI)")
    # 로컬 모델(Gemma) 기반으로 파이프라인 구조 수정
    st.markdown("""
    ```mermaid
    graph LR
        A[현장 작물 이미지] --> B(Vision Model)
        W[기상 시뮬레이션 API] --> F
        A --> DB[(SQLite: 진단 이력)]
        B --> F[RAG Engine: Gemma 2B]
        DB --> F
        F --> G[종합 경영 리포트]
    ```
    """)
    st.caption("인터넷 연결이 불안정한 농가 현장에서도 중단 없이 작동할 수 있도록 로컬 경량화 모델(Gemma:2b) 기반의 파이프라인을 구축했습니다.")

    st.markdown("---")

    st.markdown("### 단계별 추론 지연 시간 (Latency 모니터링)")
    latency = st.session_state.get('latency', {})

    df_latency = pd.DataFrame({
        "파이프라인 모듈": [
            "Vision AI (병해 판별)",
            "위성 데이터 API (NDVI)",
            "기상 데이터 수집 (Weather)",
            "맞춤형 처방 생성 (Local RAG)"
        ],
        "소요 시간 (초)": [
            round(latency.get('vision', 0), 2),
            round(latency.get('gee', 0), 2),
            round(latency.get('weather', 0), 2),
            round(latency.get('rag', 0), 2)
        ]
    })

    st.table(df_latency)

    # 벤치마크 데이터 시각화 섹션
    st.markdown("---")
    st.markdown("### 모델 성능 비교 분석 (Benchmark Results)")

    csv_path = "data/model_comparison.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        col1, col2 = st.columns(2)
        with col1:
            fig_time = px.bar(df, x="model_name", y="latency_sec", color="model_name",
                             title="모델별 응답 속도 (초)", barmode='group')
            st.plotly_chart(fig_time, use_container_width=True)

        with col2:
            fig_ram = px.bar(df, x="model_name", y="ram_usage_mb", color="model_name",
                            title="로컬 RAM 점유량 (MB)", barmode='group')
            st.plotly_chart(fig_ram, use_container_width=True)

        st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)
    else:
        st.info("벤치마크 데이터가 아직 수집되지 않았습니다. 처방전 생성을 실행하면 Gemma와 Gemini의 비교 데이터가 기록됩니다.")

    st.markdown("---")

    st.markdown("### 핵심 기술 스택")
    stack_col1, stack_col2 = st.columns(2)
    with stack_col1:
        st.markdown("**Back-end & Database**")
        st.code("- Python 3.10+\n- Streamlit\n- SQLite (Time-series History)")
    with stack_col2:
        st.markdown("**AI & Data API**")
        st.code("- PyTorch (Vision)\n- Google Earth Engine API\n- Google Gemini 1.5 Flash (LLM)")
