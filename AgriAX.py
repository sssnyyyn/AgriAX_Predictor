import streamlit as st

from app.state import init_session_state
from app.tabs import tab1_vision, tab2_rag, tab3_finance, tab4_report, tab5_tech
from src.earth_engine import EarthEngineManager
from src.vision_model import VisionAnalyzer

def main():
    st.set_page_config(page_title="AgriAX Predictor", layout="wide")

    st.markdown("""
        <style>
        .main {background-color: #f8f9fa;}
        h1, h2, h3 {color: #2c3e50;}
        .stMetric {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);}
        </style>
        """, unsafe_allow_html=True)

    # 전역 상태 초기화
    init_session_state()

    # 백그라운드 인프라 및 모델 로드
    v_model, v_device = VisionAnalyzer.load_model()
    gee_ready, gee_msg = EarthEngineManager.initialize()
    st.session_state['v_model'] = v_model
    st.session_state['v_device'] = v_device
    st.session_state['gee_ready'] = gee_ready

    st.title("스마트 농작물 건강 관리 시스템")
    st.markdown("인공지능 모델을 통해 병해를 실시간 진단하고 최적의 방제 전략과 재무 리스크 정보를 제공합니다")

    # 탭 라우팅
    tab_titles = [
        "작물 병해 판별",
        "맞춤형 처방 가이드",
        "방제 경제성 시뮬레이션",
        "종합 요약 보고서",
        "시스템 및 성능 지표"
    ]
    t1, t2, t3, t4, t5 = st.tabs(tab_titles)

    with t1:
        tab1_vision.render()
    with t2:
        tab2_rag.render()
    with t3:
        tab3_finance.render()
    with t4:
        tab4_report.render()
    with t5:
        tab5_tech.render()

if __name__ == "__main__":
    main()
# uvicorn.run(app="main:app", host="0.0.0"...)
