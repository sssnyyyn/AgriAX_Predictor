import streamlit as st

def init_session_state():
    # 1. 진단 상태
    if 'diagnosis' not in st.session_state:
        st.session_state['diagnosis'] = None
    if 'pred_idx' not in st.session_state:
        st.session_state['pred_idx'] = -1

    # 2. 이미지 및 GEE 데이터
    if 'uploaded_img' not in st.session_state:
        st.session_state['uploaded_img'] = None
    if 'real_ndvi_seq' not in st.session_state:
        st.session_state['real_ndvi_seq'] = None

    # 3. 재무 데이터 입력 상태 (Tab 4 분기 처리용)
    if 'finance_done' not in st.session_state:
        st.session_state['finance_done'] = False
    if 'financial_results' not in st.session_state:
        st.session_state['financial_results'] = {}

    # 4. 성능 모니터링 (Latency)
    if 'latency' not in st.session_state:
        st.session_state['latency'] = {
            'vision': 0.0,
            'rag': 0.0,
            'gee': 0.0,
            'roi': 0.0
        }
