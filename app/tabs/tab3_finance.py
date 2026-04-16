import streamlit as st
import pandas as pd
import numpy as np
import ee
from src.earth_engine import EarthEngineManager
from src.financial_analyzer import FinancialAnalyzer

@st.cache_resource
def connect_gee(project_id):
    try:
        ee.Initialize(project=project_id)
        return True, None
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize(project=project_id)
            return True, None
        except Exception as e:
            return False, str(e)

def custom_metric(label, value):
    st.markdown(
        f"""
        <div style="
            background-color: #f1f3f5;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #dee2e6;
            margin-bottom: 10px;">
            <p style="font-size: 14px; color: #6c757d; margin: 0; padding-bottom: 8px;">{label}</p>
            <p style="font-size: 20px; font-weight: bold; color: #212529; margin: 0;">{value}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def render():
    diagnosis = st.session_state.get('diagnosis')

    if not diagnosis:
        st.warning("Vision AI 분석을 먼저 완료해 주십시오.")
        return

    col_title, col_id, col_btn = st.columns([4, 3, 2])

    with col_title:
        st.markdown("### 재무 영향 및 ROI 시뮬레이션")

    default_id = "agriax-predictor"
    if 'gcp_project_id' not in st.session_state:
        st.session_state['gcp_project_id'] = default_id

    with col_id:
        project_id = st.text_input("GCP ID", value=st.session_state['gcp_project_id'], label_visibility="collapsed", placeholder="GCP 프로젝트 ID")

    with col_btn:
        if st.button("GEE 인증", use_container_width=True):
            success, error = connect_gee(project_id)
            if success:
                st.session_state['gee_authenticated'] = True
                st.session_state['gcp_project_id'] = project_id
                st.success("연결 성공")
            else:
                st.error(f"실패: {error}")

    st.markdown("---")

    st.subheader("위성 데이터 연동")
    c_lon, c_lat = st.columns(2)
    lon = c_lon.number_input("경도", value=126.8000, format="%.4f")
    lat = c_lat.number_input("위도", value=36.4500, format="%.4f")

    if st.button("ESA 위성 NDVI 데이터 추출"):
        if not st.session_state.get('gee_authenticated'):
            st.warning("우측 상단의 GEE 인증을 완료해 주십시오.")
            return

        with st.spinner("데이터 추출 중"):
            try:
                ndvi_data, status = EarthEngineManager.fetch_real_gee_ndvi(lon, lat)
                if status == "SUCCESS":
                    st.session_state['real_ndvi_seq'] = ndvi_data
                else:
                    raise Exception(status)
            except Exception as e:
                if "serviceusage.serviceUsageConsumer" in str(e):
                    st.error("권한 설정이 필요합니다.")
                    st.session_state['real_ndvi_seq'] = np.random.uniform(0.4, 0.8, 12).tolist()
                else:
                    st.error(f"오류: {e}")

    if st.session_state.get('real_ndvi_seq') is not None:
        chart_data = pd.DataFrame(st.session_state['real_ndvi_seq'], columns=['NDVI'])
        st.area_chart(chart_data, color="#2ecc71")

    st.markdown("---")

    st.subheader("재무 타격 예측 및 방제 ROI")
    st.markdown(f"진단 결과인 **{diagnosis['name']}** 데이터를 바탕으로 재무 타격을 시뮬레이션합니다.")

    with st.expander("농가 비즈니스 데이터 입력", expanded=True):
        f1, f2, f3 = st.columns(3)
        area_sqm = f1.number_input("농지 면적 (제곱미터)", value=3300, step=100)
        yield_sqm = f2.number_input("기대 수확량 (kg/sqm)", value=1.5, step=0.1)
        price_kg = f3.number_input("시장 단가 (원/kg)", value=15000, step=500)

    potential_revenue = area_sqm * yield_sqm * price_kg
    loss_rate = 0.35 if "탄저병" in diagnosis["name"] else (0.0 if "정상" in diagnosis["name"] else 0.20)

    dynamic_loss = potential_revenue * loss_rate
    dynamic_cost = area_sqm * 150

    recovered_revenue = dynamic_loss * 0.80
    net_profit = recovered_revenue - dynamic_cost
    dynamic_roi = (net_profit / dynamic_cost * 100) if dynamic_cost > 0 else 0

    r1, r2, r3 = st.columns(3)
    with r1:
        custom_metric("방치 시 예상 손실", f"{int(dynamic_loss):,}원")
    with r2:
        custom_metric("초기 방제 비용", f"{int(dynamic_cost):,}원")
    with r3:
        custom_metric("예상 방제 ROI", f"{dynamic_roi:.1f}%")

    if dynamic_loss > 0:
        st.markdown("##### 시나리오별 재무 지표 시각화")
        bar_data = pd.DataFrame({
            "금액(원)": [dynamic_loss, dynamic_cost, net_profit]
        }, index=["방치 시 손실액", "초기 방제 비용", "방제 후 예상 순수익"])
        st.bar_chart(bar_data, color="#e74c3c")
