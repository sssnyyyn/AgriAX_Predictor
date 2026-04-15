import streamlit as st
import time
import pandas as pd
from src.earth_engine import EarthEngineManager
from src.financial_analyzer import FinancialAnalyzer

def render():
    st.header("재무 영향 및 ROI 시뮬레이션")
    diagnosis = st.session_state.get('diagnosis')

    if not diagnosis:
        st.warning("Vision AI 분석을 먼저 완료해 주십시오.")
        return

    st.markdown("병해가 발생했습니다. 이제 농가 맞춤형 재무 타격을 계산합니다.")

    with st.expander("농가 비즈니스 데이터 입력", expanded=True):
        c1, c2, c3 = st.columns(3)
        area_sqm = c1.number_input("농지 면적 (제곱미터)", value=3300, step=100)
        yield_sqm = c2.number_input("기대 수확량 (kg/sqm)", value=1.5, step=0.1)
        price_kg = c3.number_input("시장 단가 (원/kg)", value=15000, step=500)

        c4, c5 = st.columns(2)
        lon = c4.number_input("경도", value=126.8000, format="%.4f")
        lat = c5.number_input("위도", value=36.4500, format="%.4f")

    st.subheader("위성 데이터 연동")
    if st.button("ESA 위성 NDVI 데이터 추출"):
        start_time = time.time()
        with st.spinner("ESA Sentinel-2 데이터를 추출 중입니다"):
            try:
                ndvi_data, status = EarthEngineManager.fetch_real_gee_ndvi(lon, lat)
                st.session_state['latency']['gee'] = time.time() - start_time

                if status == "SUCCESS":
                    st.session_state['real_ndvi_seq'] = ndvi_data
                else:
                    st.error(f"위성 연동 실패: {status}")
            except Exception as e:
                if "serviceusage.serviceUsageConsumer" in str(e):
                    st.error("### GCP 권한 설정 필요")
                    st.markdown("위성 연동을 위해 GCP 프로젝트의 권한 설정이 필요합니다. GCP Console에 접속하여 서비스 계정에 Service Usage Consumer 역할을 추가해 주십시오.")
                else:
                    st.error(f"오류 발생: {e}")

    if st.session_state.get('real_ndvi_seq') is not None:
        chart_data = pd.DataFrame(st.session_state['real_ndvi_seq'], columns=['NDVI'])
        st.area_chart(chart_data, color="#2ecc71")

        st.markdown("---")
        st.subheader("재무 타격 예측 및 방제 ROI")
        if st.button("시나리오 분석 실행", type="primary"):
            start_time = time.time()
            with st.spinner("방치 시나리오와 방제 시나리오의 경제성을 시뮬레이션합니다"):
                analyzer = FinancialAnalyzer()
                roi_result = analyzer.generate_roi_scenario(diagnosis["name"], area_sqm)

                st.session_state['latency']['roi'] = time.time() - start_time
                st.session_state['finance_done'] = True
                st.session_state['financial_results'] = {
                    'area_sqm': area_sqm,
                    'roi_result': roi_result
                }

            r1, r2, r3 = st.columns(3)
            r1.metric("방치 시 예상 손실", roi_result.get("no_action_loss", "0원"))
            r2.metric("초기 방제 비용", roi_result.get("early_action_cost", "0원"))
            r3.metric("예상 방제 ROI", roi_result.get("roi_percentage", "0%"))
