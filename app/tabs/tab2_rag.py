import streamlit as st
import time
from src.rag_engine import AgriDoctorRAG
from src.weather_api import WeatherManager

def render():
    st.header("농작물 맞춤형 처방전")

    diagnosis = st.session_state.get('diagnosis')
    uploaded_img = st.session_state.get('uploaded_img')

    if not diagnosis:
        st.warning("작물 병해 판별 결과가 없습니다. 먼저 분석을 진행해 주세요.")
        return

    # 정상 작물 처리
    if "정상" in diagnosis['name']:
        st.success(f"진단 결과: {diagnosis['name']} - 현재 특별한 병해가 발견되지 않았습니다.")
        if uploaded_img:
            st.image(uploaded_img, caption="분석 대상", width=300)
        return

    prescription = st.session_state.get('prescription')
    current_disease = diagnosis['name']

    # 처방 데이터 생성 및 갱신
    if not prescription or prescription.get('disease_name') != current_disease:
        with st.spinner("로컬 AI 엔진을 통해 맞춤형 처방 데이터를 구성 중입니다."):
            try:
                start_time = time.time()

                weather_data = WeatherManager.get_current_weather()
                rag_engine = AgriDoctorRAG()
                prescription = rag_engine.generate_prescription(current_disease, weather_data)

                st.session_state['prescription'] = prescription

                # 기술 모니터링을 위한 Latency 기록
                if 'latency' not in st.session_state:
                    st.session_state['latency'] = {}
                st.session_state['latency']['rag'] = time.time() - start_time

            except Exception as e:
                st.error(f"처방 프로세스 오류: {e}")
                return

    # 기상 상황 정보
    weather = prescription.get('weather', {})
    if weather:
        st.info(f"현재 농장 기상 상황: 온도 {weather.get('temperature')}°C | 습도 {weather.get('humidity')}% | {weather.get('weather_desc')}")

    st.markdown(f"### {current_disease} 분석 보고")

    # 전문가 소견 상단 배치 (중복 출력 방지 및 정보 전달력 강화)
    expert_advice = prescription.get('llm_narrative', '')
    if expert_advice:
        st.warning(f"시스템 처방 소견:\n\n{expert_advice}")

    col_text, col_img = st.columns([2, 1])
    with col_text:
        st.markdown("#### 질환 상세 및 발생 환경")
        st.write(prescription.get('environment_and_symptoms', "상세 정보가 누락되었습니다."))
    with col_img:
        if uploaded_img:
            st.image(uploaded_img, caption="분석 대상 이미지", width='stretch')
        else:
            st.image("https://img.icons8.com/illustrations/external-pack-flaticons-lineal-color-flat-icons/512/external-botany-agriculture-pack-flaticons-lineal-color-flat-icons-2.png", width='stretch')

    st.markdown("---")
    st.markdown("#### 방제 솔루션")

    c1, c2 = st.columns(2)
    with c1:
        st.success("##### 재배적 방제")
        st.write(prescription.get("cultural_control", "정보가 없습니다."))
    with c2:
        st.error("##### 화학적 방제")
        chemical_data = prescription.get("chemical_control", [])
        if chemical_data:
            st.table(chemical_data)
