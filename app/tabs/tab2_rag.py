import streamlit as st
from src.rag_engine import AgriDoctorRAG

def render():
    st.header("Agri-Doctor 맞춤형 처방전")
    diagnosis = st.session_state.get('diagnosis')

    if not diagnosis or diagnosis['name'] == "정상 (모든 작물)":
        st.info("AI 진단 결과 병해가 감지되면 전문 처방전이 이곳에 생성됩니다.")
        return

    prescription = st.session_state.get('prescription', {})

    st.markdown(f"### {diagnosis['name']} 분석 보고")

    st.warning(f"**요약:** {prescription.get('environment_and_symptoms', '').split('.')[0]}.")

    col_text, col_img = st.columns([2, 1])
    with col_text:
        st.markdown("#### 질환 상세 및 발생 환경")
        st.write(prescription.get('environment_and_symptoms', "상세 정보 로딩 중"))
    with col_img:
        st.image("https://img.icons8.com/illustrations/external-pack-flaticons-lineal-color-flat-icons/512/external-botany-agriculture-pack-flaticons-lineal-color-flat-icons-2.png", use_container_width=True)

    st.markdown("---")
    st.markdown("#### 투트랙(Two-Track) 방제 솔루션")
    c1, c2 = st.columns(2)
    with c1:
        st.success("##### 재배적 방제")
        st.write(prescription.get("cultural_control", ""))
    with c2:
        st.error("##### 화학적 방제")
        st.table(prescription.get("chemical_control", []))
