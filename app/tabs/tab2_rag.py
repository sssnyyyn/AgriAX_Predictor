import streamlit as st
from src.rag_engine import AgriDoctorRAG

def render():
    st.header("Agri-Doctor 맞춤형 처방전")

    diagnosis = st.session_state.get('diagnosis')
    uploaded_img = st.session_state.get('uploaded_img')

    if not diagnosis:
        st.warning("'작물 병해 판별' 탭에서 작물 이미지를 분석해 주세요.")
        return

    if diagnosis['name'] == "정상" or "정상" in diagnosis['name']:
        st.success(f"진단 결과: **{diagnosis['name']}** 입니다. 현재 특별한 병해가 발견되지 않았습니다.")
        # 정상일 때도 분석했던 이미지는 상단에 노출
        if uploaded_img:
            st.image(uploaded_img, caption="분석된 이미지", width=300)
        return

    prescription = st.session_state.get('prescription')
    current_disease = diagnosis['name']

    if not prescription or prescription.get('disease_name') != current_disease:
        with st.spinner(f"'{current_disease}'에 대한 맞춤형 처방전을 작성 중입니다"):
            try:
                rag_engine = AgriDoctorRAG()
                new_prescription = rag_engine.generate_prescription(current_disease)
                new_prescription['disease_name'] = current_disease

                st.session_state['prescription'] = new_prescription
                prescription = new_prescription
            except Exception as e:
                st.error(f"처방전을 생성하는 중 오류가 발생했습니다: {e}")
                return

    st.markdown(f"### {current_disease} 분석 보고")

    summary = prescription.get('environment_and_symptoms', '').split('.')[0]
    st.warning(f"**요약:** {summary}.")

    col_text, col_img = st.columns([2, 1])
    with col_text:
        st.markdown("#### 질환 상세 및 발생 환경")
        st.write(prescription.get('environment_and_symptoms', "상세 정보 로딩 중"))
    with col_img:
        if uploaded_img:
            st.image(uploaded_img, caption="분석 대상 이미지", use_container_width=True)
        else:
            st.image("https://img.icons8.com/illustrations/external-pack-flaticons-lineal-color-flat-icons/512/external-botany-agriculture-pack-flaticons-lineal-color-flat-icons-2.png", use_container_width=True)

    st.markdown("---")
    st.markdown("#### 투트랙(Two-Track) 방제 솔루션")

    c1, c2 = st.columns(2)
    with c1:
        st.success("##### 재배적 방제")
        st.write(prescription.get("cultural_control", "정보가 없습니다."))

    with c2:
        st.error("##### 화학적 방제")
        chemical_data = prescription.get("chemical_control", [])
        if chemical_data:
            st.table(chemical_data)
        else:
            st.write("등록된 화학적 방제 정보가 없습니다.")
