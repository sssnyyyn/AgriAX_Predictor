import streamlit as st
from PIL import Image
import os
import time
from src.vision_model import VisionAnalyzer
from src.disease_db import DiseaseDictionary

def load_original_image(image_input):
    img = Image.open(image_input)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

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
    st.header("AI 병해 진단 분석")
    st.write("학습된 작물 잎 이미지를 업로드하거나 샘플 이미지를 선택하여 AI 진단 성능을 테스트해 보세요.")

    st.markdown("#### 샘플 이미지로 테스트하기")
    sample_col1, sample_col2 = st.columns(2)

    current_file_path = os.path.abspath(__file__)
    tabs_dir = os.path.dirname(current_file_path)
    app_dir = os.path.dirname(tabs_dir)
    root_dir = os.path.dirname(app_dir)

    samples = [
        {"name": "고추 탄저병", "path": os.path.join(root_dir, "data", "samples", "chili_anthracnose.jpg")},
        {"name": "정상 작물 (고추)", "path": os.path.join(root_dir, "data", "samples", "healthy_leaf.jpg")}
    ]

    sample_img = None
    for i, col in enumerate([sample_col1, sample_col2]):
        with col:
            if os.path.exists(samples[i]["path"]):
                preview_img = Image.open(samples[i]["path"])
                st.image(preview_img, caption=samples[i]["name"], use_container_width=True)

                if st.button(f"{samples[i]['name']} 선택", key=f"sample_{i}"):
                    sample_img = load_original_image(samples[i]["path"])
            else:
                st.caption(f"샘플 {i+1} 준비 중")

    st.markdown("---")
    uploaded_file = st.file_uploader("이미지 업로드 (Drag & Drop)", type=["jpg", "png", "jpeg"])

    img = None
    if uploaded_file:
        img = load_original_image(uploaded_file)
    elif sample_img:
        img = sample_img

    if img:
        st.session_state['uploaded_img'] = img

        with st.spinner("AI 엔진 분석 중"):
            start_time_vision = time.time()

            v_model = st.session_state.get('v_model')
            v_device = st.session_state.get('v_device')

            pred_idx, conf = VisionAnalyzer.predict(img, v_model, v_device)
            diagnosis = DiseaseDictionary.get_info(pred_idx)
            st.session_state['diagnosis'] = diagnosis
            st.session_state['pred_idx'] = pred_idx

            col_orig, col_grad = st.columns(2)

            with col_orig:
                st.image(img, caption="분석 대상 원본 이미지", use_container_width=True)

            with col_grad:
                if pred_idx > 0:
                    grad_img = VisionAnalyzer.generate_gradcam(img, v_model, v_device, pred_idx)
                    st.image(grad_img, caption="AI 판단 근거 (Grad-CAM)", use_container_width=True)
                else:
                    st.info("특이사항이 발견되지 않아 시각화 맵을 생성하지 않습니다.")

            end_time_vision = time.time()

            if 'latency' in st.session_state:
                st.session_state['latency']['vision'] = end_time_vision - start_time_vision

            st.markdown("### AI 판별 결과")
            m1, m2, m3 = st.columns(3)
            with m1:
                custom_metric("질병명", diagnosis["name"])
            with m2:
                custom_metric("위험 등급", diagnosis["status"])
            with m3:
                custom_metric("모델 신뢰도", f"{conf*100:.1f}%")
