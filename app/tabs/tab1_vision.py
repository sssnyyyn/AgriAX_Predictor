import streamlit as st
import time
from PIL import Image
import os
from src.vision_model import VisionAnalyzer
from src.disease_db import DiseaseDictionary

def optimize_image(image_input, max_size=(600, 600)):
    """
    이미지 해상도를 제한하여 메모리 사용량을 줄이고 로딩 속도를 최적화하는 함수
    """
    img = Image.open(image_input)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img

def render():
    st.header("AI 병해 진단 분석")
    st.write("학습된 작물 잎 이미지를 업로드하거나 샘플 이미지를 선택하여 AI 진단 성능을 테스트해 보세요")

    st.markdown("#### 샘플 이미지로 테스트하기")
    # 샘플 개수에 맞춰 2개의 컬럼으로 레이아웃 수정
    sample_col1, sample_col2 = st.columns(2)

    current_file_path = os.path.abspath(__file__)
    tabs_dir = os.path.dirname(current_file_path)
    app_dir = os.path.dirname(tabs_dir)
    root_dir = os.path.dirname(app_dir)

    # 모델이 확실히 판별할 수 있는 2가지 샘플만 구성
    samples = [
        {"name": "고추 탄저병", "path": os.path.join(root_dir, "data", "samples", "chili_anthracnose.jpg")},
        {"name": "정상 작물 (고추)", "path": os.path.join(root_dir, "data", "samples", "healthy_leaf.jpg")}
    ]

    sample_img = None
    for i, col in enumerate([sample_col1, sample_col2]):
        with col:
            if os.path.exists(samples[i]["path"]):
                preview_img = optimize_image(samples[i]["path"], max_size=(400, 400))
                st.image(preview_img, caption=samples[i]["name"], use_container_width=True)

                if st.button(f"{samples[i]['name']} 선택", key=f"sample_{i}"):
                    sample_img = optimize_image(samples[i]["path"])
            else:
                st.caption(f"샘플 {i+1} 준비 중")

    st.markdown("---")
    uploaded_file = st.file_uploader("이미지 업로드 (Drag & Drop)", type=["jpg", "png", "jpeg"])

    img = None
    if uploaded_file:
        img = optimize_image(uploaded_file)
    elif sample_img:
        img = sample_img

    if img:
        st.session_state['uploaded_img'] = img
        col_orig, col_grad = st.columns(2)

        with st.spinner("AI 엔진 분석 중"):
            v_model = st.session_state.get('v_model')
            v_device = st.session_state.get('v_device')

            pred_idx, conf = VisionAnalyzer.predict(img, v_model, v_device)
            diagnosis = DiseaseDictionary.get_info(pred_idx)
            st.session_state['diagnosis'] = diagnosis
            st.session_state['pred_idx'] = pred_idx

            with col_orig:
                st.image(img, caption="분석 대상 이미지", use_container_width=True)
            with col_grad:
                if pred_idx > 0:
                    grad_img = VisionAnalyzer.generate_gradcam(img, v_model, v_device, pred_idx)
                    st.image(grad_img, caption="AI 판단 근거 (Grad-CAM)", use_container_width=True)
                else:
                    st.info("특이사항이 발견되지 않아 시각화 맵을 생성하지 않습니다")

            st.markdown("### AI 판별 결과")
            m1, m2, m3 = st.columns(3)
            m1.metric("질병명", diagnosis["name"])
            m2.metric("위험 등급", diagnosis["status"])
            m3.metric("모델 신뢰도", f"{conf*100:.1f}%")
