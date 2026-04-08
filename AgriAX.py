import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import time
import os
import ee

# PyTorch (Vision)
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Function

# TensorFlow (Time-Series)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, BatchNormalization

# -----------------------------------------------------------------------------
# 1. PyTorch 모델 아키텍처 (DANN)
# -----------------------------------------------------------------------------
class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def build_resnet50_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

class AgriAX_DANN(nn.Module):
    def __init__(self, base_model, num_classes):
        super(AgriAX_DANN, self).__init__()
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.class_classifier = nn.Linear(base_model.fc.in_features, num_classes)
        self.domain_classifier = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x, alpha=1.0):
        features = torch.flatten(self.feature_extractor(x), 1)
        class_output = self.class_classifier(features)
        return class_output, None

# -----------------------------------------------------------------------------
# 2. 시스템 자원 관리 (모델 로드 및 GEE 초기화)
# -----------------------------------------------------------------------------
@st.cache_resource
def init_google_earth_engine(project_id=""):
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        return True, "SUCCESS"
    except Exception as e:
        return False, str(e)

@st.cache_resource
def load_vision_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'dann_best_model.pth'

    base_model = build_resnet50_model(num_classes=38)
    model = AgriAX_DANN(base_model=base_model, num_classes=15)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    return None, device

@st.cache_resource
def load_timeseries_model():
    model_path = 'bilstm_best_model.h5'
    if os.path.exists(model_path):
        model = Sequential()
        model.add(Input(shape=(14, 1)))
        model.add(Bidirectional(LSTM(64)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))

        model.load_weights(model_path)
        return model
    return None

# -----------------------------------------------------------------------------
# 3. 비즈니스 로직 및 위성 데이터 추출 함수
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_real_gee_ndvi(lon, lat):
    try:
        poi = ee.Geometry.Point([lon, lat])
        end_date = ee.Date(int(time.time() * 1000))
        start_date = end_date.advance(-40, 'day')

        s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
               .filterBounds(poi) \
               .filterDate(start_date, end_date) \
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

        def get_ndvi(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            return image.addBands(ndvi)

        s2_ndvi = s2.map(get_ndvi)
        ndvi_list = s2_ndvi.select('NDVI').getRegion(poi, 10).getInfo()

        if len(ndvi_list) <= 1:
            return None, "위성 데이터가 존재하지 않거나 구름이 너무 많습니다."

        df = pd.DataFrame(ndvi_list[1:], columns=ndvi_list[0])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df = df.groupby('time').mean(numeric_only=True).reset_index()
        df = df.sort_values('time')

        ndvi_values = df['NDVI'].dropna().values
        if len(ndvi_values) == 0:
            return None, "유효한 NDVI 픽셀을 찾을 수 없습니다."

        if len(ndvi_values) < 14:
            pad_length = 14 - len(ndvi_values)
            padded = np.pad(ndvi_values, (pad_length, 0), mode='edge')
            return padded, "SUCCESS"
        else:
            return ndvi_values[-14:], "SUCCESS"

    except Exception as e:
        return None, str(e)

def get_disease_info(class_idx):
    if class_idx == -1:
        return {"name": "판별 불가", "status": "분석 보류", "base_loss": 0.0, "urgency": "전문가 확인", "guide": "1. 미학습 데이터이거나 화질이 낮습니다.\n2. 재촬영을 권장합니다."}

    mapping = {
        0: {"name": "고추 병해 (세균성점무늬병/탄저병 등)", "status": "위험", "base_loss": 0.15, "urgency": "높음", "guide": "1. 병든 잎 및 과실 조기 제거\n2. 등록 약제 살포"},
        1: {"name": "고추 정상", "status": "안전", "base_loss": 0.0, "urgency": "불필요", "guide": "1. 특이사항 없음\n2. 현재 상태 유지"},
        2: {"name": "감자 조기마름병", "status": "경고", "base_loss": 0.08, "urgency": "보통", "guide": "1. 하엽 제거로 통풍 개선\n2. 보호살균제 살포"},
        3: {"name": "감자 역병", "status": "심각", "base_loss": 0.20, "urgency": "매우 높음", "guide": "1. 감염 개체 소각\n2. 침투이행성 약제 살포"},
        4: {"name": "감자 정상", "status": "안전", "base_loss": 0.0, "urgency": "불필요", "guide": "1. 특이사항 없음\n2. 현재 상태 유지"},
        5: {"name": "토마토 세균성점무늬병", "status": "위험", "base_loss": 0.10, "urgency": "높음", "guide": "1. 병든 잎 제거\n2. 약제 방제"},
        6: {"name": "토마토 조기마름병", "status": "경고", "base_loss": 0.08, "urgency": "보통", "guide": "1. 질소비료 과용 금지\n2. 초기 병반 발견 시 약제 살포"},
        7: {"name": "토마토 역병", "status": "심각", "base_loss": 0.25, "urgency": "매우 높음", "guide": "1. 다습 환경 개선\n2. 발병 초 방제"},
        8: {"name": "토마토 잎곰팡이병", "status": "경고", "base_loss": 0.08, "urgency": "보통", "guide": "1. 환기 철저\n2. 적엽으로 통풍 개선"},
        9: {"name": "토마토 점무늬병", "status": "경고", "base_loss": 0.05, "urgency": "보통", "guide": "1. 병든 잎 조기 제거\n2. 등록 약제 살포"},
        10: {"name": "토마토 점박이응애", "status": "위험", "base_loss": 0.12, "urgency": "높음", "guide": "1. 살비제 살포\n2. 계통이 다른 약제 교차 살포"},
        11: {"name": "토마토 겹무늬병", "status": "위험", "base_loss": 0.10, "urgency": "높음", "guide": "1. 환기 및 채광 개선\n2. 적용 약제 살포"},
        12: {"name": "토마토 황화잎말이바이러스(TYLCV)", "status": "심각", "base_loss": 0.30, "urgency": "매우 높음", "guide": "1. 매개충(담배가루이) 방제\n2. 발병 개체 소각"},
        13: {"name": "토마토 모자이크바이러스", "status": "심각", "base_loss": 0.20, "urgency": "매우 높음", "guide": "1. 종자 소독 철저\n2. 작업 도구 소독"},
        14: {"name": "토마토 정상", "status": "안전", "base_loss": 0.0, "urgency": "불필요", "guide": "1. 특이사항 없음\n2. 현재 상태 유지"}
    }
    return mapping.get(class_idx, {"name": "시스템 에러", "status": "에러", "base_loss": 0.0, "urgency": "-", "guide": "-"})

def predict_image(image, model, device, threshold=0.75):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        class_output, _ = model(img_tensor)
        probabilities = torch.nn.functional.softmax(class_output[0], dim=0)
        max_prob, predicted_idx = torch.max(probabilities, 0)

        if max_prob.item() < threshold:
            return -1, max_prob.item()
    return predicted_idx.item(), max_prob.item()

def predict_timeseries_loss(ts_model, base_loss, real_ndvi_seq):
    if base_loss == 0.0 or ts_model is None or real_ndvi_seq is None:
        return base_loss

    seq_input = real_ndvi_seq.reshape(1, 14, 1)
    predicted_ndvi = ts_model.predict(seq_input, verbose=0)[0][0]

    recent_trend = real_ndvi_seq[-1] - real_ndvi_seq[-7]
    trend_factor = abs(recent_trend) * 3 if recent_trend < 0 else 0
    severity_multiplier = 1.0 + (0.6 - predicted_ndvi) + trend_factor

    severity_multiplier = max(1.0, min(severity_multiplier, 2.5))

    final_loss = base_loss * severity_multiplier
    return round(final_loss, 3)

def generate_real_gradcam(pil_img, model, device, class_idx):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(pil_img.convert('RGB')).unsqueeze(0).to(device)
    img_tensor.requires_grad_(True)

    gradients = []
    activations = []

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    def fwd_hook(module, input, output):
        activations.append(output.detach())

    target_layer = model.feature_extractor[7]
    handle_fwd = target_layer.register_forward_hook(fwd_hook)
    handle_bwd = target_layer.register_full_backward_hook(bwd_hook)

    model.eval()
    model.zero_grad()
    class_output, _ = model(img_tensor)

    score = class_output[0, class_idx]
    score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    if not gradients or not activations:
        return pil_img

    grads = gradients[0].cpu().numpy()[0]
    acts = activations[0].cpu().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (pil_img.width, pil_img.height))
    cam -= np.min(cam)
    cam_max = np.max(cam)
    if cam_max != 0:
        cam /= cam_max

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    orig_np = np.array(pil_img.convert('RGB'))
    superimposed = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)

    return Image.fromarray(superimposed)

# -----------------------------------------------------------------------------
# 4. UI 렌더링
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AgriAX Predictor", layout="wide")

with st.sidebar:
    st.title("AgriAX 시스템")
    st.info("실시간 위성 + 드론 멀티모달 예측")
    st.markdown("---")

    st.header("Google Earth Engine 설정")

    # Secrets에서 Project ID를 가져오고, 없으면 빈 값 처리
    try:
        default_id = st.secrets["GCP_PROJECT_ID"]
    except:
        default_id = ""

    gcp_project_id = st.text_input("GCP Project ID", value=default_id, help="비밀 설정 파일(.streamlit/secrets.toml)에서 자동으로 읽어옵니다.")

    gee_ready, gee_msg = init_google_earth_engine(gcp_project_id)

    if gee_ready:
        st.success("Google Earth Engine 연동 완료")
    else:
        st.error(f"GEE 초기화 실패: {gee_msg}")

    v_model, v_device = load_vision_model()
    if v_model:
        st.success("Vision 모델(DANN) 로드 완료")
    else:
        st.error("DANN 가중치 파일 누락")

    ts_model = load_timeseries_model()
    if ts_model:
        st.success("Time-Series 모델(BiLSTM) 로드 완료")
    else:
        st.warning("BiLSTM 가중치 파일 누락")

    st.markdown("---")
    st.header("농가 위치 설정")
    farm_lon = st.number_input("경도 (Longitude)", value=126.8000, format="%.4f")
    farm_lat = st.number_input("위도 (Latitude)", value=36.4500, format="%.4f")

    st.markdown("---")
    threshold_slider = st.slider("신뢰 구간 임계치", 50, 99, 75) / 100.0

st.title("AgriAX Predictor 대시보드")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    drone_file = st.file_uploader("잎 단위 고해상도 이미지 업로드", type=["jpg", "png", "jpeg"])
    if drone_file:
        img = Image.open(drone_file)
        st.image(img, caption="입력 데이터", width='stretch')

with col2:
    st.write("실시간 Sentinel-2 위성 식생지수 (최근 14일)")
    st.caption(f"타겟 좌표: [{farm_lat}, {farm_lon}]")

    real_ndvi_seq = None
    if gee_ready:
        with st.spinner("ESA 위성 데이터 추출 중..."):
            ndvi_data, status = fetch_real_gee_ndvi(farm_lon, farm_lat)
            if status == "SUCCESS":
                real_ndvi_seq = ndvi_data
                chart_data = pd.DataFrame(real_ndvi_seq, columns=['NDVI (실측)'])
                st.line_chart(chart_data)
            else:
                st.error(f"위성 데이터 로드 실패: {status}")
    else:
        st.warning("GEE 인증 문제로 위성 데이터를 불러올 수 없습니다.")

if st.button("통합 분석 시작", type="primary", width='stretch'):
    if not drone_file:
        st.error("이미지를 업로드해 주십시오.")
    elif v_model is None:
        st.error("Vision 모델이 로드되지 않았습니다.")
    elif not gee_ready or real_ndvi_seq is None:
        st.error("위성 데이터 추출이 완료되지 않아 통합 분석을 수행할 수 없습니다.")
    else:
        with st.spinner("멀티모달 데이터 추론 중..."):
            pred_idx, conf = predict_image(img, v_model, v_device, threshold=threshold_slider)
            diagnosis = get_disease_info(pred_idx)

            final_loss_ratio = predict_timeseries_loss(ts_model, diagnosis["base_loss"], real_ndvi_seq)

            farm_area = 3300
            yield_sqm = 1.5
            price_kg = 15000

            total_yield = farm_area * yield_sqm
            lost_yield = total_yield * final_loss_ratio
            financial_loss = lost_yield * price_kg

            st.markdown("---")
            res_col1, res_col2, res_col3 = st.columns(3)

            with res_col1:
                st.metric("병충해 판별", diagnosis["name"], diagnosis["status"])
                st.write(f"진단 신뢰도: {conf*100:.1f}%")

            with res_col2:
                delta_color = "inverse" if final_loss_ratio > 0 else "off"
                st.metric("예상 수확 감소", f"-{final_loss_ratio*100:.1f}%", delta_color=delta_color)
                st.write(f"예상 손실액: 약 {int(financial_loss):,}원")

            with res_col3:
                st.metric("방제 시급성", diagnosis["urgency"])

            st.markdown("---")
            xai_col1, xai_col2 = st.columns(2)

            with xai_col1:
                st.subheader("Grad-CAM 시각화")
                if final_loss_ratio > 0 and pred_idx != -1:
                    gradcam_img = generate_real_gradcam(img, v_model, v_device, pred_idx)
                    st.image(gradcam_img, caption="AI 판단 근거 (활성화 맵)", width='stretch')
                else:
                    st.image(img, caption="특이사항 없음 / 판별 불가", width='stretch')

            with xai_col2:
                st.subheader("현장 대응 가이드")
                if final_loss_ratio > 0:
                    st.warning(diagnosis["guide"])
                else:
                    st.success(diagnosis["guide"])
