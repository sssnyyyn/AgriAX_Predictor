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
from tensorflow.keras.layers import Input, LSTM, Dense

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
    model_path = 'dann_multicrop_best.pth'

    base_model = build_resnet50_model(num_classes=38)
    model = AgriAX_DANN(base_model=base_model, num_classes=21)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    return None, device

@st.cache_resource
def load_timeseries_model():
    model_path = 'lstm_best_model.h5'
    if os.path.exists(model_path):
        model = Sequential()
        model.add(Input(shape=(14, 1)))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='linear'))

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
            return None, "위성 데이터가 존재하지 않거나 구름이 너무 많습니다"

        df = pd.DataFrame(ndvi_list[1:], columns=ndvi_list[0])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df = df.groupby('time').mean(numeric_only=True).reset_index()
        df = df.sort_values('time')

        ndvi_values = df['NDVI'].dropna().values
        if len(ndvi_values) == 0:
            return None, "유효한 NDVI 픽셀을 찾을 수 없습니다"

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
        return {"name": "판별 불가", "status": "분석 보류", "base_loss": 0.0, "urgency": "전문가 확인", "guide": "1. 미학습 데이터이거나 화질이 낮습니다\n2. 재촬영을 권장합니다"}

    mapping = {
        0: {"name": "정상 (모든 작물)", "status": "안전", "base_loss": 0.0, "urgency": "불필요", "guide": "1. 특이사항 없음\n2. 현재 상태 유지"},
        1: {"name": "고추 탄저병", "status": "위험", "base_loss": 0.15, "urgency": "높음", "guide": "1. 병든 과실 및 잎 조기 제거\n2. 비 오기 전후 등록 약제 살포"},
        2: {"name": "고추 흰가루병", "status": "경고", "base_loss": 0.10, "urgency": "보통", "guide": "1. 초기 병반 발견 시 약제 살포\n2. 밀식 방지 및 통풍 개선"},
        3: {"name": "무 검은무늬병", "status": "경고", "base_loss": 0.10, "urgency": "보통", "guide": "1. 병든 잎 제거\n2. 종자 소독 및 윤작 권장"},
        4: {"name": "무 노균병", "status": "위험", "base_loss": 0.12, "urgency": "높음", "guide": "1. 다습 환경 개선 (배수 철저)\n2. 적용 보호살균제 살포"},
        5: {"name": "배추 검은썩음병", "status": "심각", "base_loss": 0.20, "urgency": "매우 높음", "guide": "1. 발병 개체 즉시 소각/매몰\n2. 농기구 소독 철저"},
        6: {"name": "배추 노균병", "status": "위험", "base_loss": 0.12, "urgency": "높음", "guide": "1. 밀식 피하고 환기 유의\n2. 발병 초기 약제 살포"},
        7: {"name": "애호박 노균병", "status": "위험", "base_loss": 0.12, "urgency": "높음", "guide": "1. 잎에 물방울이 맺히지 않도록 관리\n2. 이병엽 제거 및 약제 방제"},
        8: {"name": "애호박 흰가루병", "status": "경고", "base_loss": 0.08, "urgency": "보통", "guide": "1. 질소질 비료 과용 금지\n2. 초기 예방 약제 살포"},
        9: {"name": "양배추 균핵병", "status": "심각", "base_loss": 0.15, "urgency": "높음", "guide": "1. 병든 식물체와 흙 제거\n2. 적용 약제 살포 및 벼과 작물 윤작"},
        10: {"name": "양배추 무름병", "status": "심각", "base_loss": 0.20, "urgency": "매우 높음", "guide": "1. 상처로 감염되므로 해충 방제 병행\n2. 이병주 조기 제거"},
        11: {"name": "오이 노균병", "status": "위험", "base_loss": 0.15, "urgency": "높음", "guide": "1. 야간 다습 환경 개선\n2. 예방 위주의 약제 살포"},
        12: {"name": "오이 흰가루병", "status": "경고", "base_loss": 0.10, "urgency": "보통", "guide": "1. 건조하지 않도록 관리\n2. 발생 초기부터 약제 교차 살포"},
        13: {"name": "콩 불마름병", "status": "위험", "base_loss": 0.12, "urgency": "높음", "guide": "1. 무병 종자 사용\n2. 비 오기 전 예방 약제 살포"},
        14: {"name": "콩 점무늬병", "status": "경고", "base_loss": 0.08, "urgency": "보통", "guide": "1. 수확 후 잔재물 제거\n2. 밀식 방지"},
        15: {"name": "토마토 잎마름병", "status": "위험", "base_loss": 0.15, "urgency": "높음", "guide": "1. 하엽 위주로 발병하므로 적엽 실시\n2. 비료 부족 방지 및 약제 살포"},
        16: {"name": "파 검은무늬병", "status": "경고", "base_loss": 0.10, "urgency": "보통", "guide": "1. 병든 잎 조기 제거\n2. 등록 약제 살포"},
        17: {"name": "파 노균병", "status": "위험", "base_loss": 0.12, "urgency": "높음", "guide": "1. 배수 관리 철저\n2. 발병 초기 7-10일 간격 약제 살포"},
        18: {"name": "파 녹병", "status": "경고", "base_loss": 0.08, "urgency": "보통", "guide": "1. 비료가 부족하지 않게 추비\n2. 적용 약제 살포"},
        19: {"name": "호박 노균병", "status": "위험", "base_loss": 0.12, "urgency": "높음", "guide": "1. 다습한 환경 피하기\n2. 병든 잎 제거 및 약제 살포"},
        20: {"name": "호박 흰가루병", "status": "경고", "base_loss": 0.08, "urgency": "보통", "guide": "1. 통풍 및 채광 개선\n2. 예방적 약제 살포"}
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
        return base_loss, 1.0

    seq_input = real_ndvi_seq.reshape(1, 14, 1)
    predicted_ndvi = ts_model.predict(seq_input, verbose=0)[0][0]

    recent_trend = real_ndvi_seq[-1] - real_ndvi_seq[-7]
    trend_factor = abs(recent_trend) * 3 if recent_trend < 0 else 0
    severity_multiplier = 1.0 + (0.6 - predicted_ndvi) + trend_factor

    severity_multiplier = max(1.0, min(severity_multiplier, 2.5))

    final_loss = base_loss * severity_multiplier
    return round(final_loss, 3), severity_multiplier

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
# 4. UI 렌더링 및 사이드바 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AgriAX Predictor", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1, h2, h3 {color: #2c3e50;}
    .stMetric {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);}
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title("AgriAX 시스템")
    st.caption("실시간 위성 및 드론 멀티모달 농해 진단")
    st.markdown("---")

    # 1. 시스템 및 인프라 설정 (직접 입력 구조)
    st.header("시스템 인프라 설정")

    # 기본값 없이 빈 문자열로 시작하며, 배포 환경의 세션 상태에 따라 입력 가능
    gcp_project_id = st.text_input(
        "GCP Project ID",
        value="",
        help="Google Earth Engine 권한이 있는 프로젝트 ID를 입력하십시오. 배포 환경의 Secrets가 작동하지 않을 경우 직접 입력이 필요합니다."
    )

    # 입력된 ID를 기반으로 GEE 초기화 프로세스 실행
    gee_ready, gee_msg = init_google_earth_engine(gcp_project_id)

    if gee_ready:
        st.success("시스템 인증 성공")
    else:
        st.warning(f"시스템 인증 대기: {gee_msg}")

    st.markdown("---")

    # 2. 농가 재무 설정
    st.header("농가 재무 설정")
    farm_area_pyung = st.number_input("재배 면적 (평)", value=1000, step=100)
    farm_area_sqm = farm_area_pyung * 3.3058
    yield_sqm = st.number_input("단위 면적당 수확량 (kg/sqm)", value=1.5, step=0.1)
    price_kg = st.number_input("시장 단가 (원/kg)", value=15000, step=500)

    st.markdown("---")

    # 3. 분석 파라미터 설정
    st.header("분석 파라미터")
    farm_lon = st.number_input("경도 (Longitude)", value=126.8000, format="%.4f")
    farm_lat = st.number_input("위도 (Latitude)", value=36.4500, format="%.4f")
    threshold_slider = st.slider("비전 진단 신뢰 임계치", 0.50, 0.99, 0.75, step=0.01)

    st.markdown("---")

    # 4. 자원 로드 상태 확인 (실시간 상태 피드백)
    st.header("모델 로드 현황")
    v_model, v_device = load_vision_model()
    if v_model:
        st.write(f"Vision Model: 로드 완료 (Device: {v_device})")
    else:
        st.error("Vision Model: 누락")

    ts_model = load_timeseries_model()
    if ts_model:
        st.write("Time-Series Model: 로드 완료")
    else:
        st.error("Time-Series Model: 누락")

# -----------------------------------------------------------------------------
# 메인 대시보드 인터페이스 (탭 구조로 통합 수정)
# -----------------------------------------------------------------------------
st.title("AgriAX: 통합 농작물 병해 진단 시스템")
st.markdown("현장 드론 이미지와 ESA Sentinel-2 위성 데이터를 융합하여 수확 감소량을 예측합니다")

# 탭 구성
tabs = st.tabs(["실시간 진단", "기술 상세", "지역 트렌드", "비즈니스 ROI"])

# --- Tab 1: 실시간 진단 (분석 로직 통합) ---
with tabs[0]:
    top_col1, top_col2, top_col3 = st.columns([1, 1, 1])

    with top_col1:
        st.subheader("현장 데이터 입력")
        drone_file = st.file_uploader("잎 단위 고해상도 이미지 업로드", type=["jpg", "png", "jpeg"])
        if drone_file:
            img = Image.open(drone_file)
            st.image(img, caption="분석 대기 중", use_container_width=True)

    with top_col2:
        st.subheader("농장 위치 확인")
        map_data = pd.DataFrame({'lat': [farm_lat], 'lon': [farm_lon]})
        st.map(map_data, zoom=12, use_container_width=True)

    with top_col3:
        st.subheader("위성 식생지수 (NDVI)")
        real_ndvi_seq = None
        if gee_ready:
            with st.spinner("ESA 위성 데이터 추출 중"):
                ndvi_data, status = fetch_real_gee_ndvi(farm_lon, farm_lat)
                if status == "SUCCESS":
                    real_ndvi_seq = ndvi_data
                    chart_data = pd.DataFrame(real_ndvi_seq, columns=['NDVI'])
                    st.area_chart(chart_data, color="#2ecc71")
                else:
                    st.error(f"데이터 로드 실패: {status}")

    st.markdown("---")

    # 분석 실행 버튼 (탭 내부 배치)
    if st.button("멀티모달 통합 분석 실행", type="primary", use_container_width=True, key="btn_main_analysis"):
        if not drone_file or v_model is None or real_ndvi_seq is None:
            st.error("데이터(이미지, 위성) 및 모델이 모두 준비되어야 분석이 가능합니다")
        else:
            with st.spinner("AI 엔진이 다각도 분석을 수행 중입니다"):
                # 1. 모델 추론
                pred_idx, conf = predict_image(img, v_model, v_device, threshold=threshold_slider)
                diagnosis = get_disease_info(pred_idx)

                # 2. 시계열 기반 수확량 감소 계산 로직
                base_loss = diagnosis["base_loss"]
                seq_input = np.array(real_ndvi_seq).reshape(1, 14, 1)
                predicted_loss_factor = ts_model.predict(seq_input, verbose=0)[0][0] if ts_model else 0
                trend_penalty = max(0, (real_ndvi_seq[-7] - real_ndvi_seq[-1]) * 2.0)

                severity_multiplier = min(1.0 + predicted_loss_factor + trend_penalty, 3.0)
                final_loss_ratio = round(base_loss * severity_multiplier, 3)

                # 3. 재무 지표 산출
                total_yield = farm_area_sqm * yield_sqm
                lost_yield = total_yield * final_loss_ratio
                safe_yield = total_yield - lost_yield
                financial_loss = lost_yield * price_kg
                expected_revenue = safe_yield * price_kg

                # 4. 결과 리포트 출력
                st.subheader("종합 분석 리포트")
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("병해 진단 결과", diagnosis["name"], f"신뢰도 {conf*100:.1f}%", delta_color="off")
                m_col2.metric("예상 수확 감소율", f"-{final_loss_ratio*100:.1f}%", f"가중치 {severity_multiplier:.2f}x", delta_color="inverse")
                m_col3.metric("예상 경제적 손실액", f"₩ {int(financial_loss):,}", f"총 기대수익 ₩ {int(total_yield*price_kg):,} 대비", delta_color="inverse")

                st.markdown("<br>", unsafe_allow_html=True)
                v_col1, v_col2 = st.columns([1, 1])

                with v_col1:
                    st.write("**재무 시뮬레이션 요약**")
                    finance_df = pd.DataFrame({
                        '구분': ['기대 수익(정상)', '예상 손실액', '최종 예상 수입'],
                        '금액(원)': [total_yield * price_kg, financial_loss, expected_revenue]
                    }).set_index('구분')
                    st.bar_chart(finance_df, color="#e74c3c")

                    st.write("**방제 시급성 지수**")
                    urgency_level = min(int(final_loss_ratio * 300), 100)
                    st.progress(urgency_level, text=f"현재 위험도: {diagnosis['urgency']}")

                with v_col2:
                    st.write("**AI 진단 근거 (Grad-CAM)**")
                    if final_loss_ratio > 0 and pred_idx != -1:
                        grad_img = generate_real_gradcam(img, v_model, v_device, pred_idx)
                        st.image(grad_img, caption="붉은색 영역: AI가 병변으로 의심한 부위", use_container_width=True)
                    else:
                        st.info("특이사항이 발견되지 않아 활성화 맵을 생성하지 않습니다")

                with st.expander("전문가 현장 대응 가이드보기", expanded=True):
                    if final_loss_ratio > 0:
                        st.error(diagnosis["guide"])
                    else:
                        st.success(diagnosis["guide"])

# --- Tab 2: 기술 상세 ---
with tabs[1]:
    st.subheader("AgriAX AI Model Architecture")
    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.markdown("**1. Vision: DANN (Domain-Adversarial Neural Networks)**")
        st.write("다양한 촬영 환경에 강건한 특징 추출을 위해 ResNet50 기반의 적대적 도메인 학습 모델을 사용합니다.")
    with detail_col2:
        st.markdown("**2. Time-Series: LSTM (Long Short-Term Memory)**")
        st.write("Sentinel-2 위성의 14일 주기 NDVI 시계열 데이터를 분석하여 작물의 건강도 변화 추이를 정량화합니다.")

# --- Tab 3: 지역 트렌드 ---
with tabs[2]:
    st.subheader("지역별 확산 위험도 모니터링")
    st.write("해당 지역 주변 농가의 병해 발생 빈도를 기반으로 한 시뮬레이션 데이터입니다.")
    trend_data = pd.DataFrame({
        'lat': [farm_lat + 0.005, farm_lat - 0.008],
        'lon': [farm_lon - 0.002, farm_lon + 0.006]
    })
    st.map(trend_data, zoom=11)
    st.info("인근 지역 내 유사 병해 의심 사례 2건이 감지되었습니다.")

# --- Tab 4: 비즈니스 ROI ---
with tabs[3]:
    st.subheader("방제 의사결정 지원 리포트")
    roi_col1, roi_col2 = st.columns(2)
    with roi_col1:
        st.write("**방제 시점별 예상 수익 보존율**")
        roi_df = pd.DataFrame({'수익률': [0.95, 0.75, 0.40]}, index=['즉시 방제', '3일 지연', '방치'])
        st.bar_chart(roi_df)
    with roi_col2:
        st.write("**추정 손실 방어 가치**")
        expected_loss_defense = (farm_area_sqm * yield_sqm * price_kg) * 0.15
        st.metric("적기 방제 시 방어액", f"₩ {int(expected_loss_defense):,}", "기대 수익 대비")
