import streamlit as st
import pandas as pd
import plotly.express as px

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
            <p style="font-size: 18px; font-weight: bold; color: #212529; margin: 0;">{value}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def calculate_fallback_finance(diagnosis_name):
    """
    Tab 3를 방문하지 않았을 경우를 대비한 기본 재무 계산 로직
    """
    default_area = 3300
    default_yield = 1.5
    default_price = 15000

    potential_revenue = default_area * default_yield * default_price
    loss_rate = 0.35 if "탄저병" in diagnosis_name else (0.0 if "정상" in diagnosis_name else 0.20)

    loss = potential_revenue * loss_rate
    cost = default_area * 150
    roi = ((loss * 0.8 - cost) / cost * 100) if cost > 0 else 0

    return {
        'no_action_loss': f"{int(loss):,}원",
        'early_action_cost': f"{int(cost):,}원",
        'roi_percentage': f"{roi:.1f}%",
        'loss_val': loss,
        'cost_val': cost
    }

def parse_money(value):
    if isinstance(value, (int, float)):
        return value
    return int(str(value).replace('원', '').replace(',', '').strip())

def render():
    st.header("Executive Summary")

    diagnosis = st.session_state.get('diagnosis')

    if not diagnosis:
        st.warning("'작물 병해 판별' 탭에서 작물 이미지를 분석해 주세요.")
        return

    # Tab 3 데이터가 없으면 fallback 로직 실행
    finance_data = st.session_state.get('financial_results')
    if finance_data:
        results = finance_data.get('roi_result', {})
        loss_val = parse_money(results.get('no_action_loss', 0))
        cost_val = parse_money(results.get('early_action_cost', 0))
        roi_pct = results.get('roi_percentage', '0%')
    else:
        fallback = calculate_fallback_finance(diagnosis['name'])
        results = fallback
        loss_val = fallback['loss_val']
        cost_val = fallback['cost_val']
        roi_pct = fallback['roi_percentage']

    st.markdown("### 경제적 손실 및 방제 효과 분석")

    try:
        profit_val = (loss_val * 0.8) - cost_val if loss_val > 0 else 0

        loss_data = pd.DataFrame({
            "구분": ["방치 시 손실", "방제 비용", "방제 후 기대수익"],
            "금액": [loss_val, cost_val, profit_val]
        })

        fig = px.bar(loss_data, x="구분", y="금액", color="구분",
                     color_discrete_map={"방치 시 손실": "#e74c3c", "방제 비용": "#3498db", "방제 후 기대수익": "#2ecc71"})

        fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"차트 생성 중 오류 발생: {e}")

    st.success("비즈니스 종합 분석 보고서 생성이 완료되었습니다.")

    st.markdown("#### 핵심 요약 지표")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        custom_metric("진단 작물", diagnosis['name'])
    with m2:
        custom_metric("방치 시 손실", results.get('no_action_loss', '0원'))
    with m3:
        custom_metric("방제 비용", results.get('early_action_cost', '0원'))
    with m4:
        custom_metric("예상 ROI", roi_pct)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 상세 진단 정보")
        st.info(f"**판별 결과:** {diagnosis['name']}\n\n**위험 등급:** {diagnosis.get('status', '보통')}")
        st.write("AI 모델의 분석 신뢰도와 작물 상태를 종합할 때 조속한 대응이 권고됩니다.")

    with c2:
        st.markdown("#### 경제성 평가 요약")
        summary = "방제 시 시나리오 기반 기대 순수익이 방치 시 손실액보다 높게 산출되었습니다. 초기 방제를 통해 잠재적 손실을 최소화하는 전략이 유효합니다."
        st.warning(summary)
