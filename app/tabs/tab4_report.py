import streamlit as st
import pandas as pd
import plotly.express as px

def render():
    st.header("Executive Summary")
    diagnosis = st.session_state.get('diagnosis')
    results = st.session_state.get('financial_results', {}).get('roi_result', {})

    if not diagnosis or not st.session_state.get('finance_done'):
        st.info("진단 및 재무 분석이 완료되면 종합 보고서가 생성됩니다.")
        return

    st.markdown("### 경제적 손실 및 방제 효과 분석")

    loss_data = pd.DataFrame({
        "구분": ["방치 시 손실", "방제 비용", "방제 후 기대 수익"],
        "금액": [
            int(results.get('no_action_loss', '0').replace('원','').replace(',','')),
            int(results.get('early_action_cost', '0').replace('원','').replace(',','')),
            int(results.get('total_yield_value', '0').replace('원','').replace(',',''))
        ]
    })

    fig = px.bar(loss_data, x="구분", y="금액", color="구분",
                 title="방제 여부에 따른 경제성 비교",
                 color_discrete_map={"방치 시 손실": "#e74c3c", "방제 비용": "#3498db", "방제 후 기대 수익": "#2ecc71"})
    st.plotly_chart(fig, use_container_width=True)

    st.success("비즈니스 종합 분석이 완료되었습니다.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### AI 진단 요약")
        st.write(f"- 판별: {diagnosis['name']}")
        st.write(f"- 시급성: {diagnosis['urgency']}")
        st.write(f"- 기본 가이드: {diagnosis['guide'].split(chr(10))[0]}")

    with col2:
        st.markdown("#### 재무적 타격 및 의사결정")
        st.write(f"- 방치 시 손실: {results.get('no_action_loss', '0원')}")
        st.write(f"- 방제 소요 비용: {results.get('early_action_cost', '0원')}")
        st.write(f"- 투자 대비 수익률(ROI): {results.get('roi_percentage', '0%')}")

    st.markdown("---")
    st.markdown("#### 방제 전략 요약")
    st.write(results.get("scenario_summary", "요약 데이터가 없습니다."))
