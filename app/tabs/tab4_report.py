import streamlit as st
import pandas as pd
import plotly.express as px
import time
from src.db_manager import DatabaseManager

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
    default_area = 3300
    default_yield = 1.5
    default_price = 15000

    potential_revenue = default_area * default_yield * default_price
    loss_rate = 0.35 if "탄저병" in diagnosis_name else (0.0 if "정상" in diagnosis_name else 0.20)

    loss = potential_revenue * loss_rate
    cost = default_area * 150 if loss > 0 else 0
    profit = (loss * 0.8) - cost if loss > 0 else 0

    return loss, cost, profit

def render():
    st.header("종합 리포트")

    diagnosis = st.session_state.get('diagnosis')

    if not diagnosis:
        st.warning("분석된 데이터가 없습니다. '작물 병해 판별' 탭에서 진단을 먼저 수행해 주십시오.")
        return

    # 데이터베이스 초기화 및 데모 데이터 삽입 (최초 1회)
    DatabaseManager.init_db()
    # DatabaseManager.insert_mock_data()

    loss_val, cost_val, profit_val = calculate_fallback_finance(diagnosis['name'])

    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        st.markdown(f"**현재 진단 상태:** {diagnosis['name']}")
    with col_btn2:
        if st.button("현재 진단 결과 DB 저장", use_container_width=True):
            try:
                DatabaseManager.insert_record(diagnosis['name'], loss_val, cost_val, profit_val)
                st.success("데이터베이스에 정상적으로 기록되었습니다.")
            except Exception as e:
                st.error(f"DB 저장 오류: {e}")

    st.markdown("---")

    # 1. 시계열 트렌드 분석 섹션
    st.subheader("과거 진단 이력 및 재무 트렌드")

    df_history = DatabaseManager.get_history()

    if not df_history.empty:
        df_history['record_date'] = pd.to_datetime(df_history['record_date'])

        # 선 그래프 (시계열 예상 손실액)
        fig_trend = px.line(
            df_history,
            x="record_date",
            y="loss_amount",
            color="disease_name",
            markers=True,
            labels={"record_date": "진단 일시", "loss_amount": "예상 손실액 (원)", "disease_name": "질병명"},
            title="최근 진단별 예상 손실액 추이"
        )
        fig_trend.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig_trend, use_container_width=True, key="tab4_history_trend_chart")

        # 히스토리 데이터프레임
        with st.expander("상세 진단 이력 데이터 보기"):
            st.dataframe(
                df_history.sort_values(by="record_date", ascending=False),
                column_config={
                    "record_date": "기록 일시",
                    "disease_name": "상태",
                    "loss_amount": "손실액",
                    "cost_amount": "방제비",
                    "net_profit": "순이익"
                },
                hide_index=True,
                use_container_width=True
            )
    else:
        st.info("기록된 데이터가 없습니다.")

    st.markdown("---")

    # 2. 경영진 요약 리포트 생성 (Prompt Chaining 구조)
    st.subheader("LLM 기반 의사결정 요약서")

    if st.button("자동 인사이트 리포트 생성"):
        with st.spinner("과거 데이터 및 현재 진단 결과를 분석하여 경영진 리포트를 작성 중입니다."):
            time.sleep(1.5) # LLM 체이닝 딜레이 시뮬레이션

            # 실제 LLM 연동 시 이 영역에 프롬프트 체이닝 결과값을 매핑합니다.
            st.markdown("#### 1. 현황 요약 및 위험도 평가")
            st.write(f"최근 시계열 데이터 분석 결과, **{diagnosis['name']}** 발생 빈도가 유지되고 있습니다. 현재 예상되는 최대 재무 손실은 {int(loss_val):,}원이며, 이는 초기 대응이 지연될 경우 구역 전체로 확산될 위험이 있습니다.")

            st.markdown("#### 2. 방제 시나리오 및 우선순위 제안")
            st.write(f"추천 액션: **집중 화학 방제 및 환경 제어**. 초기 방제 비용 {int(cost_val):,}원 투입 시, 기대 수익 {int(profit_val):,}원을 보존할 수 있어 약 {(profit_val/cost_val*100):.1f}%의 ROI가 산출됩니다.")

            st.markdown("#### 3. 차주 관리 지침")
            st.write("방제 후 3일 간격으로 위성 NDVI 데이터를 지속 모니터링하고, 토양 습도를 현행 대비 15% 낮추는 재배적 관리가 병행되어야 합니다.")

        st.download_button(
            label="리포트 PDF 다운로드",
            data=b"PDF_Dummy_Data_Export",
            file_name=f"AgriAX_Report_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )
