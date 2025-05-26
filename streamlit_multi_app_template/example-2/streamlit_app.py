import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="생산라인 병목 시뮬레이션", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏭 생산라인 병목 시뮬레이션 + 시나리오 비교 + Plotly 시각화")

mode = st.radio("분석 모드 선택", ["📂 실제 데이터 업로드", "🧪 시뮬레이션 실행"], index=0)

if mode == "📂 실제 데이터 업로드":
    uploaded_file = st.file_uploader("CSV 파일 업로드 (Scenario, A, B, C 열 포함)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("업로드된 데이터 미리보기")
        st.dataframe(df.head())

        st.subheader("⏱️ 시나리오별 생산 시간 분포 (Plotly)")
        result_df = pd.DataFrame()
        for scenario in df['Scenario'].unique():
            row = df[df['Scenario'] == scenario].iloc[0]
            A = np.random.exponential(row['A'], 500)
            B = np.random.exponential(row['B'], 500)
            C = np.random.exponential(row['C'], 500)
            total_time = A + B + C
            temp = pd.DataFrame({'총생산시간': total_time, '시나리오': scenario})
            result_df = pd.concat([result_df, temp], axis=0)

        fig = px.histogram(result_df, x="총생산시간", color="시나리오", nbins=50, barmode="overlay", marginal="violin")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 시나리오별 통계 비교")
        summary = result_df.groupby("시나리오")["총생산시간"].agg(['mean', 'std', 'min', 'max']).reset_index()
        summary.columns = ['시나리오', '평균', '표준편차', '최소', '최대']
        st.dataframe(summary)

else:
    st.markdown("👉 시나리오 1: 공정 B 느림 / 시나리오 2: 공정 B 개선안")

    col1, col2 = st.columns(2)
    with col1:
        time_A = st.slider("공정 A 처리시간", 5, 20, 10)
        time_B_1 = st.slider("공정 B 처리시간 - 시나리오 1", 5, 30, 20)
        time_B_2 = st.slider("공정 B 처리시간 - 시나리오 2", 5, 30, 12)
        time_C = st.slider("공정 C 처리시간", 5, 20, 10)
    with col2:
        n = st.slider("샘플 수", 100, 3000, 1000)

    A = np.random.exponential(time_A, n)
    B1 = np.random.exponential(time_B_1, n)
    B2 = np.random.exponential(time_B_2, n)
    C = np.random.exponential(time_C, n)
    total_1 = A + B1 + C
    total_2 = A + B2 + C

    df_compare = pd.DataFrame({
        "총생산시간": np.concatenate([total_1, total_2]),
        "시나리오": ["시나리오 1"] * n + ["시나리오 2"] * n
    })

    st.subheader("📊 Plotly 기반 생산시간 분포 비교")
    fig2 = px.histogram(df_compare, x="총생산시간", color="시나리오", barmode="overlay", nbins=50)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📈 시나리오별 통계 비교")
    summary2 = df_compare.groupby("시나리오")["총생산시간"].agg(['mean', 'std', 'min', 'max']).reset_index()
    summary2.columns = ['시나리오', '평균', '표준편차', '최소', '최대']
    st.dataframe(summary2)
