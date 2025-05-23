pip install matplotlib

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="LLM 기반 자동차 연비 분석", layout="centered")

st.title("🚗 자동차 데이터 연비 분석 (Streamlit + GPT)")
st.markdown("""
이 앱은 업로드한 CSV 데이터를 기반으로 제조 연도별 평균 연비를 분석하고 시각화합니다.
또한 이상치(비정상적으로 낮은 연비)를 감지하여 강조합니다.
""")

uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # 연비와 연도 관련 컬럼 추출
    if 'year' not in df.columns or 'mpg' not in df.columns:
        st.error("데이터에 'year' 및 'mpg' 열이 포함되어 있어야 합니다.")
    else:
        st.subheader("📊 연도별 평균 연비")
        year_avg = df.groupby('year')['mpg'].mean().reset_index()

        # 이상치 탐지 (평균보다 1.5표준편차 이상 낮은 mpg)
        mean_mpg = df['mpg'].mean()
        std_mpg = df['mpg'].std()
        outliers = df[df['mpg'] < mean_mpg - 1.5 * std_mpg]

        fig, ax = plt.subplots()
        ax.plot(year_avg['year'], year_avg['mpg'], marker='o', label='연도별 평균 연비')
        ax.set_xlabel("제조 연도")
        ax.set_ylabel("평균 연비 (mpg)")
        ax.set_title("연도별 평균 연비 추이")
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("### 🚨 이상치 요약")
        st.dataframe(outliers[['year', 'mpg']])
        st.success(f"총 {len(outliers)}개의 이상치가 감지되었습니다.")
