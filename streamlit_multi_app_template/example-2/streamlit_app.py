import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ✅ 구글 웹폰트 설정 (Noto Sans KR)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 앱 구성 예시
st.title("🚗 자동차 데이터 연비 분석 (Streamlit + GPT)")

st.markdown("""
이 앱은 업로드한 CSV 데이터를 기반으로 제조 연도별 평균 연비를 분석하고 시각화합니다.  
또한 이상치(비정상적으로 낮은 연비)를 감지하여 강조합니다.
""")

uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'year' not in df.columns or 'mpg' not in df.columns:
        st.error("데이터에 'year' 및 'mpg' 열이 포함되어 있어야 합니다.")
    else:
        # 연도별 평균 연비 시각화
        st.subheader("📊 연도별 평균 연비")
        year_avg = df.groupby('year')['mpg'].mean().reset_index()

        fig, ax = plt.subplots()
        sns.barplot(data=year_avg, x='year', y='mpg', ax=ax, palette="Blues_d")
        ax.set_title("연도별 평균 연비")
        ax.set_xlabel("제조 연도")
        ax.set_ylabel("평균 연비 (mpg)")
        st.pyplot(fig)

        # 이상치 탐지
        mean_mpg = df['mpg'].mean()
        std_mpg = df['mpg'].std()
        outliers = df[df['mpg'] < mean_mpg - 1.5 * std_mpg]

        st.subheader("🚨 이상치 탐지 결과")
        st.dataframe(outliers)
