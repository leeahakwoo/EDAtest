import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO

# ✅ 한글 폰트 설정 (NanumGothic이 서버에 설치되어 있어야 함)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# ✅ Streamlit 페이지 기본 설정
st.set_page_config(page_title="생산라인 병목 시뮬레이션", layout="wide")

# ✅ 제목 영역
st.title("🏭 생산 라인 병목 시뮬레이션 (Streamlit 인터랙티브)")
st.markdown("""
이 시뮬레이션은 공정 A → 공정 B → 공정 C의 연속 작업에서 공정 B의 처리 시간이 전체 생산 시간에 미치는 영향을 분석합니다.  
**업로드된 시나리오 데이터를 기반으로 병목 해결 전략을 시각적으로 비교**합니다.
""")

# ✅ 시나리오 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요 (컬럼: Scenario, A, B, C)", type=["csv"])

if uploaded_file is not None:
    # ✅ 데이터 읽기
    df = pd.read_csv(uploaded_file)

    # ✅ 시뮬레이션 함수
    def simulate_from_dataframe(df):
        results = {}
        for scenario in df['Scenario'].unique():
            row = df[df['Scenario'] == scenario].iloc[0]
            A = np.random.exponential(row['A'], 1000)
            B = np.random.exponential(row['B'], 1000)
            C = np.random.exponential(row['C'], 1000)
            total_time = A + B + C
            results[scenario] = total_time
        return pd.DataFrame(results)

    # ✅ 시뮬레이션 실행
    df_result = simulate_from_dataframe(df)

    # ✅ 시각화: 생산시간 분포
    st.subheader("⏱️ 전체 생산 시간 분포 비교")
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in df_result.columns:
        sns.kdeplot(df_result[col], label=col, fill=True, ax=ax)
    ax.set_xlabel("총 생산 시간 (분)")
    ax.set_ylabel("밀도")
    ax.set_title("시나리오별 생산 시간 분포")
    ax.legend()
    st.pyplot(fig)

    # ✅ 통계 요약
    st.subheader("📊 시나리오 비교 요약")
    summary = df_result.describe().T[["mean", "std", "min", "max"]].rename(columns={
        "mean": "평균",
        "std": "표준편차",
        "min": "최소값",
        "max": "최대값"
    })
    st.dataframe(summary)

else:
    st.info("시뮬레이션을 시작하려면 CSV 파일을 업로드하세요.")
