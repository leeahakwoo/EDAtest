import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 페이지 설정
st.set_page_config(page_title="생산라인 병목 시뮬레이션", layout="wide")

# 제목
st.title("🏭 생산 라인 병목 시뮬레이션 (Streamlit 인터랙티브)")
st.markdown("""
이 시뮬레이션은 공정 A → 공정 B → 공정 C의 연속 작업에서 공정 B의 처리 시간이 전체 생산 시간에 미치는 영향을 분석합니다.
시나리오 비교를 통해 병목 해결 전략을 시각적으로 비교할 수 있습니다.
""")

# 사용자 입력 UI
col1, col2 = st.columns(2)
with col1:
    time_A = st.slider("공정 A 처리시간 (분)", 5, 20, 10)
    time_B_1 = st.slider("공정 B 처리시간 - 시나리오 1 (분)", 5, 30, 20)
    time_B_2 = st.slider("공정 B 처리시간 - 시나리오 2 (분)", 5, 30, 12)
    time_C = st.slider("공정 C 처리시간 (분)", 5, 20, 10)

with col2:
    num_samples = st.number_input("시뮬레이션 반복 횟수", min_value=100, max_value=5000, value=1000)
    st.markdown("👉 시나리오 1: 공정 B 기본값  \n👉 시나리오 2: 공정 B 개선안", unsafe_allow_html=True)

# 시뮬레이션 함수
def simulate(time_A, time_B, time_C, n=1000):
    A = np.random.exponential(time_A, n)
    B = np.random.exponential(time_B, n)
    C = np.random.exponential(time_C, n)
    total_time = A + B + C
    return total_time

# 시뮬레이션 실행
total_1 = simulate(time_A, time_B_1, time_C, num_samples)
total_2 = simulate(time_A, time_B_2, time_C, num_samples)

# 데이터프레임 생성
df_result = pd.DataFrame({
    f"시나리오 1 (B = {time_B_1}분)": total_1,
    f"시나리오 2 (B = {time_B_2}분)": total_2
})

# 시각화: 분포 그래프
st.subheader("⏱️ 전체 생산 시간 분포 비교")
fig, ax = plt.subplots(figsize=(10, 5))
for col in df_result.columns:
    sns.kdeplot(df_result[col], label=col, fill=True, ax=ax)
ax.set_xlabel("총 생산 시간 (분)")
ax.set_ylabel("밀도")
ax.set_title("시나리오별 생산 시간 분포")
ax.legend()
st.pyplot(fig)

# 통계 요약 테이블
st.subheader("📊 시나리오 비교 요약")
summary = df_result.describe().T[["mean", "std", "min", "max"]].rename(columns={
    "mean": "평균",
    "std": "표준편차",
    "min": "최소값",
    "max": "최대값"
})
st.dataframe(summary)
