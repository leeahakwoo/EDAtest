import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ 웹폰트 (한글 깨짐 방지)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="생산라인 병목 시뮬레이션", layout="wide")
st.title("🏭 생산 라인 병목 시뮬레이션 (업로드 or 시뮬레이션 선택 가능)")

mode = st.radio("분석 모드 선택", ["📂 실제 데이터 업로드", "🧪 시뮬레이션 실행"], index=0)

# 📂 실제 데이터 업로드
if mode == "📂 실제 데이터 업로드":
    uploaded_file = st.file_uploader("CSV 파일 업로드 (예: 공정별 처리 시간)", type=["csv"])

    if uploaded_file:
        df_real = pd.read_csv(uploaded_file)
        st.success("데이터 업로드 완료!")
        st.write("🔍 데이터 미리보기", df_real.head())

        st.subheader("⏱️ 공정별 생산 시간 히스토그램")
        fig, ax = plt.subplots()
        df_real.plot.hist(alpha=0.6, bins=30, ax=ax)
        plt.title("공정별 생산 시간 분포")
        st.pyplot(fig)

        st.subheader("📊 통계 요약")
        st.dataframe(df_real.describe().T)
    else:
        st.warning("CSV 파일을 업로드해주세요. 열 이름은 '공정A', '공정B', '공정C' 등이 포함되어야 합니다.")

# 🧪 시뮬레이션 모드
elif mode == "🧪 시뮬레이션 실행":
    col1, col2 = st.columns(2)
    with col1:
        time_A = st.slider("공정 A 처리시간 (분)", 5, 20, 10)
        time_B_1 = st.slider("공정 B 처리시간 - 시나리오 1 (분)", 5, 30, 20)
        time_B_2 = st.slider("공정 B 처리시간 - 시나리오 2 (분)", 5, 30, 12)
        time_C = st.slider("공정 C 처리시간 (분)", 5, 20, 10)
    with col2:
        num_workers_B_1 = st.slider("공정 B 작업자 수 - 시나리오 1", 1, 5, 1)
        num_workers_B_2 = st.slider("공정 B 작업자 수 - 시나리오 2", 1, 5, 2)
        num_samples = st.number_input("시뮬레이션 반복 횟수", min_value=100, max_value=5000, value=1000)

    st.markdown("👉 시나리오 1: 공정 B 기본 조건  
👉 시나리오 2: 공정 B 개선안", unsafe_allow_html=True)

    def simulate(time_A, time_B, time_C, workers_B, n=1000):
        A = np.random.exponential(time_A, n)
        B = np.random.exponential(time_B / workers_B, n)
        C = np.random.exponential(time_C, n)
        return A + B + C

    total_1 = simulate(time_A, time_B_1, time_C, num_workers_B_1, num_samples)
    total_2 = simulate(time_A, time_B_2, time_C, num_workers_B_2, num_samples)

    df_result = pd.DataFrame({
        f"시나리오 1 (B = {time_B_1}분, 작업자 {num_workers_B_1})": total_1,
        f"시나리오 2 (B = {time_B_2}분, 작업자 {num_workers_B_2})": total_2
    })

    st.subheader("⏱️ 전체 생산 시간 분포 비교")
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in df_result.columns:
        sns.kdeplot(df_result[col], label=col, fill=True, ax=ax)
    ax.set_xlabel("총 생산 시간 (분)")
    ax.set_ylabel("밀도")
    ax.set_title("시나리오별 생산 시간 분포")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📊 시나리오 통계 요약")
    summary = df_result.describe().T[["mean", "std", "min", "max"]].rename(columns={
        "mean": "평균",
        "std": "표준편차",
        "min": "최소값",
        "max": "최대값"
    })
    st.dataframe(summary)

    st.subheader("🧠 자동 분석 요약")
    col_names = df_result.columns.tolist()
    m1, m2 = df_result[col_names[0]].mean(), df_result[col_names[1]].mean()
    diff = round(m1 - m2, 2)

    if diff > 0:
        st.markdown(f"✅ **시나리오 2**가 더 효율적입니다. 평균 생산 시간이 **{diff}분** 더 짧습니다.")
    elif diff < 0:
        st.markdown(f"⚠️ **시나리오 1**이 더 빠릅니다. 평균 생산 시간이 **{-diff}분** 더 짧습니다.")
    else:
        st.markdown("⏸️ 두 시나리오의 평균 생산 시간이 동일합니다.")
