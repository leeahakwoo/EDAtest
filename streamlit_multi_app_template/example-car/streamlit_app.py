import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="다양한 자동차 데이터 분석", layout="wide")
st.title("🚘 다양한 자동차 데이터 분석 데모")

uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("데이터 업로드 완료!")
    st.write("### 🔍 데이터 미리보기")
    st.dataframe(df.head())

    # 기본 통계
    st.write("### 📊 기본 통계")
    st.dataframe(df.describe())

    # 연도별 평균 연비
    st.write("### 📈 연도별 평균 연비")
    fig, ax = plt.subplots()
    df.groupby("year")["mpg"].mean().plot(kind="line", marker="o", ax=ax)
    plt.ylabel("평균 MPG")
    plt.grid(True)
    st.pyplot(fig)

    # 제조사별 평균 연비
    st.write("### 🏭 제조사별 평균 연비")
    fig, ax = plt.subplots()
    df.groupby("make")["mpg"].mean().sort_values().plot(kind="barh", ax=ax, color="skyblue")
    plt.xlabel("평균 MPG")
    st.pyplot(fig)

    # 연비와 배기량의 관계
    st.write("### ⚙️ 배기량과 연비 관계 (산점도)")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="displacement", y="mpg", hue="origin", ax=ax)
    st.pyplot(fig)

    # 연료 종류별 연비 분포 (박스플롯)
    st.write("### ⛽ 연료 타입별 연비 분포")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="fuel_type", y="mpg", ax=ax)
    st.pyplot(fig)

    # 상관 관계 히트맵
    st.write("### 🧠 수치 변수 간 상관관계")
    fig, ax = plt.subplots()
    sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
