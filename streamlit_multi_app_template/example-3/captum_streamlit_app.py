# ───────────────────────────────────────────────
#  범용 XAI 진단 도구  (IG + SHAP, CSV 4-컬럼 입력)
#  · Python 3.11  · requirements.txt 참고
# ───────────────────────────────────────────────
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
import shap

# 1. PyTorch 모델 ────────────────────────────────
class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(4, 16), nn.Linear(16, 3)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

FEATURE_NAMES = ["SepLen", "SepWid", "PetLen", "PetWid"]

# 2. Streamlit 페이지 설정 ───────────────────────
st.set_page_config(page_title="범용 XAI 진단", layout="centered")
st.title("🧠 범용 XAI 진단 도구")
st.caption(
    "PyTorch **state_dict(.pt)** 가중치와 숫자 4-열 **CSV** 를 업로드하면 "
    "Integrated Gradients·SHAP 전역 중요도를 시각화합니다."
)

# 3. 모델 업로드 ─────────────────────────────────
up_model = st.file_uploader("📂 PyTorch state_dict (.pt)", type=["pt"])
model = None
if up_model:
    try:
        model = IrisNet()
        state_dict = torch.load(up_model, weights_only=True, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        st.success("✅ 모델 로드 완료")
    except Exception as e:
        st.error(f"❌ 모델 로딩 실패: {e}")
        st.stop()

# 4. CSV 업로드 ──────────────────────────────────
up_csv = st.file_uploader("📄 CSV (숫자 4열, 헤더 허용)", type=["csv"])
if up_csv:
    try:
        df = pd.read_csv(up_csv)
        if df.shape[1] > 4:                       # 열이 4개 초과하면 앞 4열만 사용
            df = df.iloc[:, :4]
        st.dataframe(df.head())
        tensor = torch.tensor(df.values, dtype=torch.float32)
        if tensor.ndim == 1:                      # 1행 CSV 인 경우 shape 맞추기
            tensor = tensor.view(1, -1)
    except Exception as e:
        st.error(f"❌ CSV 처리 오류: {e}")
        st.stop()

# 5. 예측 + Integrated Gradients + SHAP ──────────
if model and up_csv:
    pred = torch.argmax(model(tensor), 1)

    st.markdown("### ✅ 예측 결과")
    st.write(pred.numpy())

    # (A) Integrated Gradients ───────────────────
    st.markdown("## 🔍 Integrated Gradients (샘플별 히트맵)")
    ig = IntegratedGradients(model)
    attr, _ = ig.attribute(tensor, target=pred, return_convergence_delta=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(attr.detach().numpy(), aspect="auto", cmap="hot")
    ax.set_xticks(range(4))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right")
    ax.set_ylabel("샘플 index")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

    # (B) SHAP 글로벌 중요도 ──────────────────────
    st.markdown("## 📊 SHAP 글로벌 중요도 (GradientExplainer)")

    explainer  = shap.GradientExplainer(model, tensor)
    raw_vals   = explainer.shap_values(tensor)    # list[C][N,4] 또는 ndarray[N,4]

    if isinstance(raw_vals, list):                # 다중 클래스
        shap_arr = np.stack(raw_vals, axis=0)     # (C, N, 4)
        mean_abs = np.mean(np.abs(shap_arr), axis=(0, 1))
    else:                                         # 단일 출력
        mean_abs = np.mean(np.abs(raw_vals), axis=0)

    shap_df = (
        pd.DataFrame({"feature": FEATURE_NAMES, "mean_abs": mean_abs})
        .sort_values("mean_abs", ascending=True)
    )

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.barh(shap_df["feature"], shap_df["mean_abs"])
    ax2.set_xlabel("평균 |SHAP 값|")
    ax2.set_ylabel("feature")
    st.pyplot(fig2)
