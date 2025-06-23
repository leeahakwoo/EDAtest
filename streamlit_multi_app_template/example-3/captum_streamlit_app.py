import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
import shap

# ── 모델 정의 ───────────────────────────────
class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(4, 16), nn.Linear(16, 3)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

FEATURE_NAMES = ["SepLen", "SepWid", "PetLen", "PetWid"]

# ── Streamlit 설정 ──────────────────────────
st.set_page_config(page_title="범용 XAI 진단", layout="centered")
st.title("🧠 범용 XAI 진단 도구")

# ── 1. 모델 업로드 ──────────────────────────
up_model = st.file_uploader("📂 PyTorch state_dict (.pt)", type=["pt"])
model = None
if up_model:
    model = IrisNet()
    sd = torch.load(up_model, weights_only=True, map_location="cpu")
    model.load_state_dict(sd); model.eval()
    st.success("✅ 모델 로드 완료")

# ── 2. CSV 업로드 ───────────────────────────
up_csv = st.file_uploader("📄 CSV (숫자 4열, 헤더 허용)", type=["csv"])
if up_csv:
    df = pd.read_csv(up_csv)
    if df.shape[1] > 4: df = df.iloc[:, :4]
    st.dataframe(df.head())
    tensor = torch.tensor(df.values, dtype=torch.float32)
    if tensor.ndim == 1: tensor = tensor.view(1, -1)

# ── 3. 예측 + IG + SHAP ─────────────────────
if model and up_csv:
    pred = torch.argmax(model(tensor), 1)
    st.markdown("### ✅ 예측 결과")
    st.write(pred.numpy())

    # (A) Integrated Gradients
    ig = IntegratedGradients(model)
    attr, _ = ig.attribute(tensor, target=pred, return_convergence_delta=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(attr.detach().numpy(), aspect="auto", cmap="hot")
    ax.set_xticks(range(4)); ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right")
    ax.set_ylabel("샘플 index"); fig.colorbar(im, ax=ax)
    st.pyplot(fig)

    # (B) SHAP 글로벌 중요도 (GradientExplainer)
    st.markdown("## 📊 SHAP 글로벌 중요도")
    expl = shap.GradientExplainer(model, tensor)
    shap_list = expl.shap_values(tensor)           # list[num_classes][N, 4]
    shap_arr  = np.stack(shap_list, axis=0)        # shape: [C, N, 4]
    mean_abs  = np.mean(np.abs(shap_arr), axis=(0, 1))   # 길이 = 4

    shap_df = pd.DataFrame({"feature": FEATURE_NAMES, "mean_abs": mean_abs}) \
                .sort_values("mean_abs", ascending=True)

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.barh(shap_df["feature"], shap_df["mean_abs"])
    ax2.set_xlabel("평균 |SHAP 값|"); ax2.set_ylabel("feature")
    st.pyplot(fig2)
