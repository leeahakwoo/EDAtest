# 범용 XAI 진단 도구  – Integrated Gradients + SHAP
import streamlit as st, torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, matplotlib.pyplot as plt, shap, numpy as np
from captum.attr import IntegratedGradients

# 1. 모델 정의 ───────────────────────────────────────────
class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(4, 16), nn.Linear(16, 3)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

FEATURE_NAMES = ["SepLen", "SepWid", "PetLen", "PetWid"]

# 2. 페이지 설정 ────────────────────────────────────────
st.set_page_config(page_title="범용 XAI 진단", layout="centered")
st.title("🧠 범용 XAI 진단 도구")
st.write("PyTorch **state_dict(.pt)** + CSV(숫자 4열)를 올리면 예측과 두 가지 XAI(IG·SHAP) 시각화를 제공합니다.")

# 3. 모델 업로드 ────────────────────────────────────────
up_model = st.file_uploader("📂 가중치(.pt) 업로드", type=["pt"])
model = None
if up_model:
    try:
        model = IrisNet()
        sd = torch.load(up_model, weights_only=True, map_location="cpu")
        model.load_state_dict(sd); model.eval()
        st.success("✅  모델 로딩 완료")
    except Exception as e:
        st.error(f"❌  모델 로딩 실패: {e}")
        st.stop()

# 4. CSV 업로드 ─────────────────────────────────────────
up_csv = st.file_uploader("📄 CSV (숫자 4열) 업로드", type=["csv"])
if up_csv:
    try:
        df = pd.read_csv(up_csv)
        if df.shape[1] > 4: df = df.iloc[:, :4]
        st.dataframe(df.head())
        tensor = torch.tensor(df.values, dtype=torch.float32)
        if tensor.ndim == 1: tensor = tensor.view(1, -1)
    except Exception as e:
        st.error(f"❌  CSV 처리 오류: {e}")
        st.stop()

# 5. 예측 + XAI ─────────────────────────────────────────
if model and up_csv:
    with torch.no_grad():
        logits = model(tensor)
        pred   = torch.argmax(logits, 1)

    st.markdown("### ✅ 예측 결과")
    st.write(pred.numpy())

    # ── (1) Integrated Gradients ──────────────────────
    st.markdown("## 🔍 Integrated Gradients 시각화")
    # ▸ 사용자 선택: 자동(pred) vs 특정 클래스
    class_opt = st.selectbox("기여도 대상 클래스",
                             options=["auto(pred)", 0, 1, 2])
    target_idx = pred if class_opt == "auto(pred)" else torch.tensor([class_opt]*len(pred))

    ig = IntegratedGradients(model)
    attr, _ = ig.attribute(tensor, target=target_idx, return_convergence_delta=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(attr.detach().numpy(), aspect="auto", cmap="hot")
    ax.set_xticks(np.arange(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right")
    ax.set_ylabel("샘플 index")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

    # ── (2) SHAP Global Feature Bar ───────────────────
    st.markdown("## 📊 SHAP 글로벌 중요도")
    explainer = shap.GradientExplainer(model, tensor)  # baseline = 입력 자체
    shap_vals = explainer.shap_values(tensor)
    # shap_values 는 list[ num_class ][num_sample, num_feature ]
    # → 평균 |값| 계산
    mean_abs = np.mean(np.abs(shap_vals[0]), axis=0)
    shap_df  = pd.DataFrame({"feature": FEATURE_NAMES, "mean_abs": mean_abs})
    shap_df.sort_values("mean_abs", ascending=False, inplace=True)

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.barh(shap_df["feature"], shap_df["mean_abs"])
    ax2.invert_yaxis(); ax2.set_xlabel("평균 |SHAP 값|")
    st.pyplot(fig2)
