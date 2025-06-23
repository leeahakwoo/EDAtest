# 범용 XAI 진단 도구 - 표형 데이터 + Captum 기반
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

# ───────────────────────────────────────────────────────────
# 1. 모델 클래스 정의
# ───────────────────────────────────────────────────────────
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1  = nn.Linear(4, 16)
        self.fc2  = nn.Linear(16, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ───────────────────────────────────────────────────────────
# 2. Streamlit 페이지 설정
# ───────────────────────────────────────────────────────────
st.set_page_config(page_title="범용 XAI 진단 도구", layout="centered")
st.title("🧠 범용 XAI 진단 도구")
st.markdown(
    "PyTorch **가중치(state_dict) 파일**(.pt)과 CSV 입력 데이터를 업로드하면 "
    "예측 결과와 Captum XAI 시각화를 제공합니다."
)

# ───────────────────────────────────────────────────────────
# 3. 모델 업로드
# ───────────────────────────────────────────────────────────
uploaded_model = st.file_uploader("📂 PyTorch 가중치 (.pt) 업로드", type=["pt"])
model = None

if uploaded_model is not None:
    try:
        # ❶ 빈 모델 인스턴스를 만들고
        model = IrisNet()
        # ❷ weights_only=True 옵션으로 state_dict 만 로드
        state_dict = torch.load(
            uploaded_model,
            weights_only=True,              # ← 핵심!
            map_location=torch.device("cpu")
        )
        model.load_state_dict(state_dict)
        model.eval()
        st.success("✅ 모델 로딩 완료!")
    except Exception as e:
        st.error(f"❌ 모델 로딩 실패: {e}")
        st.stop()

# ───────────────────────────────────────────────────────────
# 4. CSV 입력 데이터 업로드
# ───────────────────────────────────────────────────────────
uploaded_csv = st.file_uploader(
    "📄 CSV 입력 데이터 업로드 (특성 4개, 숫자만)", type=["csv"]
)

if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
        st.markdown("### 📊 업로드된 입력 데이터 (상위 5행)")
        st.dataframe(df.head())

        # Tensor 변환 & shape 맞추기
        input_tensor = torch.tensor(df.values, dtype=torch.float32)
        if input_tensor.ndim == 1:      # 행 1개인 경우
            input_tensor = input_tensor.view(1, -1)
    except Exception as e:
        st.error(f"❌ CSV 처리 오류: {e}")
        st.stop()

# ───────────────────────────────────────────────────────────
# 5. 예측 + Captum XAI
# ───────────────────────────────────────────────────────────
if model and uploaded_csv:
    with torch.no_grad():
        logits = model(input_tensor)
        pred   = torch.argmax(logits, dim=1)

    st.markdown("### ✅ 예측 결과")
    st.write(pred.numpy())

    # Integrated Gradients 기여도
    ig = IntegratedGradients(model)
    attr, _ = ig.attribute(
        input_tensor,
        target=pred,
        return_convergence_delta=True
    )

    st.markdown("### 🔍 Integrated Gradients 시각화")
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(attr.detach().numpy(), aspect="auto", cmap="hot")
    ax.set_xlabel("입력 특성 index")
    ax.set_ylabel("샘플 index")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)
