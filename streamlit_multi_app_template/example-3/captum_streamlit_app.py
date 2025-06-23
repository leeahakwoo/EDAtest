# 범용 XAI 진단 도구 - 표형 데이터 + Captum 기반
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

# 모델 클래스 정의
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# 모델 로딩
model = IrisNet()
state_dict = torch.load(uploaded_model, map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.eval()

# 2. 앱 설정
st.set_page_config(page_title="범용 XAI 진단 도구", layout="centered")
st.title("🧠 범용 XAI 진단 도구")
st.markdown("PyTorch 모델 (.pt)과 CSV 입력 데이터를 함께 업로드하면 예측과 XAI 시각화를 제공합니다.")

# 3. 모델 업로드
uploaded_model = st.file_uploader("📂 PyTorch 모델 업로드 (.pt)", type=["pt"])
model = None

if uploaded_model is not None:
    try:
        model = IrisNet()
        model.load_state_dict(torch.load(uploaded_model, map_location=torch.device("cpu")))
        model.eval()
        st.success("✅ 모델 로딩 완료!")
    except Exception as e:
        st.error(f"❌ 모델 로딩 실패: {e}")

# 4. 입력 CSV 업로드
uploaded_csv = st.file_uploader("📄 CSV 입력 데이터 업로드", type=["csv"])

if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
        st.dataframe(df.head())
        input_tensor = torch.tensor(df.values, dtype=torch.float32)
    except Exception as e:
        st.error(f"❌ CSV 처리 오류: {e}")

# 5. 예측 + Captum XAI
if uploaded_model and uploaded_csv and model:
    with torch.no_grad():
        pred = model(input_tensor)
        pred_label = torch.argmax(pred, dim=1)

    st.markdown("### ✅ 예측 결과")
    st.write(pred_label.numpy())

    # Integrated Gradients
    ig = IntegratedGradients(model)
    attr, _ = ig.attribute(input_tensor, target=pred_label, return_convergence_delta=True)

    st.markdown("### 🔍 Integrated Gradients 시각화")
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(attr.detach().numpy(), aspect="auto", cmap="hot")
    ax.set_xlabel("입력 특성")
    ax.set_ylabel("샘플 인덱스")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)
