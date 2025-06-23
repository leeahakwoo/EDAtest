# 범용 XAI 진단 도구  –  표형 데이터 + Captum
import streamlit as st, torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

# 1. 모델 클래스
class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(4, 16), nn.Linear(16, 3)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# 2. 페이지 설정
st.set_page_config(page_title="범용 XAI 진단", layout="centered")
st.title("🧠 범용 XAI 진단 도구")
st.write("PyTorch **가중치(state_dict)** 파일(.pt)과 CSV(숫자 4열)를 업로드하면 예측과 Captum 시각화를 제공합니다.")

# 3. 모델 업로드 (state_dict 전용)
up_model = st.file_uploader("📂 PyTorch 가중치 (.pt)", type=["pt"])
model = None
if up_model:
    try:
        model = IrisNet()
        # ――― 핵심: pickled 객체 차단 & 가중치만 로드
        sd = torch.load(up_model, weights_only=True, map_location="cpu")
        model.load_state_dict(sd)
        model.eval()
        st.success("✅  모델 로딩 완료")
    except Exception as e:
        st.error(f"❌  모델 로딩 실패: {e}")
        st.stop()

# 4. CSV 업로드
up_csv = st.file_uploader("📄 CSV 입력 데이터 (숫자 4열)", type=["csv"])
if up_csv:
    try:
        df = pd.read_csv(up_csv)
        if df.shape[1] > 4:          # 열이 4개 초과하면 앞 4열만 사용
            df = df.iloc[:, :4]
        st.dataframe(df.head())
        tensor = torch.tensor(df.values, dtype=torch.float32)
    except Exception as e:
        st.error(f"❌  CSV 처리 오류: {e}")
        st.stop()

# 5. 예측 + Integrated Gradients
if model and up_csv:
    with torch.no_grad():
        logits = model(tensor)
        pred   = torch.argmax(logits, 1)

    st.markdown("### ✅ 예측 결과")
    st.write(pred.numpy())

    ig = IntegratedGradients(model)
    attr, _ = ig.attribute(tensor, target=pred, return_convergence_delta=True)

    st.markdown("### 🔍 Integrated Gradients 시각화")
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(attr.detach().numpy(), aspect="auto", cmap="hot")
    ax.set_xlabel("특성 index"); ax.set_ylabel("샘플 index")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)
