# captum_streamlit_demo.py

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

# ---------------------
# 모델 정의
# ---------------------
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

@st.cache_resource
def load_model():
    model = SimpleNet()
    model.eval()
    return model

# ---------------------
# 이미지 처리 함수
# ---------------------
def preprocess_image(image):
    image = image.convert('L').resize((28, 28))  # 흑백 변환 + 28x28
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    return image_tensor

def show_attribution_map(attributions):
    attr = attributions.squeeze().detach().numpy()
    fig, ax = plt.subplots()
    ax.imshow(attr, cmap='hot')
    ax.axis('off')
    st.pyplot(fig)

# ---------------------
# Streamlit 앱 시작
# ---------------------
st.title("🧠 Captum XAI 데모 (Streamlit)")
st.markdown("PyTorch 모델에 대해 Integrated Gradients로 **설명가능성 시각화**를 수행합니다.")

uploaded_file = st.file_uploader("🎨 손글씨 이미지 업로드 (MNIST 스타일, 흑백)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="입력 이미지", width=150)

    input_tensor = preprocess_image(image)
    model = load_model()

    with torch.no_grad():
        output = model(input_tensor)
        pred_label = torch.argmax(output, dim=1).item()

    st.write(f"✅ 모델 예측 결과: **{pred_label}**")

    ig = IntegratedGradients(model)
    attributions, _ = ig.attribute(input_tensor, target=pred_label, return_convergence_delta=True)

    st.subheader("🧭 Integrated Gradients 시각화")
    show_attribution_map(attributions)
else:
    st.info("🖼 이미지를 업로드하면 예측과 XAI 결과를 확인할 수 있어요!")
