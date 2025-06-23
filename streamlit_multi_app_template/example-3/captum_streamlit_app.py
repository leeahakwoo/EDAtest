
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import numpy as np

# ---------------------
# 모델 업로드
# ---------------------
@st.cache_resource
def load_model(uploaded_file):
    try:
        model = torch.load(uploaded_file, map_location=torch.device("cpu"))
        if not isinstance(model, nn.Module):
            raise ValueError("이 파일은 모델 클래스 정의를 포함하지 않습니다.")
        model.eval()
        return model
    except Exception as e:
        st.error(f"모델 로딩 실패: {e}")
        return None

# ---------------------
# 이미지 전처리
# ---------------------
def preprocess_image(image):
    image = image.convert('L').resize((28, 28))
    return transforms.ToTensor()(image).unsqueeze(0)

# ---------------------
# 표형 데이터 전처리
# ---------------------
def preprocess_tabular(df):
    return torch.tensor(df.values, dtype=torch.float32)

# ---------------------
# Captum 시각화
# ---------------------
def show_attribution_map(attributions, input_tensor, data_type):
    if data_type == "이미지":
        attr = attributions.squeeze().detach().numpy()
        fig, ax = plt.subplots()
        ax.imshow(attr, cmap='hot')
        ax.axis('off')
        st.pyplot(fig)
    elif data_type == "표형 데이터 (CSV)":
        attr = attributions.squeeze().detach().numpy()
        df_attr = pd.DataFrame(attr.reshape(1, -1), columns=[f"feature_{i}" for i in range(attr.shape[0])])
        st.bar_chart(df_attr.T)

# ---------------------
# Streamlit 앱 UI
# ---------------------
st.set_page_config(page_title="범용 XAI 분석 도구", page_icon="🧠")
st.title("🧠 범용 XAI 진단 도구")

# 모델 업로드
model_file = st.file_uploader("📦 PyTorch 모델 업로드 (.pt/.pth)", type=["pt", "pth"])
model = load_model(model_file) if model_file else None

# 입력 데이터 업로드
data_type = st.selectbox("입력 데이터 유형 선택", ["이미지", "표형 데이터 (CSV)"])

if data_type == "이미지":
    uploaded_image = st.file_uploader("🖼️ 이미지 업로드", type=["png", "jpg", "jpeg"])
    if uploaded_image and model:
        image = Image.open(uploaded_image)
        st.image(image, caption="입력 이미지", width=150)
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        st.success(f"✅ 모델 예측 결과: {pred}")

        ig = IntegratedGradients(model)
        attr, _ = ig.attribute(input_tensor, target=pred, return_convergence_delta=True)
        st.subheader("🧭 Integrated Gradients 시각화")
        show_attribution_map(attr, input_tensor, data_type)

elif data_type == "표형 데이터 (CSV)":
    uploaded_csv = st.file_uploader("📄 CSV 파일 업로드", type=["csv"])
    if uploaded_csv and model:
        try:
            df = pd.read_csv(uploaded_csv)
            st.dataframe(df.head())
            input_tensor = preprocess_tabular(df.iloc[0:1])
            with torch.no_grad():
                output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            st.success(f"✅ 모델 예측 결과: {pred}")

            ig = IntegratedGradients(model)
            attr, _ = ig.attribute(input_tensor, target=pred, return_convergence_delta=True)
            st.subheader("🧭 Integrated Gradients 시각화")
            show_attribution_map(attr, input_tensor, data_type)
        except Exception as e:
            st.error(f"❌ 입력 또는 예측 처리 실패: {e}")
