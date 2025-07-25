
# person_reid_app.py
# Simple person re-identification example using torchreid and Streamlit

import streamlit as st
import torchreid
import os
from PIL import Image
from torchvision import transforms
import torch

st.title("🎥 Person Re-Identification")
st.write("Upload two person images (possibly from different cameras), and we'll check if they match.")

# Load model
@st.cache_resource
def load_model():
    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=1000,
        pretrained=True
    )
    model.eval()
    return model

model = load_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Upload images
img1 = st.file_uploader("Upload image from Camera A", type=["jpg", "png"], key="img1")
img2 = st.file_uploader("Upload image from Camera B", type=["jpg", "png"], key="img2")

if img1 and img2:
    image1 = Image.open(img1).convert("RGB")
    image2 = Image.open(img2).convert("RGB")

    st.image([image1, image2], caption=["Camera A", "Camera B"], width=200)

    with torch.no_grad():
        tensor1 = transform(image1).unsqueeze(0)
        tensor2 = transform(image2).unsqueeze(0)

        feat1 = model(tensor1)
        feat2 = model(tensor2)

        similarity = torch.nn.functional.cosine_similarity(feat1, feat2).item()

    st.write(f"🔍 Cosine Similarity: **{similarity:.4f}**")
    if similarity > 0.7:
        st.success("✅ Likely same person")
    else:
        st.error("❌ Likely different persons")
else:
    st.info("Please upload both images to begin re-identification.")
