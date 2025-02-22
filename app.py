import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from models.cnn import SimpleCNN
from models.unet import UNet
from models.vit import get_vit_model
from transformers import ViTImageProcessor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import rasterio
from io import BytesIO

# Load Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = SimpleCNN(num_classes=10).to(device)
cnn_model.load_state_dict(torch.load("models/cnn_model.pth", map_location=device))
cnn_model.eval()

vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224", do_rescale=False)
vit_model = get_vit_model(num_classes=10).to(device)
vit_model.load_state_dict(torch.load("models/vit_model.pth", map_location=device))
vit_model.eval()

unet_model = UNet(in_channels=3, out_channels=9).to(device)
unet_model.load_state_dict(torch.load("models/unet_model.pth", map_location=device))
unet_model.eval()

# Classes
classes = ["AnnualCrop", "Forest", "Highway", "Industrial", "Pasture", 
           "PermanentCrop", "River", "SeaLake", "Urban", "HerbaceousVegetation"]
dw_classes = ["Water", "Trees", "Grass", "Flooded Veg", "Crops", "Shrub/Scrub", "Built", "Bare", "Snow/Ice"]
colors = ["blue", "green", "lightgreen", "cyan", "yellow", "brown", "gray", "beige", "white"]

# Preprocessing
cnn_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
vit_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
unet_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Helper function to load images (GeoTIFF or standard)
def load_image(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.endswith('.tif'):  # Handle GeoTIFF
        with rasterio.open(BytesIO(file_bytes)) as src:
            img = src.read([1, 2, 3])  # R, G, B (bands 4, 3, 2 from Sentinel-2)
            img = np.moveaxis(img, 0, -1)  # [C, H, W] → [H, W, C]
            img = (img / img.max() * 255).astype(np.uint8)  # Normalize to 0-255
        return Image.fromarray(img)
    else:  # Handle standard images (jpg, png)
        return Image.open(BytesIO(file_bytes)).convert("RGB")

# Inference Functions
def classify_cnn(img):
    img_tensor = cnn_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = cnn_model(img_tensor)
        prediction = torch.argmax(output, dim=1).item()
    return classes[prediction]

def classify_vit(img):
    img_tensor = vit_transform(img)
    inputs = vit_processor(images=img_tensor, return_tensors="pt", do_rescale=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = vit_model(**inputs).logits
        prediction = torch.argmax(output, dim=1).item()
    return classes[prediction]

def segment_image(img):
    transformed = unet_transform(image=np.array(img))
    img_tensor = transformed["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        output = unet_model(img_tensor)
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    return mask

# Streamlit UI
st.title("Satellite Image Analysis (Dynamic World)")
st.sidebar.title("About")
st.sidebar.write("Using Dynamic World + Sentinel-2 for land use and change detection.")

tab1, tab2 = st.tabs(["Land Use Analysis", "Change Detection"])

# Tab 1: Land Use Analysis
with tab1:
    st.header("Land Use Analysis")
    uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "png", "tif"], key="class")
    
    if uploaded_file:
        image = load_image(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Classify (CNN)"):
                with st.spinner("Running CNN..."):
                    pred_cnn = classify_cnn(image)
                    st.write(f"Predicted (CNN): **{pred_cnn}**")
        with col2:
            if st.button("Classify (ViT)"):
                with st.spinner("Running ViT..."):
                    pred_vit = classify_vit(image)
                    st.write(f"Predicted (ViT): **{pred_vit}**")
        with col3:
            if st.button("Segment (U-Net)"):
                with st.spinner("Running U-Net..."):
                    mask = segment_image(image)
                    plt.imshow(mask, cmap="tab10")
                    plt.title("Segmentation Map")
                    st.pyplot(plt)
                    st.write("Legend:")
                    for cls, color in zip(dw_classes, colors):
                        st.write(f"{cls}: {color}")

# Tab 2: Change Detection
with tab2:
    st.header("Change Detection")
    file1 = st.file_uploader("Image 1 (Before)", type=["jpg", "png", "tif"], key="before")
    file2 = st.file_uploader("Image 2 (After)", type=["jpg", "png", "tif"], key="after")
    
    if file1 and file2:
        img1 = load_image(file1)
        img2 = load_image(file2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img1, caption="Before", use_column_width=True)
        with col2:
            st.image(img2, caption="After", use_column_width=True)
        
        if st.button("Detect Changes"):
            with st.spinner("Analyzing changes..."):
                mask1 = segment_image(img1)
                mask2 = segment_image(img2)
                
                diff = (mask1 != mask2).astype(np.uint8)
                change_area = np.sum(diff) / (256 * 256) * 100
                
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                ax1.imshow(mask1, cmap="tab10")
                ax1.set_title("Before")
                ax2.imshow(mask2, cmap="tab10")
                ax2.set_title("After")
                ax3.imshow(diff, cmap="Reds")
                ax3.set_title("Change Map (Red = Changed)")
                st.pyplot(fig)
                
                st.write(f"Change Percentage: **{change_area:.2f}%**")
                changes = {}
                for i, cls in enumerate(dw_classes):
                    area_before = np.sum(mask1 == i)
                    area_after = np.sum(mask2 == i)
                    if area_before != area_after:
                        change = (area_after - area_before) / (256 * 256) * 100
                        changes[cls] = change
                if changes:
                    st.write("Changes Detected:")
                    for cls, ch in changes.items():
                        st.write(f"{cls}: {'+' if ch > 0 else ''}{ch:.2f}%")
