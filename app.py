import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import numpy as np
import torch
import os
import io

# ✅ First Streamlit command
st.set_page_config(page_title="🔬 BCCD Detector Dashboard", layout="wide", page_icon="🩸")

# Custom CSS styling for better look and accent color
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: white;
    }
    .block-container {
        padding-top: 2rem;
        color: white;
    }
    .css-1d391kg { 
        background-color: #1e293b; 
        border-radius: 10px;
        padding: 20px; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        color: white;
    }
    h1 {
        color: white; 
        background-color: #334155;
        padding: 15px;
        border-radius: 8px;
    }
    p, .stSidebar, .stSlider, .stMarkdown, .stDataFrame, .stButton, .stDownloadButton {
        color: white !important;
    }
    .css-1cpxqw2 { /* Sidebar background */
        background-color: #1e293b !important;
        color: white !important;
    }
    /* Accent color for buttons and sliders */
    .stButton > button, .stDownloadButton > button {
        background-color: #2563eb !important; /* blue */
        color: white !important;
        border-radius: 8px;
        padding: 10px 16px;
    }
   
    </style>
""", unsafe_allow_html=True)

# Load fine-tuned model
@st.cache_resource
def load_model():
    model = YOLO("models/finetuned_yolov10s.pt")  # Updated path to models folder
    return model

model = load_model()
class_names = ['WBC', 'RBC', 'Platelets']

# Streamlit UI
st.markdown("""
    <h1 style='text-align: center;'>🩸 Blood Cell Counter & Detection Dashboard</h1>
    <p style='text-align: center; font-size:18px;'>Detect <strong>WBCs</strong>, <strong>RBCs</strong>, and <strong>Platelets</strong> from blood smear images using YOLOv10.</p>
""", unsafe_allow_html=True)

# Sidebar with settings
with st.sidebar:
    st.header("🔧 Settings")
    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, step=0.01)
    st.markdown("---")
    uploaded_images = st.file_uploader("📤 Upload Blood Cell Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Main body
if uploaded_images:
    for img_file in uploaded_images:
        st.markdown(f"### 🖼️ Image: {img_file.name}")
        image = Image.open(img_file).convert("RGB")

        # Layout in columns
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        # Model prediction
        results = model.predict(image)
        result = results[0]
        boxes = result.boxes

        # Annotated image
        annotated_img = result.plot()
        with col2:
            st.subheader("Detected Cells")
            st.image(annotated_img, use_container_width=True)

        # Prediction summary
        if boxes and boxes.cls is not None:
            classes = [class_names[int(cls)] for cls in boxes.cls.cpu().numpy()]
            confidences = [round(float(c), 3) for c in boxes.conf.cpu().numpy()]
            df = pd.DataFrame({"Class": classes, "Confidence": confidences})

            # Filter based on confidence threshold
            filtered_df = df[df['Confidence'] >= threshold]
            
            st.markdown("### 📊 Filtered Detections")
            st.dataframe(filtered_df, use_container_width=True)

            # Class-wise stats
            st.markdown("### 🧮 Class-wise Detection Stats")
            summary = filtered_df.groupby('Class').agg(
                Total_Detections=('Class', 'count'),
                Avg_Confidence=('Confidence', 'mean')
            ).reset_index()
            st.dataframe(summary, use_container_width=True)

            # Download CSV option
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Detection Report",
                data=csv,
                file_name=f'{img_file.name}_detection_report.csv',
                mime='text/csv',
                use_container_width=True
            )
        else:
            st.warning("No objects detected. Try adjusting the confidence threshold.")
else:
    st.info("📌 Upload one or more images from the sidebar to start detection.")