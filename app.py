import streamlit as st
import os
import subprocess

try:
    import cv2
except ImportError:
    # Streamlit Cloud workaround: Ultralytics installs the GUI version of opencv-python
    # which fails to load due to missing system libraries. We forcefully replace it with headless.
    subprocess.check_call(["pip", "uninstall", "-y", "opencv-python", "opencv-python-headless"])
    subprocess.check_call(["pip", "install", "opencv-python-headless"])
    import cv2

import numpy as np
from PIL import Image
from collections import Counter
from ultralytics import YOLO
from pathlib import Path

# UI Setup
st.set_page_config(page_title="Blood Cell Detector", page_icon="🩸", layout="wide")
st.title("🩸 Blood Cell Detector & Classifier")
st.markdown("Upload a blood smear image to detect and classify blood cells. The model detects **RBC**, **Platelets**, and 5 **WBC subtypes** (Neutrophil, Lymphocyte, Monocyte, Eosinophil, Basophil).")

@st.cache_resource
def load_model():
    # Use absolute or relative path where the model is expected
    model_path = Path(__file__).parent / "blood_detector_model.pt"
    return YOLO(str(model_path))

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

WBC_SUBTYPES = {"Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"}

# RGB Colors since we will display using Streamlit (which expects RGB)
COLOR_RBC = (255, 0, 0)        # Red
COLOR_PLATELETS = (0, 200, 0)  # Green
COLOR_WBC = (0, 80, 255)       # Blue

def color_for(name: str) -> tuple[int, int, int]:
    if name == "RBC":
        return COLOR_RBC
    if name == "Platelets":
        return COLOR_PLATELETS
    if name in WBC_SUBTYPES:
        return COLOR_WBC
    return (200, 200, 200)

def annotate(img, boxes_xyxy, classes, names) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft, bt = 0.5, 1, 1
    for (x1, y1, x2, y2), c in zip(boxes_xyxy, classes):
        name = names[int(c)]
        col = color_for(name)
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), col, bt)
        
        # Draw label background and text
        (tw, th), _ = cv2.getTextSize(name, font, fs, ft)
        ly = y1 - 2
        if ly - th - 2 < 0:
            ly = y1 + th + 4
        cv2.rectangle(img, (x1, ly - th - 2), (x1 + tw + 2, ly + 1), col, -1)
        cv2.putText(img, name, (x1 + 1, ly - 1), font, fs, (255, 255, 255), ft, cv2.LINE_AA)

st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.15, 0.05)
iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.70, 0.05)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.write("Processing...")
    
    # Run prediction
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
        save=False,
        verbose=False
    )
    
    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    
    # Convert PIL Image to numpy array for annotation
    img_np = np.array(image)
    
    # Annotate the image
    annotate(img_np, boxes, classes, r.names)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(img_np, caption="Detected Cells", use_column_width=True)
        
    with col2:
        st.subheader("Detection Summary")
        if len(boxes) == 0:
            st.write("No cells detected.")
        else:
            counts = Counter(r.names[int(c)] for c in classes)
            st.metric("Total Cells", len(boxes))
            
            for cell_type in ["RBC", "Platelets"] + list(WBC_SUBTYPES):
                if counts[cell_type] > 0:
                    st.write(f"- **{cell_type}**: {counts[cell_type]}")
            
            # Show other cell types if any exist outside standard names
            for cell_type, count in counts.items():
                if cell_type not in ["RBC", "Platelets"] and cell_type not in WBC_SUBTYPES:
                    st.write(f"- **{cell_type}**: {count}")
