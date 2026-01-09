import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# --- 1. CONFIGURATION & THEME ---
st.set_page_config(
    page_title="MediScan AI - Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Medical/X-Ray Theme
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    .main-header { font-size: 2.5rem; font-weight: 700; color: #4da6ff; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #b3b3b3; text-align: center; margin-bottom: 2rem; }
    .metric-card { background-color: #1c212c; border: 1px solid #2d3748; border-radius: 8px; padding: 15px; text-align: center; }
    .stButton>button { width: 100%; background-color: #4da6ff; color: white; border: none; border-radius: 5px; }
    .stButton>button:hover { background-color: #0066cc; }
    </style>
""", unsafe_allow_html=True)

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/x-ray.png", width=80)
    st.title("⚙️ System Config")
    st.divider()
    
    model_source = st.radio("Select Model Source", ["Standard YOLOv8 (Demo)", "Custom Fracture Model (.pt)"])
    
    if model_source == "Standard YOLOv8 (Demo)":
        model_path = 'yolov8n.pt'
        st.info("ℹ️ Using standard YOLOv8n. Upload a custom .pt file for medical accuracy.")
    else:
        uploaded_model = st.file_uploader("Upload Trained Model (.pt)", type=['pt'])
        if uploaded_model:
            with open("custom_model.pt", "wb") as f:
                f.write(uploaded_model.getbuffer())
            model_path = "custom_model.pt"
        else:
            model_path = None

    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# --- 4. MAIN APP LOGIC ---
st.markdown('<div class="main-header">🦴 MediScan AI: Fracture Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated X-Ray Analysis & Fracture Localization</div>', unsafe_allow_html=True)

if model_path:
    model = load_model(model_path)
else:
    model = None

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown("### 📤 Upload X-Ray")
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Original X-Ray", use_column_width=True)

with col2:
    st.markdown("### 🔍 Analysis Results")
    
    if uploaded_file and model:
        if st.button("RUN DIAGNOSIS", type="primary"):
            with st.spinner("Analyzing bone structure..."):
                img_array = np.array(image)
                results = model.predict(img_array, conf=confidence_threshold)
                result = results[0]
                
                # --- A. MANUAL DRAWING (Clean Red Box Only) ---
                # We copy the original image array so we can draw on it
                annotated_frame = img_array.copy()
                
                # OpenCV uses BGR, but Streamlit uses RGB. 
                # Since we converted to RGB earlier, 'annotated_frame' is RGB.
                # So for RED, we need (255, 0, 0).
                
                detections = []
                for box in result.boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Draw Rectangle: Image, Start_Point, End_Point, Color(R,G,B), Thickness
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)
                    
                    # Store data for the table
                    conf = float(box.conf[0])
                    detections.append({
                        "Confidence": f"{conf:.1%}",
                        "Coordinates": f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
                    })
                
                st.image(annotated_frame, caption="Detected Region", use_column_width=True)
                
                # --- B. REPORTING ---
                st.divider()
                st.markdown(f"#### 📊 Findings Report")
                
                if len(detections) > 0:
                    st.warning(f"⚠️ {len(detections)} Potential Anomalies Detected")
                    st.table(detections)
                else:
                    st.success("✅ No fractures detected.")

    elif not uploaded_file:
        st.info("👈 Waiting for X-ray upload...")
    elif not model:
        st.error("⚠️ Model not loaded.")