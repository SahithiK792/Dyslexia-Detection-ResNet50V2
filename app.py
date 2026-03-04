import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Dyslexia Symptom Detector",
    page_icon="🧠",
    layout="centered"
)

# --- 2. STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .motto-box {
        background-color: #e3f2fd;
        border-left: 6px solid #2196f3;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .motto-text {
        font-size: 20px;
        font-style: italic;
        color: #1565c0;
        font-weight: 500;
        text-align: center;
        margin: 0;
    }
    .result-card {
        background-color: white;
        padding: 40px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .risk-header { 
        font-size: 36px; 
        font-weight: 800; 
        margin-bottom: 15px; 
        text-transform: uppercase; 
        letter-spacing: 1px;
    }
    .result-message {
        font-size: 20px;
        color: #555;
        line-height: 1.5;
    }
    .stButton>button {
        background-color: #2c3e50; color: white; width: 100%; height: 55px;
        font-size: 18px; border-radius: 8px; border: none; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #34495e; box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HEADER ---
st.title("🧠 Dyslexia Symptom Detector")

st.markdown("""
    <div class="motto-box">
        <p class="motto-text">"Decoding the Signs: AI-Powered Handwriting Analysis for Early Dyslexia Screening."</p>
    </div>
""", unsafe_allow_html=True)

# --- 4. LOAD MODEL ---
@st.cache_resource
def load_classifier_model():
    try:
        # Ensure 'resnet50_final_winner.keras' is in the folder
        return tf.keras.models.load_model('resnet50_oversampled.keras')
    except:
        return None

with st.spinner('Initializing AI Engine...'):
    model = load_classifier_model()

# --- 5. PROCESSING LOGIC ---
def process_and_predict(original_image, model):
    # 1. Convert to OpenCV format
    open_cv_image = np.array(original_image.convert('RGB')) 
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    
    # 2. Auto-Invert Check (For Black Backgrounds)
    is_dark_bg = np.mean(gray) < 127
    if is_dark_bg:
        # If dark bg, text is white. Threshold normal.
        thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    else:
        # If white paper, text is dark. Adaptive threshold handles shadows/lines.
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 5)
    
    # 3. Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    class_counts = {'Normal': 0, 'Reversal': 0, 'Corrected': 0}
    valid_crops = 0
    class_names = ['Normal', 'Reversal', 'Corrected']
    
    total_img_area = open_cv_image.shape[0] * open_cv_image.shape[1]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = w * h
        
        # FILTERS: Ignore noise, lines, and huge blobs
        if w > 10 and h > 15 and aspect_ratio < 4.0 and (area < total_img_area * 0.1):
            
            roi = open_cv_image[y:y+h, x:x+w]
            if is_dark_bg: roi = cv2.bitwise_not(roi)
            if roi.size == 0: continue

            roi_pil = Image.fromarray(roi)
            
            # Resize & Preprocess
            roi_resized = ImageOps.fit(roi_pil, (128, 128), Image.Resampling.LANCZOS)
            img_array = np.asarray(roi_resized).astype('float32')
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Predict
            pred = model.predict(img_array, verbose=0)
            predicted_idx = np.argmax(pred[0])
            predicted_label = class_names[predicted_idx]
            confidence = np.max(pred[0])
            
            # STRICTER LOGIC: Benefit of doubt to Normal
            if predicted_label != 'Normal' and confidence < 0.85:
                predicted_label = 'Normal'
            
            class_counts[predicted_label] += 1
            valid_crops += 1

    return class_counts, valid_crops

# --- 6. MAIN INTERFACE ---
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### Uploaded Sample")
        st.image(image, caption='Input Image', use_container_width=True)
    
    with col2:
        st.markdown("### AI Analysis")
        
        if st.button('🔍 Scan Handwriting'):
            if model is None:
                st.error("Model Error: 'resnet50_final_winner.keras' not found.")
            else:
                with st.spinner('Analyzing handwriting patterns...'):
                    
                    counts, total_valid = process_and_predict(image, model)
                    
                    if total_valid == 0:
                        st.warning("No clear handwriting detected. Try cropping the image closer to the text.")
                    else:
                        # --- RISK LOGIC ---
                        symptom_count = counts['Reversal'] + counts['Corrected']
                        symptom_ratio = symptom_count / total_valid
                        
                        # Threshold: 15% (0.15)
                        RISK_THRESHOLD = 0.15
                        
                        if symptom_ratio > RISK_THRESHOLD:
                            risk_status = "HIGH RISK"
                            card_color = "#dc3545" # Red
                            icon = "⚠️"
                            msg = "Significant indicators of reversals or corrections detected in the handwriting sample."
                        else:
                            risk_status = "LOW RISK"
                            card_color = "#28a745" # Green
                            icon = "✅"
                            msg = "Handwriting patterns appear consistent with standard development. No significant indicators found."

                        # --- CLEAN RESULT DISPLAY (No extra stats) ---
                        st.markdown(f"""
                            <div class="result-card" style="border-top: 10px solid {card_color};">
                                <div class="risk-header" style="color: {card_color};">{icon} {risk_status}</div>
                                <div class="result-message">{msg}</div>
                            </div>
                        """, unsafe_allow_html=True)