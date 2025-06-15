import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time



st.markdown("""
    <style>
    .main-title {
        font-size: 44px;
        font-weight: bold;
        color: #4a6fa5;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .subtext {
        text-align: center;
        font-size: 18px;
        color: #3a506b;
    }
    .stRadio > div {
        flex-direction: row !important;
    }
    .stButton button {
        background-color: #a1c6ea;
        color: black;
        border-radius: 10px;
        padding: 0.5em 1em;
        border: none;
    }
    .stButton button:hover {
        background-color: #89abe3;
        color: white;
    }
    img {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)



# ---------- CONFIG ----------
MODEL_PATH = 'sign_language_model.keras'
IMAGE_SIZE = 64
CLASSES = [chr(i) for i in range(65, 91)]  # A-Z
# ----------------------------

# Custom styling
st.markdown("<div class='main-title'>🌼 Sign Language Detection</div>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Detect American Sign Language using AI 💙</p>", unsafe_allow_html=True)

# Load model
@st.cache_resource(show_spinner=False)
def load_trained_model(path):
    return load_model(path, compile=False)

model = load_trained_model(MODEL_PATH)



# ---------- OPTIONS ----------
st.markdown("---")
option = st.radio("📸 Choose Input Method:", ("Upload Image", "Use Webcam"), horizontal=True)
st.markdown("---")

# ---------- HELPER FUNCTIONS ----------
def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
    norm = resized.astype("float32") / 255.0
    return norm.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

def predict_letter(img_tensor: np.ndarray):
    preds = model.predict(img_tensor, verbose=0)
    idx = int(np.argmax(preds))
    conf = float(preds[0, idx])
    return CLASSES[idx], conf

# ---------- IMAGE UPLOAD ----------
if option == "Upload Image":
    st.subheader("📁 Upload a hand sign image")
    file = st.file_uploader("", type=["png", "jpg", "jpeg"])
    if file:
        img = Image.open(file).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

        letter, conf = predict_letter(preprocess(img_bgr))
        st.success(f"🔤 **Predicted Letter:** `{letter}`  \n📊 **Confidence:** `{conf*100:.2f}%`")

# ---------- WEBCAM ----------
elif option == "Use Webcam":
    st.subheader("🎥 Real-time Sign Detection from Webcam")
    run = st.checkbox("▶️ Start Webcam")
    frame_window = st.image([])

    cap = None
    fps_time = time.time()

    while run:
        if cap is None:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Could not access webcam.")
                break

        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Failed to capture frame.")
            break

        # Prediction
        letter, conf = predict_letter(preprocess(frame))
        cv2.putText(frame, f"{letter} ({conf*100:.1f}%)", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # FPS
        new_time = time.time()
        fps = 1 / (new_time - fps_time)
        fps_time = new_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show
        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        run = st.session_state.get('Start Webcam', False)

    if cap:
        cap.release()
    cv2.destroyAllWindows()

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("Made with ❤️ by Jumana", unsafe_allow_html=True)
