import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time

# ----------------- CONFIG -----------------
MODEL_PATH   = 'sign_language_model.keras'     # change if needed
IMAGE_SIZE   = 64                              # must match training
CLASSES      = [chr(i) for i in range(65, 91)] # ['A'‥'Z']
# -------------------------------------------

# Load once at start
@st.cache_resource(show_spinner=False)
def load_trained_model(path):
    return load_model(path, compile=False)

model = load_trained_model(MODEL_PATH)

st.title("🖐️ Sign Language Detection")

st.markdown(
    "Upload an image **or** use your webcam to detect **American Sign Language** letters."
)

option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

# ---------- helper: preprocessing ----------
def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to model-ready tensor shape (1,64,64,1)."""
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
    norm    = resized.astype("float32") / 255.0
    return norm.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

# ---------- helper: prediction ----------
def predict_letter(img_tensor: np.ndarray):
    """Return (predicted_letter, confidence_float)"""
    preds = model.predict(img_tensor, verbose=0)
    idx   = int(np.argmax(preds))
    conf  = float(preds[0, idx])
    return CLASSES[idx], conf

# =============== 1) UPLOAD IMAGE ===========
if option == "Upload Image":
    file = st.file_uploader("Upload a hand-sign image", type=["png", "jpg", "jpeg"])
    if file:
        img = Image.open(file).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        letter, conf = predict_letter(preprocess(img_bgr))
        st.success(f"### Predicted Letter: **{letter}**  \nConfidence: **{conf*100:.1f}%**")

# =============== 2) USE WEBCAM =============
elif option == "Use Webcam":
    run = st.checkbox("Start / Stop Webcam")
    frame_window = st.image([])   # placeholder for video frames

    # Delay webcam initialisation until user checks the box
    cap = None
    fps_time = time.time()

    while run:
        if cap is None:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Could not open webcam.")
                break

        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ No frame captured.")
            break

        # Predict current frame
        letter, conf = predict_letter(preprocess(frame))

        # Overlay prediction on frame
        cv2.putText(frame, f"{letter} ({conf*100:.1f}%)", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # FPS counter (optional)
        new_time = time.time()
        fps = 1 / (new_time - fps_time)
        fps_time = new_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show in Streamlit (convert BGR → RGB)
        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Check if checkbox has been unticked mid-loop
        run = st.session_state.get('Start / Stop Webcam', False)

    # Clean-up when loop ends
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
