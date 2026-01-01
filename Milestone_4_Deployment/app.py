import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="AI SpillGuard", layout="wide")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/oil_spill_final_model.h5", compile=False)

model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.title("⚙️ System Configuration")
st.sidebar.success("Device: CPU")
oil_thresh = st.sidebar.slider("Oil Spill Alert (%)", 0, 100, 5)

st.title("🛢️ AI SpillGuard – Oil Spill Detection System")

tab1, tab2, tab3 = st.tabs(["Detection", "History & Analytics", "API Info"])

COLOR_MAP = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [255, 255, 0],
    3: [0, 0, 255]
}

def preprocess(img):
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def decode(mask):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k, v in COLOR_MAP.items():
        out[mask == k] = v
    return out

with tab1:
    file = st.file_uploader("Upload Satellite Image", type=["jpg","png","jpeg"])
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, width=500)

        if st.button("Detect Oil Spill"):
            pred = model.predict(preprocess(img))[0]
            mask = np.argmax(pred, axis=-1)
            oil_pct = round(np.mean(mask == 1) * 100, 2)
            alert = "Yes" if oil_pct >= oil_thresh else "No"

            st.session_state.history.append({
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Image": file.name,
                "Oil (%)": oil_pct,
                "Alert": alert
            })

            color_mask = decode(mask)
            overlay = cv2.addWeighted(
                cv2.resize(np.array(img), (256,256)), 0.7,
                color_mask, 0.3, 0
            )

            st.success(f"Oil Spill Detected: {oil_pct}%")
            st.image(color_mask, caption="Segmentation Mask")
            st.image(overlay, caption="Overlay Result")

with tab2:
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        st.download_button("Export CSV", df.to_csv(index=False), "history.csv")
    else:
        st.info("No detections yet.")

with tab3:
    st.code("POST /predict\nInput: Image\nOutput: Mask + Statistics")
