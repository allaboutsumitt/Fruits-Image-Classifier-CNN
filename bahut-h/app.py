# app.py
import os, json, numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Fruit Classifier (All Classes)", page_icon="🍎", layout="centered")

MODEL_PATH = "models/fruit_mobilenetv2_half_clean.keras"
CLASS_MAP_PATH = "class_indices.json"

@st.cache_resource
def load_keras_model(path):
    # Clean model: no Lambda or custom preprocess required
    return load_model(path, compile=False)

@st.cache_data
def load_class_names(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    return [idx_to_class[i] for i in sorted(idx_to_class)]

model = load_keras_model(MODEL_PATH)
class_names = load_class_names(CLASS_MAP_PATH)

# Infer input size
try:
    input_h, input_w = model.input_shape[1], model.input_shape[2]
except Exception:
    input_h, input_w = 160, 160
IMG_SIZE = (input_w, input_h)

# The training model contains a 'preproc' Rescaling layer; don't rescale here
HAS_PREPROC = any(("preproc" in (getattr(l, "name", "") or "")) for l in model.layers)

st.title("🍓 Fruit Image Classifier")
st.caption(f"Trained on {len(class_names)} Fruits-360 classes. Upload an image or use your camera.")

def preprocess_for_model(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    x = np.array(img).astype("float32")
    if not HAS_PREPROC:
        x = x / 255.0
    return np.expand_dims(x, axis=0)

def predict_topk(pil_img, k=5):
    x = preprocess_for_model(pil_img)
    probs = model.predict(x, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:k]
    return [(class_names[i], float(probs[i])) for i in top_idx]

tab1, tab2 = st.tabs(["Upload Image", "Use Camera"])

with tab1:
    uploaded = st.file_uploader("Upload JPG/PNG", type=["jpg","jpeg","png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_container_width=True)
        topk = predict_topk(img, k=5)
        st.subheader(f"Prediction: {topk[0][0]} ({topk[0][1]*100:.1f}%)")
        with st.expander("Top-5 probabilities"):
            for name, p in topk:
                st.write(f"- {name}: {p*100:.1f}%")

with tab2:
    cam = st.camera_input("Take a photo")
    if cam:
        img = Image.open(cam)
        st.image(img, caption="Captured image", use_container_width=True)
        topk = predict_topk(img, k=5)
        st.subheader(f"Prediction: {topk[0][0]} ({topk[0][1]*100:.1f}%)")
        with st.expander("Top-5 probabilities"):
            for name, p in topk:
                st.write(f"- {name}: {p*100:.1f}%")

st.markdown("---")
st.caption("Model and class_indices.json must be from the same training run.")