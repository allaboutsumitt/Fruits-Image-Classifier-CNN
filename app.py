# app.py
import json, numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

st.set_page_config(page_title="Fruit Classifier", page_icon="🍎", layout="centered")

# 1) Config – update paths if your filenames differ
IMG_SIZE = (128, 128)  # must match training
WEIGHTS_PATH = "models/fruit_mnet128_half.weights.h5"  # your weights-only file
CLASS_MAP_PATH = "class_indices.json"

# 2) Helpers
@st.cache_data
def load_class_names(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    return [idx_to_class[i] for i in sorted(idx_to_class)]

def build_model(num_classes, img_size=(128,128)):
    inputs = layers.Input(shape=img_size + (3,), name="image_rgb")
    # Preprocessing to [-1,1] (MobileNetV2 expectation)
    x = layers.Rescaling(1.0/127.5, offset=-1.0, name="preproc")(inputs)
    base = MobileNetV2(weights=None, include_top=False, input_shape=img_size + (3,))
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, activation="relu", name="dense_512")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return models.Model(inputs, outputs, name="fruit_classifier_mnet_clean")

# 3) Load classes
class_names = load_class_names(CLASS_MAP_PATH)

# 4) Build model and load weights  <<< PASTE THESE LINES HERE >>>
model = build_model(num_classes=len(class_names), img_size=IMG_SIZE)
model.load_weights(WEIGHTS_PATH)

# 5) Inference helpers and UI
def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    x = np.array(img).astype("float32")
    # Do NOT divide by 255 here; model has a Rescaling layer already.
    return np.expand_dims(x, axis=0)

def predict_topk(pil_img, k=5):
    x = preprocess(pil_img)
    probs = model.predict(x, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:k]
    return [(class_names[i], float(probs[i])) for i in top_idx]

st.title("🍓 Fruit Image Classifier")
st.caption(f"Trained on {len(class_names)} Fruits-360 classes.")

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