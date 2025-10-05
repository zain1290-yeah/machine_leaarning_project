import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
from PIL import Image

# -------------------------------
# Load model and labels safely
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model("best_finetuned_rgb.keras", compile=False)
        with open("labels.json", "r") as f:
            class_names = json.load(f)
        return model, class_names
    except Exception as e:
        st.error(f"⚠️ Failed to load model or labels. Error: {e}")
        st.stop()

model, class_names = load_model()

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("🌾 Crop Disease Classifier")
st.write("Upload a leaf image to detect the type of disease or if it’s healthy.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    try:
        preds = model.predict(img_array)
        class_idx = np.argmax(preds)
        confidence = float(np.max(preds))
        label = class_names[class_idx]

        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {confidence:.2%}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
