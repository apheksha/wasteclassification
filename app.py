import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="♻️ Smart Waste Classifier", page_icon="🗑️", layout="centered")

st.title("♻️ Smart Waste Classifier using AI")
st.write("Upload an image of waste to identify if it's **Organic** or **Recyclable**.")

uploaded_file = st.file_uploader("📸 Upload a waste image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = np.expand_dims(np.array(image) / 255.0, axis=0).astype(np.float32)

    interpreter = tf.lite.Interpreter(model_path="waste_model_fp16.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = [line.strip() for line in open("classes.txt")]
    pred_class = classes[int(np.argmax(preds))]
    confidence = float(np.max(preds) * 100)

    st.success(f"✅ Prediction: **{pred_class}** ({confidence:.2f}% confidence)")
    st.progress(confidence / 100)
