import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("custom_cnn.keras")

st.title("Coronary Artery Disease Identification")

uploaded_file = st.file_uploader(
    "Upload CT Scan Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    if st.button("Predict"):

        prediction = model.predict(img)
        

        if prediction[0][0] > 0.5:
            st.success("Prediction: Normal")
            st.write("• Maintain a healthy lifestyle.")
            
            
        else:
            st.error("Prediction: Abnormal")
            st.write("• Please consult a Cardiologist.")