import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
model = tf.keras.models.load_model('model')

# Streamlit app
st.title("Food Classification App with VGG 19")

# File uploader
uploaded_file = st.file_uploader("Choose a food image...", type="jpg")

if uploaded_file is not None:
    # Preprocess the image
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))  # Adjust the size according to your model's input size
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make prediction
    prediction = model.predict(image)

    # Display the result
    print(prediction)
    class_names = ["Yoruba", "Hausa", "Igbo"]  # Replace with your actual class names
    class_index = np.argmax(prediction)
    st.image(image, caption=f"Predicted class: {class_names[class_index]}", use_column_width=True)
