import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Load the model
model_path = os.path.join(os.getcwd(), 'cat_classifier.h5')
model = tf.keras.models.load_model(model_path)

# Define the class names
class_names = ['Abyssinian', 'Bengal', 'Birman', 'Bombay',
               'British Shorthair', 'Egyptian Mau', 'Maine Coon',
               'Norweigian forest', 'Persian', 'Ragdoll',
               'Russian Blue', 'Siamese', 'Sphynx']

st.title("Cat Breed Classifier")
st.write("Upload an image of a cat to classify its breed:")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png"])

if uploaded_file is not None:
    # Read the uploaded file
    image = Image.open(io.BytesIO(uploaded_file.getvalue()))

    # Preprocess the uploaded image
    image = tf.image.resize(np.array(image), (224, 224))
    image = image / 255.0

    # Make predictions
    predictions = model.predict(image[None, ...])

    # Get the top-1 prediction
    top_prediction = np.argmax(predictions)

    # Display the result
    st.write(f"Predicted breed: {class_names[top_prediction]}")
