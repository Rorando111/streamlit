import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

# Constants
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['Abyssinian', 'Bengal', 'Birman', 'Bombay',
               'British Shorthair', 'Egyptian Mau', 'Maine Coon',
               'Norweigian forest', 'Persian', 'Ragdoll',
               'Russian Blue', 'Siamese', 'Sphynx']

@st.cache(allow_output_mutation=True)
def load_model() -> tf.keras.Model:
    """Load the cat breed classifier model"""
    model = tf.keras.models.load_model('cat_classifier.h5')
    return model

def import_and_resize_image(image_data: bytes) -> Image:
    """Import and resize the image"""
    image = Image.open(image_data)
    image = ImageOps.fit(image, IMAGE_SIZE, Image.ANTIALIAS)
    return image

def preprocess_image(image: Image) -> np.ndarray:
    """Preprocess the image for prediction"""
    img = np.asarray(image)
    img = img[np.newaxis, ...]
    return img

def make_prediction(image: np.ndarray, model: tf.keras.Model) -> np.ndarray:
    """Make a prediction using the model"""
    prediction = model.predict(image)
    return prediction

def main():
    st.write("""
# Cat Breed Classifier
""")
    file = st.file_uploader("Choose a cat photo from your computer", type=["jpg", "png"])

    if file is None:
        st.text("Please upload an image file")
    else:
        try:
            image = import_and_resize_image(file)
            st.image(image, use_column_width=True)
            image_data = preprocess_image(image)
            model = load_model()
            prediction = make_prediction(image_data, model)
            class_index = np.argmax(prediction)
            output_string = f"OUTPUT: {CLASS_NAMES[class_index]}"
            st.success(output_string)
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
