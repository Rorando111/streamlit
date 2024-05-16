import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('cat_classifier.h5')
    return model

model = load_model()

st.write("""
# Cat Breed Classifier
""")

file = st.file_uploader("Choose a cat photo from your computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img = img / 255.0  # Normalize the image
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Abyssinian', 'Bengal', 'Birman', 'Bombay',
                   'British Shorthair', 'Egyptian Mau', 'Maine Coon',
                   'Norweigian forest', 'Persian', 'Ragdoll',
                   'Russian Blue', 'Siamese', 'Sphynx']
    string = "Predicted breed: " + class_names[np.argmax(prediction)]
    st.success(string)
