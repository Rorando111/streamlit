# catApp.py
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

# Load the model
model = tf.keras.models.load_model('cat_classifier.h5')

# Define the class names
class_names = ['Abyssinian', 'Bengal', 'Birman', 'Bombay',
               'British Shorthair', 'Egyptian Mau', 'Maine Coon',
               'Norweigian forest', 'Persian', 'Ragdoll',
               'Russian Blue', 'Siamese', 'Sphynx']

def main():
    print("Cat Breed Classifier")
    print("Upload an image of a cat to classify its breed:")

    # Get the uploaded file
    uploaded_file = input("Enter the file path: ")

    # Read the uploaded file
    image = Image.open(uploaded_file)

    # Display the uploaded image
    plt.imshow(image)
    plt.title("Uploaded Image")
    plt.show()

    # Preprocess the uploaded image
    image = tf.image.resize(np.array(image), (224, 224))
    image = image / 255.0

    # Make predictions
    predictions = model.predict(image[None, ...])

    # Get the top-1 prediction
    top_prediction = np.argmax(predictions)

    # Display the result
    print(f"Predicted breed: {class_names[top_prediction]}")

if __name__ == "__main__":
    main()
