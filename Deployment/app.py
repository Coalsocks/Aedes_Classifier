import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import gdown
import time

st.title('Aedes Classifier')

st.markdown("Welcome to this simple web application that classifies mosquitoes. The classes include: Aedes and Non-Aedes.")

# Cache the model loading to optimize performance
@st.cache_resource
def download_and_load_model():
    url = 'https://drive.google.com/uc?id=1AbCdEfGhiJKLm'  # Replace with your file ID
    output = 'model_classifier.h5'

    st.write("Downloading the model...")
    gdown.download(url, output, quiet=False)

    st.write("Loading the model...")
    model = load_model(output, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})

    st.success("Model loaded successfully!")
    return model

# Prediction function
def predict(image, model):
    IMAGE_SHAPE = (224, 224)
    test_image = image.resize(IMAGE_SHAPE)
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = ['Aedes', 'Non-Aedes']
    
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    
    result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence."
    return result

def main():
    model = download_and_load_model()  # Ensure the model is loaded when the app starts

    file_uploaded = st.file_uploader("Choose File", type=["png", "jpg", "jpeg"])
    class_btn = st.button("Classify")
    
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                fig = plt.figure()
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image, model)  # Pass the model to the predict function
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
