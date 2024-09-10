import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import urllib.request
import time

st.title('Aedes Classifier')

st.markdown("Welcome to this simple web application that classifies mosquitoes. The classes include: Aedes and Non-Aedes.")

# Cache the model loading to optimize performance
@st.cache_resource

model_url = "https://drive.google.com/uc?export=download&id=1NbxywH92PygxyGwFzCEMFX4469PC1Rah"
urllib.request.urlretrieve(model_url, 'model_classifier.h5')

def load_classification_model():
    classifier_model = "model_classifier.h5"
    model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    return model

model = load_classification_model()

def predict(image):
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
                fig = plt.figure()  # Create the figure inside the function
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
