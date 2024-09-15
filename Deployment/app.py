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

st.title('Mosquito Classifier')

# User Guide
st.sidebar.title('User Guide')
st.sidebar.markdown("""
1. **Upload Image**: Upload a mosquito image in PNG, JPG, or JPEG format.
2. **Image Quality**: Make sure the image is clear and shows a single mosquito for best results.
3. **Classification**: The model will classify the image into one of four categories: Aedes, Anopheles, Culex, or Other.
4. **Batch Classification**: You can upload multiple images at once to classify them in one go.
5. **Feedback**: Provide feedback on the classification to help improve future versions of the model.
""")

# Class Descriptions
st.sidebar.title('Class Descriptions')
st.sidebar.markdown("""
- **Aedes**: Known for spreading diseases like dengue, Zika, and chikungunya. They have white markings on their legs and a pattern of white scales on their thorax.
- **Anopheles**: Primary vectors for malaria. They are recognized by their spotted wings and resting position at a 45-degree angle.
- **Culex**: Common in both urban and rural areas, and are vectors for West Nile virus. They have a blunt abdomen and uniform color.
- **Other**: Any other mosquito species not included in the above categories or other insects that resemble mosquitoes.
""")

# Cache the model loading to optimize performance
@st.cache_resource
def download_and_load_model():
    url = 'https://drive.google.com/uc?id=1NbxywH92PygxyGwFzCEMFX4469PC1Rah'  # Replace with your file ID
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
    class_names = ['Aedes', 'Anopheles', 'Culex', 'Other']
    
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    
    # Get the predicted class and its confidence
    predicted_class = class_names[np.argmax(scores)]
    confidence = (100 * np.max(scores)).round(2)
    
    result = f"{predicted_class} with a {confidence}% confidence."
    return result, scores, class_names

# Image Quality Check
def image_quality_check(image):
    min_width, min_height = 100, 100  # Minimum acceptable resolution
    if image.width < min_width or image.height < min_height:
        return False
    return True

def main():
    model = download_and_load_model()  # Ensure the model is loaded when the app starts

    # User uploads one or multiple images
    file_uploaded = st.file_uploader("Choose File(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    class_btn = st.button("Classify")

    images = []
    if file_uploaded is not None:
        for uploaded_file in file_uploaded:
            image = Image.open(uploaded_file)
            if image_quality_check(image):
                images.append(image)
            else:
                st.warning(f"Image {uploaded_file.name} is too small, please upload a higher resolution image.")
    
        if images:
            st.image(images, caption=[f'Uploaded Image {i+1}' for i in range(len(images))], use_column_width=True)
    
    if class_btn:
        if not images:
            st.write("Invalid command, please upload at least one valid image.")
        else:
            results = []
            with st.spinner('Model working....'):
                for i, image in enumerate(images):
                    fig, ax = plt.subplots()
                    ax.imshow(image)
                    ax.axis("off")
                    
                    # Get classification result
                    classification_result, scores, class_names = predict(image, model)
                    results.append(classification_result)
                    
                    # Display all confidence scores as a bar chart
                    fig, ax_bar = plt.subplots()
                    ax_bar.barh(class_names, scores, color='blue')
                    ax_bar.set_xlim(0, 1)
                    ax_bar.set_title(f'Confidence Scores for Image {i+1}')
                    st.pyplot(fig)

                for i, result in enumerate(results):
                    st.write(f"Result for Image {i+1}: {result}")
            
            # User Feedback Mechanism
            st.subheader('Provide Feedback')
            feedback = st.text_area('Was the classification accurate? Any comments or suggestions?')
            if st.button('Submit Feedback'):
                if feedback:
                    st.write("Thank you for your feedback!")
                    # Here you could add code to save the feedback to a file or database
                else:
                    st.warning("Please enter some feedback before submitting.")

if __name__ == "__main__":
    main()
