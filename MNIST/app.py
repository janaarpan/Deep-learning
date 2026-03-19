import streamlit as st


import numpy as np
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model('mnist_mlp1_model1.h5')


st.header('MLP 1')
st.subheader('Upload a image')
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image) / 255.0  # Normalize
    image_flattened = image_array.flatten().reshape(1, -1)  # Flatten and reshape for model

    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        prediction = model.predict(image_flattened)
        predicted_class = np.argmax(prediction)
        st.write(f'Predicted Digit: {predicted_class}')
        st.success(f'Digit is {predicted_class}')