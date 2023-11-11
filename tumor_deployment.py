import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Conv2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
import streamlit as st
from tensorflow.keras.models import load_model
from model_architecture import create_model

# @st.cache_data()
def load_model():
    model = create_model()
    model.load_weights('model_weights.h5')
    return model
model = load_model()

#model = load_model('model.keras')

st.title('Tumor Detection Application')
uploaded_file = st.file_uploader(label='Choose an image...', type=['jpg', 'jpeg', 'png'])

def classify_image(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image =  cv2.resize(image, (75, 75))
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction


if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype='uint8')
    opencv_image = cv2.imdecode(file_bytes, flags=1)
    st.image(opencv_image, channels='BGR')
    prediction = classify_image(opencv_image, model)
    if prediction[0][0] > 0.5:
        st.write('Tumor detected')
    else:
        st.write('No Tumor detected')

