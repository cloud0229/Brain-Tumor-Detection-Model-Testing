import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2
from PIL import Image, ImageOps
import numpy as np

model = tf.keras.models.load_model('model_vgg16.hdf5')

st.write("""
         # Brain Tumor Prediction
         """
         )
st.write("This is a simple image classification web app to predict brain tumor")
file = st.file_uploader("Please upload an image file", type=["jpg", "png", "tif"])

import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    
        size = (150, 150)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is a glioma!")
    elif np.argmax(prediction) == 1:
        st.write("It is a meningioma!")
    elif np.argmax(prediction) == 2:
        st.write("It is not a tumor!")       
    else:
        st.write("It is a pituitary!")
    
    st.text("Probability (0: glioma, 1: meningioma, 2: notumor, 3: pituitary")
    st.write(prediction)