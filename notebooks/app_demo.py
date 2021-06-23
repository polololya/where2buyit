from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import streamlit as st

st.set_option('deprecation.showfileUploaderEncoding', False)

model = ResNet50(weights='imagenet')

st.title('Where2ButIt')
st.header('Upload your image and  find similar goods:')

img_data = st.file_uploader(label='', type=['png', 'jpg', 'jpeg'])

if img_data is not None:

    uploaded_image = Image.open(img_data)
    st.image(uploaded_image, width=250)

    img_path = f'/Users/va_sc/Desktop/wheretobuyit/{img_data.name}'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    prediction = model.predict(x)
    make_prediction = decode_predictions(prediction)

    st.header('Here is what we found:')
    for i in make_prediction:
        for j in i:
            st.write(j[1])