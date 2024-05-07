import streamlit as st
import tensorflow as tf
from PIL import Image,ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)

def load_model():
  model=tf.keras.models.load_model('CNN_Model_7.h5')
  return model
model=load_model()

st.write("""
## Weather Classification"""
)
file=st.file_uploader("Choose weather photo from computer",type=["jpg","png"])

def import_and_predict(image_data,model):
    size=(100,100)
    image=ImageOps.fit(image_data,size)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")

else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['Sunrise','Shine', 'Rain', 'Cloudy']
    prediction = prediction*100000
    prediction = np.array(prediction,dtype=int)
    prediction[0][2]=prediction[0][2]%100
    
    string="The weather is "+class_names[np.argmax(prediction)]
    st.success(string)
