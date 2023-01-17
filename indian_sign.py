import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
from gtts import gTTS
# To install this module, run:
# python -m pip install Pillow
from io import BytesIO
from PIL import Image
from PIL import ImageDraw
import json
import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.utils import load_img
from keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np
from keras.models import load_model

st.set_page_config(layout="wide")
#st.sidebar.image('images/Azure_Image.png', width=300)
st.sidebar.header('A website for classifying Indian Sign Language')
st.sidebar.markdown('Used CNN algorithm')


app_mode = st.sidebar.radio(
    "",
    ("About Me","Image to Text","Image to Speech","Text to Image"),
)


st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.sidebar.markdown('---')
st.sidebar.write('N.V.Suresh Krishna | sureshkrishnanv24@gmail.com https://github.com/sureshkrishna123')

if app_mode =='About Me':
    #st.image('images/wp4498220.jpg', use_column_width=True)
    st.markdown('''
              # About Me \n 
                Hey this is ** N.V.Suresh Krishna **. \n
                
                
                Also check me out on Social Media
                - [git-Hub](https://github.com/sureshkrishna123)
                - [LinkedIn](https://www.linkedin.com/in/suresh-krishna-nv/)
                - [Instagram](https://www.instagram.com/worldofsuresh._/)
                - [Portfolio](https://sureshkrishna123.github.io/sureshportfolio/)
                - [Blog](https://ingenious-point.blogspot.com/)\n
                If you are interested in building more about Microsoft Azure then   [click here](https://azure.microsoft.com/en-in/)\n
                ''')
               

if app_mode=='Image to Text':
  #st.image('ind.png'),use_column_width=True )
  st.title("Final year project")
  st.image('ind.png')
  st.header('Indian Sign Language Detection')
  st.markdown("Using CNN algorithm, the hand sign images are classified and gives the text as an output.")
  st.text("")
  
  image_file =  st.file_uploader("Upload Images (less than 1mb)", type=["png","jpg","jpeg"])
  if image_file is not None:
    img = Image.open(image_file)
    st.image(image_file,width=250,caption='Uploaded image')
    byte_io = BytesIO()
    img.save(byte_io, 'PNG')#PNG
    image = byte_io.getvalue()


  button_translate=st.button('Click me',help='To give the image')

  if button_translate and image_file :
    class_names= ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    img_height,img_width=180,180
    model = load_model('indian_sign.h5')
    #class_names = model.class_names
    demo_image_path = image_file
    img = tf.keras.utils.load_img(demo_image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    st.text("The hand sign of the above image is : ")
    st.subheader(class_names[np.argmax(score)])
    #word=class_names[np.argmax(score)]
    #sound_file = BytesIO()
    #tts = gTTS(word)
    #tts.write_to_fp(sound_file)
    #st.audio(sound_file)

    
    
if app_mode=='Image to Speech':
  #st.image('ind.png'),use_column_width=True )
  st.title("Final year project")
  st.image('ind.png')
  st.header('Indian Sign Language Detection')
  st.markdown("Using CNN algorithm and googletrans, the hand sign images are classified and gives the audio as an output.")
  st.text("")
  
  image_file =  st.file_uploader("Upload Images (less than 1mb)", type=["png","jpg","jpeg"])
  if image_file is not None:
    img = Image.open(image_file)
    st.image(image_file,width=250,caption='Uploaded image')
    byte_io = BytesIO()
    img.save(byte_io, 'PNG')#PNG
    image = byte_io.getvalue()


  button_translate=st.button('Click me',help='To give the image')

  if button_translate and image_file :
    class_names= ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    img_height,img_width=180,180
    model = load_model('indian_sign.h5')
    #class_names = model.class_names
    demo_image_path = image_file
    img = tf.keras.utils.load_img(demo_image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    st.text("The hand sign of the above image is : ")
    #st.subheader(class_names[np.argmax(score)])
    word=class_names[np.argmax(score)]
    sound_file = BytesIO()
    tts = gTTS(word)
    tts.write_to_fp(sound_file)
    st.audio(sound_file)
    
  
if app_mode=='Text to Image':
  #st.image('ind.png'),use_column_width=True )
  st.title("Final year project")
  st.image('ind.png')
  st.header('Indian Sign Language Detection')
  st.markdown("Using CNN algorithm, the hand sign images are classified and gives the text as an output.")
  st.text("")
  letter=st.text_input("enter the letter")  
 

  button_translate=st.button('Click me',help='To give the image')

  if button_translate and letter :
        out_path=train_data_dir+"/"+letter+"/7.jpg"
        image = Image.open(out_path)
        st.image(image, caption=letter)
