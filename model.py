#!/usr/bin/env python
# coding: utf-8
# import files
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
import re
from tensorflow.keras.models import Model
import streamlit as st
from PIL import Image
import numpy as np
from numpy import  vstack
import zipfile
import io
#loading the model and compiling it with optimizer and loss function
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.1, beta_2=0.99)

def genLossFunc(Disc_Output):
     return K.mean(tf.math.log(1 - Disc_Output))

generator = tf.keras.models.load_model('Generator/Final_model.keras',compile=False)

generator.compile(optimizer=generator_optimizer, loss=genLossFunc)

#Generating the output
def generate_image(model,number,all=False,number1=1):
    if all==False:
        noise = tf.random.normal([number1,100])    
        number = np.full((number1), number, dtype=int)
        
        generated_image = model.predict([noise,number])
    else:
        noise = tf.random.normal([len(number),100])    

        generated_image = model.predict([noise,number])
    images = vstack(generated_image)
    images = (images+1)/2.0
    return images


st.title("Synthetic Data Generator") # the title  
# asking for the options
option = st.selectbox(
    'Would you like to Generate Every Digit',
    ('Yes', 'No'))
if option=="No":
    number = st.number_input('Insert a number you want to generate?',max_value=9,step=1)
    number1 = st.number_input('How many digits do you want to generate?',min_value=1,step=1)
else:
    number =  np.asarray([x for _ in range(100) for x in range(10)])

if st.button("Generate"): # generate the images
    st.subheader("Synthetic Number")
    images=[]
    if option=="No":
        images = generate_image(generator,int(number),False,number1)
        images = images.reshape(number1,28,28,1)
    else:
        images = generate_image(generator,number,True)
        images = images.reshape(1000,28,28,1)
    st.image(images)
#    Saving the images in a zip and giving the option to download
    images *= 255  # Convert to the range of 0-255
    images  = images.astype(np.uint8) 
    image_files = []
    in_memory_zip = io.BytesIO()
    with zipfile.ZipFile(in_memory_zip, 'w') as zipf:
        for i, image_data in enumerate(images):
            image = Image.fromarray(image_data.squeeze(), mode='L')
            image_file = io.BytesIO()
            image.save(image_file, format='jpeg')
            image_file.seek(0)
            zipf.writestr(f'image_{i}.jpeg', image_file.read())
    btn = st.download_button(
            label="Download image",
            data=in_memory_zip,
            file_name="Images.zip",
            mime="application/zip"
          )






