#!/usr/bin/env python
# coding: utf-8

# In[18]:


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

# In[15]:


def generate_image(model,number,all=False,number1=1):
    """
    Function to load image given image path
    :param model: The GAN model 
    :param image: The image to apply the GAN model on
    :return: generated images
    """

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

# In[8]:


discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.1, beta_2=0.99)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.1, beta_2=0.99)

# Loss function
def genLossFunc(Disc_Output):
     return K.mean(tf.math.log(1 - Disc_Output))
def discLossFunc(real_output,fake_output):
    # Compute the average log(D(x)) and log(1 - D(G(z)))
    real_loss = tf.reduce_mean(tf.math.log(real_output))
    fake_loss = tf.reduce_mean(tf.math.log(1 - fake_output))
    
    # Total discriminator loss
    disc_loss = -0.5*(real_loss + fake_loss)
    return disc_loss

# discriminator.compile(optimizer=discriminator_optimizer, loss=discLossFunc)

# Compile the generator


# In[9]:


generator = tf.keras.models.load_model('Generator/Final_model.h5',compile=False)

generator.compile(optimizer=generator_optimizer, loss=genLossFunc)


# In[16]:




# In[17]:


st.title("Synthetic Data Generator") # Set the title  
# st.image('./horses_zebras.png') # set the featured image of the web application 
option = st.selectbox(
    'Would you like to Generate Every Digit',
    ('Yes', 'No'))
if option=="No":
    number = st.number_input('Insert a number you want to generate?',max_value=9,step=1)
    number1 = st.number_input('How many digits do you want to generate?',min_value=1,step=1)
else:
    number =  np.asarray([x for _ in range(100) for x in range(10)])

if st.button("Generate"): # if the user selected to use the default image
    st.subheader("Synthetic Number")
    images=[]
    if option=="No":
        images = generate_image(generator,int(number),False,number1)
        images = images.reshape(number1,28,28,1)
    else:
        images = generate_image(generator,number,True)
        images = images.reshape(1000,28,28,1)
    st.image(images)
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






