import streamlit as st 
import os 
import imageio

import tensorflow as tf 
from utils import load_data, num_to_char ,char_to_num 
from modelutil import load_model1, CTCLoss

st.set_page_config(layout='wide')
ffmpeg_path = r'C:\Users\banou\Downloads\ffmpeg-6.1-essentials_build\ffmpeg-6.1-essentials_build\bin\ffmpeg.exe'
with st.sidebar:
    st.image('https://images.unsplash.com/photo-1485115905815-74a5c9fda2f5?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTJ8fHNpZ24lMjBwbGFjZW1lbnR8ZW58MHx8MHx8fDA%3D')
    st.title('Lip Reading')
    st.info('This application helps us to convert video into text.')

options = os. listdir(os.path.join('..','data','s1'))
options.sort()
options.remove('Thumbs.db')
selected_video = st.selectbox('Choose Video',options)

col1, col2= st.columns(2)

if options:
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join(r'C:\Users\banou\Downloads\data', 's1', selected_video)
        os.system(f'{ffmpeg_path} -i {file_path} -vcodec libx264 test_video.mp4 -y')
        # Rendering inside of the app
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)
        
    with col2:

        st. info('This is all the machine learning model sees when making a prediction')
       
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        video = tf.squeeze(video)
        video = (video * 255).numpy().astype('uint8')
        
        imageio.mimsave('animation.gif', video, fps=10) 
        st.image('animation.gif', width=400)

        st.info('This is the output of the machine learning model as tokens')
        model = load_model1()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoded=tf.keras.backend.ctc_decode(yhat,input_length=[75],greedy=True)[0][0].numpy()
        st.text(decoded)
        
        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded]
        st.text(converted_prediction[0].numpy().decode('utf-8'))


