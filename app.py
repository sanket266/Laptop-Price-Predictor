import streamlit as st
import pickle
import numpy as np
import pandas as pd

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title('Laptop Predictor')

# Brand
Company = st.selectbox('Brand', df['company'].unique())

# Type
Type = st.selectbox('Type', df['typename'].unique())

# RAM
RAM = st.selectbox('RAM(in GB)', [2,4,8,16,32,64]) #int

# Weight
Weight = st.number_input('Weight')

# Touchscreen
Touchscreen = st.selectbox('touchscreen', ['Yes', 'No'])

# ips
ips = st.selectbox('IPS', ['Yes', 'No'])

# Screen size
Screensize = st.number_input('Screen size')

# Resolution
Resolution = st.selectbox('resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

# cpu
CPU = st.selectbox('CPU', df['cpu_name'].unique())

# hdd
HDD = st.selectbox('HDD', [0,128,256,512,1024,2048]) #int

# ssd
SSD = st.selectbox('SSD', [0,128,256,512,1024,2048]) #int

# gpu
GPU = st.selectbox('GPU', df['gpu_name'].unique())

# Operating System
OperatingSystem = st.selectbox('OS', df['ops'].unique())

if st.button('test'):
    st.title('Success')

if st.button('Predict Price'):
    
    # ppi= None

    if Touchscreen == 'Yes':
        Touchscreen = 1
    else:
        Touchscreen = 0
    
    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
    
    x_res = int(Resolution.split('x')[0])
    y_res = int(Resolution.split('x')[1])
    ppi = (((x_res**2) + (y_res**2))**0.5)/Screensize

    query = np.array([Company, Type, RAM, Weight, ips, Touchscreen, ppi, CPU, HDD, SSD, GPU, OperatingSystem], dtype=object)
    query = query.reshape(1,12)
    

    # st.title('The approx price of the Laptop is' + int(np.exp(int(pipe.predict(query)[0]))))
    st.title(f"Approx price of Laptop: {np.exp(pipe.predict(query)[0])}")

    # st.title(type(query))

    # try:
    #     prediction = pipe.predict(query)[0]
    #     st.title(f'The approx price of the Laptop is: {int(np.exp(prediction))} INR')
    # except ValueError as e:
    #     st.error(f"This error occurred: {e}")

# print(type(query[0][2]))

# print('okay')