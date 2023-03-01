#!/usr/bin/env python
# coding: utf-8
pip install keras
import time
import streamlit as st
import numpy as np
import pandas as pd
import keras
import cv2
import os
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import pydicom
from sklearn.svm import SVC
import joblib
import matplotlib.pyplot as plt
from tf_keras_vis.gradcam import Gradcam,GradcamPlusPlus
from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
import tensorflow as tf
import keras.backend as K
#from keras.applications.resnet50 import preprocess_input
#from keras.preprocessing.image import img_to_array
from matplotlib import cm
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import numpy as np
import tensorflow as tf
from tensorflow import keras
try:
    from PIL import Image
except:
    import Image
# Display
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

st.set_page_config(layout="wide")
st.title('COVID-19 Mortality Risk Prediction')
col1, col2, col3, col4 = st.columns((1,1,1,1))



with col1:
    
    # BMI
    BMI = st.number_input(label = 'BMI', min_value = 0.0, step = 0.1)
    # Systolic_blood_pressure
    Systolic_blood_pressure = st.number_input(label = 'Systolic blood pressure (mmHg)', min_value = 0.0, step = 0.1)
    # Diastolic_blood_pressure
    Diastolic_blood_pressure = st.number_input(label = 'Diastolic blood pressure (mmHg)', min_value = 0.0, step = 0.1)
    # Heart Rate
    Heart_Rate = st.number_input(label = 'Heart Rate (bpm)', min_value = 0.0, step = 0.1)
    # Temperature
    Temperature = st.number_input(label = 'Temperature (°C)', min_value = 0.0, step = 0.1)
    # Na
    Na = st.number_input(label = 'Na (mmol/L)', min_value = 0.0, step = 0.1)
    # K
    K = st.number_input(label = 'K (mEq/L)', min_value = 0.0, step = 0.1)
    # Glucose
    Glucose = st.number_input(label = 'Glucose (mg/dL)', min_value = 0.0, step = 0.1)
    # Blood_urea_nitrogen
    Blood_urea_nitrogen = st.number_input(label = 'Blood urea nitrogen (mg/dL)', min_value = 0.0, step = 0.1)
     # D-Dimer
    D_Dimer = st.number_input(label = 'D-Dimer (mg/L FEU)', min_value = 0.0, step = 0.1)

with col2:    
    # WBC
    WBC = st.number_input(label = 'WBC (/ul)', min_value = 0.0, step = 0.1)
    # Neutrophils(SEG)
    SEG = st.number_input(label = 'Neutrophils(SEG) (%)', min_value = 0.0)
    # Lymphocyte_count
    Lymphocyte_count = st.number_input(label = 'Lymphocyte percentage(%)', min_value = 0.0, step = 0.1)
    # BAND
    BAND = st.number_input(label = 'BAND', min_value = 0.0, step = 0.1)
    # Hemoglobin
    Hemoglobin = st.number_input(label = 'Hemoglobin (g/dL)', min_value = 0.0, step = 0.1)
    # Platelet
    Platelet = st.number_input(label = 'Platelet count (x109/L)', min_value = 0.0, step = 0.1)
    # eGFR
    eGFR = st.number_input(label = 'eGFR (mL/min/1.73m2)', min_value = 0.0, step = 0.1)
    # CRE
    CRE = st.number_input(label = 'CRE', min_value = 0.0, step = 0.1)
    # CRP
    CRP = st.number_input(label = 'CRP (mg/dL)', min_value = 0.0, step = 0.1)

with col3:
    # upload X-ray image and return score(1-5)
    uploaded_file = st.file_uploader("請上傳一張X光圖(.jpeg, .dicom)：")
    uploaded_image = []
    if uploaded_file is not None and st.button('Submit'):
        st.write("Loading....")
        image = Image.open(uploaded_file)
        st.image(image, caption='X-ray', width = 420)


with col4:
    st.write('結果：')
    # upload X-ray image and return score(1-5)
    #uploaded_file = st.file_uploader("請上傳一張X光圖(.jpeg, .dicom)：")
    #uploaded_image = []
    # upload X-ray image and return score(1-5)
    # uploaded_file = st.file_uploader("請上傳一張X光圖(.jpeg, .dicom)：")
    uploaded_image = []
    #grad cam function
    def get_img_array(img_path, size):
        # `img` is a PIL image of size 299x299
        img = keras.preprocessing.image.load_img(img_path, target_size=size)
        # `array` is a float32 Numpy array of shape (299, 299, 3)
        array = keras.preprocessing.image.img_to_array(img)
        # We add a dimension to transform our array into a "batch"
        # of size (1, 299, 299, 3)
        array = np.expand_dims(array, axis=0)
        return array


    def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=2.0):        # Load the original image
        img1 = keras.preprocessing.image.load_img(img_path)
        img = keras.preprocessing.image.img_to_array(img1)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
        
        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
        # Display Grad CAM
        # display(Image(cam_path))
        blendImg = Image.blend(img1,superimposed_img,alpha = 0.8)
        st.image(blendImg, caption='X-ray', width = 420) #
        return blendImg
    
    if uploaded_file is not None: # and st.button('Submit'):
        #st.write("Loading....")
        image = Image.open(uploaded_file)
        #st.image(image, caption='X-ray', width = 420)
        
        image = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (299, 299))
        uploaded_image.append(image/255.0)

        model = keras.models.load_model('xception_mse_mse0.9446_val_mse1.1390.h5')
        pred = model.predict(np.array(uploaded_image))
        if pred > 5.0:  pred = 5.0
        if pred < 1.0:  pred = 1.0      

        from matplotlib import cm
        
        dataset = pd.DataFrame([[BAND, CRP, Diastolic_blood_pressure, BMI, Lymphocyte_count, Platelet, Heart_Rate, D_Dimer, Glucose, Systolic_blood_pressure, eGFR, WBC, Na, K, Blood_urea_nitrogen, Hemoglobin, SEG, Temperature, CRE]], 
        columns = ['BAND', 'CRPs', 'DBP', 'BMI', 'LYM.', 'PLT', 'HeartRate', 'D.DIMER', 'GLUs', 'SBP', 'eGFR', 'WBC', 'Na', 'K', 'BUN', 'HB', 'SEG', 'Temperature', 'CRE'])
        
        with open('LightGBM_clinicaldata_AUC0.8554.pickle', 'rb') as f:
            LightGBM = pickle.load(f)
        LightGBM_result = LightGBM.predict_proba(dataset)[:, 1]

        df = pd.DataFrame({"pred":[float(pred)],"LightGBM":[float(LightGBM_result)]})
        clf = joblib.load('LateFusion_LightGBM_SVC_AUC0.9979.pickle')
        SVC_result = clf.predict_proba(df)[:, 1]
        
        st.info("CXR嚴重度(1-5)評分為{:.2f}".format(float(pred)))
        #st.write(result)  # alive within 30 days(0 = false/1 = true)
        st.info("三十天內的存活率為{:.2f}%".format(SVC_result[0]*100))
        
        #grad-cam
        model_builder = keras.applications.xception.Xception
        img_size = (299, 299)
        preprocess_input = keras.applications.xception.preprocess_input
        decode_predictions = keras.applications.xception.decode_predictions

        last_conv_layer_name = "block14_sepconv2_act"
        import pickle
        from keras.models import load_model
        model = load_model('xception_mse_mse0.9446_val_mse1.1390.h5')
        img_path = uploaded_file       
        #st.write(img_path)
        # Prepare image
        img_array = preprocess_input(get_img_array(img_path, size=img_size))
        # Remove last layer's softmax
        model.layers[-1].activation = None
        # Print what the top predicted class is
        preds = model.predict(img_array)
        #print("Predicted:", decode_predictions(preds, top=1)[0])
        # Generate class activation heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        # Display heatmap
        plt.matshow(heatmap)
        # plt.show()
        save_and_display_gradcam(img_path, heatmap)
        #st.image(blendImg, width = 290) #, caption='X-ray'
        
        
        
