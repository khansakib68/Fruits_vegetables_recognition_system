import streamlit as st
import tensorflow as tf
import numpy as np
import base64



##################MODEL PREDICTION#################
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model2.h5")
    
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) 
    predictions = model.predict(input_arr)
    
    return np.argmax(predictions) 


st.set_page_config(layout='wide')
st.html('<h1 style="text-align:center ; color:red">FRUITS AND VEGETABLES RECOGNITION SYSTEM</htm>')

test_image =st.file_uploader('Please Choose an Image')


if(test_image is not None and st.button('CLASSIFY')):

    c1,c2,c3= st.columns([1,1,1])
    result_index=model_prediction(test_image)
    
    

    with c1:
        st.info('Your Uploaded Image')
        #if(st.button('Show Image')):
        st.image(test_image,width=1, use_column_width=True)

    with c2:
        st.info('Prediction')
        #if( st.button('Predict')):
           
           #result_index, score=model_prediction(test_image)

        with open('label.txt') as f:
            content =f.readlines()
        label=[]

        for i in content:
            label.append(i[:-1])
        nme = label[result_index].upper()
        st.header( f':orange[{      nme}]') 
    
    with c3:
        st.info('Category')
        with open('label.txt') as f:
               content =f.readlines()
        label=[]

        for i in content:
              label.append(i[:-1])
        if(label[result_index] =='apple' or label[result_index] =='banana' or label[result_index] =='grapes' or label[result_index] =='kiwi' or label[result_index] =='lemon' or label[result_index] =='mango' or label[result_index] =='orange' or label[result_index] =='pear' or label[result_index] =='pineapple' or label[result_index] =='pomegranate' or label[result_index] =='watermelon' ):
            st.html("<h1 style='color: #ff0066 ;text-align:center'>FRUIT</h1>")
        else:
            st.html("<h1 style='color:green ;text-align:center'> VEGETABLE</h1>")

        



custom_css = """
<style>
div.stButton button {
    color:black;
    font-weight: bold;
    font-size:25px;
    border:false;
    background-color: #b3d9ff;
    width: 100%;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

