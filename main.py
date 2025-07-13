import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Detection"])

#Main Page
if(app_mode=="Home"):
    st.header("TOMATO PLANT DISEASE DETECTION SYSTEM")
    image_path = "home_page.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Tomato Plant Disease Detection System! 🌿🔍
    
    Our mission is to help in identifying tomato plant diseases efficiently. Upload an image of a tomato plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Detection** page and upload an image of a tomato plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.


    ### Get Started
    Click on the **Disease Detection** page in the sidebar to upload an image and experience the power of our Tomato Plant Disease Detection System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on Kaggle website.
                This dataset consists of 18456 rgb images of healthy and diseased crop leaves which is categorized into 10 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
               

                """)

#Prediction Page
elif(app_mode=="Disease Detection"):
    st.header("Disease Detection")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = [ 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))