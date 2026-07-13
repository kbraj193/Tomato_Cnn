import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import json


#load the model

model=load_model("tomato_resnet50.keras")

#load class names

with open("class_names.json","r") as f:
    class_indices=json.load(f)
    

class_names = {v: k for k, v in class_indices.items()}
    
st.write(class_names)

def predict( image ):
    image=image.resize((224,224))
    image=np.array(image).astype(np.float32)
    
    image=np.expand_dims(image,axis=0) 
    image=preprocess_input(image)
    
    prediction=model.predict( image,verbose=0 )
    index=np.argmax(prediction)
    disease=class_names[index]
    confidence=np.max(prediction)*100
    return disease,confidence

st.title("Tomato plant disease prediction using Resnet50")

uploaded_file=st.file_uploader(
    "Upload tomato leaf image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:
    
    image=Image.open(uploaded_file).convert("RGB")
    
    #display image
    
    st.image(image,caption="Uploaded Tomato Leaf",widht=300)
    
    # prediction
    
    disease,confidence=predict(image)
    
    # result display
    
    st.success(f"Prediction : {disease}")
    st.info(f"Confidence : {confidence} %")