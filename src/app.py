import streamlit as st
import cv2
from detector import model2_detect, model2
import numpy as np
from PIL import Image

st.title("Object Detection with Color Recognition")

uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Run detection
    results = model2_detect(img_array)
    annotated = results.render()[0]
    
    st.image(annotated, caption="Detected Objects")
    
    # Show detection data
    detections = results.xyxy[0]
    st.write(f"Total objects detected: {len(detections)}")
