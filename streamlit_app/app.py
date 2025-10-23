import streamlit as st
import numpy as np
from PIL import Image

st.title("Simple MNIST Classifier")
st.write("This is a demo interface. Model loading is simulated for demonstration.")

uploaded_file = st.file_uploader("Upload digit image", type=['png', 'jpg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption='Uploaded Digit', width=150)
    
    # Simulate prediction (replace with actual model in final version)
    simulated_prediction = np.random.randint(0, 10)
    confidence = np.random.uniform(0.85, 0.99)
    
    st.success(f"**Prediction**: {simulated_prediction}")
    st.info(f"**Confidence**: {confidence:.1%}")
    st.warning("Note: This is a simulated result for demo purposes")