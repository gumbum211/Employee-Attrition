input_features = np.array([[2,3]]])  # Replace with actual input features from Streamlit UI 

import streamlit as st
import pickle
import numpy as np

# Load the serialized model and scaler
@st.cache_resource
def load_artifacts():
    with open('logistic_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

model, scaler = load_artifacts()

# Example Streamlit UI execution
if st.button("Predict Attrition Risk"):
    # Assuming 'input_features' is captured from Streamlit UI forms
    scaled_data = scaler.transform(input_features)
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1] * 100
    
    st.write(f"Attrition Probability: {probability:.2f}%")

