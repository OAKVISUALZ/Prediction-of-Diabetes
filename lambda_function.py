import streamlit as st
import boto3
import joblib
import os
import numpy as np

# 1. Caching the model so it doesn't download on every click
@st.cache_resource
def load_model_from_s3():
    # It's safer to get these from st.secrets on Streamlit Cloud
    BUCKET_NAME = st.secrets.get("BUCKET_NAME", "my-diabetes-model-bucket")
    MODEL_FILE_NAME = st.secrets.get("MODEL_FILE_NAME", "diabetes_predictor.pkl")
    download_path = f'/tmp/{MODEL_FILE_NAME}'
    
    if not os.path.exists(download_path):
        s3 = boto3.client(
            's3',
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
        s3.download_file(BUCKET_NAME, MODEL_FILE_NAME, download_path)
    
    return joblib.load(download_path)

# --- Streamlit UI ---
st.title("Diabetes Prediction App")
st.write("Enter the patient details below to get a prediction.")

# Form for user inputs
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        preg = st.number_input("Pregnancies", min_value=0)
        gluc = st.number_input("Glucose", min_value=0)
        bp = st.number_input("Blood Pressure", min_value=0)
        skin = st.number_input("Skin Thickness", min_value=0)
    with col2:
        ins = st.number_input("Insulin", min_value=0)
        bmi = st.number_input("BMI", min_value=0.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
        age = st.number_input("Age", min_value=1)
    
    submit = st.form_submit_button("Predict")

if submit:
    try:
        model = load_model_from_s3()
        features = np.array([[preg, gluc, bp, skin, ins, bmi, dpf, age]])
        prediction = model.predict(features)
        
        if prediction[0] == 1:
            st.error("Prediction: Diabetic")
        else:
            st.success("Prediction: Non-Diabetic")
    except Exception as e:
        st.error(f"Error: {e}")
